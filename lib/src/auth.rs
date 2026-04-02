//! Authentication strategies for LLM providers.
//!
//! Providers need different auth mechanisms depending on the backend:
//! - **API key** — Azure OpenAI classic (`api-key` header)
//! - **Bearer token** — OpenAI, static tokens (`Authorization: Bearer ...`)
//! - **Dynamic token** — Microsoft Foundry with Entra ID (tokens expire,
//!   need refresh)
//!
//! # Example
//! ```
//! use agentive::AuthStrategy;
//!
//! // Static API key (Azure OpenAI)
//! let auth = AuthStrategy::ApiKey("my-api-key".into());
//!
//! // Static Bearer token (OpenAI)
//! let auth = AuthStrategy::Bearer("sk-...".into());
//!
//! // Dynamic token with refresh (Microsoft Foundry + Entra)
//! use std::sync::{Arc, Mutex};
//! let cached_token = Arc::new(Mutex::new("initial-token".to_string()));
//! let token_ref = cached_token.clone();
//! let auth = AuthStrategy::Dynamic(Arc::new(move || {
//!     token_ref.lock().unwrap().clone()
//! }));
//! ```

use std::sync::Arc;

/// Authentication strategy for LLM API requests.
///
/// Determines how the `Authorization` or `api-key` header is set.
#[derive(Clone)]
pub enum AuthStrategy {
    /// Azure-style API key — sent as `api-key: <key>` header.
    ApiKey(String),
    /// Bearer token — sent as `Authorization: Bearer <token>`.
    /// Use for OpenAI direct, or static tokens.
    Bearer(String),
    /// Dynamic token provider — called before each request.
    /// Use for Entra ID / OAuth tokens that expire and need refresh.
    /// The closure should return a valid Bearer token string.
    /// Token refresh/caching is the caller's responsibility.
    Dynamic(Arc<dyn Fn() -> String + Send + Sync>),
}

impl AuthStrategy {
    /// Apply this auth strategy to a reqwest RequestBuilder.
    pub(crate) fn apply(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self {
            AuthStrategy::ApiKey(key) => req.header("api-key", key),
            AuthStrategy::Bearer(token) => req.bearer_auth(token),
            AuthStrategy::Dynamic(provider) => req.bearer_auth(provider()),
        }
    }
}

impl std::fmt::Debug for AuthStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthStrategy::ApiKey(_) => write!(f, "AuthStrategy::ApiKey(...)"),
            AuthStrategy::Bearer(_) => write!(f, "AuthStrategy::Bearer(...)"),
            AuthStrategy::Dynamic(_) => write!(f, "AuthStrategy::Dynamic(...)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_debug() {
        let auth = AuthStrategy::ApiKey("secret".into());
        let debug = format!("{:?}", auth);
        assert!(!debug.contains("secret")); // shouldn't leak the key
        assert!(debug.contains("ApiKey"));
    }

    #[test]
    fn test_bearer_debug() {
        let auth = AuthStrategy::Bearer("sk-secret".into());
        let debug = format!("{:?}", auth);
        assert!(!debug.contains("sk-secret"));
    }

    #[test]
    fn test_dynamic_calls_closure() {
        use std::sync::Mutex;
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        let auth = AuthStrategy::Dynamic(Arc::new(move || {
            let mut c = counter_clone.lock().unwrap();
            *c += 1;
            format!("token-{}", c)
        }));

        // Each call should invoke the closure
        if let AuthStrategy::Dynamic(ref f) = auth {
            assert_eq!(f(), "token-1");
            assert_eq!(f(), "token-2");
        }
        assert_eq!(*counter.lock().unwrap(), 2);
    }

    #[test]
    fn test_clone_dynamic() {
        let auth = AuthStrategy::Dynamic(Arc::new(|| "token".to_string()));
        let auth2 = auth.clone();
        if let AuthStrategy::Dynamic(ref f) = auth2 {
            assert_eq!(f(), "token");
        }
    }
}
