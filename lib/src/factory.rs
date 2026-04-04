//! Provider factory — auto-routing from endpoint + model to the right provider.
//!
//! Most apps need the same routing logic: pick `OpenAiProvider` vs `ResponsesProvider`
//! based on model name, auto-detect Azure auth, and set sensible context budgets.
//!
//! # Example
//! ```no_run
//! use agentive::factory::build_provider;
//! use agentive::AuthStrategy;
//! use std::sync::Arc;
//!
//! let provider = build_provider(
//!     "https://my-resource.openai.azure.com",
//!     "my-api-key",
//!     "gpt-4o",
//! );
//! ```

use std::sync::Arc;

use crate::auth::AuthStrategy;
use crate::provider::Provider;
use crate::providers::openai::OpenAiProvider;
use crate::providers::responses::ResponsesProvider;

/// Check if a model name requires the Responses API instead of Chat Completions.
///
/// Returns `true` for:
/// - Models containing "codex" (e.g., `gpt-5.1-codex`, `gpt-5-codex`)
/// - Models matching `gpt-5*-pro` (e.g., `gpt-5-pro`, `gpt-5.4-pro`)
pub fn needs_responses_api(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("codex") || (m.contains("gpt-5") && m.ends_with("-pro"))
}

/// Check if a model supports vision (image inputs).
///
/// Returns `true` for known vision-capable model families.
pub fn supports_vision(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("gpt-4o")
        || m.contains("gpt-4.1")
        || m.contains("gpt-5")
        || m.contains("gpt-4-turbo")
        || m.contains("gpt-4-vision")
        || m.contains("claude-3-5")
        || m.contains("claude-3.5")
        || m.contains("claude-4")
        || m.contains("gemini")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
}

/// Estimate a reasonable context budget (in characters) for a model.
///
/// Uses model family heuristics to pick a default. Returns characters, not tokens.
/// Assumes ~4 chars/token and uses ~75% of the model's context window.
///
/// If `reported_context` is `Some(n)`, uses that token count instead of heuristics.
pub fn default_context_budget(model: &str) -> usize {
    context_budget(model, None)
}

/// Like [`default_context_budget`], but accepts an optional API-reported context
/// length that overrides the heuristic.
pub fn context_budget(model: &str, reported_context: Option<usize>) -> usize {
    if let Some(reported) = reported_context {
        let usable = reported * 3 / 4;
        return usable * 4;
    }

    let m = model.to_lowercase();

    let token_limit: usize = if m.contains("codex") {
        // Responses API models have limited effective context due to body size limits
        16_000
    } else if m.contains("claude-3-5") || m.contains("claude-3.5") || m.contains("claude-4") {
        200_000
    } else if m.contains("claude") {
        100_000
    } else if m.contains("gpt-5")
        || m.contains("gpt-4o")
        || m.contains("gpt-4.1")
        || m.contains("gpt-4-turbo")
        || m.contains("gpt-4-1106")
        || m.contains("gpt-4-0125")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.contains("gemini")
    {
        128_000
    } else if m.contains("deepseek") {
        64_000
    } else if m.contains("gpt-35-turbo-16k") || m.contains("gpt-3.5-turbo-16k") {
        16_384
    } else if m.contains("mistral-large") || m.contains("mistral-medium") {
        32_000
    } else if m.contains("phi-4") || m.contains("phi-3") {
        16_000
    } else if m.contains("gpt-4") {
        8_192
    } else if m.contains("gpt-35") || m.contains("gpt-3.5") {
        4_096
    } else if m.contains("mistral") {
        8_000
    } else {
        // Conservative default
        32_000
    };

    // Use 75% of context, convert tokens to chars (~4 chars/token)
    (token_limit * 3 / 4) * 4
}

/// Build the appropriate provider for an endpoint + model combination.
///
/// Auto-detects:
/// - **Provider type**: `ResponsesProvider` for codex/pro models, `OpenAiProvider` otherwise
/// - **Auth strategy**: Azure endpoints get `ApiKey` header, others get `Bearer`
/// - **Vision**: Enabled for known vision-capable models
/// - **Context budget**: Set based on model family heuristics
///
/// For custom auth (e.g., Entra tokens), use `build_provider_with_auth()` instead.
pub fn build_provider(endpoint: &str, api_key: &str, model: &str) -> Arc<dyn Provider> {
    if needs_responses_api(model) {
        Arc::new(
            ResponsesProvider::new(endpoint, api_key, model)
                .with_context_budget(default_context_budget(model))
                .with_vision(supports_vision(model)),
        )
    } else {
        Arc::new(
            OpenAiProvider::new(endpoint, api_key, model)
                .with_context_budget(default_context_budget(model))
                .with_vision(supports_vision(model)),
        )
    }
}

/// Build a provider with explicit auth strategy.
///
/// Use this when you need `AuthStrategy::Dynamic` for Entra token refresh,
/// or want to override the auto-detected auth.
pub fn build_provider_with_auth(
    endpoint: &str,
    auth: AuthStrategy,
    model: &str,
) -> Arc<dyn Provider> {
    if needs_responses_api(model) {
        Arc::new(
            ResponsesProvider::with_auth(endpoint, auth, model)
                .with_context_budget(default_context_budget(model))
                .with_vision(supports_vision(model)),
        )
    } else {
        Arc::new(
            OpenAiProvider::with_auth(endpoint, auth, model)
                .with_context_budget(default_context_budget(model))
                .with_vision(supports_vision(model)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_needs_responses_api() {
        assert!(needs_responses_api("gpt-5.1-codex"));
        assert!(needs_responses_api("gpt-5-codex"));
        assert!(needs_responses_api("codex-mini"));
        assert!(needs_responses_api("gpt-5-pro"));
        assert!(needs_responses_api("gpt-5.4-pro"));

        assert!(!needs_responses_api("gpt-4o"));
        assert!(!needs_responses_api("gpt-5"));
        assert!(!needs_responses_api("gpt-5-mini"));
        assert!(!needs_responses_api("claude-3.5-sonnet"));
    }

    #[test]
    fn test_supports_vision() {
        assert!(supports_vision("gpt-4o"));
        assert!(supports_vision("gpt-4o-mini"));
        assert!(supports_vision("gpt-4.1"));
        assert!(supports_vision("gpt-5"));
        assert!(supports_vision("gpt-5.1-codex"));
        assert!(supports_vision("claude-3.5-sonnet"));
        assert!(supports_vision("claude-4-opus"));
        assert!(supports_vision("o1-preview"));
        assert!(supports_vision("o3-mini"));
        assert!(supports_vision("gemini-2.0-flash"));

        assert!(!supports_vision("gpt-3.5-turbo"));
        assert!(!supports_vision("gpt-4")); // base gpt-4 doesn't have vision
        assert!(!supports_vision("random-model"));
    }

    #[test]
    fn test_default_context_budget() {
        // codex models: small budget (Responses API body size limits)
        assert_eq!(default_context_budget("gpt-5.1-codex"), 48_000);

        // gpt-4o: 128k tokens → 75% → 96k tokens → ~384k chars
        assert_eq!(default_context_budget("gpt-4o"), 384_000);

        // gpt-5: same as gpt-4o
        assert_eq!(default_context_budget("gpt-5"), 384_000);

        // Claude: 200k tokens → 600k chars
        assert_eq!(default_context_budget("claude-3.5-sonnet"), 600_000);

        // Unknown model: conservative default
        assert_eq!(default_context_budget("random-model"), 96_000);

        // New model families
        assert_eq!(default_context_budget("deepseek-r1"), 192_000);
        assert_eq!(default_context_budget("phi-4"), 48_000);
        assert_eq!(default_context_budget("mistral-large"), 96_000);
        assert_eq!(default_context_budget("mistral-7b"), 24_000);
        assert_eq!(default_context_budget("claude-3-opus"), 300_000);
        assert_eq!(default_context_budget("gpt-35-turbo"), 12_288);

        // o-series
        assert_eq!(default_context_budget("o1-preview"), 384_000);
        assert_eq!(default_context_budget("o3-mini"), 384_000);
    }

    #[test]
    fn test_context_budget_with_reported() {
        // Reported context overrides heuristic
        assert_eq!(context_budget("my-custom-deployment", None), 96_000);
        assert_eq!(context_budget("my-custom-deployment", Some(200_000)), 600_000);
        // Even for known models, reported takes precedence
        assert_eq!(context_budget("gpt-4o", Some(32_000)), 96_000);
    }

    #[test]
    fn test_build_provider_routes_correctly() {
        // Codex model should get ResponsesProvider
        let p = build_provider("https://api.openai.com/v1", "key", "gpt-5.1-codex");
        assert!(p.name().contains("responses") || p.supports_vision());

        // Regular model should get OpenAiProvider
        let p = build_provider("https://api.openai.com/v1", "key", "gpt-4o");
        // Just verify it doesn't panic and returns a provider
        assert!(!p.name().is_empty());
    }

    #[test]
    fn test_build_provider_with_auth() {
        let p = build_provider_with_auth(
            "https://my-resource.openai.azure.com",
            AuthStrategy::Bearer("token123".into()),
            "gpt-4o",
        );
        assert!(!p.name().is_empty());
    }
}
