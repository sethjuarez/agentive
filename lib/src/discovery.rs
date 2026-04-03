//! Azure / OpenAI model and deployment discovery.
//!
//! Lists available models or deployments from an endpoint, handling the
//! different response formats across Azure OpenAI, Azure AI Foundry,
//! and standard OpenAI.
//!
//! # Example
//! ```no_run
//! # async fn example() -> Result<(), String> {
//! use agentive::{AuthStrategy, discovery};
//!
//! let models = discovery::list_models(
//!     "https://my-resource.openai.azure.com",
//!     &AuthStrategy::ApiKey("my-key".into()),
//! ).await?;
//!
//! for m in &models {
//!     println!("{} ({})", m.id, m.owned_by.as_deref().unwrap_or("unknown"));
//! }
//! # Ok(())
//! # }
//! ```

use crate::auth::AuthStrategy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Information about an available model or deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model or deployment ID (what you pass as the model name).
    pub id: String,
    /// The underlying model name (for deployments that wrap a model).
    #[serde(default)]
    pub owned_by: Option<String>,
    /// Capability flags (e.g., `"chat_completion": "true"`).
    #[serde(default)]
    pub capabilities: Option<HashMap<String, String>>,
    /// Max context window in tokens, if reported.
    #[serde(default)]
    pub context_length: Option<usize>,
}

// ---------------------------------------------------------------------------
// Response shapes (private)
// ---------------------------------------------------------------------------

/// Standard OpenAI format: `{ "data": [{ "id": "gpt-4o", ... }] }`
#[derive(Debug, Deserialize)]
struct OpenAiModelsResponse {
    #[serde(default)]
    data: Vec<OpenAiModel>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModel {
    id: String,
    #[serde(default)]
    owned_by: Option<String>,
}

/// Azure OpenAI deployments: `{ "data": [{ "id": "my-gpt4", "model": "gpt-4" }] }`
#[derive(Debug, Deserialize)]
struct AzureDeploymentsResponse {
    #[serde(default)]
    data: Vec<AzureDeployment>,
}

#[derive(Debug, Deserialize)]
struct AzureDeployment {
    id: String,
    #[serde(default)]
    model: Option<String>,
}

/// Foundry catalog: `{ "value": [{ "name": "gpt-4o", "modelName": "...", "capabilities": {...} }] }`
#[derive(Debug, Deserialize)]
struct FoundryCatalogResponse {
    #[serde(default)]
    value: Vec<FoundryCatalogEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FoundryCatalogEntry {
    name: String,
    #[serde(default)]
    model_name: Option<String>,
    #[serde(default)]
    capabilities: Option<HashMap<String, String>>,
}

/// Foundry project deployments: `{ "value": [{ "name": "...", "properties": { "model": { "name": "..." } } }] }`
#[derive(Debug, Deserialize)]
struct FoundryProjectDeploymentsResponse {
    #[serde(default)]
    value: Vec<FoundryProjectDeployment>,
}

#[derive(Debug, Deserialize)]
struct FoundryProjectDeployment {
    name: String,
    #[serde(default)]
    properties: Option<FoundryDeploymentProperties>,
}

#[derive(Debug, Deserialize)]
struct FoundryDeploymentProperties {
    #[serde(default)]
    model: Option<FoundryDeploymentModel>,
}

#[derive(Debug, Deserialize)]
struct FoundryDeploymentModel {
    name: String,
}

// ---------------------------------------------------------------------------
// Endpoint detection & normalization
// ---------------------------------------------------------------------------

/// Returns true if the endpoint looks like Azure AI Foundry.
fn is_foundry(endpoint: &str) -> bool {
    endpoint.contains(".services.ai.azure.com")
}

/// Returns true if the endpoint looks like Azure OpenAI (not Foundry).
fn is_azure_openai(endpoint: &str) -> bool {
    endpoint.contains(".openai.azure.com") || endpoint.contains(".cognitiveservices.azure.com")
}

/// Rewrite `*.cognitiveservices.azure.com` → `*.openai.azure.com`.
///
/// The ARM API often returns the generic Cognitive Services endpoint, but the
/// `/openai/deployments` API only lives on the `openai.azure.com` host.
fn normalize_azure_endpoint(endpoint: &str) -> String {
    endpoint.replace(".cognitiveservices.azure.com", ".openai.azure.com")
}

/// Strip any `/api/projects/...` suffix from a Foundry endpoint.
fn foundry_base(endpoint: &str) -> &str {
    let base = endpoint.trim_end_matches('/');
    if let Some(idx) = base.find("/api/projects") {
        &base[..idx]
    } else {
        base
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// List available models/deployments from the given endpoint.
///
/// Automatically detects the endpoint type (Azure OpenAI, Foundry, or OpenAI)
/// and calls the appropriate API.
pub async fn list_models(
    endpoint: &str,
    auth: &AuthStrategy,
) -> Result<Vec<ModelInfo>, String> {
    let endpoint = endpoint.trim_end_matches('/');

    if is_foundry(endpoint) {
        list_models_foundry(endpoint, auth).await
    } else if is_azure_openai(endpoint) {
        list_models_azure(endpoint, auth).await
    } else {
        list_models_openai(endpoint, auth).await
    }
}

/// Azure OpenAI: GET /openai/deployments?api-version=2024-10-21
async fn list_models_azure(
    endpoint: &str,
    auth: &AuthStrategy,
) -> Result<Vec<ModelInfo>, String> {
    let endpoint = normalize_azure_endpoint(endpoint);
    let url = format!("{endpoint}/openai/deployments?api-version=2024-10-21");
    let body = fetch_body(&url, auth).await?;
    parse_azure_deployments(&body)
}

/// Standard OpenAI: GET /v1/models
async fn list_models_openai(
    endpoint: &str,
    auth: &AuthStrategy,
) -> Result<Vec<ModelInfo>, String> {
    let base = if endpoint.is_empty() {
        "https://api.openai.com"
    } else {
        endpoint
    };
    let url = format!("{}/v1/models", base);
    let body = fetch_body(&url, auth).await?;
    parse_openai_models(&body)
}

/// Foundry: try project deployments first, then catalog.
async fn list_models_foundry(
    endpoint: &str,
    auth: &AuthStrategy,
) -> Result<Vec<ModelInfo>, String> {
    // Try project-level deployments first (small, deployed-only list)
    if endpoint.contains("/api/projects") {
        let url = format!("{}/deployments?api-version=v1", endpoint);
        if let Ok(body) = fetch_body(&url, auth).await {
            if let Ok(models) = parse_foundry_project_deployments(&body) {
                if !models.is_empty() {
                    return Ok(models);
                }
            }
        }
    }

    // Fallback: catalog (filtered to chat-capable)
    let base = foundry_base(endpoint);
    let url = format!("{}/openai/models?api-version=2024-10-21", base);
    let body = fetch_body(&url, auth).await?;
    parse_foundry_catalog(&body, true)
}

// ---------------------------------------------------------------------------
// HTTP + parsing helpers
// ---------------------------------------------------------------------------

async fn fetch_body(url: &str, auth: &AuthStrategy) -> Result<String, String> {
    let http = reqwest::Client::new();
    let req = http
        .get(url)
        .timeout(std::time::Duration::from_secs(30));
    let req = auth.apply(req);

    let resp = req
        .send()
        .await
        .map_err(|e| format!("Failed to list models: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Model list failed ({status}) at {url}: {body}"));
    }

    resp.text()
        .await
        .map_err(|e| format!("Failed to read models response: {e}"))
}

fn parse_openai_models(body: &str) -> Result<Vec<ModelInfo>, String> {
    let parsed: OpenAiModelsResponse =
        serde_json::from_str(body).map_err(|e| format!("Failed to parse models: {e}"))?;
    Ok(parsed
        .data
        .into_iter()
        .map(|m| ModelInfo {
            id: m.id,
            owned_by: m.owned_by,
            capabilities: None,
            context_length: None,
        })
        .collect())
}

fn parse_azure_deployments(body: &str) -> Result<Vec<ModelInfo>, String> {
    // Try the Azure format first, fall back to OpenAI format
    if let Ok(parsed) = serde_json::from_str::<AzureDeploymentsResponse>(body) {
        if !parsed.data.is_empty() {
            return Ok(parsed
                .data
                .into_iter()
                .map(|d| ModelInfo {
                    id: d.id,
                    owned_by: d.model,
                    capabilities: None,
                    context_length: None,
                })
                .collect());
        }
    }
    // Some Azure endpoints return OpenAI format
    parse_openai_models(body)
}

fn parse_foundry_catalog(body: &str, filter_chat: bool) -> Result<Vec<ModelInfo>, String> {
    let parsed: FoundryCatalogResponse =
        serde_json::from_str(body).map_err(|e| format!("Failed to parse Foundry catalog: {e}"))?;
    Ok(parsed
        .value
        .into_iter()
        .filter(|m| {
            if !filter_chat {
                return true;
            }
            m.capabilities
                .as_ref()
                .and_then(|c| c.get("chat_completion"))
                .map(|v| v == "true")
                .unwrap_or(true)
        })
        .map(|m| ModelInfo {
            id: m.name,
            owned_by: m.model_name,
            capabilities: m.capabilities,
            context_length: None,
        })
        .collect())
}

fn parse_foundry_project_deployments(body: &str) -> Result<Vec<ModelInfo>, String> {
    let parsed: FoundryProjectDeploymentsResponse = serde_json::from_str(body)
        .map_err(|e| format!("Failed to parse Foundry deployments: {e}"))?;
    Ok(parsed
        .value
        .into_iter()
        .map(|d| ModelInfo {
            id: d.name,
            owned_by: d.properties.and_then(|p| p.model).map(|m| m.name),
            capabilities: None,
            context_length: None,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_foundry() {
        assert!(is_foundry("https://my-project.services.ai.azure.com/api/projects/proj1"));
        assert!(!is_foundry("https://my-resource.openai.azure.com"));
        assert!(!is_foundry("https://api.openai.com"));
    }

    #[test]
    fn test_is_azure_openai() {
        assert!(is_azure_openai("https://my-resource.openai.azure.com"));
        assert!(is_azure_openai("https://my-resource.cognitiveservices.azure.com"));
        assert!(!is_azure_openai("https://my-project.services.ai.azure.com"));
        assert!(!is_azure_openai("https://api.openai.com"));
    }

    #[test]
    fn test_foundry_base() {
        assert_eq!(
            foundry_base("https://host.services.ai.azure.com/api/projects/p1"),
            "https://host.services.ai.azure.com"
        );
        assert_eq!(
            foundry_base("https://host.services.ai.azure.com"),
            "https://host.services.ai.azure.com"
        );
        assert_eq!(
            foundry_base("https://host.services.ai.azure.com/"),
            "https://host.services.ai.azure.com"
        );
    }

    #[test]
    fn test_parse_openai_models() {
        let body = r#"{"data":[{"id":"gpt-4o","owned_by":"openai"},{"id":"gpt-3.5-turbo","owned_by":"openai"}]}"#;
        let models = parse_openai_models(body).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gpt-4o");
        assert_eq!(models[1].id, "gpt-3.5-turbo");
    }

    #[test]
    fn test_parse_azure_deployments() {
        let body = r#"{"data":[{"id":"my-gpt4","model":"gpt-4"},{"id":"my-embed","model":"text-embedding-ada-002"}]}"#;
        let models = parse_azure_deployments(body).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "my-gpt4");
        assert_eq!(models[0].owned_by.as_deref(), Some("gpt-4"));
    }

    #[test]
    fn test_parse_foundry_catalog_all() {
        let body = r#"{"value":[
            {"name":"gpt-4o","modelName":"gpt-4o","capabilities":{"chat_completion":"true"}},
            {"name":"text-embedding","modelName":"embed","capabilities":{"embeddings":"true"}}
        ]}"#;
        let models = parse_foundry_catalog(body, false).unwrap();
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn test_parse_foundry_catalog_chat_only() {
        let body = r#"{"value":[
            {"name":"gpt-4o","modelName":"gpt-4o","capabilities":{"chat_completion":"true"}},
            {"name":"text-embedding","modelName":"embed","capabilities":{"chat_completion":"false","embeddings":"true"}}
        ]}"#;
        let models = parse_foundry_catalog(body, true).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gpt-4o");
    }

    #[test]
    fn test_parse_foundry_project_deployments() {
        let body = r#"{"value":[
            {"name":"my-gpt4","properties":{"model":{"name":"gpt-4o"}}},
            {"name":"my-embed","properties":{"model":{"name":"text-embedding-3-small"}}}
        ]}"#;
        let models = parse_foundry_project_deployments(body).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "my-gpt4");
        assert_eq!(models[0].owned_by.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn test_parse_empty_responses() {
        assert!(parse_openai_models(r#"{"data":[]}"#).unwrap().is_empty());
        assert!(parse_foundry_project_deployments(r#"{"value":[]}"#).unwrap().is_empty());
    }

    #[test]
    fn test_normalize_azure_endpoint_cognitiveservices() {
        assert_eq!(
            normalize_azure_endpoint("https://my-resource.cognitiveservices.azure.com"),
            "https://my-resource.openai.azure.com"
        );
    }

    #[test]
    fn test_normalize_azure_endpoint_with_trailing_slash() {
        assert_eq!(
            normalize_azure_endpoint("https://my-resource.cognitiveservices.azure.com/"),
            "https://my-resource.openai.azure.com/"
        );
    }

    #[test]
    fn test_normalize_azure_endpoint_already_openai() {
        // Should be a no-op when it's already openai.azure.com
        assert_eq!(
            normalize_azure_endpoint("https://my-resource.openai.azure.com"),
            "https://my-resource.openai.azure.com"
        );
    }

    #[test]
    fn test_normalize_azure_endpoint_non_azure() {
        // Should be a no-op for non-Azure endpoints
        assert_eq!(
            normalize_azure_endpoint("https://api.openai.com"),
            "https://api.openai.com"
        );
    }
}
