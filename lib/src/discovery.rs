//! Model and deployment discovery across OpenAI-compatible, Azure, Foundry, and Anthropic endpoints.
//!
//! Lists available models or deployments from an endpoint, handling the
//! different response formats across Azure OpenAI, Azure AI Foundry,
//! standard OpenAI, and Anthropic.
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
use crate::factory::{context_budget, needs_responses_api, supports_vision};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const ANTHROPIC_VERSION: &str = "2023-06-01";

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

/// Preferred Anthropic family to select from discovered models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnthropicModelTier {
    /// Prefer Claude Sonnet models.
    Sonnet,
    /// Prefer Claude Opus models.
    Opus,
    /// Prefer Claude Haiku models.
    Haiku,
    /// Pick the first available Anthropic model.
    Any,
}

/// Result of validating a persisted model ID against a live discovery result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSelection {
    /// Model ID from caller settings.
    pub requested: String,
    /// Model ID to use now.
    #[serde(default)]
    pub selected: Option<String>,
    /// True when `requested` was present in the live model list.
    pub available: bool,
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

/// Anthropic Models API: `{ "data": [{ "id": "...", ... }], "has_more": true, "last_id": "..." }`
#[derive(Debug, Deserialize)]
struct AnthropicModelsResponse {
    #[serde(default)]
    data: Vec<AnthropicModel>,
    #[serde(default)]
    has_more: bool,
    #[serde(default)]
    last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicModel {
    id: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    max_input_tokens: Option<usize>,
    #[serde(default)]
    capabilities: Option<AnthropicCapabilities>,
}

#[derive(Debug, Deserialize)]
struct AnthropicCapabilities {
    #[serde(default)]
    image_input: Option<AnthropicCapabilitySupport>,
}

#[derive(Debug, Deserialize)]
struct AnthropicCapabilitySupport {
    #[serde(default)]
    supported: bool,
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

/// Returns true if the endpoint looks like Anthropic.
fn is_anthropic(endpoint: &str) -> bool {
    endpoint.contains("anthropic.com")
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

fn versionless_base(endpoint: &str) -> &str {
    endpoint.trim_end_matches('/').trim_end_matches("/v1")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// List available models/deployments from the given endpoint.
///
/// Automatically detects the endpoint type (Azure OpenAI, Foundry, Anthropic, or OpenAI)
/// and calls the appropriate API.
pub async fn list_models(endpoint: &str, auth: &AuthStrategy) -> Result<Vec<ModelInfo>, String> {
    let endpoint = endpoint.trim_end_matches('/');

    if is_foundry(endpoint) {
        list_models_foundry(endpoint, auth).await
    } else if is_azure_openai(endpoint) {
        list_models_azure(endpoint, auth).await
    } else if is_anthropic(endpoint) {
        list_models_anthropic(endpoint, auth).await
    } else {
        list_models_openai(endpoint, auth).await
    }
}

/// Return Agentive's recommended OpenAI default from a live model list.
///
/// OpenAI's `/v1/models` response does not include a provider default flag, so
/// this is intentionally an Agentive-owned policy layered over live availability.
pub fn recommended_openai_model(models: &[ModelInfo]) -> Option<String> {
    recommended_by_preference(
        models,
        &["gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o", "gpt-4-turbo"],
    )
    .or_else(|| {
        models
            .iter()
            .find(|m| {
                capability_is_true(m, "chat_completion")
                    && !m.id.to_lowercase().contains("codex")
                    && !m.id.to_lowercase().ends_with("-pro")
            })
            .map(|m| m.id.clone())
    })
}

/// Return Agentive's recommended Anthropic default, preferring Sonnet.
pub fn recommended_anthropic_default_model(models: &[ModelInfo]) -> Option<String> {
    recommended_anthropic_model(models, AnthropicModelTier::Sonnet)
        .or_else(|| recommended_anthropic_model(models, AnthropicModelTier::Any))
}

/// Return Agentive's recommended Anthropic model for a family/tier.
///
/// Anthropic's Models API returns newer models first and does not expose a
/// default flag, so this preserves response ordering within the requested tier.
pub fn recommended_anthropic_model(
    models: &[ModelInfo],
    tier: AnthropicModelTier,
) -> Option<String> {
    models
        .iter()
        .find(|m| match tier {
            AnthropicModelTier::Sonnet => m.id.contains("sonnet"),
            AnthropicModelTier::Opus => m.id.contains("opus"),
            AnthropicModelTier::Haiku => m.id.contains("haiku"),
            AnthropicModelTier::Any => true,
        })
        .map(|m| m.id.clone())
}

/// Validate a saved model ID and select a replacement when it is stale.
pub fn validate_or_recommend_model(
    saved_model: &str,
    models: &[ModelInfo],
    recommended: Option<String>,
) -> ModelSelection {
    if models.iter().any(|m| m.id == saved_model) {
        return ModelSelection {
            requested: saved_model.to_string(),
            selected: Some(saved_model.to_string()),
            available: true,
        };
    }

    ModelSelection {
        requested: saved_model.to_string(),
        selected: recommended.or_else(|| models.first().map(|m| m.id.clone())),
        available: false,
    }
}

/// Azure OpenAI: GET /openai/deployments?api-version=2024-10-21
async fn list_models_azure(endpoint: &str, auth: &AuthStrategy) -> Result<Vec<ModelInfo>, String> {
    let endpoint = normalize_azure_endpoint(endpoint);
    let url = format!("{endpoint}/openai/deployments?api-version=2024-10-21");
    let body = fetch_body(&url, auth).await?;
    parse_azure_deployments(&body)
}

/// Standard OpenAI: GET /v1/models
async fn list_models_openai(endpoint: &str, auth: &AuthStrategy) -> Result<Vec<ModelInfo>, String> {
    let base = if endpoint.is_empty() {
        "https://api.openai.com"
    } else {
        versionless_base(endpoint)
    };
    let url = format!("{}/v1/models", base);
    let body = fetch_body(&url, auth).await?;
    parse_openai_models(&body)
}

/// Anthropic: GET /v1/models?limit=1000
async fn list_models_anthropic(
    endpoint: &str,
    auth: &AuthStrategy,
) -> Result<Vec<ModelInfo>, String> {
    let base = if endpoint.is_empty() {
        "https://api.anthropic.com"
    } else {
        versionless_base(endpoint)
    };

    let mut after_id: Option<String> = None;
    let mut models = Vec::new();

    loop {
        let url = match &after_id {
            Some(after) => format!(
                "{base}/v1/models?limit=1000&after_id={}",
                urlencoding::encode(after)
            ),
            None => format!("{base}/v1/models?limit=1000"),
        };
        let body = fetch_body_anthropic(&url, auth).await?;
        let (mut page, next_after_id) = parse_anthropic_models(&body)?;
        models.append(&mut page);

        if let Some(next) = next_after_id {
            after_id = Some(next);
        } else {
            break;
        }
    }

    Ok(models)
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
    let req = http.get(url).timeout(std::time::Duration::from_secs(30));
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

async fn fetch_body_anthropic(url: &str, auth: &AuthStrategy) -> Result<String, String> {
    let http = reqwest::Client::new();
    let req = http
        .get(url)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .timeout(std::time::Duration::from_secs(30));
    let req = match auth {
        AuthStrategy::ApiKey(key) | AuthStrategy::Bearer(key) => req.header("x-api-key", key),
        AuthStrategy::Dynamic(provider) => req.header("x-api-key", provider()),
    };

    let resp = req
        .send()
        .await
        .map_err(|e| format!("Failed to list Anthropic models: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!(
            "Anthropic model list failed ({status}) at {url}: {body}"
        ));
    }

    resp.text()
        .await
        .map_err(|e| format!("Failed to read Anthropic models response: {e}"))
}

fn parse_openai_models(body: &str) -> Result<Vec<ModelInfo>, String> {
    let parsed: OpenAiModelsResponse =
        serde_json::from_str(body).map_err(|e| format!("Failed to parse models: {e}"))?;
    Ok(parsed
        .data
        .into_iter()
        .map(|m| enrich_openai_model(m.id, m.owned_by))
        .collect())
}

fn parse_anthropic_models(body: &str) -> Result<(Vec<ModelInfo>, Option<String>), String> {
    let parsed: AnthropicModelsResponse =
        serde_json::from_str(body).map_err(|e| format!("Failed to parse Anthropic models: {e}"))?;
    let next_after_id = if parsed.has_more {
        parsed.last_id.clone()
    } else {
        None
    };

    Ok((
        parsed
            .data
            .into_iter()
            .map(enrich_anthropic_model)
            .collect(),
        next_after_id,
    ))
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

fn enrich_openai_model(id: String, owned_by: Option<String>) -> ModelInfo {
    let context_length = estimated_context_tokens(&id);
    ModelInfo {
        capabilities: Some(openai_capabilities(&id, context_length)),
        context_length: Some(context_length),
        id,
        owned_by,
    }
}

fn enrich_anthropic_model(model: AnthropicModel) -> ModelInfo {
    let mut capabilities = HashMap::new();
    capabilities.insert("chat_completion".to_string(), "true".to_string());
    capabilities.insert("tool_calling".to_string(), "true".to_string());
    capabilities.insert("streaming".to_string(), "true".to_string());
    capabilities.insert("structured_outputs".to_string(), "false".to_string());
    capabilities.insert("reasoning_effort".to_string(), "false".to_string());

    if let Some(display_name) = model.display_name {
        capabilities.insert("display_name".to_string(), display_name);
    }
    if let Some(created_at) = model.created_at {
        capabilities.insert("created_at".to_string(), created_at);
    }

    let vision = model
        .capabilities
        .and_then(|c| c.image_input)
        .map(|image| image.supported)
        .unwrap_or_else(|| supports_vision(&model.id));
    capabilities.insert("vision".to_string(), vision.to_string());

    ModelInfo {
        id: model.id,
        owned_by: Some("Anthropic".to_string()),
        capabilities: Some(capabilities),
        context_length: model.max_input_tokens,
    }
}

fn openai_capabilities(model: &str, context_length: usize) -> HashMap<String, String> {
    let mut capabilities = HashMap::new();
    let chat = is_openai_chat_model(model);
    capabilities.insert("chat_completion".to_string(), chat.to_string());
    capabilities.insert(
        "responses_api".to_string(),
        needs_responses_api(model).to_string(),
    );
    capabilities.insert("tool_calling".to_string(), chat.to_string());
    capabilities.insert("vision".to_string(), supports_vision(model).to_string());
    capabilities.insert("streaming".to_string(), chat.to_string());
    capabilities.insert("structured_outputs".to_string(), chat.to_string());
    capabilities.insert(
        "reasoning_effort".to_string(),
        supports_reasoning_effort(model).to_string(),
    );
    capabilities.insert("context_length".to_string(), context_length.to_string());
    capabilities.insert(
        "context_budget_chars".to_string(),
        context_budget(model, Some(context_length)).to_string(),
    );
    capabilities
}

fn is_openai_chat_model(model: &str) -> bool {
    let m = model.to_lowercase();
    !m.contains("embedding")
        && !m.contains("whisper")
        && !m.contains("tts")
        && !m.contains("dall-e")
        && !m.contains("moderation")
}

fn supports_reasoning_effort(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("gpt-5") || m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4")
}

fn estimated_context_tokens(model: &str) -> usize {
    let m = model.to_lowercase();
    if m.contains("codex") {
        16_000
    } else if m.contains("gpt-5")
        || m.contains("gpt-4o")
        || m.contains("gpt-4.1")
        || m.contains("gpt-4-turbo")
        || m.contains("gpt-4-1106")
        || m.contains("gpt-4-0125")
        || m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
    {
        128_000
    } else if m.contains("gpt-3.5-turbo-16k") {
        16_384
    } else if m.contains("gpt-4") {
        8_192
    } else if m.contains("gpt-3.5") {
        4_096
    } else {
        32_000
    }
}

fn recommended_by_preference(models: &[ModelInfo], preferences: &[&str]) -> Option<String> {
    preferences.iter().find_map(|preferred| {
        models
            .iter()
            .find(|m| m.id == *preferred)
            .map(|m| m.id.clone())
    })
}

fn capability_is_true(model: &ModelInfo, name: &str) -> bool {
    model
        .capabilities
        .as_ref()
        .and_then(|c| c.get(name))
        .map(|v| v == "true")
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_foundry() {
        assert!(is_foundry(
            "https://my-project.services.ai.azure.com/api/projects/proj1"
        ));
        assert!(!is_foundry("https://my-resource.openai.azure.com"));
        assert!(!is_foundry("https://api.openai.com"));
    }

    #[test]
    fn test_is_azure_openai() {
        assert!(is_azure_openai("https://my-resource.openai.azure.com"));
        assert!(is_azure_openai(
            "https://my-resource.cognitiveservices.azure.com"
        ));
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
    fn test_versionless_base() {
        assert_eq!(
            versionless_base("https://api.openai.com/v1"),
            "https://api.openai.com"
        );
        assert_eq!(
            versionless_base("https://api.anthropic.com/v1/"),
            "https://api.anthropic.com"
        );
        assert_eq!(
            versionless_base("https://api.openai.com"),
            "https://api.openai.com"
        );
    }

    #[test]
    fn test_parse_openai_models() {
        let body = r#"{"data":[{"id":"gpt-4o","owned_by":"openai"},{"id":"gpt-3.5-turbo","owned_by":"openai"}]}"#;
        let models = parse_openai_models(body).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "gpt-4o");
        assert_eq!(models[1].id, "gpt-3.5-turbo");
        assert_eq!(models[0].context_length, Some(128_000));
        let caps = models[0].capabilities.as_ref().unwrap();
        assert_eq!(
            caps.get("chat_completion").map(String::as_str),
            Some("true")
        );
        assert_eq!(caps.get("vision").map(String::as_str), Some("true"));
        assert_eq!(caps.get("responses_api").map(String::as_str), Some("false"));
    }

    #[test]
    fn test_parse_openai_models_marks_non_chat_models() {
        let body = r#"{"data":[{"id":"text-embedding-3-small","owned_by":"openai"}]}"#;
        let models = parse_openai_models(body).unwrap();
        let caps = models[0].capabilities.as_ref().unwrap();
        assert_eq!(
            caps.get("chat_completion").map(String::as_str),
            Some("false")
        );
        assert_eq!(caps.get("tool_calling").map(String::as_str), Some("false"));
    }

    #[test]
    fn test_openai_recommendation_policy() {
        let body = r#"{"data":[
            {"id":"text-embedding-3-small","owned_by":"openai"},
            {"id":"gpt-4o","owned_by":"openai"},
            {"id":"gpt-5","owned_by":"openai"}
        ]}"#;
        let models = parse_openai_models(body).unwrap();
        assert_eq!(recommended_openai_model(&models).as_deref(), Some("gpt-5"));
    }

    #[test]
    fn test_openai_recommendation_falls_back_to_chat_model() {
        let body = r#"{"data":[
            {"id":"text-embedding-3-small","owned_by":"openai"},
            {"id":"custom-chat-model","owned_by":"openai"}
        ]}"#;
        let models = parse_openai_models(body).unwrap();
        assert_eq!(
            recommended_openai_model(&models).as_deref(),
            Some("custom-chat-model")
        );
    }

    #[test]
    fn test_parse_anthropic_models() {
        let body = r#"{
            "data": [
                {
                    "id": "claude-sonnet-4-6",
                    "display_name": "Claude Sonnet 4.6",
                    "created_at": "2026-01-01T00:00:00Z",
                    "max_input_tokens": 200000,
                    "capabilities": {
                        "image_input": { "supported": true }
                    }
                },
                {
                    "id": "claude-haiku-4-5",
                    "max_input_tokens": 200000,
                    "capabilities": {
                        "image_input": { "supported": false }
                    }
                }
            ],
            "has_more": true,
            "last_id": "claude-haiku-4-5"
        }"#;

        let (models, next) = parse_anthropic_models(body).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(next.as_deref(), Some("claude-haiku-4-5"));
        assert_eq!(models[0].owned_by.as_deref(), Some("Anthropic"));
        assert_eq!(models[0].context_length, Some(200_000));
        let caps = models[0].capabilities.as_ref().unwrap();
        assert_eq!(caps.get("vision").map(String::as_str), Some("true"));
        assert_eq!(
            caps.get("display_name").map(String::as_str),
            Some("Claude Sonnet 4.6")
        );
    }

    #[test]
    fn test_anthropic_recommendation_policy() {
        let body = r#"{"data":[
            {"id":"claude-opus-4-8","max_input_tokens":200000},
            {"id":"claude-sonnet-4-6","max_input_tokens":200000},
            {"id":"claude-haiku-4-5","max_input_tokens":200000}
        ]}"#;
        let (models, _) = parse_anthropic_models(body).unwrap();

        assert_eq!(
            recommended_anthropic_default_model(&models).as_deref(),
            Some("claude-sonnet-4-6")
        );
        assert_eq!(
            recommended_anthropic_model(&models, AnthropicModelTier::Opus).as_deref(),
            Some("claude-opus-4-8")
        );
    }

    #[test]
    fn test_validate_or_recommend_model() {
        let body = r#"{"data":[{"id":"gpt-5","owned_by":"openai"}]}"#;
        let models = parse_openai_models(body).unwrap();

        assert_eq!(
            validate_or_recommend_model("gpt-5", &models, recommended_openai_model(&models)),
            ModelSelection {
                requested: "gpt-5".to_string(),
                selected: Some("gpt-5".to_string()),
                available: true,
            }
        );

        assert_eq!(
            validate_or_recommend_model("stale-model", &models, recommended_openai_model(&models)),
            ModelSelection {
                requested: "stale-model".to_string(),
                selected: Some("gpt-5".to_string()),
                available: false,
            }
        );
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
        assert!(parse_foundry_project_deployments(r#"{"value":[]}"#)
            .unwrap()
            .is_empty());
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
