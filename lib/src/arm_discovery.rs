//! Azure Resource Manager (ARM) discovery for AI resources.
//!
//! Lists Azure subscriptions and AI Foundry / Cognitive Services resources
//! using the ARM API. Requires a token minted for the
//! `https://management.azure.com/.default` scope.
//!
//! # Example
//! ```no_run
//! # async fn example() -> Result<(), String> {
//! use agentive::arm_discovery;
//!
//! let mgmt_token = "..."; // from azure_oauth with AZURE_MANAGEMENT_SCOPE
//!
//! let subs = arm_discovery::list_subscriptions(mgmt_token).await?;
//! for s in &subs {
//!     println!("{}: {}", s.display_name, s.subscription_id);
//!     let resources = arm_discovery::list_ai_resources(mgmt_token, &s.subscription_id).await?;
//!     for r in &resources {
//!         println!("  {} → {}", r.name, r.endpoint);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An Azure subscription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    /// The subscription GUID.
    pub subscription_id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Subscription state (e.g. "Enabled", "Disabled").
    pub state: String,
}

/// An Azure AI resource (Cognitive Services account).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiResource {
    /// Resource name.
    pub name: String,
    /// Resource kind (e.g. "AIServices", "OpenAI").
    pub kind: String,
    /// The inference endpoint URL.
    pub endpoint: String,
    /// Azure region (e.g. "eastus").
    pub location: String,
    /// Resource group name.
    pub resource_group: String,
    /// For AI Services resources, the Foundry base URL
    /// (`https://{name}.services.ai.azure.com`). `None` for pure OpenAI resources.
    #[serde(default)]
    pub foundry_url: Option<String>,
}

/// A Foundry project within an AI Services hub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundryProject {
    /// Project workspace name (used in the endpoint path).
    pub name: String,
    /// Human-readable display name.
    pub display_name: String,
    /// The full project inference endpoint
    /// (`https://{hub}.services.ai.azure.com/api/projects/{name}`).
    pub endpoint: String,
}

// ---------------------------------------------------------------------------
// ARM response shapes (private)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ArmListResponse<T> {
    value: Vec<T>,
    #[serde(default, rename = "nextLink")]
    next_link: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ArmSubscription {
    subscription_id: String,
    display_name: String,
    state: String,
}

#[derive(Debug, Deserialize)]
struct ArmCogAccount {
    name: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    location: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    properties: Option<ArmCogProperties>,
}

#[derive(Debug, Deserialize)]
struct ArmCogProperties {
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    endpoints: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct ArmMlWorkspace {
    name: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    properties: Option<ArmMlWorkspaceProperties>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ArmMlWorkspaceProperties {
    #[serde(default)]
    friendly_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ArmCogProject {
    name: String,
    #[serde(default)]
    properties: Option<ArmCogProjectProperties>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ArmCogProjectProperties {
    #[serde(default)]
    display_name: Option<String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const ARM_BASE: &str = "https://management.azure.com";

/// List Azure subscriptions accessible to the authenticated user.
pub async fn list_subscriptions(token: &str) -> Result<Vec<Subscription>, String> {
    let url = format!("{ARM_BASE}/subscriptions?api-version=2022-12-01");
    let items = fetch_all_pages::<ArmSubscription>(token, &url).await?;
    Ok(items
        .into_iter()
        .filter(|s| s.state == "Enabled")
        .map(|s| Subscription {
            subscription_id: s.subscription_id,
            display_name: s.display_name,
            state: s.state,
        })
        .collect())
}

/// List AI resources (Cognitive Services accounts) in a subscription.
///
/// Filters to resources of kind `AIServices` or `OpenAI` that have
/// a usable inference endpoint.
pub async fn list_ai_resources(
    token: &str,
    subscription_id: &str,
) -> Result<Vec<AiResource>, String> {
    let url = format!(
        "{ARM_BASE}/subscriptions/{subscription_id}/providers/\
         Microsoft.CognitiveServices/accounts?api-version=2023-05-01"
    );
    let items = fetch_all_pages::<ArmCogAccount>(token, &url).await?;
    Ok(items
        .into_iter()
        .filter_map(|a| {
            let kind = a.kind.unwrap_or_default();
            if kind != "AIServices" && kind != "OpenAI" {
                return None;
            }
            let endpoint = a
                .properties
                .as_ref()
                .and_then(|p| {
                    // Prefer the OpenAI-specific endpoint (*.openai.azure.com)
                    // over the generic cognitiveservices endpoint which doesn't
                    // serve the /openai/deployments API.
                    p.endpoints
                        .as_ref()
                        .and_then(|e| e.get("OpenAI Language Model Instance API").cloned())
                        .or_else(|| p.endpoint.clone())
                })
                .unwrap_or_default();
            if endpoint.is_empty() {
                return None;
            }
            // Extract resource group from the ARM resource id
            let rg = a
                .id
                .as_deref()
                .and_then(extract_resource_group)
                .unwrap_or_default();
            Some(AiResource {
                foundry_url: if kind == "AIServices" {
                    Some(format!("https://{}.services.ai.azure.com", a.name))
                } else {
                    None
                },
                name: a.name,
                kind,
                endpoint,
                location: a.location.unwrap_or_default(),
                resource_group: rg,
            })
        })
        .collect())
}

/// List Foundry projects for an AI Services resource.
///
/// Tries two discovery strategies:
/// 1. **New Foundry**: `CognitiveServices/accounts/{name}/projects` (direct child resources)
/// 2. **Classic hub**: `MachineLearningServices/workspaces` (kind=Project, subscription-wide)
///
/// Results are merged, with new Foundry projects taking priority.
pub async fn list_foundry_projects(
    token: &str,
    subscription_id: &str,
    resource_group: &str,
    resource_name: &str,
) -> Result<Vec<FoundryProject>, String> {
    let base_url = format!("https://{resource_name}.services.ai.azure.com");
    let mut projects = Vec::new();

    // Strategy 1: New Foundry projects (CognitiveServices subresource)
    let cog_url = format!(
        "{ARM_BASE}/subscriptions/{subscription_id}/resourceGroups/{resource_group}/\
         providers/Microsoft.CognitiveServices/accounts/{resource_name}/\
         projects?api-version=2025-04-01-preview"
    );
    if let Ok(items) = fetch_all_pages::<ArmCogProject>(token, &cog_url).await {
        projects.extend(items.into_iter().map(|p| {
            // ARM returns subresource names as "parent/child" — strip the parent prefix
            let short_name = p.name.rsplit('/').next().unwrap_or(&p.name).to_string();
            let display = p
                .properties
                .as_ref()
                .and_then(|props| props.display_name.clone())
                .unwrap_or_else(|| short_name.clone());
            FoundryProject {
                endpoint: format!("{base_url}/api/projects/{short_name}"),
                display_name: display,
                name: short_name,
            }
        }));
    }

    // Strategy 2 (fallback): Classic hub-based projects (ML workspaces, subscription-wide).
    // Only used when the CogServices query returned nothing (e.g. older hub-based setup).
    if projects.is_empty() {
        let ml_url = format!(
            "{ARM_BASE}/subscriptions/{subscription_id}/\
             providers/Microsoft.MachineLearningServices/workspaces?api-version=2024-10-01"
        );
        if let Ok(items) = fetch_all_pages::<ArmMlWorkspace>(token, &ml_url).await {
            let classic_projects: Vec<_> = items
                .into_iter()
                .filter(|w| w.kind.as_deref() == Some("Project"))
                .map(|w| {
                    let display = w
                        .properties
                        .as_ref()
                        .and_then(|p| p.friendly_name.clone())
                        .unwrap_or_else(|| w.name.clone());
                    FoundryProject {
                        endpoint: format!("{base_url}/api/projects/{}", w.name),
                        display_name: display,
                        name: w.name,
                    }
                })
                .collect();
            projects.extend(classic_projects);
        }
    }

    Ok(projects)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the resource group from an ARM resource ID.
fn extract_resource_group(id: &str) -> Option<String> {
    // Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/...
    let lower = id.to_lowercase();
    let idx = lower.find("/resourcegroups/")?;
    let rest = &id[idx + "/resourceGroups/".len()..];
    Some(rest.split('/').next()?.to_string())
}

/// Fetch all pages from a paginated ARM API response.
async fn fetch_all_pages<T: serde::de::DeserializeOwned>(
    token: &str,
    initial_url: &str,
) -> Result<Vec<T>, String> {
    let http = reqwest::Client::new();
    let mut all = Vec::new();
    let mut url = initial_url.to_string();

    loop {
        let resp = http
            .get(&url)
            .bearer_auth(token)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("ARM request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("ARM API error ({status}): {body}"));
        }

        let page: ArmListResponse<T> = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse ARM response: {e}"))?;

        all.extend(page.value);

        match page.next_link {
            Some(next) if !next.is_empty() => url = next,
            _ => break,
        }
    }

    Ok(all)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_resource_group() {
        let id = "/subscriptions/abc-123/resourceGroups/my-rg/providers/Microsoft.CognitiveServices/accounts/my-ai";
        assert_eq!(
            extract_resource_group(id),
            Some("my-rg".to_string())
        );
    }

    #[test]
    fn test_extract_resource_group_case_insensitive() {
        let id = "/subscriptions/abc/resourcegroups/MyRG/providers/Foo";
        assert_eq!(
            extract_resource_group(id),
            Some("MyRG".to_string())
        );
    }

    #[test]
    fn test_extract_resource_group_missing() {
        assert_eq!(extract_resource_group("/subscriptions/abc"), None);
        assert_eq!(extract_resource_group(""), None);
    }

    #[test]
    fn test_parse_subscriptions_response() {
        let body = r#"{"value":[
            {"subscriptionId":"sub-1","displayName":"Dev","state":"Enabled"},
            {"subscriptionId":"sub-2","displayName":"Prod","state":"Enabled"},
            {"subscriptionId":"sub-3","displayName":"Disabled","state":"Disabled"}
        ]}"#;
        let parsed: ArmListResponse<ArmSubscription> = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.value.len(), 3);
        assert_eq!(parsed.value[0].subscription_id, "sub-1");
        assert_eq!(parsed.value[0].display_name, "Dev");
    }

    #[test]
    fn test_parse_cog_accounts_response() {
        let body = r#"{"value":[
            {
                "name":"my-ai-services",
                "kind":"AIServices",
                "location":"eastus",
                "id":"/subscriptions/sub-1/resourceGroups/my-rg/providers/Microsoft.CognitiveServices/accounts/my-ai-services",
                "properties":{
                    "endpoint":"https://my-ai-services.cognitiveservices.azure.com/"
                }
            },
            {
                "name":"my-openai",
                "kind":"OpenAI",
                "location":"westus",
                "id":"/subscriptions/sub-1/resourceGroups/rg2/providers/Microsoft.CognitiveServices/accounts/my-openai",
                "properties":{
                    "endpoint":"https://my-openai.openai.azure.com/"
                }
            },
            {
                "name":"my-search",
                "kind":"CognitiveSearch",
                "location":"eastus",
                "id":"/subscriptions/sub-1/resourceGroups/rg3/providers/Microsoft.CognitiveServices/accounts/my-search",
                "properties":{}
            }
        ]}"#;
        let parsed: ArmListResponse<ArmCogAccount> = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.value.len(), 3);

        // Filter like list_ai_resources does
        let resources: Vec<_> = parsed
            .value
            .into_iter()
            .filter_map(|a| {
                let kind = a.kind.unwrap_or_default();
                if kind != "AIServices" && kind != "OpenAI" {
                    return None;
                }
                let endpoint = a.properties.as_ref().and_then(|p| p.endpoint.clone()).unwrap_or_default();
                if endpoint.is_empty() { return None; }
                let rg = a.id.as_deref().and_then(extract_resource_group).unwrap_or_default();
                let foundry_url = if kind == "AIServices" {
                    Some(format!("https://{}.services.ai.azure.com", a.name))
                } else {
                    None
                };
                Some(AiResource {
                    name: a.name,
                    kind,
                    endpoint,
                    location: a.location.unwrap_or_default(),
                    resource_group: rg,
                    foundry_url,
                })
            })
            .collect();

        assert_eq!(resources.len(), 2);
        assert_eq!(resources[0].name, "my-ai-services");
        assert_eq!(resources[0].kind, "AIServices");
        assert_eq!(
            resources[0].endpoint,
            "https://my-ai-services.cognitiveservices.azure.com/"
        );
        assert_eq!(resources[0].resource_group, "my-rg");
        assert_eq!(
            resources[0].foundry_url.as_deref(),
            Some("https://my-ai-services.services.ai.azure.com")
        );
        assert_eq!(resources[1].name, "my-openai");
        assert_eq!(resources[1].kind, "OpenAI");
        assert_eq!(resources[1].foundry_url, None);
    }

    #[test]
    fn test_parse_cog_accounts_with_endpoints_map() {
        // When both endpoint and endpoints map exist, prefer the OpenAI-specific one
        let body = r#"{"value":[{
            "name":"foundry-svc",
            "kind":"AIServices",
            "location":"eastus2",
            "id":"/subscriptions/s1/resourceGroups/rg1/providers/Microsoft.CognitiveServices/accounts/foundry-svc",
            "properties":{
                "endpoint":"https://foundry-svc.cognitiveservices.azure.com/",
                "endpoints":{
                    "OpenAI Language Model Instance API":"https://foundry-svc.openai.azure.com/"
                }
            }
        }]}"#;
        let parsed: ArmListResponse<ArmCogAccount> = serde_json::from_str(body).unwrap();
        let a = &parsed.value[0];
        let endpoint = a
            .properties
            .as_ref()
            .and_then(|p| {
                p.endpoints
                    .as_ref()
                    .and_then(|e| e.get("OpenAI Language Model Instance API").cloned())
                    .or_else(|| p.endpoint.clone())
            })
            .unwrap_or_default();
        assert_eq!(endpoint, "https://foundry-svc.openai.azure.com/");
    }

    #[test]
    fn test_parse_empty_response() {
        let body = r#"{"value":[]}"#;
        let parsed: ArmListResponse<ArmSubscription> = serde_json::from_str(body).unwrap();
        assert!(parsed.value.is_empty());
        assert!(parsed.next_link.is_none());
    }

    #[test]
    fn test_parse_with_next_link() {
        let body = r#"{"value":[{"subscriptionId":"s1","displayName":"A","state":"Enabled"}],"nextLink":"https://management.azure.com/next?page=2"}"#;
        let parsed: ArmListResponse<ArmSubscription> = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.value.len(), 1);
        assert_eq!(
            parsed.next_link.as_deref(),
            Some("https://management.azure.com/next?page=2")
        );
    }

    #[test]
    fn test_parse_ml_workspaces_projects() {
        let body = r#"{"value":[
            {"name":"dev-models","kind":"Project","properties":{"friendlyName":"Dev Models"}},
            {"name":"my-hub","kind":"Hub","properties":{"friendlyName":"My Hub"}},
            {"name":"prod-models","kind":"Project","properties":{}}
        ]}"#;
        let parsed: ArmListResponse<ArmMlWorkspace> = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.value.len(), 3);

        // Filter like list_foundry_projects does
        let base = "https://seth-foundry-dev.services.ai.azure.com";
        let projects: Vec<_> = parsed
            .value
            .into_iter()
            .filter(|w| w.kind.as_deref() == Some("Project"))
            .map(|w| {
                let display = w
                    .properties
                    .as_ref()
                    .and_then(|p| p.friendly_name.clone())
                    .unwrap_or_else(|| w.name.clone());
                FoundryProject {
                    endpoint: format!("{base}/api/projects/{}", w.name),
                    display_name: display,
                    name: w.name,
                }
            })
            .collect();

        assert_eq!(projects.len(), 2);
        assert_eq!(projects[0].name, "dev-models");
        assert_eq!(projects[0].display_name, "Dev Models");
        assert_eq!(
            projects[0].endpoint,
            "https://seth-foundry-dev.services.ai.azure.com/api/projects/dev-models"
        );
        // When no friendly_name, falls back to workspace name
        assert_eq!(projects[1].name, "prod-models");
        assert_eq!(projects[1].display_name, "prod-models");
    }

    #[test]
    fn test_parse_cog_projects_response() {
        // ARM returns subresource names as "parent/child"
        let body = r#"{"value":[
            {"name":"seth-foundry-dev/dev-models","properties":{"displayName":"Dev Models"}},
            {"name":"seth-foundry-dev/staging","properties":{}}
        ]}"#;
        let parsed: ArmListResponse<ArmCogProject> = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.value.len(), 2);

        let base = "https://seth-foundry-dev.services.ai.azure.com";
        let projects: Vec<_> = parsed
            .value
            .into_iter()
            .map(|p| {
                let short_name = p.name.rsplit('/').next().unwrap_or(&p.name).to_string();
                let display = p
                    .properties
                    .as_ref()
                    .and_then(|props| props.display_name.clone())
                    .unwrap_or_else(|| short_name.clone());
                FoundryProject {
                    endpoint: format!("{base}/api/projects/{short_name}"),
                    display_name: display,
                    name: short_name,
                }
            })
            .collect();

        assert_eq!(projects.len(), 2);
        assert_eq!(projects[0].name, "dev-models");
        assert_eq!(projects[0].display_name, "Dev Models");
        assert_eq!(
            projects[0].endpoint,
            "https://seth-foundry-dev.services.ai.azure.com/api/projects/dev-models"
        );
        // When no displayName, falls back to short name
        assert_eq!(projects[1].name, "staging");
        assert_eq!(projects[1].display_name, "staging");
    }
}
