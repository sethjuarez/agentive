//! OpenAI-compatible provider.
//!
//! Supports OpenAI, Azure OpenAI, Microsoft Foundry, and any endpoint
//! that implements the `/chat/completions` SSE streaming protocol.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use futures_util::StreamExt;
use reqwest::Client;
use tokio::sync::mpsc;

use crate::auth::AuthStrategy;
use crate::cancel::CancellationToken;
use crate::error::AgentError;
use crate::provider::Provider;
use crate::providers::request_compaction::{
    compact_items_to_request_limit, default_azure_max_request_bytes,
    default_azure_request_reserved_bytes,
};
#[cfg(test)]
use crate::providers::request_compaction::{
    effective_request_budget_bytes, DEFAULT_AZURE_MAX_REQUEST_BYTES,
    DEFAULT_AZURE_REQUEST_RESERVED_BYTES,
};
use crate::providers::sse::SseParser;
use crate::types::*;

/// OpenAI-compatible chat completion provider.
///
/// Works with OpenAI, Azure OpenAI, Microsoft Foundry, and any endpoint
/// that speaks the OpenAI chat completions SSE protocol.
pub struct OpenAiProvider {
    endpoint: String,
    auth: AuthStrategy,
    model: String,
    client: Client,
    context_budget: usize,
    vision: bool,
    max_request_bytes: usize,
    reserved_request_bytes: usize,
}

impl OpenAiProvider {
    /// Create a new provider pointing at an OpenAI-compatible endpoint.
    ///
    /// Auth is auto-detected: Azure endpoints use `api-key` header,
    /// others use `Bearer` token. For Entra/OAuth tokens, use
    /// [`with_auth`](Self::with_auth) instead.
    pub fn new(endpoint: &str, api_key: &str, model: &str) -> Self {
        let trimmed = endpoint.trim_end_matches('/');
        let auth = if trimmed.contains("azure.com") {
            AuthStrategy::ApiKey(api_key.to_string())
        } else {
            AuthStrategy::Bearer(api_key.to_string())
        };
        Self {
            endpoint: trimmed.to_string(),
            auth,
            model: model.to_string(),
            client: Client::new(),
            context_budget: 200_000,
            vision: false,
            max_request_bytes: default_azure_max_request_bytes(trimmed),
            reserved_request_bytes: default_azure_request_reserved_bytes(trimmed),
        }
    }

    /// Create a provider with an explicit auth strategy.
    ///
    /// Use this for Microsoft Foundry with Entra ID tokens:
    /// ```no_run
    /// use agentive::{OpenAiProvider, AuthStrategy};
    /// use std::sync::{Arc, Mutex};
    ///
    /// let token = Arc::new(Mutex::new("initial-token".to_string()));
    /// let token_ref = token.clone();
    /// let provider = OpenAiProvider::with_auth(
    ///     "https://my-resource.services.ai.azure.com",
    ///     AuthStrategy::Dynamic(Arc::new(move || token_ref.lock().unwrap().clone())),
    ///     "gpt-4o",
    /// );
    /// ```
    pub fn with_auth(endpoint: &str, auth: AuthStrategy, model: &str) -> Self {
        let trimmed = endpoint.trim_end_matches('/');
        Self {
            endpoint: trimmed.to_string(),
            auth,
            model: model.to_string(),
            client: Client::new(),
            context_budget: 200_000,
            vision: false,
            max_request_bytes: default_azure_max_request_bytes(trimmed),
            reserved_request_bytes: default_azure_request_reserved_bytes(trimmed),
        }
    }

    /// Set the context budget in characters.
    pub fn with_context_budget(mut self, chars: usize) -> Self {
        self.context_budget = chars;
        self
    }

    /// Enable vision support.
    pub fn with_vision(mut self, enabled: bool) -> Self {
        self.vision = enabled;
        self
    }

    /// Set the maximum serialized request body size in bytes.
    ///
    /// Azure OpenAI-compatible gateways can truncate request bodies around
    /// ~79KB, causing misleading 400 JSON parse errors. Azure endpoints default
    /// to 64KB; non-Azure endpoints default to no byte cap. Set to
    /// `usize::MAX` to disable the guard.
    pub fn with_max_request_bytes(mut self, bytes: usize) -> Self {
        self.max_request_bytes = bytes;
        self
    }

    fn chat_url(&self) -> String {
        if self.endpoint.contains("/chat/completions") {
            // Already a fully-qualified URL — use as-is
            self.endpoint.clone()
        } else if self.endpoint.contains("/api/projects/") {
            // Foundry project: strip /api/projects/... to get the resource base,
            // then use deployment-based URL
            let base = match self.endpoint.find("/api/projects") {
                Some(idx) => &self.endpoint[..idx],
                None => &self.endpoint,
            };
            format!(
                "{}/openai/deployments/{}/chat/completions?api-version=2024-10-21",
                base, self.model
            )
        } else if self.endpoint.contains("azure.com") {
            // Plain Azure OpenAI endpoint (e.g. https://my-resource.openai.azure.com)
            // Uses deployment-based URL where model name = deployment name
            format!(
                "{}/openai/deployments/{}/chat/completions?api-version=2024-10-21",
                self.endpoint, self.model
            )
        } else {
            format!("{}/chat/completions", self.endpoint)
        }
    }

    /// Build the JSON request body, compacting older conversation messages if
    /// the serialized size exceeds `max_request_bytes`.
    fn request_body(
        &self,
        messages: &[ChatMessage],
        tools: &Option<Vec<Tool>>,
        stream: bool,
        response_format: &Option<ResponseFormat>,
    ) -> Result<serde_json::Value, AgentError> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "tools": tools,
        });

        if stream {
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }

        if let Some(rf) = response_format {
            body["response_format"] = serde_json::to_value(rf).map_err(|e| {
                AgentError::Stream(format!("Failed to serialize response_format: {e}"))
            })?;
        }

        Ok(body)
    }

    fn build_body_within_limit(
        &self,
        messages: &mut Vec<ChatMessage>,
        tools: &Option<Vec<Tool>>,
        stream: bool,
        response_format: &Option<ResponseFormat>,
    ) -> Result<serde_json::Value, AgentError> {
        let make_body =
            |messages: &[ChatMessage]| self.request_body(messages, tools, stream, response_format);

        compact_items_to_request_limit(
            messages,
            self.max_request_bytes,
            self.reserved_request_bytes,
            "OpenAI chat",
            "message",
            make_body,
            |message| is_preserved_prefix_role(&message.role),
            remove_message_group,
        )
    }
}

#[async_trait::async_trait]
impl Provider for OpenAiProvider {
    async fn chat(
        &self,
        request: ChatRequest,
        tx: mpsc::Sender<ChatEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), AgentError> {
        let stream = request.stream;
        let tools = request.tools;
        let response_format = request.response_format;
        let mut messages = request.messages;
        let body = self.build_body_within_limit(&mut messages, &tools, stream, &response_format)?;

        let started = Instant::now();
        log::debug!(
            "[agentive::openai] request start stream={} model={}",
            stream,
            self.model
        );

        let mut req = self.client.post(self.chat_url()).json(&body);
        req = self.auth.apply(req);

        let response = req.send().await?;
        log::debug!(
            "[agentive::openai] response headers status={} elapsed={}ms",
            response.status().as_u16(),
            started.elapsed().as_millis()
        );

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            log::warn!(
                "[agentive::openai] request failed status={} elapsed={}ms",
                status,
                started.elapsed().as_millis()
            );
            return Err(AgentError::Api {
                status,
                message: text,
            });
        }

        if !stream {
            return self
                .handle_non_streaming_response(response, tx, started)
                .await;
        }

        let mut stream = response.bytes_stream();
        let mut parser = SseParser::new();

        // Accumulate tool calls across chunks (keyed by index)
        let mut tool_calls: HashMap<u32, PendingToolCall> = HashMap::new();
        let mut full_content = String::new();
        let mut usage: Option<Usage> = None;
        let mut saw_first_chunk = false;

        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            let chunk = chunk.map_err(|e| AgentError::Stream(e.to_string()))?;
            if !saw_first_chunk {
                saw_first_chunk = true;
                log::debug!(
                    "[agentive::openai] first stream chunk elapsed={}ms bytes={}",
                    started.elapsed().as_millis(),
                    chunk.len()
                );
            }
            let data_lines = parser.feed(&chunk);

            for data in data_lines {
                if data == "[DONE]" {
                    let final_tool_calls: Option<Vec<ToolCall>> = if tool_calls.is_empty() {
                        None
                    } else {
                        let mut calls: Vec<_> = tool_calls.drain().collect();
                        calls.sort_by_key(|(idx, _)| *idx);
                        Some(
                            calls
                                .into_iter()
                                .map(|(_, tc)| tc.into_tool_call())
                                .collect(),
                        )
                    };

                    let tool_call_count = final_tool_calls
                        .as_ref()
                        .map(|calls| calls.len())
                        .unwrap_or(0);
                    let message = if let Some(calls) = final_tool_calls {
                        let mut msg = ChatMessage::assistant_with_tool_calls(calls);
                        if !full_content.is_empty() {
                            msg.content = Some(MessageContent::Text(full_content.clone()));
                        }
                        msg
                    } else {
                        ChatMessage::assistant(&full_content)
                    };

                    let _ = tx
                        .send(ChatEvent::Done {
                            response: ChatResponse { message, usage },
                        })
                        .await;
                    log::debug!(
                        "[agentive::openai] stream done elapsed={}ms response_chars={} tool_calls={}",
                        started.elapsed().as_millis(),
                        full_content.chars().count(),
                        tool_call_count
                    );
                    return Ok(());
                }

                let parsed: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Extract usage if present
                if let Some(u) = parsed.get("usage") {
                    if let Ok(u) = serde_json::from_value::<Usage>(u.clone()) {
                        usage = Some(u);
                    }
                }

                let Some(choice) = parsed.get("choices").and_then(|c| c.get(0)) else {
                    continue;
                };
                let Some(delta) = choice.get("delta") else {
                    continue;
                };

                // Content tokens
                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        full_content.push_str(content);
                        let _ = tx
                            .send(ChatEvent::Token {
                                token: content.to_string(),
                            })
                            .await;
                    }
                }

                // Reasoning/thinking tokens (OpenAI o-series)
                if let Some(thinking) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
                    if !thinking.is_empty() {
                        let _ = tx
                            .send(ChatEvent::Thinking {
                                token: thinking.to_string(),
                            })
                            .await;
                    }
                }

                // Tool call deltas — accumulate by index
                if let Some(tc_array) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tc_array {
                        let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as u32;

                        let entry = tool_calls.entry(index).or_default();

                        // Emit tool_call start when we first get the name
                        if entry.update_from_delta(tc) {
                            let _ = tx
                                .send(ChatEvent::ToolCallStart {
                                    tool_call: entry.clone().into_tool_call(),
                                })
                                .await;
                        }
                    }
                }
            }
        }

        log::warn!(
            "[agentive::openai] stream ended without done elapsed={}ms response_chars={}",
            started.elapsed().as_millis(),
            full_content.chars().count()
        );
        Err(AgentError::Stream("Stream ended without [DONE]".into()))
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn context_budget_chars(&self) -> usize {
        self.context_budget
    }

    fn supports_vision(&self) -> bool {
        self.vision
    }

    fn request_budget_bytes(&self) -> Option<usize> {
        (self.max_request_bytes != usize::MAX).then_some(self.max_request_bytes)
    }

    fn estimate_request_bytes(&self, request: &ChatRequest) -> Result<Option<usize>, AgentError> {
        if self.request_budget_bytes().is_none() {
            return Ok(None);
        }

        self.request_body(
            &request.messages,
            &request.tools,
            request.stream,
            &request.response_format,
        )
        .and_then(|body| serde_json::to_string(&body).map_err(AgentError::from))
        .map(|body| Some(body.len()))
    }
}

impl OpenAiProvider {
    async fn handle_non_streaming_response(
        &self,
        response: reqwest::Response,
        tx: mpsc::Sender<ChatEvent>,
        started: Instant,
    ) -> Result<(), AgentError> {
        let parsed: serde_json::Value = response.json().await?;
        let usage = parsed
            .get("usage")
            .cloned()
            .map(serde_json::from_value::<Usage>)
            .transpose()?;

        let Some(message_value) = parsed
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
        else {
            return Err(AgentError::Stream(
                "Non-streaming response missing choices[0].message".into(),
            ));
        };

        let message = parse_chat_completion_message(message_value)?;
        let _ = tx
            .send(ChatEvent::Done {
                response: ChatResponse { message, usage },
            })
            .await;
        log::debug!(
            "[agentive::openai] non-streaming done elapsed={}ms response_chars={}",
            started.elapsed().as_millis(),
            parsed
                .get("choices")
                .and_then(|choices| choices.get(0))
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(|content| content.as_str())
                .map(|content| content.chars().count())
                .unwrap_or(0)
        );
        Ok(())
    }
}

// -- Helpers -----------------------------------------------------------------

fn is_preserved_prefix_role(role: &str) -> bool {
    matches!(role, "system" | "developer")
}

fn remove_message_group(messages: &mut Vec<ChatMessage>, idx: usize) -> usize {
    if idx >= messages.len() {
        return 0;
    }

    let removed = messages.remove(idx);
    let mut count = 1usize;

    if removed.role == "user" {
        while idx < messages.len()
            && !matches!(messages[idx].role.as_str(), "user" | "system" | "developer")
        {
            messages.remove(idx);
            count += 1;
        }
    } else if removed.role == "assistant" {
        count += remove_matching_tool_results(messages, idx, &removed);
    }

    while idx < messages.len()
        && messages[idx].role == "user"
        && messages[idx]
            .text()
            .is_some_and(|text| text.starts_with("[Images from the tool result above"))
    {
        messages.remove(idx);
        count += 1;
    }

    count
}

fn remove_matching_tool_results(
    messages: &mut Vec<ChatMessage>,
    idx: usize,
    assistant: &ChatMessage,
) -> usize {
    let call_ids = assistant
        .tool_calls
        .as_ref()
        .into_iter()
        .flatten()
        .map(|call| call.id.clone())
        .collect::<HashSet<_>>();
    let mut count = 0usize;

    while !call_ids.is_empty()
        && idx < messages.len()
        && messages[idx].role == "tool"
        && messages[idx]
            .tool_call_id
            .as_ref()
            .is_some_and(|id| call_ids.contains(id))
    {
        messages.remove(idx);
        count += 1;
    }

    count
}

fn parse_chat_completion_message(value: &serde_json::Value) -> Result<ChatMessage, AgentError> {
    let role = value
        .get("role")
        .and_then(|role| role.as_str())
        .unwrap_or("assistant")
        .to_string();

    let content = match value.get("content") {
        Some(serde_json::Value::String(text)) => Some(MessageContent::Text(text.clone())),
        Some(serde_json::Value::Array(parts)) => {
            let parts = parts
                .iter()
                .filter_map(parse_content_part)
                .collect::<Vec<_>>();
            if parts.is_empty() {
                None
            } else {
                Some(MessageContent::Parts(parts))
            }
        }
        _ => None,
    };

    let tool_calls = value
        .get("tool_calls")
        .cloned()
        .map(serde_json::from_value::<Vec<ToolCall>>)
        .transpose()?;

    Ok(ChatMessage {
        role,
        content,
        tool_calls,
        tool_call_id: None,
    })
}

fn parse_content_part(value: &serde_json::Value) -> Option<ContentPart> {
    match value.get("type").and_then(|kind| kind.as_str()) {
        Some("text") => value
            .get("text")
            .and_then(|text| text.as_str())
            .map(|text| ContentPart::Text {
                text: text.to_string(),
            }),
        Some("image_url") => value
            .get("image_url")
            .cloned()
            .and_then(|image_url| serde_json::from_value::<ImageUrl>(image_url).ok())
            .map(|image_url| ContentPart::ImageUrl { image_url }),
        _ => None,
    }
}

#[derive(Clone, Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
    start_emitted: bool,
}

impl PendingToolCall {
    fn update_from_delta(&mut self, tc: &serde_json::Value) -> bool {
        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
            self.id = id.to_string();
        }
        if let Some(func) = tc.get("function") {
            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                self.name = name.to_string();
            }
            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                self.arguments.push_str(args);
            }
        }

        if !self.name.is_empty() && !self.start_emitted {
            self.start_emitted = true;
            return true;
        }
        false
    }

    fn into_tool_call(self) -> ToolCall {
        ToolCall {
            id: self.id,
            call_type: "function".into(),
            function: FunctionCall {
                name: self.name,
                arguments: self.arguments,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_url_foundry_project() {
        let p = OpenAiProvider::new(
            "https://my-resource.services.ai.azure.com/api/projects/my-project",
            "key",
            "gpt-4o",
        );
        // Strips /api/projects/... and uses deployment-based URL
        assert_eq!(
            p.chat_url(),
            "https://my-resource.services.ai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn test_chat_url_no_suffix() {
        let p = OpenAiProvider::new("https://api.openai.com/v1", "key", "gpt-4o");
        assert_eq!(p.chat_url(), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn test_chat_url_with_suffix() {
        let p = OpenAiProvider::new(
            "https://my.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-10-21",
            "key",
            "gpt-4o",
        );
        assert!(p.chat_url().contains("chat/completions"));
    }

    #[test]
    fn test_auth_auto_detection() {
        // Azure endpoints should get ApiKey auth
        let p = OpenAiProvider::new("https://my-resource.openai.azure.com/v1", "k", "m");
        assert!(matches!(p.auth, AuthStrategy::ApiKey(_)));
        // Non-Azure endpoints should get Bearer auth
        let p = OpenAiProvider::new("https://api.openai.com/v1", "k", "m");
        assert!(matches!(p.auth, AuthStrategy::Bearer(_)));
    }

    #[test]
    fn test_request_size_guard_defaults_to_azure_only() {
        let azure = OpenAiProvider::new("https://my-resource.openai.azure.com", "key", "gpt-4o");
        assert_eq!(azure.max_request_bytes, DEFAULT_AZURE_MAX_REQUEST_BYTES);
        assert_eq!(
            azure.reserved_request_bytes,
            DEFAULT_AZURE_REQUEST_RESERVED_BYTES
        );

        let openai = OpenAiProvider::new("https://api.openai.com/v1", "key", "gpt-4o");
        assert_eq!(openai.max_request_bytes, usize::MAX);
        assert_eq!(openai.reserved_request_bytes, 0);
    }

    #[test]
    fn test_build_body_within_limit_drops_oldest_messages() {
        let provider = OpenAiProvider::new("https://api.openai.com/v1", "key", "gpt-4o")
            .with_max_request_bytes(900);
        let mut messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user(&"old user ".repeat(80)),
            ChatMessage::assistant("old assistant"),
            ChatMessage::user("recent question"),
            ChatMessage::assistant("recent answer"),
        ];

        let body = provider
            .build_body_within_limit(&mut messages, &None, true, &None)
            .unwrap();
        let serialized = serde_json::to_string(&body).unwrap();

        assert!(serialized.len() <= 900);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].text(), Some("recent question"));
        assert_eq!(messages[2].text(), Some("recent answer"));
    }

    #[test]
    fn test_build_body_within_limit_drops_tool_call_group() {
        let provider = OpenAiProvider::new("https://api.openai.com/v1", "key", "gpt-4o")
            .with_max_request_bytes(1_300);
        let mut messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                id: "call_old".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: r#"{"path":"old.txt"}"#.into(),
                },
            }]),
            ChatMessage::tool_result("call_old", &"old tool output ".repeat(80)),
            ChatMessage::user("recent question"),
            ChatMessage::assistant("recent answer"),
        ];

        let body = provider
            .build_body_within_limit(&mut messages, &None, true, &None)
            .unwrap();
        let serialized = serde_json::to_string(&body).unwrap();

        assert!(serialized.len() <= 1_300);
        assert_eq!(
            messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            ["system", "user", "assistant"]
        );
        assert_eq!(messages[1].text(), Some("recent question"));
    }

    #[test]
    fn test_build_body_within_limit_errors_when_prefix_is_too_large() {
        let provider = OpenAiProvider::new("https://api.openai.com/v1", "key", "gpt-4o")
            .with_max_request_bytes(100);
        let mut messages = vec![
            ChatMessage::system(&"large system ".repeat(40)),
            ChatMessage::user("recent question"),
            ChatMessage::assistant("recent answer"),
        ];

        let err = provider
            .build_body_within_limit(&mut messages, &None, true, &None)
            .unwrap_err();

        assert!(err.to_string().contains("too large after compaction"));
        assert!(err.to_string().contains("Reduce attached files"));
    }

    #[test]
    fn test_build_body_within_limit_uses_reserved_azure_overhead() {
        let sizing_provider =
            OpenAiProvider::new("https://my-resource.openai.azure.com", "key", "gpt-4o")
                .with_max_request_bytes(usize::MAX);
        let mut original_messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user(&"old web context ".repeat(70)),
            ChatMessage::assistant("middle answer"),
            ChatMessage::user("recent question"),
        ];
        let body_at_nominal_limit = sizing_provider
            .request_body(&original_messages, &None, true, &None)
            .unwrap();
        let nominal_cap = serde_json::to_string(&body_at_nominal_limit).unwrap().len() + 8;
        let effective_cap =
            effective_request_budget_bytes(nominal_cap, DEFAULT_AZURE_REQUEST_RESERVED_BYTES);
        assert!(serde_json::to_string(&body_at_nominal_limit).unwrap().len() <= nominal_cap);
        assert!(serde_json::to_string(&body_at_nominal_limit).unwrap().len() > effective_cap);

        let provider = OpenAiProvider::new("https://my-resource.openai.azure.com", "key", "gpt-4o")
            .with_max_request_bytes(nominal_cap);
        let body = provider
            .build_body_within_limit(&mut original_messages, &None, true, &None)
            .unwrap();
        let compacted_len = serde_json::to_string(&body).unwrap().len();

        assert!(compacted_len <= effective_cap);
        assert!(!original_messages.iter().any(|message| message
            .text()
            .is_some_and(|text| text.contains("old web context"))));
        assert_eq!(
            original_messages.last().and_then(ChatMessage::text),
            Some("recent question")
        );
    }

    #[test]
    fn test_request_budget_defaults_are_provider_aware() {
        let openai = OpenAiProvider::new("https://api.openai.com/v1", "key", "gpt-4o");
        assert_eq!(openai.request_budget_bytes(), None);
        assert_eq!(openai.max_request_bytes, usize::MAX);
        assert_eq!(openai.reserved_request_bytes, 0);

        let azure = OpenAiProvider::new("https://my-resource.openai.azure.com", "key", "gpt-4o");
        assert_eq!(
            azure.request_budget_bytes(),
            Some(DEFAULT_AZURE_MAX_REQUEST_BYTES)
        );
        assert_eq!(
            effective_request_budget_bytes(azure.max_request_bytes, azure.reserved_request_bytes),
            DEFAULT_AZURE_MAX_REQUEST_BYTES - DEFAULT_AZURE_REQUEST_RESERVED_BYTES
        );
    }

    #[test]
    fn test_with_auth_explicit() {
        use std::sync::Arc;
        let p = OpenAiProvider::with_auth(
            "https://my-foundry.services.ai.azure.com",
            AuthStrategy::Dynamic(Arc::new(|| "my-entra-token".to_string())),
            "gpt-4o",
        );
        assert!(matches!(p.auth, AuthStrategy::Dynamic(_)));
    }

    #[test]
    fn test_builder_methods() {
        let p = OpenAiProvider::new("https://api.openai.com/v1", "key", "gpt-4o")
            .with_context_budget(100_000)
            .with_vision(true)
            .with_max_request_bytes(1_024);
        assert_eq!(p.context_budget_chars(), 100_000);
        assert!(p.supports_vision());
        assert_eq!(p.max_request_bytes, 1_024);
    }

    #[test]
    fn test_chat_url_plain_azure() {
        // Plain Azure OpenAI endpoint (not Foundry) uses deployment-based URL
        let p = OpenAiProvider::new("https://my-resource.openai.azure.com", "key", "gpt-4o");
        assert_eq!(
            p.chat_url(),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn test_chat_url_azure_services() {
        // Azure AI Services endpoint (not Foundry project, no /api/projects/)
        let p = OpenAiProvider::new("https://my-resource.services.ai.azure.com", "key", "gpt-4o");
        assert_eq!(
            p.chat_url(),
            "https://my-resource.services.ai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn test_chat_url_with_query_params() {
        // Azure URLs with api-version query params should not get /chat/completions appended
        let p = OpenAiProvider::new(
            "https://my.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-10-21",
            "key",
            "gpt-4o",
        );
        assert_eq!(
            p.chat_url(),
            "https://my.azure.com/openai/deployments/gpt4/chat/completions?api-version=2024-10-21"
        );
    }

    #[tokio::test]
    async fn test_parse_sse_content_tokens() {
        let (tx, mut rx) = mpsc::channel(32);

        tokio::spawn(async move {
            let mut full_content = String::new();
            let sse_data = vec![
                r#"data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}"#,
                r#"data: {"choices":[{"delta":{"content":" world"},"index":0}]}"#,
                r#"data: [DONE]"#,
            ];

            for line in sse_data {
                let data = line.strip_prefix("data: ").unwrap();
                if data == "[DONE]" {
                    let _ = tx
                        .send(ChatEvent::Done {
                            response: ChatResponse {
                                message: ChatMessage::assistant(&full_content),
                                usage: None,
                            },
                        })
                        .await;
                    break;
                }
                let parsed: serde_json::Value = serde_json::from_str(data).unwrap();
                if let Some(content) = parsed["choices"][0]["delta"]["content"].as_str() {
                    full_content.push_str(content);
                    let _ = tx
                        .send(ChatEvent::Token {
                            token: content.to_string(),
                        })
                        .await;
                }
            }
        });

        let mut events = vec![];
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        assert_eq!(events.len(), 3);
        match &events[0] {
            ChatEvent::Token { token } => assert_eq!(token, "Hello"),
            _ => panic!("Expected Token"),
        }
        match &events[2] {
            ChatEvent::Done { response } => {
                assert_eq!(response.message.text(), Some("Hello world"));
            }
            _ => panic!("Expected Done"),
        }
    }

    #[test]
    fn test_parse_non_streaming_message_content() {
        let value = serde_json::json!({
            "role": "assistant",
            "content": "Hello world"
        });

        let message = parse_chat_completion_message(&value).unwrap();
        assert_eq!(message.role, "assistant");
        assert_eq!(message.text(), Some("Hello world"));
        assert!(message.tool_calls.is_none());
    }

    #[test]
    fn test_parse_non_streaming_message_tool_calls() {
        let value = serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"test.txt\"}"
                }
            }]
        });

        let message = parse_chat_completion_message(&value).unwrap();
        assert_eq!(message.role, "assistant");
        assert!(message.text().is_none());
        let calls = message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, "{\"path\":\"test.txt\"}");
    }

    #[tokio::test]
    async fn test_parse_sse_tool_calls() {
        let (tx, mut rx) = mpsc::channel(32);

        tokio::spawn(async move {
            let sse_data = vec![
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":""}}]},"index":0}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\""}}]},"index":0}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"test.txt\"}"}}]},"index":0}]}"#,
                r#"data: [DONE]"#,
            ];

            let mut tool_calls: HashMap<u32, PendingToolCall> = HashMap::new();

            for line in sse_data {
                let data = line.strip_prefix("data: ").unwrap();
                if data == "[DONE]" {
                    let mut calls: Vec<_> = tool_calls.drain().collect();
                    calls.sort_by_key(|(idx, _)| *idx);
                    let final_calls: Vec<ToolCall> = calls
                        .into_iter()
                        .map(|(_, tc)| tc.into_tool_call())
                        .collect();
                    let msg = ChatMessage::assistant_with_tool_calls(final_calls);
                    let _ = tx
                        .send(ChatEvent::Done {
                            response: ChatResponse {
                                message: msg,
                                usage: None,
                            },
                        })
                        .await;
                    break;
                }

                let parsed: serde_json::Value = serde_json::from_str(data).unwrap();
                if let Some(tc_array) = parsed["choices"][0]["delta"]["tool_calls"].as_array() {
                    for tc in tc_array {
                        let index = tc["index"].as_u64().unwrap_or(0) as u32;
                        let entry = tool_calls.entry(index).or_default();
                        entry.update_from_delta(tc);
                    }
                }
            }
        });

        let mut events = vec![];
        while let Some(event) = rx.recv().await {
            events.push(event);
        }

        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatEvent::Done { response } => {
                let calls = response.message.tool_calls.as_ref().unwrap();
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].function.name, "read_file");
                assert_eq!(calls[0].function.arguments, "{\"path\":\"test.txt\"}");
            }
            _ => panic!("Expected Done with tool calls"),
        }
    }

    #[test]
    fn pending_tool_call_start_emits_once() {
        let mut pending = PendingToolCall::default();

        assert!(pending.update_from_delta(&serde_json::json!({
            "id": "call_1",
            "function": { "name": "read_file", "arguments": "" }
        })));
        assert!(!pending.update_from_delta(&serde_json::json!({
            "function": { "name": "read_file", "arguments": "{\"path\"" }
        })));
        assert!(!pending.update_from_delta(&serde_json::json!({
            "function": { "arguments": ":\"test.txt\"}" }
        })));

        let call = pending.into_tool_call();
        assert_eq!(call.id, "call_1");
        assert_eq!(call.function.name, "read_file");
        assert_eq!(call.function.arguments, "{\"path\":\"test.txt\"}");
    }
}
