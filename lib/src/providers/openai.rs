//! OpenAI-compatible provider.
//!
//! Supports OpenAI, Azure OpenAI, Microsoft Foundry, and any endpoint
//! that implements the `/chat/completions` SSE streaming protocol.

use std::collections::HashMap;

use futures_util::StreamExt;
use reqwest::Client;
use tokio::sync::mpsc;

use crate::auth::AuthStrategy;
use crate::cancel::CancellationToken;
use crate::error::AgentError;
use crate::provider::Provider;
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
        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            auth,
            model: model.to_string(),
            client: Client::new(),
            context_budget: 200_000,
            vision: false,
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

    fn chat_url(&self) -> String {
        if self.endpoint.contains("/chat/completions") {
            self.endpoint.clone()
        } else {
            format!("{}/chat/completions", self.endpoint)
        }
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
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": request.messages,
            "stream": true,
            "stream_options": {"include_usage": true},
            "tools": request.tools,
        });

        // Add response_format if specified
        if let Some(ref rf) = request.response_format {
            body["response_format"] = serde_json::to_value(rf)
                .map_err(|e| AgentError::Stream(format!("Failed to serialize response_format: {}", e)))?;
        }

        let mut req = self.client.post(self.chat_url()).json(&body);
        req = self.auth.apply(req);

        let response = req.send().await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::Api {
                status,
                message: text,
            });
        }

        let mut stream = response.bytes_stream();
        let mut parser = SseParser::new();

        // Accumulate tool calls across chunks (keyed by index)
        let mut tool_calls: HashMap<u32, PendingToolCall> = HashMap::new();
        let mut full_content = String::new();
        let mut usage: Option<Usage> = None;

        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            let chunk = chunk.map_err(|e| AgentError::Stream(e.to_string()))?;
            let data_lines = parser.feed(&chunk);

            for data in data_lines {
                if data == "[DONE]" {
                    let final_tool_calls = if tool_calls.is_empty() {
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
                if let Some(thinking) = delta
                    .get("reasoning_content")
                    .and_then(|c| c.as_str())
                {
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
                        let index =
                            tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as u32;

                        let entry =
                            tool_calls
                                .entry(index)
                                .or_insert_with(|| PendingToolCall {
                                    id: String::new(),
                                    name: String::new(),
                                    arguments: String::new(),
                                });

                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            entry.id = id.to_string();
                        }
                        if let Some(func) = tc.get("function") {
                            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                entry.name = name.to_string();
                            }
                            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                entry.arguments.push_str(args);
                            }
                        }

                        // Emit tool_call start when we first get the name
                        if !entry.name.is_empty()
                            && tc.get("function").and_then(|f| f.get("name")).is_some()
                        {
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

        Err(AgentError::Stream("Stream ended without [DONE]".into()))
    }

    fn name(&self) -> &str {
        "openai"
    }

    fn context_budget_chars(&self) -> usize {
        self.context_budget
    }

    fn supports_vision(&self) -> bool {
        self.vision
    }
}

// -- Helpers -----------------------------------------------------------------

#[derive(Clone)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl PendingToolCall {
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
            .with_vision(true);
        assert_eq!(p.context_budget_chars(), 100_000);
        assert!(p.supports_vision());
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
                    let final_calls: Vec<ToolCall> =
                        calls.into_iter().map(|(_, tc)| tc.into_tool_call()).collect();
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
                        let entry =
                            tool_calls
                                .entry(index)
                                .or_insert_with(|| PendingToolCall {
                                    id: String::new(),
                                    name: String::new(),
                                    arguments: String::new(),
                                });
                        if let Some(id) = tc["id"].as_str() {
                            entry.id = id.to_string();
                        }
                        if let Some(name) = tc["function"]["name"].as_str() {
                            entry.name = name.to_string();
                        }
                        if let Some(args) = tc["function"]["arguments"].as_str() {
                            entry.arguments.push_str(args);
                        }
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
}
