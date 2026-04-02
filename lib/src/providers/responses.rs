//! OpenAI Responses API provider.
//!
//! Supports the newer `/v1/responses` endpoint used by OpenAI, Azure OpenAI,
//! and Microsoft Foundry. This API uses a different request/response format
//! and different SSE event types compared to Chat Completions.
//!
//! The provider transparently converts between agentive's internal types
//! (`ChatMessage`, `ToolCall`, etc.) and the Responses API format.

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

/// OpenAI Responses API provider.
///
/// Unlike [`super::openai::OpenAiProvider`] which uses `/chat/completions`,
/// this provider uses the `/v1/responses` endpoint with its different
/// input/output format and SSE event types.
pub struct ResponsesProvider {
    endpoint: String,
    auth: AuthStrategy,
    model: String,
    client: Client,
    context_budget: usize,
    vision: bool,
}

impl ResponsesProvider {
    /// Create a new Responses API provider.
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

    /// Enable vision/image support.
    pub fn with_vision(mut self, enabled: bool) -> Self {
        self.vision = enabled;
        self
    }

    fn responses_url(&self) -> String {
        let is_azure = self.endpoint.contains("azure.com");
        if is_azure {
            format!("{}/openai/v1/responses", self.endpoint)
        } else {
            format!("{}/v1/responses", self.endpoint)
        }
    }

    /// Convert agentive ChatMessages to Responses API input items.
    fn messages_to_input(&self, messages: &[ChatMessage]) -> Vec<serde_json::Value> {
        let mut input = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    input.push(serde_json::json!({
                        "role": "developer",
                        "content": msg.text().unwrap_or("")
                    }));
                }
                "user" => {
                    match &msg.content {
                        Some(MessageContent::Text(text)) => {
                            input.push(serde_json::json!({
                                "role": "user",
                                "content": text
                            }));
                        }
                        Some(MessageContent::Parts(parts)) => {
                            let content: Vec<serde_json::Value> = parts
                                .iter()
                                .map(|p| match p {
                                    ContentPart::Text { text } => {
                                        serde_json::json!({
                                            "type": "input_text",
                                            "text": text
                                        })
                                    }
                                    ContentPart::ImageUrl { image_url } => {
                                        serde_json::json!({
                                            "type": "input_image",
                                            "image_url": image_url.url
                                        })
                                    }
                                })
                                .collect();
                            input.push(serde_json::json!({
                                "role": "user",
                                "content": content
                            }));
                        }
                        None => {}
                    }
                }
                "assistant" => {
                    // Text content as message item
                    if let Some(text) = msg.text() {
                        if !text.is_empty() {
                            input.push(serde_json::json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": text
                                }]
                            }));
                        }
                    }
                    // Tool calls as separate function_call items
                    if let Some(ref tool_calls) = msg.tool_calls {
                        for tc in tool_calls {
                            input.push(serde_json::json!({
                                "type": "function_call",
                                "call_id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }));
                        }
                    }
                }
                "tool" => {
                    // Tool result → function_call_output
                    if let Some(ref tc_id) = msg.tool_call_id {
                        input.push(serde_json::json!({
                            "type": "function_call_output",
                            "call_id": tc_id,
                            "output": msg.text().unwrap_or("")
                        }));
                    }
                }
                _ => {}
            }
        }
        input
    }

    /// Convert agentive Tool definitions to Responses API format.
    /// Responses API uses flattened `{ type, name, description, parameters }`
    /// instead of `{ type: "function", function: { name, ... } }`.
    fn tools_to_responses_format(&self, tools: &[Tool]) -> Vec<serde_json::Value> {
        tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters
                })
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl Provider for ResponsesProvider {
    async fn chat(
        &self,
        request: ChatRequest,
        tx: mpsc::Sender<ChatEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), AgentError> {
        let input = self.messages_to_input(&request.messages);

        let mut body = serde_json::json!({
            "model": self.model,
            "input": input,
            "stream": true,
        });

        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                body["tools"] = serde_json::json!(self.tools_to_responses_format(tools));
            }
        }

        let mut req = self.client.post(self.responses_url()).json(&body);
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

        // Track pending tool calls by output_index
        let mut pending_tool_calls: HashMap<u32, PendingToolCall> = HashMap::new();
        let mut full_content = String::new();
        let mut usage: Option<Usage> = None;
        let mut got_completed = false;

        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            let chunk = chunk.map_err(|e| AgentError::Stream(e.to_string()))?;
            let data_lines = parser.feed(&chunk);

            for data in data_lines {
                if data == "[DONE]" {
                    break;
                }

                let parsed: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let event_type = parsed["type"].as_str().unwrap_or("");

                match event_type {
                    "response.output_text.delta" => {
                        if let Some(delta) = parsed["delta"].as_str() {
                            full_content.push_str(delta);
                            let _ = tx
                                .send(ChatEvent::Token {
                                    token: delta.to_string(),
                                })
                                .await;
                        }
                    }
                    "response.output_item.added" => {
                        let item = &parsed["item"];
                        if item["type"].as_str() == Some("function_call") {
                            let output_index =
                                parsed["output_index"].as_u64().unwrap_or(0) as u32;
                            let call_id =
                                item["call_id"].as_str().unwrap_or("").to_string();
                            let name =
                                item["name"].as_str().unwrap_or("").to_string();

                            pending_tool_calls.insert(
                                output_index,
                                PendingToolCall {
                                    id: call_id.clone(),
                                    name: name.clone(),
                                    arguments: String::new(),
                                },
                            );

                            let _ = tx
                                .send(ChatEvent::ToolCallStart {
                                    tool_call: ToolCall {
                                        id: call_id,
                                        call_type: "function".into(),
                                        function: FunctionCall {
                                            name,
                                            arguments: String::new(),
                                        },
                                    },
                                })
                                .await;
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        let output_index =
                            parsed["output_index"].as_u64().unwrap_or(0) as u32;
                        if let Some(delta) = parsed["delta"].as_str() {
                            if let Some(tc) = pending_tool_calls.get_mut(&output_index) {
                                tc.arguments.push_str(delta);
                            }
                        }
                    }
                    "response.completed" => {
                        got_completed = true;
                        // Extract usage if present
                        if let Some(resp_usage) = parsed.get("response").and_then(|r| r.get("usage")) {
                            let input_tokens = resp_usage["input_tokens"].as_u64().unwrap_or(0) as u32;
                            let output_tokens = resp_usage["output_tokens"].as_u64().unwrap_or(0) as u32;
                            usage = Some(Usage {
                                prompt_tokens: input_tokens,
                                completion_tokens: output_tokens,
                                total_tokens: input_tokens + output_tokens,
                            });
                        }
                    }
                    _ => {
                        // Ignore unknown event types
                    }
                }
            }
        }

        if !got_completed {
            return Err(AgentError::Stream(
                "Stream ended without response.completed".into(),
            ));
        }

        // Build final response
        let tool_calls = if pending_tool_calls.is_empty() {
            None
        } else {
            let mut calls: Vec<_> = pending_tool_calls.drain().collect();
            calls.sort_by_key(|(idx, _)| *idx);
            Some(
                calls
                    .into_iter()
                    .map(|(_, tc)| ToolCall {
                        id: tc.id,
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: tc.name,
                            arguments: tc.arguments,
                        },
                    })
                    .collect(),
            )
        };

        let message = ChatMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Text(full_content)),
            tool_calls,
            tool_call_id: None,
        };

        let _ = tx
            .send(ChatEvent::Done {
                response: ChatResponse { message, usage },
            })
            .await;

        Ok(())
    }

    fn name(&self) -> &str {
        "responses"
    }

    fn context_budget_chars(&self) -> usize {
        self.context_budget
    }

    fn supports_vision(&self) -> bool {
        self.vision
    }
}

/// Pending tool call accumulator for streaming.
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_messages_to_input_basic() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o");

        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];

        let input = provider.messages_to_input(&messages);

        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "developer");
        assert_eq!(input[0]["content"], "You are helpful");
        assert_eq!(input[1]["role"], "user");
        assert_eq!(input[1]["content"], "Hello");
    }

    #[test]
    fn test_messages_to_input_tool_calls() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o");

        let messages = vec![
            ChatMessage::user("Read the file"),
            ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                id: "call_123".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: r#"{"path":"test.txt"}"#.into(),
                },
            }]),
            ChatMessage::tool_result("call_123", "file contents here"),
        ];

        let input = provider.messages_to_input(&messages);

        assert_eq!(input.len(), 3);
        // User message
        assert_eq!(input[0]["role"], "user");
        // Tool call → function_call item
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_123");
        assert_eq!(input[1]["name"], "read_file");
        // Tool result → function_call_output item
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_123");
        assert_eq!(input[2]["output"], "file contents here");
    }

    #[test]
    fn test_tools_to_responses_format() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o");

        let tools = vec![Tool::function(
            "read_file",
            "Read a file",
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )];

        let formatted = provider.tools_to_responses_format(&tools);
        assert_eq!(formatted.len(), 1);
        assert_eq!(formatted[0]["type"], "function");
        assert_eq!(formatted[0]["name"], "read_file");
        assert_eq!(formatted[0]["description"], "Read a file");
        // No nested "function" wrapper
        assert!(formatted[0].get("function").is_none());
    }

    #[test]
    fn test_responses_url_openai() {
        let p = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o");
        assert_eq!(p.responses_url(), "https://api.openai.com/v1/responses");
    }

    #[test]
    fn test_responses_url_azure() {
        let p = ResponsesProvider::new(
            "https://my-resource.openai.azure.com",
            "key",
            "gpt-4o",
        );
        assert_eq!(
            p.responses_url(),
            "https://my-resource.openai.azure.com/openai/v1/responses"
        );
    }

    #[test]
    fn test_assistant_text_and_tool_calls() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o");

        // Assistant with both text and tool calls
        let mut msg = ChatMessage::assistant("I'll read that for you");
        msg.tool_calls = Some(vec![ToolCall {
            id: "c1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "read".into(),
                arguments: "{}".into(),
            },
        }]);

        let input = provider.messages_to_input(&[msg]);

        // Should produce both a message item and a function_call item
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "assistant");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "c1");
    }

    #[test]
    fn test_multimodal_parts_conversion() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o");

        let msg = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What's in this image?".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/photo.png".into(),
                        detail: None,
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
        };

        let input = provider.messages_to_input(&[msg]);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");

        let content = input[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "What's in this image?");
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["image_url"], "https://example.com/photo.png");
    }

    #[test]
    fn test_empty_messages_conversion() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o");
        let input = provider.messages_to_input(&[]);
        assert!(input.is_empty());
    }

    #[test]
    fn test_user_none_content_skipped() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o");
        let msg = ChatMessage {
            role: "user".into(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let input = provider.messages_to_input(&[msg]);
        assert!(input.is_empty()); // None content user messages are skipped
    }
}
