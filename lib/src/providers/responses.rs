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
///
/// **Important:** The Azure Responses API endpoint silently truncates request
/// bodies at ~79KB, causing JSON parse errors. This provider defaults to a
/// `max_request_bytes` of 64KB and will automatically drop oldest conversation
/// items to stay under that limit. Override with
/// [`with_max_request_bytes`](Self::with_max_request_bytes) if your endpoint
/// supports larger payloads.
pub struct ResponsesProvider {
    endpoint: String,
    auth: AuthStrategy,
    model: String,
    client: Client,
    context_budget: usize,
    vision: bool,
    max_request_bytes: usize,
}

/// Default max request body size (64KB) — safely below the ~79KB Azure limit.
const DEFAULT_MAX_REQUEST_BYTES: usize = 64 * 1024;

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
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
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
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
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

    /// Set the maximum request body size in bytes.
    ///
    /// The Azure Responses API silently truncates at ~79KB. Default is 64KB.
    /// Set to `usize::MAX` to disable the guard.
    pub fn with_max_request_bytes(mut self, bytes: usize) -> Self {
        self.max_request_bytes = bytes;
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

    /// Build the JSON request body, compacting input items if the serialized
    /// size exceeds `max_request_bytes`. Drops oldest non-system items first,
    /// keeping tool_call/function_call_output pairs together.
    fn build_body_within_limit(
        &self,
        input: &mut Vec<serde_json::Value>,
        tools_json: &Option<serde_json::Value>,
        stream: bool,
    ) -> Result<serde_json::Value, AgentError> {
        let make_body = |input: &[serde_json::Value]| -> serde_json::Value {
            let mut b = serde_json::json!({
                "model": self.model,
                "input": input,
                "stream": stream,
            });
            if let Some(ref tools) = tools_json {
                b["tools"] = tools.clone();
            }
            b
        };

        let body = make_body(input);
        let serialized = serde_json::to_string(&body)
            .map_err(|e| AgentError::Stream(format!("Failed to serialize request: {e}")))?;

        if serialized.len() <= self.max_request_bytes {
            return Ok(body);
        }

        // Body too large — drop oldest non-system input items.
        // Skip system/developer messages at the front.
        let sys_end = input
            .iter()
            .position(|item| {
                item.get("role")
                    .and_then(|r| r.as_str())
                    .map(|r| r != "system" && r != "developer")
                    .unwrap_or(true)
            })
            .unwrap_or(input.len());

        let mut dropped = 0;
        while input.len() > sys_end + 2 {
            let trial = make_body(input);
            let size = serde_json::to_string(&trial)
                .map(|s| s.len())
                .unwrap_or(usize::MAX);
            if size <= self.max_request_bytes {
                break;
            }

            // Remove the first conversation item (after system prefix)
            let removed = input.remove(sys_end);
            dropped += 1;

            // If we removed a function_call item, also remove its matching
            // function_call_output items to avoid orphaned tool results
            if removed.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                if let Some(call_id) = removed.get("call_id").and_then(|c| c.as_str()) {
                    while input.len() > sys_end
                        && input[sys_end].get("type").and_then(|t| t.as_str())
                            == Some("function_call_output")
                        && input[sys_end].get("call_id").and_then(|c| c.as_str())
                            == Some(call_id)
                    {
                        input.remove(sys_end);
                        dropped += 1;
                    }
                }
            }
        }

        if dropped > 0 {
            log::warn!(
                "[agentive] Responses body exceeded {}KB — dropped {} input items to fit",
                self.max_request_bytes / 1024,
                dropped
            );
        }

        Ok(make_body(input))
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
        let mut input = self.messages_to_input(&request.messages);

        let tools_json = request.tools.as_ref().and_then(|tools| {
            if tools.is_empty() {
                None
            } else {
                Some(serde_json::json!(self.tools_to_responses_format(tools)))
            }
        });

        // Build body and enforce byte limit.
        // The Azure Responses API silently truncates at ~79KB — we compact
        // by dropping oldest non-system input items until the serialized
        // body fits. This uses the actual byte count, not character estimates.
        let body = self.build_body_within_limit(&mut input, &tools_json, request.stream)?;

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

        if !request.stream {
            return handle_non_streaming_response(response, tx).await;
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

async fn handle_non_streaming_response(
    response: reqwest::Response,
    tx: mpsc::Sender<ChatEvent>,
) -> Result<(), AgentError> {
    let parsed: serde_json::Value = response.json().await?;
    let usage = parsed.get("usage").map(parse_responses_usage).transpose()?;
    let (content, tool_calls) = parse_responses_output(&parsed)?;

    let message = ChatMessage {
        role: "assistant".into(),
        content: Some(MessageContent::Text(content)),
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

fn parse_responses_usage(value: &serde_json::Value) -> Result<Usage, AgentError> {
    let input_tokens = value.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let output_tokens = value.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    Ok(Usage {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        total_tokens: input_tokens + output_tokens,
    })
}

fn parse_responses_output(
    parsed: &serde_json::Value,
) -> Result<(String, Option<Vec<ToolCall>>), AgentError> {
    let Some(output) = parsed.get("output").and_then(|output| output.as_array()) else {
        return Err(AgentError::Stream(
            "Non-streaming response missing output array".into(),
        ));
    };

    let mut full_content = String::new();
    let mut tool_calls = Vec::new();

    for item in output {
        match item.get("type").and_then(|kind| kind.as_str()) {
            Some("message") => {
                if let Some(content) = item.get("content").and_then(|content| content.as_array()) {
                    for part in content {
                        if part.get("type").and_then(|kind| kind.as_str()) == Some("output_text") {
                            if let Some(text) = part.get("text").and_then(|text| text.as_str()) {
                                full_content.push_str(text);
                            }
                        }
                    }
                }
            }
            Some("function_call") => {
                tool_calls.push(ToolCall {
                    id: item
                        .get("call_id")
                        .and_then(|call_id| call_id.as_str())
                        .unwrap_or("")
                        .to_string(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: item
                            .get("name")
                            .and_then(|name| name.as_str())
                            .unwrap_or("")
                            .to_string(),
                        arguments: item
                            .get("arguments")
                            .and_then(|arguments| arguments.as_str())
                            .unwrap_or("")
                            .to_string(),
                    },
                });
            }
            _ => {}
        }
    }

    Ok((
        full_content,
        if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
    ))
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

    #[test]
    fn test_body_within_limit_no_compaction() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o")
            .with_max_request_bytes(64 * 1024);

        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let mut input = provider.messages_to_input(&messages);
        let body = provider.build_body_within_limit(&mut input, &None, true).unwrap();

        // Small body should pass through unchanged
        assert_eq!(input.len(), 2);
        assert!(body.get("input").unwrap().as_array().unwrap().len() == 2);
    }

    #[test]
    fn test_body_honors_non_streaming_request() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o");
        let messages = vec![ChatMessage::user("Hello")];
        let mut input = provider.messages_to_input(&messages);

        let body = provider.build_body_within_limit(&mut input, &None, false).unwrap();

        assert_eq!(body["stream"], false);
    }

    #[test]
    fn test_parse_non_streaming_responses_output() {
        let parsed = serde_json::json!({
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Hello"
                }, {
                    "type": "output_text",
                    "text": " world"
                }]
            }, {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": "{\"path\":\"test.txt\"}"
            }]
        });

        let (content, tool_calls) = parse_responses_output(&parsed).unwrap();

        assert_eq!(content, "Hello world");
        let calls = tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, "{\"path\":\"test.txt\"}");
    }

    #[test]
    fn test_body_exceeds_limit_compacts() {
        // Use a very small limit to force compaction
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o")
            .with_max_request_bytes(200);

        let messages = vec![
            ChatMessage::system("System prompt"),
            ChatMessage::user(&"A".repeat(100)),
            ChatMessage::user(&"B".repeat(100)),
            ChatMessage::user("Last"),
        ];
        let mut input = provider.messages_to_input(&messages);
        let original_len = input.len();

        let body = provider.build_body_within_limit(&mut input, &None, true).unwrap();

        // Should have dropped some items
        let final_len = body.get("input").unwrap().as_array().unwrap().len();
        assert!(final_len < original_len, "Expected compaction: {} < {}", final_len, original_len);
        // System/developer message should be preserved
        assert_eq!(
            body["input"][0]["role"].as_str().unwrap(),
            "developer"
        );
    }

    #[test]
    fn test_body_compaction_preserves_system_messages() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o")
            .with_max_request_bytes(300);

        let messages = vec![
            ChatMessage::system("Important system prompt"),
            ChatMessage::user(&"A".repeat(200)),
            ChatMessage::user("Short"),
        ];
        let mut input = provider.messages_to_input(&messages);
        let _ = provider.build_body_within_limit(&mut input, &None, true).unwrap();

        // Developer (system) message should never be dropped
        assert!(input[0].get("role").unwrap().as_str().unwrap() == "developer");
    }

    #[test]
    fn test_body_compaction_drops_tool_pairs() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o")
            .with_max_request_bytes(400);

        // Build input items directly to simulate function_call + output pairs
        let mut input = vec![
            serde_json::json!({"role": "developer", "content": "sys"}),
            serde_json::json!({"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}" }),
            serde_json::json!({"type": "function_call_output", "call_id": "c1", "output": "A".repeat(300)}),
            serde_json::json!({"role": "user", "content": "ok"}),
        ];

        let _ = provider.build_body_within_limit(&mut input, &None, true).unwrap();

        // The function_call + its output should be dropped together
        // No orphaned function_call_output should remain
        for item in &input {
            if item.get("type").and_then(|t| t.as_str()) == Some("function_call_output") {
                // If a function_call_output remains, its matching function_call must also remain
                let call_id = item["call_id"].as_str().unwrap();
                assert!(
                    input.iter().any(|i| {
                        i.get("type").and_then(|t| t.as_str()) == Some("function_call")
                            && i.get("call_id").and_then(|c| c.as_str()) == Some(call_id)
                    }),
                    "Orphaned function_call_output for call_id={call_id}"
                );
            }
        }
    }

    #[test]
    fn test_with_max_request_bytes_disabled() {
        let provider = ResponsesProvider::new("https://api.openai.com", "key", "gpt-4o")
            .with_max_request_bytes(usize::MAX);

        let messages = vec![
            ChatMessage::system("System"),
            ChatMessage::user(&"A".repeat(100_000)),
        ];
        let mut input = provider.messages_to_input(&messages);
        let body = provider.build_body_within_limit(&mut input, &None, true).unwrap();

        // With max disabled, no compaction should occur
        assert_eq!(body["input"].as_array().unwrap().len(), 2);
    }
}
