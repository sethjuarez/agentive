//! Anthropic Messages API provider with SSE streaming.
//!
//! Handles the Anthropic-specific message format, system prompt extraction,
//! tool_use/tool_result content blocks, and the distinct SSE event protocol.

use futures_util::StreamExt;
use reqwest::Client;
use tokio::sync::mpsc;

use crate::cancel::CancellationToken;
use crate::error::AgentError;
use crate::provider::Provider;
use crate::providers::sse::SseParser;
use crate::types::*;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider.
pub struct AnthropicProvider {
    api_key: String,
    model: String,
    client: Client,
    context_budget: usize,
}

impl AnthropicProvider {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: Client::new(),
            context_budget: 200_000,
        }
    }

    /// Set the context budget in characters.
    pub fn with_context_budget(mut self, chars: usize) -> Self {
        self.context_budget = chars;
        self
    }

    /// Convert unified messages to Anthropic format.
    /// Anthropic requires system as a top-level param, not in messages.
    /// Tool results use a different format than OpenAI.
    fn prepare_request(
        &self,
        request: &ChatRequest,
    ) -> (
        Option<String>,
        Vec<serde_json::Value>,
        Option<Vec<serde_json::Value>>,
    ) {
        let mut system_prompt = None;
        let mut messages = Vec::new();

        for msg in &request.messages {
            match msg.role.as_str() {
                "system" => {
                    system_prompt = msg.text().map(String::from);
                }
                "tool" => {
                    // Anthropic expects tool results as user messages with tool_result content blocks
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.text(),
                        }]
                    }));
                }
                "assistant" => {
                    if let Some(ref tool_calls) = msg.tool_calls {
                        let mut content: Vec<serde_json::Value> = Vec::new();
                        if let Some(text) = msg.text() {
                            if !text.is_empty() {
                                content.push(serde_json::json!({
                                    "type": "text",
                                    "text": text,
                                }));
                            }
                        }
                        for tc in tool_calls {
                            let args: serde_json::Value =
                                serde_json::from_str(&tc.function.arguments)
                                    .unwrap_or(serde_json::json!({}));
                            content.push(serde_json::json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": args,
                            }));
                        }
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": content,
                        }));
                    } else {
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": msg.text(),
                        }));
                    }
                }
                _ => {
                    // User messages — handle multimodal content
                    match &msg.content {
                        Some(MessageContent::Parts(parts)) => {
                            let content: Vec<serde_json::Value> = parts
                                .iter()
                                .map(|p| match p {
                                    ContentPart::Text { text } => {
                                        serde_json::json!({ "type": "text", "text": text })
                                    }
                                    ContentPart::ImageUrl { image_url } => {
                                        serde_json::json!({
                                            "type": "image",
                                            "source": {
                                                "type": "url",
                                                "url": image_url.url,
                                            }
                                        })
                                    }
                                })
                                .collect();
                            messages.push(serde_json::json!({
                                "role": msg.role,
                                "content": content,
                            }));
                        }
                        _ => {
                            messages.push(serde_json::json!({
                                "role": msg.role,
                                "content": msg.text(),
                            }));
                        }
                    }
                }
            }
        }

        // Convert tools to Anthropic format
        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.function.name,
                        "description": t.function.description,
                        "input_schema": t.function.parameters,
                    })
                })
                .collect()
        });

        (system_prompt, messages, tools)
    }
}

#[async_trait::async_trait]
impl Provider for AnthropicProvider {
    async fn chat(
        &self,
        request: ChatRequest,
        tx: mpsc::Sender<ChatEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), AgentError> {
        let (system_prompt, messages, tools) = self.prepare_request(&request);

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": true,
        });

        if let Some(sys) = &system_prompt {
            body["system"] = serde_json::json!(sys);
        }
        if let Some(t) = &tools {
            body["tools"] = serde_json::json!(t);
        }

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

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

        // Accumulate state across events
        let mut full_content = String::new();
        let mut pending_tool_uses: Vec<PendingToolUse> = Vec::new();
        let mut input_usage: Option<u32> = None;
        let mut output_usage: Option<u32> = None;

        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            let chunk = chunk.map_err(|e| AgentError::Stream(e.to_string()))?;
            let data_lines = parser.feed(&chunk);

            for data in data_lines {
                let parsed: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let event_type = parsed
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");

                match event_type {
                    "message_start" => {
                        if let Some(usage_obj) =
                            parsed.get("message").and_then(|m| m.get("usage"))
                        {
                            input_usage = usage_obj
                                .get("input_tokens")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32);
                        }
                    }
                    "content_block_start" => {
                        if let Some(block) = parsed.get("content_block") {
                            let block_type = block
                                .get("type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("text");

                            if block_type == "tool_use" {
                                let id = block
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let name = block
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                pending_tool_uses.push(PendingToolUse {
                                    id: id.clone(),
                                    name: name.clone(),
                                    arguments_json: String::new(),
                                });

                                let _ = tx
                                    .send(ChatEvent::ToolCallStart {
                                        tool_call: ToolCall {
                                            id,
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
                    }
                    "content_block_delta" => {
                        if let Some(delta) = parsed.get("delta") {
                            let delta_type =
                                delta.get("type").and_then(|t| t.as_str()).unwrap_or("");

                            match delta_type {
                                "text_delta" => {
                                    if let Some(text) =
                                        delta.get("text").and_then(|t| t.as_str())
                                    {
                                        full_content.push_str(text);
                                        let _ = tx
                                            .send(ChatEvent::Token {
                                                token: text.to_string(),
                                            })
                                            .await;
                                    }
                                }
                                "thinking_delta" => {
                                    if let Some(text) =
                                        delta.get("thinking").and_then(|t| t.as_str())
                                    {
                                        let _ = tx
                                            .send(ChatEvent::Thinking {
                                                token: text.to_string(),
                                            })
                                            .await;
                                    }
                                }
                                "input_json_delta" => {
                                    if let Some(json_str) =
                                        delta.get("partial_json").and_then(|j| j.as_str())
                                    {
                                        if let Some(last) = pending_tool_uses.last_mut() {
                                            last.arguments_json.push_str(json_str);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    "content_block_stop" => {}
                    "message_delta" => {
                        if let Some(usage_obj) = parsed.get("usage") {
                            output_usage = usage_obj
                                .get("output_tokens")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32);
                        }
                    }
                    "message_stop" => {
                        let usage = match (input_usage, output_usage) {
                            (Some(input), Some(output)) => Some(Usage {
                                prompt_tokens: input,
                                completion_tokens: output,
                                total_tokens: input + output,
                            }),
                            _ => None,
                        };

                        let message = if pending_tool_uses.is_empty() {
                            ChatMessage::assistant(&full_content)
                        } else {
                            let tool_calls: Vec<ToolCall> = pending_tool_uses
                                .drain(..)
                                .map(|tu| tu.into_tool_call())
                                .collect();
                            let mut msg = ChatMessage::assistant_with_tool_calls(tool_calls);
                            if !full_content.is_empty() {
                                msg.content = Some(MessageContent::Text(full_content.clone()));
                            }
                            msg
                        };

                        let _ = tx
                            .send(ChatEvent::Done {
                                response: ChatResponse { message, usage },
                            })
                            .await;
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }

        Err(AgentError::Stream(
            "Stream ended without message_stop".into(),
        ))
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn context_budget_chars(&self) -> usize {
        self.context_budget
    }
}

// -- Helpers -----------------------------------------------------------------

struct PendingToolUse {
    id: String,
    name: String,
    arguments_json: String,
}

impl PendingToolUse {
    fn into_tool_call(self) -> ToolCall {
        ToolCall {
            id: self.id,
            call_type: "function".into(),
            function: FunctionCall {
                name: self.name,
                arguments: self.arguments_json,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_request_system_extraction() {
        let provider = AnthropicProvider::new("key", "claude-sonnet-4-20250514");
        let request = ChatRequest {
            messages: vec![
                ChatMessage::system("You are helpful"),
                ChatMessage::user("Hello"),
            ],
            model: "claude-sonnet-4-20250514".into(),
            tools: None,
            stream: true,
            response_format: None,
        };

        let (system, messages, tools) = provider.prepare_request(&request);
        assert_eq!(system.as_deref(), Some("You are helpful"));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert!(tools.is_none());
    }

    #[test]
    fn test_prepare_request_tool_result_conversion() {
        let provider = AnthropicProvider::new("key", "claude-sonnet-4-20250514");

        let tool_call = ToolCall {
            id: "toolu_123".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: "{\"path\":\"test.txt\"}".into(),
            },
        };

        let request = ChatRequest {
            messages: vec![
                ChatMessage::user("Read my file"),
                ChatMessage::assistant_with_tool_calls(vec![tool_call]),
                ChatMessage::tool_result("toolu_123", "File content here"),
            ],
            model: "claude-sonnet-4-20250514".into(),
            tools: None,
            stream: true,
            response_format: None,
        };

        let (_, messages, _) = provider.prepare_request(&request);
        assert_eq!(messages.len(), 3);

        // Tool result should be converted to user message with tool_result block
        assert_eq!(messages[2]["role"], "user");
        let content = messages[2]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_123");
    }

    #[test]
    fn test_prepare_request_assistant_with_tool_calls() {
        let provider = AnthropicProvider::new("key", "claude-sonnet-4-20250514");

        let request = ChatRequest {
            messages: vec![ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                id: "toolu_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: "{\"path\":\"test.txt\"}".into(),
                },
            }])],
            model: "claude-sonnet-4-20250514".into(),
            tools: None,
            stream: true,
            response_format: None,
        };

        let (_, messages, _) = provider.prepare_request(&request);
        let content = messages[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["name"], "read_file");
    }

    #[test]
    fn test_prepare_request_multimodal_user() {
        let provider = AnthropicProvider::new("key", "claude-sonnet-4-20250514");

        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Describe this".into(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/img.png".into(),
                            detail: None,
                        },
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
            }],
            model: "claude-sonnet-4-20250514".into(),
            tools: None,
            stream: true,
            response_format: None,
        };

        let (_, messages, _) = provider.prepare_request(&request);
        assert_eq!(messages.len(), 1);
        let content = messages[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Describe this");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "url");
        assert_eq!(content[1]["source"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn test_prepare_request_multiple_systems_last_wins() {
        let provider = AnthropicProvider::new("key", "claude-sonnet-4-20250514");

        let request = ChatRequest {
            messages: vec![
                ChatMessage::system("First system"),
                ChatMessage::system("Second system"),
                ChatMessage::user("Hello"),
            ],
            model: "claude-sonnet-4-20250514".into(),
            tools: None,
            stream: true,
            response_format: None,
        };

        let (system, messages, _) = provider.prepare_request(&request);
        assert_eq!(system.as_deref(), Some("Second system"));
        assert_eq!(messages.len(), 1); // only user
    }

    #[test]
    fn test_prepare_request_tool_definitions() {
        let provider = AnthropicProvider::new("key", "claude-sonnet-4-20250514");

        let request = ChatRequest {
            messages: vec![ChatMessage::user("Hello")],
            model: "claude-sonnet-4-20250514".into(),
            tools: Some(vec![Tool::function(
                "read_file",
                "Reads a file",
                serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            )]),
            stream: true,
            response_format: None,
        };

        let (_, _, tools) = provider.prepare_request(&request);
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "read_file");
        assert_eq!(tools[0]["description"], "Reads a file");
        assert!(tools[0]["input_schema"].is_object());
    }
}
