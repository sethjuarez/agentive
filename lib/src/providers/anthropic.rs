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
use crate::providers::request_compaction::compact_items_to_request_limit;
use crate::providers::sse::{SseEvent, SseEventParser};
use crate::types::*;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API provider.
pub struct AnthropicProvider {
    api_key: String,
    model: String,
    client: Client,
    context_budget: usize,
    max_request_bytes: usize,
}

impl AnthropicProvider {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: Client::new(),
            context_budget: 200_000,
            max_request_bytes: usize::MAX,
        }
    }

    /// Set the context budget in characters.
    pub fn with_context_budget(mut self, chars: usize) -> Self {
        self.context_budget = chars;
        self
    }

    /// Set the maximum serialized request body size in bytes.
    ///
    /// Anthropic does not default to a provider-specific byte cap, but callers
    /// can opt in to the same shared request compaction used by other providers.
    pub fn with_max_request_bytes(mut self, bytes: usize) -> Self {
        self.max_request_bytes = bytes;
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

    fn build_body_within_limit(
        &self,
        messages: &mut Vec<serde_json::Value>,
        system_prompt: &Option<String>,
        tools: &Option<Vec<serde_json::Value>>,
    ) -> Result<serde_json::Value, AgentError> {
        let make_body =
            |messages: &[serde_json::Value]| Ok(self.request_body(messages, system_prompt, tools));

        compact_items_to_request_limit(
            messages,
            self.max_request_bytes,
            0,
            "Anthropic",
            "message",
            make_body,
            |_| false,
            remove_anthropic_message_group,
        )
    }
}

fn remove_anthropic_message_group(messages: &mut Vec<serde_json::Value>, idx: usize) -> usize {
    if idx >= messages.len() {
        return 0;
    }

    let removed = messages.remove(idx);
    let mut dropped = 1usize;
    if removed.get("role").and_then(|r| r.as_str()) == Some("user") {
        while idx < messages.len()
            && messages[idx].get("role").and_then(|r| r.as_str()) != Some("user")
        {
            messages.remove(idx);
            dropped += 1;
        }
        return dropped;
    }

    let tool_use_ids = removed
        .get("content")
        .and_then(|content| content.as_array())
        .into_iter()
        .flatten()
        .filter(|part| part.get("type").and_then(|kind| kind.as_str()) == Some("tool_use"))
        .filter_map(|part| part.get("id").and_then(|id| id.as_str()))
        .map(String::from)
        .collect::<std::collections::HashSet<_>>();

    if tool_use_ids.is_empty() {
        return dropped;
    }

    while idx < messages.len() && anthropic_tool_result_matches(&messages[idx], &tool_use_ids) {
        messages.remove(idx);
        dropped += 1;
    }

    dropped
}

fn anthropic_tool_result_matches(
    message: &serde_json::Value,
    tool_use_ids: &std::collections::HashSet<String>,
) -> bool {
    message
        .get("content")
        .and_then(|content| content.as_array())
        .into_iter()
        .flatten()
        .any(|part| {
            part.get("type").and_then(|kind| kind.as_str()) == Some("tool_result")
                && part
                    .get("tool_use_id")
                    .and_then(|id| id.as_str())
                    .is_some_and(|id| tool_use_ids.contains(id))
        })
}

#[async_trait::async_trait]
impl Provider for AnthropicProvider {
    async fn chat(
        &self,
        request: ChatRequest,
        tx: mpsc::Sender<ChatEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), AgentError> {
        let (system_prompt, mut messages, tools) = self.prepare_request(&request);
        let body = self.build_body_within_limit(&mut messages, &system_prompt, &tools)?;

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .header("accept-encoding", "identity")
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
        let mut parser = SseEventParser::new();

        // Accumulate state across events
        let mut state = AnthropicStreamState::default();

        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            let chunk = chunk.map_err(|e| anthropic_stream_chunk_error(&e, &state))?;
            state.bytes_seen += chunk.len();
            let events = parser.feed(&chunk);

            for event in events {
                if process_anthropic_stream_event(event, &mut state, &tx).await? {
                    return Ok(());
                }
            }
        }

        for event in parser.finish() {
            if process_anthropic_stream_event(event, &mut state, &tx).await? {
                return Ok(());
            }
        }

        Err(incomplete_anthropic_stream_error(&state))
    }

    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn context_budget_chars(&self) -> usize {
        self.context_budget
    }

    fn request_budget_bytes(&self) -> Option<usize> {
        (self.max_request_bytes != usize::MAX).then_some(self.max_request_bytes)
    }

    fn estimate_request_bytes(&self, request: &ChatRequest) -> Result<Option<usize>, AgentError> {
        if self.request_budget_bytes().is_none() {
            return Ok(None);
        }

        let (system_prompt, messages, tools) = self.prepare_request(request);
        serde_json::to_string(&self.request_body(&messages, &system_prompt, &tools))
            .map(|body| Some(body.len()))
            .map_err(AgentError::from)
    }
}

// -- Helpers -----------------------------------------------------------------

impl AnthropicProvider {
    fn request_body(
        &self,
        messages: &[serde_json::Value],
        system_prompt: &Option<String>,
        tools: &Option<Vec<serde_json::Value>>,
    ) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": true,
        });

        if let Some(sys) = system_prompt {
            body["system"] = serde_json::json!(sys);
        }
        if let Some(t) = tools {
            body["tools"] = serde_json::json!(t);
        }

        body
    }
}

struct PendingToolUse {
    id: String,
    name: String,
    arguments_json: String,
}

#[derive(Default)]
struct AnthropicStreamState {
    full_content: String,
    pending_tool_uses: Vec<PendingToolUse>,
    input_usage: Option<u32>,
    output_usage: Option<u32>,
    bytes_seen: usize,
    events_seen: usize,
    saw_message_start: bool,
    last_event_type: Option<String>,
}

async fn process_anthropic_stream_event(
    event: SseEvent,
    state: &mut AnthropicStreamState,
    tx: &mpsc::Sender<ChatEvent>,
) -> Result<bool, AgentError> {
    let parsed = parse_anthropic_sse_data(&event)?;
    state.events_seen += 1;

    let event_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");
    state.last_event_type = Some(event_type.to_string());

    match event_type {
        "message_start" => {
            state.saw_message_start = true;
            if let Some(usage_obj) = parsed.get("message").and_then(|m| m.get("usage")) {
                state.input_usage = usage_obj
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
            }
        }
        "content_block_start" => {
            if let Some(block) = parsed.get("content_block") {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");

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

                    state.pending_tool_uses.push(PendingToolUse {
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
                let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");

                match delta_type {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                            state.full_content.push_str(text);
                            let _ = tx
                                .send(ChatEvent::Token {
                                    token: text.to_string(),
                                })
                                .await;
                        }
                    }
                    "thinking_delta" => {
                        if let Some(text) = delta.get("thinking").and_then(|t| t.as_str()) {
                            let _ = tx
                                .send(ChatEvent::Thinking {
                                    token: text.to_string(),
                                })
                                .await;
                        }
                    }
                    "input_json_delta" => {
                        if let Some(json_str) = delta.get("partial_json").and_then(|j| j.as_str()) {
                            if let Some(last) = state.pending_tool_uses.last_mut() {
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
                state.output_usage = usage_obj
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
            }
        }
        "message_stop" => {
            let usage = match (state.input_usage, state.output_usage) {
                (Some(input), Some(output)) => Some(Usage {
                    prompt_tokens: input,
                    completion_tokens: output,
                    total_tokens: input + output,
                }),
                _ => None,
            };

            let message = if state.pending_tool_uses.is_empty() {
                ChatMessage::assistant(&state.full_content)
            } else {
                let tool_calls: Vec<ToolCall> = state
                    .pending_tool_uses
                    .drain(..)
                    .map(|tu| tu.into_tool_call())
                    .collect();
                let mut msg = ChatMessage::assistant_with_tool_calls(tool_calls);
                if !state.full_content.is_empty() {
                    msg.content = Some(MessageContent::Text(state.full_content.clone()));
                }
                msg
            };

            let _ = tx
                .send(ChatEvent::Done {
                    response: ChatResponse { message, usage },
                })
                .await;
            return Ok(true);
        }
        _ => {}
    }

    Ok(false)
}

fn parse_anthropic_sse_data(event: &SseEvent) -> Result<serde_json::Value, AgentError> {
    let parsed: serde_json::Value = serde_json::from_str(&event.data).map_err(|e| {
        let context = if event.event.as_deref() == Some("error") {
            "Anthropic SSE error event contained malformed JSON"
        } else {
            "Anthropic SSE event contained malformed JSON"
        };
        AgentError::Stream(format!(
            "{context}: {e}; event={}; payload_prefix={}",
            event.event.as_deref().unwrap_or("message"),
            payload_prefix(&event.data)
        ))
    })?;

    if event.event.as_deref() == Some("error")
        || parsed.get("type").and_then(|t| t.as_str()) == Some("error")
    {
        return Err(anthropic_sse_error(&parsed, &event.data));
    }

    Ok(parsed)
}

fn anthropic_sse_error(parsed: &serde_json::Value, payload: &str) -> AgentError {
    let error_obj = parsed.get("error").and_then(|error| error.as_object());
    let error_type = error_obj
        .and_then(|error| error.get("type"))
        .and_then(|v| v.as_str())
        .or_else(|| parsed.get("type").and_then(|v| v.as_str()))
        .unwrap_or("unknown_error");
    let message = error_obj
        .and_then(|error| error.get("message"))
        .and_then(|v| v.as_str());
    let message = match message {
        Some(message) => message.to_string(),
        None => format!(
            "Anthropic stream returned an SSE error event with an unexpected payload shape; payload_prefix={}",
            payload_prefix(payload)
        ),
    };

    AgentError::Api {
        status: anthropic_status_for_error_type(error_type),
        message: format!("Anthropic SSE error ({error_type}): {message}"),
    }
}

fn anthropic_status_for_error_type(error_type: &str) -> u16 {
    match error_type {
        "invalid_request_error" => 400,
        "authentication_error" => 401,
        "permission_error" => 403,
        "not_found_error" => 404,
        "rate_limit_error" => 429,
        "overloaded_error" => 529,
        "api_error" => 500,
        _ => 500,
    }
}

fn anthropic_stream_chunk_error(
    error: &reqwest::Error,
    state: &AnthropicStreamState,
) -> AgentError {
    AgentError::Stream(format!(
        "Anthropic stream transport/decode error after {} bytes and {} events (last_event={}): {}. The request asks Anthropic for identity-encoded SSE; this can still happen if the upstream connection closes mid-chunk or sends an invalid body.",
        state.bytes_seen,
        state.events_seen,
        state.last_event_type.as_deref().unwrap_or("none"),
        error
    ))
}

fn incomplete_anthropic_stream_error(state: &AnthropicStreamState) -> AgentError {
    AgentError::Stream(format!(
        "Anthropic stream ended before message_stop after {} bytes and {} events (saw_message_start={}, last_event={}, accumulated_text_chars={}, pending_tool_uses={})",
        state.bytes_seen,
        state.events_seen,
        state.saw_message_start,
        state.last_event_type.as_deref().unwrap_or("none"),
        state.full_content.chars().count(),
        state.pending_tool_uses.len()
    ))
}

fn payload_prefix(payload: &str) -> String {
    payload.chars().take(200).collect()
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

    #[test]
    fn test_build_body_within_limit_opt_in_compacts_messages() {
        let provider =
            AnthropicProvider::new("key", "claude-sonnet-4-20250514").with_max_request_bytes(350);
        let request = ChatRequest {
            messages: vec![
                ChatMessage::system("Important system prompt"),
                ChatMessage::user(&"old ".repeat(100)),
                ChatMessage::assistant("old answer"),
                ChatMessage::user("recent question"),
            ],
            model: "claude-sonnet-4-20250514".into(),
            tools: None,
            stream: true,
            response_format: None,
        };
        let (system, mut messages, tools) = provider.prepare_request(&request);

        let body = provider
            .build_body_within_limit(&mut messages, &system, &tools)
            .unwrap();
        let serialized = serde_json::to_string(&body).unwrap();

        assert!(serialized.len() <= 350);
        assert_eq!(body["system"], "Important system prompt");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"], "recent question");
    }

    #[test]
    fn test_request_budget_is_opt_in_for_anthropic() {
        let uncapped = AnthropicProvider::new("key", "claude-sonnet-4-20250514");
        assert_eq!(uncapped.request_budget_bytes(), None);

        let capped =
            AnthropicProvider::new("key", "claude-sonnet-4-20250514").with_max_request_bytes(350);
        assert_eq!(capped.request_budget_bytes(), Some(350));
    }

    #[test]
    fn test_build_body_within_limit_drops_tool_use_group() {
        let provider =
            AnthropicProvider::new("key", "claude-sonnet-4-20250514").with_max_request_bytes(500);
        let request = ChatRequest {
            messages: vec![
                ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "toolu_old".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: "{\"path\":\"old.txt\"}".into(),
                    },
                }]),
                ChatMessage::tool_result("toolu_old", &"old tool output ".repeat(60)),
                ChatMessage::user("recent question"),
            ],
            model: "claude-sonnet-4-20250514".into(),
            tools: None,
            stream: true,
            response_format: None,
        };
        let (system, mut messages, tools) = provider.prepare_request(&request);

        let body = provider
            .build_body_within_limit(&mut messages, &system, &tools)
            .unwrap();
        let serialized = serde_json::to_string(&body).unwrap();

        assert!(serialized.len() <= 500);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"], "recent question");
    }

    #[test]
    fn test_parse_anthropic_sse_error_event_returns_api_error() {
        let err = parse_anthropic_sse_data(&SseEvent {
            event: Some("error".into()),
            data: serde_json::json!({
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "Overloaded"
                }
            })
            .to_string(),
        })
        .unwrap_err();

        match err {
            AgentError::Api { status, message } => {
                assert_eq!(status, 529);
                assert!(message.contains("Anthropic SSE error (overloaded_error): Overloaded"));
            }
            other => panic!("expected API error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_anthropic_sse_error_with_unexpected_shape_has_payload_context() {
        let err = parse_anthropic_sse_data(&SseEvent {
            event: Some("error".into()),
            data: serde_json::json!({
                "type": "error",
                "detail": "upstream closed"
            })
            .to_string(),
        })
        .unwrap_err();

        match err {
            AgentError::Api { status, message } => {
                assert_eq!(status, 500);
                assert!(message.contains("Anthropic SSE error (error)"));
                assert!(message.contains("unexpected payload shape"));
                assert!(message.contains("payload_prefix="));
                assert!(message.contains("upstream closed"));
            }
            other => panic!("expected API error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_anthropic_malformed_sse_data_returns_stream_error() {
        let err = parse_anthropic_sse_data(&SseEvent {
            event: Some("content_block_delta".into()),
            data: "{not-json".into(),
        })
        .unwrap_err();

        match err {
            AgentError::Stream(message) => {
                assert!(message.contains("Anthropic SSE event contained malformed JSON"));
                assert!(message.contains("event=content_block_delta"));
                assert!(message.contains("payload_prefix={not-json"));
            }
            other => panic!("expected stream error, got {other:?}"),
        }
    }

    #[test]
    fn test_incomplete_anthropic_stream_error_has_terminal_context() {
        let state = AnthropicStreamState {
            full_content: "partial response".into(),
            bytes_seen: 1234,
            events_seen: 7,
            saw_message_start: true,
            last_event_type: Some("content_block_delta".into()),
            ..Default::default()
        };

        let err = incomplete_anthropic_stream_error(&state);
        match err {
            AgentError::Stream(message) => {
                assert!(message.contains("Anthropic stream ended before message_stop"));
                assert!(message.contains("1234 bytes"));
                assert!(message.contains("last_event=content_block_delta"));
                assert!(message.contains("accumulated_text_chars=16"));
            }
            other => panic!("expected stream error, got {other:?}"),
        }
    }
}
