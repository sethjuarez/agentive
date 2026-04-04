//! Core types for agentive — messages, tools, requests, responses, and events.
//!
//! All types derive `Serialize`/`Deserialize` for easy JSON conversion.
//! `ChatMessage` supports multimodal content via [`MessageContent`].

use serde::{Deserialize, Serialize};

// -- Message content (supports text and multimodal) --------------------------

/// Content of a chat message — either plain text or structured parts
/// (for multimodal messages with images).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Structured content parts (text + images for multimodal).
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Get the text content, if this is a text message or extract text from parts.
    pub fn text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s),
            MessageContent::Parts(parts) => parts.iter().find_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            }),
        }
    }

    /// Estimate character length for context budget calculations.
    pub fn char_len(&self) -> usize {
        match self {
            MessageContent::Text(s) => s.chars().count(),
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text { text } => text.chars().count(),
                    ContentPart::ImageUrl { .. } => 200, // rough estimate for image reference
                })
                .sum(),
        }
    }
}

/// A single part of a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL reference (base64 data URI or HTTP URL).
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// Image URL details for multimodal messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Output from a tool executor — either plain text or text with images.
///
/// When a tool produces images (e.g., screenshots, diagrams), return
/// `ToolOutput::with_images(text, parts)`.  The runner will add the text as
/// a normal tool result and inject a follow-up user message with the image
/// content parts so the LLM can "see" them.
///
/// Plain `String` converts to `ToolOutput::Text` automatically via `From`,
/// so existing executors returning `Result<String, String>` keep working
/// when the signature is widened to `Result<ToolOutput, String>`.
#[derive(Debug, Clone)]
pub enum ToolOutput {
    /// Plain text tool result.
    Text(String),
    /// Text result plus image content parts to inject as a follow-up user
    /// message.  The `images` vec should contain `ContentPart::ImageUrl`
    /// entries.
    WithImages {
        text: String,
        images: Vec<ContentPart>,
    },
}

impl ToolOutput {
    /// Create a tool output with text and associated images.
    pub fn with_images(text: impl Into<String>, images: Vec<ContentPart>) -> Self {
        Self::WithImages { text: text.into(), images }
    }

    /// Get the text portion of this output.
    pub fn text(&self) -> &str {
        match self {
            Self::Text(s) => s,
            Self::WithImages { text, .. } => text,
        }
    }

    /// Get the image parts, if any.
    pub fn images(&self) -> Option<&[ContentPart]> {
        match self {
            Self::Text(_) => None,
            Self::WithImages { images, .. } => Some(images),
        }
    }
}

impl From<String> for ToolOutput {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for ToolOutput {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

// -- Chat messages -----------------------------------------------------------

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Content is serialized as `null` (not omitted) when None. The OpenAI API
    /// requires `"content": null` on assistant messages that have tool_calls.
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Create a user message with text content.
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".into(),
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a user message with text and image content parts.
    pub fn user_with_images(text: &str, images: Vec<ContentPart>) -> Self {
        let mut parts = vec![ContentPart::Text {
            text: text.into(),
        }];
        parts.extend(images);
        Self {
            role: "user".into(),
            content: Some(MessageContent::Parts(parts)),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message with text content.
    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".into(),
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a system message.
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".into(),
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a tool result message.
    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        Self {
            role: "tool".into(),
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Create an assistant message with tool calls (and optional text).
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// Get the text content of this message (convenience accessor).
    pub fn text(&self) -> Option<&str> {
        self.content.as_ref().and_then(|c| c.text())
    }
}

// -- Tool calling ------------------------------------------------------------

/// A tool call requested by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "default_tool_type")]
    pub call_type: String,
    pub function: FunctionCall,
}

fn default_tool_type() -> String {
    "function".into()
}

/// The function name and arguments for a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// A tool definition that can be passed to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

impl Tool {
    /// Create a function-type tool definition.
    pub fn function(name: &str, description: &str, parameters: serde_json::Value) -> Self {
        Self {
            tool_type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// Function metadata within a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// -- Requests & responses ----------------------------------------------------

/// Structured output format specification.
///
/// Used to constrain the LLM's response to a specific JSON format.
/// OpenAI supports this natively via `response_format`. Anthropic achieves
/// similar results through tool-use patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    /// Request raw JSON output (no specific schema).
    #[serde(rename = "json_object")]
    JsonObject,
    /// Request JSON output conforming to a specific schema.
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// Schema definition with name and strict mode.
        json_schema: JsonSchemaSpec,
    },
}

/// JSON schema specification for structured output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaSpec {
    /// Name for the schema (used by the API for identification).
    pub name: String,
    /// Whether to enforce strict schema adherence.
    #[serde(default)]
    pub strict: bool,
    /// The JSON Schema definition.
    pub schema: serde_json::Value,
}

/// A chat completion request.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    pub stream: bool,
    /// Optional structured output format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

/// A chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Token usage statistics from the API.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
        self.total_tokens += rhs.total_tokens;
    }
}

impl std::ops::Add for Usage {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

// -- Streaming events --------------------------------------------------------

/// Events emitted by a provider during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ChatEvent {
    /// A text token streamed from the LLM.
    #[serde(rename = "token")]
    Token { token: String },
    /// A reasoning/thinking token (for models that expose chain-of-thought).
    #[serde(rename = "thinking")]
    Thinking { token: String },
    /// A tool call is starting (name known, arguments may still be streaming).
    #[serde(rename = "tool_call")]
    ToolCallStart { tool_call: ToolCall },
    /// The streaming response is complete.
    #[serde(rename = "done")]
    Done { response: ChatResponse },
    /// An error occurred during streaming.
    #[serde(rename = "error")]
    Error { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_constructors() {
        let user = ChatMessage::user("hello");
        assert_eq!(user.role, "user");
        assert_eq!(user.text(), Some("hello"));
        assert!(user.tool_calls.is_none());

        let asst = ChatMessage::assistant("hi there");
        assert_eq!(asst.role, "assistant");
        assert_eq!(asst.text(), Some("hi there"));

        let sys = ChatMessage::system("you are helpful");
        assert_eq!(sys.role, "system");

        let tool = ChatMessage::tool_result("call_123", "{\"result\": \"ok\"}");
        assert_eq!(tool.role, "tool");
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_123"));
    }

    #[test]
    fn test_assistant_with_tool_calls() {
        let tc = ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: "{\"path\":\"test.txt\"}".into(),
            },
        };
        let msg = ChatMessage::assistant_with_tool_calls(vec![tc]);
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
        assert_eq!(msg.tool_calls.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_tool_function_builder() {
        let tool = Tool::function(
            "read_file",
            "Read a file",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        );
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "read_file");
    }

    #[test]
    fn test_message_content_text() {
        let content = MessageContent::Text("hello".into());
        assert_eq!(content.text(), Some("hello"));
        assert_eq!(content.char_len(), 5);
    }

    #[test]
    fn test_message_content_parts() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Look at this:".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,...".into(),
                    detail: None,
                },
            },
        ]);
        assert_eq!(content.text(), Some("Look at this:"));
        assert!(content.char_len() > 13); // text + image estimate
    }

    #[test]
    fn test_user_with_images() {
        let msg = ChatMessage::user_with_images(
            "What's in this image?",
            vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/img.png".into(),
                    detail: Some("high".into()),
                },
            }],
        );
        assert_eq!(msg.role, "user");
        match msg.content.unwrap() {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("Expected Parts"),
        }
    }

    #[test]
    fn test_chat_event_serialization() {
        let event = ChatEvent::Token {
            token: "Hello".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"token\""));

        let event = ChatEvent::Thinking {
            token: "reasoning...".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"thinking\""));
    }

    #[test]
    fn test_chat_request_serialization() {
        let req = ChatRequest {
            messages: vec![ChatMessage::user("hi")],
            model: "gpt-4o".into(),
            tools: None,
            stream: true,
            response_format: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream"], true);
        assert!(json.get("tools").is_none());
        assert!(json.get("response_format").is_none());
    }

    #[test]
    fn test_response_format_json_object() {
        let rf = ResponseFormat::JsonObject;
        let json = serde_json::to_value(&rf).unwrap();
        assert_eq!(json["type"], "json_object");
    }

    #[test]
    fn test_response_format_json_schema() {
        let rf = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: "my_schema".into(),
                strict: true,
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "answer": { "type": "string" }
                    },
                    "required": ["answer"]
                }),
            },
        };
        let json = serde_json::to_value(&rf).unwrap();
        assert_eq!(json["type"], "json_schema");
        assert_eq!(json["json_schema"]["name"], "my_schema");
        assert_eq!(json["json_schema"]["strict"], true);
        assert_eq!(json["json_schema"]["schema"]["type"], "object");
    }
}
