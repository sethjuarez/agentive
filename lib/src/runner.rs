//! Agentic runner — the core multi-turn LLM + tool execution loop.
//!
//! The [`run`] function is the main entry point. It:
//! 1. Sends messages to the LLM provider (streaming)
//! 2. Collects the response — if it contains tool calls, executes them
//! 3. Appends tool results to the conversation and loops back to step 1
//! 4. Repeats until the LLM returns a text response (no tool calls) or limits are hit
//!
//! The runner emits [`RunnerEvent`]s via a callback so apps can stream tokens
//! to the UI, persist conversation state, and show tool execution progress.
//!
//! # Example
//! ```no_run
//! use agentive::*;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), AgentError> {
//! let provider = Arc::new(OpenAiProvider::new("https://api.openai.com/v1", "sk-...", "gpt-4o"));
//! let result = agentive::run(
//!     provider,
//!     vec![ChatMessage::system("You are helpful"), ChatMessage::user("Hi")],
//!     vec![],
//!     |_call| Ok("not implemented".into()),
//!     RunnerConfig::default(),
//!     CancellationToken::new(),
//!     Steering::new(),
//!     |event| { /* handle events */ },
//! ).await?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::cancel::CancellationToken;
use crate::context::trim_to_context_window;
use crate::error::AgentError;
use crate::provider::Provider;
use crate::sanitize::sanitize_for_api;
use crate::steering::Steering;
use crate::types::*;

/// Configuration for the runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Maximum number of tool-call rounds before giving up.
    pub max_iterations: usize,
    /// Whether to retry once on 400 errors.
    pub retry_on_400: bool,
    /// Whether to trim context when it exceeds the provider's budget.
    pub auto_trim_context: bool,
    /// Whether to sanitize tool results (strip control chars, base64).
    pub sanitize_tool_results: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            retry_on_400: true,
            auto_trim_context: true,
            sanitize_tool_results: true,
        }
    }
}

/// Events emitted by the runner during execution.
/// Apps use these to update UI, persist state, log, etc.
#[derive(Debug, Clone)]
pub enum RunnerEvent {
    /// A text token streamed from the LLM.
    Token { token: String },
    /// A reasoning/thinking token from the LLM.
    Thinking { token: String },
    /// Status update (e.g., "Thinking…", "Running 3 tool calls…").
    Status { message: String },
    /// A tool is being called.
    ToolCallStart { name: String, arguments: String },
    /// A tool returned a result.
    ToolResult { name: String, result: String },
    /// The full message history was updated (after a tool round).
    /// Apps can use this to persist conversation state mid-run.
    MessagesUpdated { messages: Vec<ChatMessage> },
    /// The runner completed successfully.
    Done {
        response: String,
        messages: Vec<ChatMessage>,
    },
    /// An error occurred.
    Error { message: String },
}

/// Result of running the agentic loop.
pub struct RunnerResult {
    /// The full conversation including tool calls and results.
    pub messages: Vec<ChatMessage>,
    /// The final assistant text response.
    pub response: String,
    /// New messages generated during this run (for persistence).
    pub new_messages: Vec<ChatMessage>,
}

/// Run the agentic loop: stream LLM → execute tools → loop until done.
///
/// # Arguments
/// * `provider` - The LLM provider to use for chat completions.
/// * `messages` - Initial conversation history (including system prompt).
/// * `tools` - Tool definitions to offer the LLM.
/// * `tool_executor` - Closure that executes a tool call and returns the result.
/// * `config` - Runner configuration.
/// * `cancel` - Cancellation token for user-initiated stop.
/// * `steering` - Handle for injecting user messages mid-run (see [`Steering`]).
/// * `on_event` - Callback for runner events (streaming tokens, status, etc.).
pub async fn run<F, E>(
    provider: Arc<dyn Provider>,
    messages: Vec<ChatMessage>,
    tools: Vec<Tool>,
    tool_executor: F,
    config: RunnerConfig,
    cancel: CancellationToken,
    steering: Steering,
    on_event: E,
) -> Result<RunnerResult, AgentError>
where
    F: Fn(&ToolCall) -> Result<String, String> + Send + Sync,
    E: Fn(RunnerEvent) + Send + Sync,
{
    let mut full_messages = messages;
    let mut new_messages: Vec<ChatMessage> = Vec::new();

    for iteration in 0..config.max_iterations {
        if cancel.is_cancelled() {
            return Err(AgentError::Cancelled);
        }

        // Drain any steering messages injected by the user mid-run
        for msg in steering.drain() {
            full_messages.push(ChatMessage::user(&msg));
            new_messages.push(ChatMessage::user(&msg));
        }

        on_event(RunnerEvent::Status {
            message: if iteration == 0 {
                "Thinking…".into()
            } else {
                format!("Thinking… (round {})", iteration + 1)
            },
        });

        // Trim context if needed
        if config.auto_trim_context {
            let budget = provider.context_budget_chars();
            let (dropped_count, _dropped) = trim_to_context_window(&mut full_messages, budget);
            if dropped_count > 0 {
                on_event(RunnerEvent::Status {
                    message: format!(
                        "Compacted context — summarized {} earlier messages",
                        dropped_count
                    ),
                });
            }
        }

        // Build request
        let request = ChatRequest {
            messages: full_messages.clone(),
            model: String::new(), // Provider uses its own model
            tools: if tools.is_empty() {
                None
            } else {
                Some(tools.clone())
            },
            stream: true,
        };

        // Spawn provider in a separate task for concurrent streaming
        let (tx, mut rx) = mpsc::channel::<ChatEvent>(64);
        let provider_clone = provider.clone();
        let cancel_clone = cancel.clone();

        let provider_handle = tokio::spawn(async move {
            provider_clone.chat(request, tx, &cancel_clone).await
        });

        // Read events as they arrive
        let mut assistant_response: Option<ChatResponse> = None;

        while let Some(event) = rx.recv().await {
            match event {
                ChatEvent::Token { token } => {
                    on_event(RunnerEvent::Token {
                        token: token.clone(),
                    });
                }
                ChatEvent::Thinking { token } => {
                    on_event(RunnerEvent::Thinking {
                        token: token.clone(),
                    });
                }
                ChatEvent::ToolCallStart { tool_call } => {
                    on_event(RunnerEvent::ToolCallStart {
                        name: tool_call.function.name.clone(),
                        arguments: tool_call.function.arguments.clone(),
                    });
                }
                ChatEvent::Done { response } => {
                    assistant_response = Some(response);
                }
                ChatEvent::Error { message } => {
                    return Err(AgentError::Stream(message));
                }
            }
        }

        // Check provider task result
        let provider_result = provider_handle
            .await
            .map_err(|e| AgentError::Stream(format!("Provider task panicked: {}", e)))?;

        // Handle 400 retry
        if let Err(ref err) = provider_result {
            if config.retry_on_400 {
                if let AgentError::Api { status: 400, .. } = err {
                    on_event(RunnerEvent::Status {
                        message: "Retrying request…".into(),
                    });

                    // Retry once
                    let request2 = ChatRequest {
                        messages: full_messages.clone(),
                        model: String::new(),
                        tools: if tools.is_empty() {
                            None
                        } else {
                            Some(tools.clone())
                        },
                        stream: true,
                    };

                    let (tx2, mut rx2) = mpsc::channel::<ChatEvent>(64);
                    let provider_clone2 = provider.clone();
                    let cancel_clone2 = cancel.clone();

                    let handle2 = tokio::spawn(async move {
                        provider_clone2.chat(request2, tx2, &cancel_clone2).await
                    });

                    while let Some(event) = rx2.recv().await {
                        match event {
                            ChatEvent::Token { token } => {
                                on_event(RunnerEvent::Token { token });
                            }
                            ChatEvent::Thinking { token } => {
                                on_event(RunnerEvent::Thinking { token });
                            }
                            ChatEvent::ToolCallStart { tool_call } => {
                                on_event(RunnerEvent::ToolCallStart {
                                    name: tool_call.function.name.clone(),
                                    arguments: tool_call.function.arguments.clone(),
                                });
                            }
                            ChatEvent::Done { response } => {
                                assistant_response = Some(response);
                            }
                            ChatEvent::Error { message } => {
                                return Err(AgentError::Stream(message));
                            }
                        }
                    }

                    handle2
                        .await
                        .map_err(|e| {
                            AgentError::Stream(format!("Retry provider task panicked: {}", e))
                        })?
                        .map_err(|e| {
                            on_event(RunnerEvent::Error {
                                message: format!("Retry also failed: {}", e),
                            });
                            e
                        })?;
                } else {
                    provider_result?;
                }
            } else {
                provider_result?;
            }
        } else {
            provider_result?;
        }

        let response = assistant_response.ok_or_else(|| {
            AgentError::Stream("Provider finished without sending Done event".into())
        })?;

        // Check for tool calls
        if let Some(ref tool_calls) = response.message.tool_calls {
            if !tool_calls.is_empty() && !tools.is_empty() {
                // Add assistant message with tool calls to history
                full_messages.push(response.message.clone());
                new_messages.push(response.message.clone());

                on_event(RunnerEvent::Status {
                    message: format!("Running {} tool call(s)…", tool_calls.len()),
                });

                // Execute each tool call
                for tc in tool_calls {
                    let result = match tool_executor(tc) {
                        Ok(r) => r,
                        Err(e) => format!("Tool error: {}", e),
                    };

                    let clean_result = if config.sanitize_tool_results {
                        sanitize_for_api(&result)
                    } else {
                        result.clone()
                    };

                    on_event(RunnerEvent::ToolResult {
                        name: tc.function.name.clone(),
                        result: clean_result.clone(),
                    });

                    let tool_msg = ChatMessage::tool_result(&tc.id, &clean_result);
                    full_messages.push(tool_msg.clone());
                    new_messages.push(tool_msg);
                }

                // Emit updated messages for persistence
                on_event(RunnerEvent::MessagesUpdated {
                    messages: full_messages.clone(),
                });

                continue; // Loop back to call the LLM with tool results
            }
        }

        // No tool calls — this is the final response
        full_messages.push(response.message.clone());
        new_messages.push(response.message.clone());

        let final_text = response
            .message
            .text()
            .unwrap_or("")
            .to_string();

        on_event(RunnerEvent::Done {
            response: final_text.clone(),
            messages: full_messages.clone(),
        });

        return Ok(RunnerResult {
            messages: full_messages,
            response: final_text,
            new_messages,
        });
    }

    Err(AgentError::MaxIterations(config.max_iterations))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A mock provider for testing the runner loop.
    struct MockProvider {
        responses: Mutex<Vec<ChatResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::provider::Provider for MockProvider {
        async fn chat(
            &self,
            _request: ChatRequest,
            tx: mpsc::Sender<ChatEvent>,
            cancel: &CancellationToken,
        ) -> Result<(), AgentError> {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            let response = {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Err(AgentError::Stream("No more mock responses".into()));
                }
                responses.remove(0)
            };

            // Simulate token streaming for text content
            if let Some(text) = response.message.text() {
                for word in text.split_whitespace() {
                    let _ = tx
                        .send(ChatEvent::Token {
                            token: format!("{} ", word),
                        })
                        .await;
                }
            }

            let _ = tx.send(ChatEvent::Done { response }).await;
            Ok(())
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_simple_conversation() {
        let provider = Arc::new(MockProvider::new(vec![ChatResponse {
            message: ChatMessage::assistant("Hello! How can I help?"),
            usage: None,
        }]));

        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();

        let result = run(
            provider,
            vec![ChatMessage::user("Hi")],
            vec![],
            |_| Ok("unused".into()),
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            move |event| {
                events_clone.lock().unwrap().push(format!("{:?}", event));
            },
        )
        .await
        .unwrap();

        assert_eq!(result.response, "Hello! How can I help?");
        assert!(result.messages.len() >= 2); // user + assistant
    }

    #[tokio::test]
    async fn test_tool_call_loop() {
        let provider = Arc::new(MockProvider::new(vec![
            // First response: tool call
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "call_1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: "{\"path\":\"test.txt\"}".into(),
                    },
                }]),
                usage: None,
            },
            // Second response: final text
            ChatResponse {
                message: ChatMessage::assistant("The file contains: hello world"),
                usage: None,
            },
        ]));

        let result = run(
            provider,
            vec![
                ChatMessage::system("You are helpful"),
                ChatMessage::user("Read test.txt"),
            ],
            vec![Tool::function(
                "read_file",
                "Read a file",
                serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            )],
            |call| {
                assert_eq!(call.function.name, "read_file");
                Ok("hello world".into())
            },
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "The file contains: hello world");
        // Should have: system, user, assistant(tool_call), tool_result, assistant(final)
        assert_eq!(result.messages.len(), 5);
    }

    #[tokio::test]
    async fn test_cancellation() {
        let provider = Arc::new(MockProvider::new(vec![ChatResponse {
            message: ChatMessage::assistant("should not reach this"),
            usage: None,
        }]));

        let cancel = CancellationToken::new();
        cancel.cancel();

        let result = run(
            provider,
            vec![ChatMessage::user("Hi")],
            vec![],
            |_| Ok("".into()),
            RunnerConfig::default(),
            cancel,
            Steering::new(),
            |_| {},
        )
        .await;

        assert!(matches!(result, Err(AgentError::Cancelled)));
    }

    #[tokio::test]
    async fn test_max_iterations() {
        // Provider always returns tool calls — should hit the limit
        let mut responses = Vec::new();
        for _ in 0..15 {
            responses.push(ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "call".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "loop_tool".into(),
                        arguments: "{}".into(),
                    },
                }]),
                usage: None,
            });
        }

        let provider = Arc::new(MockProvider::new(responses));

        let result = run(
            provider,
            vec![ChatMessage::user("loop forever")],
            vec![Tool::function("loop_tool", "loops", serde_json::json!({}))],
            |_| Ok("looping".into()),
            RunnerConfig {
                max_iterations: 3,
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            |_| {},
        )
        .await;

        assert!(matches!(result, Err(AgentError::MaxIterations(3))));
    }

    #[tokio::test]
    async fn test_steering_injects_messages() {
        let provider = Arc::new(MockProvider::new(vec![
            // First response: tool call
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "call_1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "slow_tool".into(),
                        arguments: "{}".into(),
                    },
                }]),
                usage: None,
            },
            // Second response: final text (after steering message was injected)
            ChatResponse {
                message: ChatMessage::assistant("Got your redirect, here's the answer"),
                usage: None,
            },
        ]));

        let steering = Steering::new();

        // Simulate user steering mid-run: inject before round 2
        steering.send("Actually, focus on the error case");

        let result = run(
            provider,
            vec![ChatMessage::user("Do the thing")],
            vec![Tool::function("slow_tool", "a tool", serde_json::json!({}))],
            |_| Ok("tool result".into()),
            RunnerConfig::default(),
            CancellationToken::new(),
            steering,
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "Got your redirect, here's the answer");
        // The steering message should be in the conversation history
        let has_steering = result
            .messages
            .iter()
            .any(|m| m.text() == Some("Actually, focus on the error case"));
        assert!(has_steering, "Steering message should be in history");
    }
}
