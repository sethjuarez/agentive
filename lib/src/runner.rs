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
//!     Guardrails::default(),
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
use crate::guardrails::{GuardrailResult, Guardrails};
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
    /// Whether to execute multiple tool calls concurrently (default: true).
    pub parallel_tool_calls: bool,
    /// Optional structured output format (JSON mode or JSON schema).
    pub response_format: Option<ResponseFormat>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            retry_on_400: true,
            auto_trim_context: true,
            sanitize_tool_results: true,
            parallel_tool_calls: true,
            response_format: None,
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
    /// Token usage for a single LLM call.
    Usage { usage: Usage },
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
#[derive(Debug)]
pub struct RunnerResult {
    /// The full conversation including tool calls and results.
    pub messages: Vec<ChatMessage>,
    /// The final assistant text response.
    pub response: String,
    /// New messages generated during this run (for persistence).
    pub new_messages: Vec<ChatMessage>,
    /// Total token usage across all LLM calls in this run.
    pub total_usage: Usage,
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
/// * `guardrails` - Optional validation hooks (see [`Guardrails`]).
/// * `on_event` - Callback for runner events (streaming tokens, status, etc.).
pub async fn run<F, E>(
    provider: Arc<dyn Provider>,
    messages: Vec<ChatMessage>,
    tools: Vec<Tool>,
    tool_executor: F,
    config: RunnerConfig,
    cancel: CancellationToken,
    steering: Steering,
    guardrails: Guardrails,
    on_event: E,
) -> Result<RunnerResult, AgentError>
where
    F: Fn(&ToolCall) -> Result<String, String> + Send + Sync,
    E: Fn(RunnerEvent) + Send + Sync,
{
    let mut full_messages = messages;
    let mut new_messages: Vec<ChatMessage> = Vec::new();
    let mut total_usage = Usage::default();

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

        // Input guardrail — check before calling LLM
        if let GuardrailResult::Deny(reason) = guardrails.check_input(&full_messages) {
            return Err(AgentError::Guardrailed(reason));
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
            response_format: config.response_format.clone(),
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
                        response_format: config.response_format.clone(),
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

        // Track usage
        if let Some(usage) = &response.usage {
            total_usage += usage.clone();
            on_event(RunnerEvent::Usage {
                usage: usage.clone(),
            });
        }

        // Output guardrail — check LLM response before proceeding
        if let GuardrailResult::Deny(reason) = guardrails.check_output(&response.message) {
            return Err(AgentError::Guardrailed(reason));
        }

        // Check for tool calls
        if let Some(ref tool_calls) = response.message.tool_calls {
            if !tool_calls.is_empty() && !tools.is_empty() {
                // Add assistant message with tool calls to history
                full_messages.push(response.message.clone());
                new_messages.push(response.message.clone());

                on_event(RunnerEvent::Status {
                    message: format!("Running {} tool call(s)…", tool_calls.len()),
                });

                // Execute tool calls (parallel or sequential)
                let tool_results = if config.parallel_tool_calls && tool_calls.len() > 1 {
                    execute_tools_parallel(tool_calls, &tool_executor, &config, &guardrails, &on_event)?
                } else {
                    execute_tools_sequential(tool_calls, &tool_executor, &config, &guardrails, &on_event)?
                };

                for tool_msg in &tool_results {
                    full_messages.push(tool_msg.clone());
                    new_messages.push(tool_msg.clone());
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
            total_usage,
        });
    }

    Err(AgentError::MaxIterations(config.max_iterations))
}

/// Execute tool calls sequentially with panic safety and guardrails.
fn execute_tools_sequential<F, E>(
    tool_calls: &[ToolCall],
    tool_executor: &F,
    config: &RunnerConfig,
    guardrails: &Guardrails,
    on_event: &E,
) -> Result<Vec<ChatMessage>, AgentError>
where
    F: Fn(&ToolCall) -> Result<String, String> + Send + Sync,
    E: Fn(RunnerEvent) + Send + Sync,
{
    let mut results = Vec::with_capacity(tool_calls.len());
    for tc in tool_calls {
        // Tool guardrail — check before execution
        if let GuardrailResult::Deny(reason) = guardrails.check_tool(tc) {
            let denied_msg = format!("Tool denied by guardrail: {}", reason);
            on_event(RunnerEvent::ToolResult {
                name: tc.function.name.clone(),
                result: denied_msg.clone(),
            });
            results.push(ChatMessage::tool_result(&tc.id, &denied_msg));
            continue;
        }

        let raw_result = execute_tool_safe(tc, tool_executor)?;
        let clean = if config.sanitize_tool_results {
            sanitize_for_api(&raw_result)
        } else {
            raw_result
        };
        on_event(RunnerEvent::ToolResult {
            name: tc.function.name.clone(),
            result: clean.clone(),
        });
        results.push(ChatMessage::tool_result(&tc.id, &clean));
    }
    Ok(results)
}

/// Execute tool calls in parallel with panic safety and guardrails.
fn execute_tools_parallel<F, E>(
    tool_calls: &[ToolCall],
    tool_executor: &F,
    config: &RunnerConfig,
    guardrails: &Guardrails,
    on_event: &E,
) -> Result<Vec<ChatMessage>, AgentError>
where
    F: Fn(&ToolCall) -> Result<String, String> + Send + Sync,
    E: Fn(RunnerEvent) + Send + Sync,
{
    // Use std threads for parallel execution since tool_executor is Fn (sync)
    let results: Vec<Result<(String, String, String), AgentError>> = std::thread::scope(|s| {
        let handles: Vec<_> = tool_calls
            .iter()
            .map(|tc| {
                let name = tc.function.name.clone();
                let id = tc.id.clone();

                // Tool guardrail check (runs in main thread before spawning)
                let denied = match guardrails.check_tool(tc) {
                    GuardrailResult::Deny(reason) => Some(reason),
                    GuardrailResult::Allow => None,
                };

                s.spawn(move || {
                    if let Some(reason) = denied {
                        Ok((id, name, format!("Tool denied by guardrail: {}", reason)))
                    } else {
                        let raw = execute_tool_safe(tc, tool_executor)?;
                        Ok((id, name, raw))
                    }
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err(AgentError::Stream("Thread panicked".into()))))
            .collect()
    });

    let mut messages = Vec::with_capacity(tool_calls.len());
    for res in results {
        let (id, name, raw) = res?;
        let clean = if config.sanitize_tool_results {
            sanitize_for_api(&raw)
        } else {
            raw
        };
        on_event(RunnerEvent::ToolResult {
            name,
            result: clean.clone(),
        });
        messages.push(ChatMessage::tool_result(&id, &clean));
    }
    Ok(messages)
}

/// Execute a single tool call with panic safety via `catch_unwind`.
fn execute_tool_safe<F>(tc: &ToolCall, tool_executor: &F) -> Result<String, AgentError>
where
    F: Fn(&ToolCall) -> Result<String, String>,
{
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tool_executor(tc)));

    match result {
        Ok(Ok(r)) => Ok(r),
        Ok(Err(e)) => Ok(format!("Tool error: {}", e)),
        Err(panic_info) => {
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            Err(AgentError::ToolPanic {
                name: tc.function.name.clone(),
                message: panic_msg,
            })
        }
    }
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
            Guardrails::default(),
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
            Guardrails::default(),
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
            Guardrails::default(),
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
            Guardrails::default(),
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
            Guardrails::default(),
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

    #[tokio::test]
    async fn test_usage_accumulation() {
        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "c1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "tool1".into(),
                        arguments: "{}".into(),
                    },
                }]),
                usage: Some(Usage {
                    prompt_tokens: 100,
                    completion_tokens: 20,
                    total_tokens: 120,
                }),
            },
            ChatResponse {
                message: ChatMessage::assistant("done"),
                usage: Some(Usage {
                    prompt_tokens: 200,
                    completion_tokens: 30,
                    total_tokens: 230,
                }),
            },
        ]));

        let result = run(
            provider,
            vec![ChatMessage::user("go")],
            vec![Tool::function("tool1", "a tool", serde_json::json!({}))],
            |_| Ok("ok".into()),
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.total_usage.prompt_tokens, 300);
        assert_eq!(result.total_usage.completion_tokens, 50);
        assert_eq!(result.total_usage.total_tokens, 350);
    }

    #[tokio::test]
    async fn test_parallel_tool_calls() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![
                    ToolCall {
                        id: "c1".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "tool_a".into(),
                            arguments: "{}".into(),
                        },
                    },
                    ToolCall {
                        id: "c2".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "tool_b".into(),
                            arguments: "{}".into(),
                        },
                    },
                    ToolCall {
                        id: "c3".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "tool_c".into(),
                            arguments: "{}".into(),
                        },
                    },
                ]),
                usage: None,
            },
            ChatResponse {
                message: ChatMessage::assistant("All done"),
                usage: None,
            },
        ]));

        let call_count = Arc::new(AtomicUsize::new(0));
        let count_clone = call_count.clone();

        let result = run(
            provider,
            vec![ChatMessage::user("run three tools")],
            vec![
                Tool::function("tool_a", "a", serde_json::json!({})),
                Tool::function("tool_b", "b", serde_json::json!({})),
                Tool::function("tool_c", "c", serde_json::json!({})),
            ],
            move |tc| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(format!("result from {}", tc.function.name))
            },
            RunnerConfig {
                parallel_tool_calls: true,
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "All done");
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        // Should have tool results in the conversation
        let tool_results: Vec<_> = result
            .messages
            .iter()
            .filter(|m| m.role == "tool")
            .collect();
        assert_eq!(tool_results.len(), 3);
    }

    #[tokio::test]
    async fn test_tool_panic_safety() {
        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "c1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "panicking_tool".into(),
                        arguments: "{}".into(),
                    },
                }]),
                usage: None,
            },
        ]));

        let result = run(
            provider,
            vec![ChatMessage::user("call the bad tool")],
            vec![Tool::function(
                "panicking_tool",
                "panics",
                serde_json::json!({}),
            )],
            |_| panic!("tool went boom"),
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await;

        match result {
            Err(AgentError::ToolPanic { name, message }) => {
                assert_eq!(name, "panicking_tool");
                assert!(message.contains("tool went boom"));
            }
            other => panic!("Expected ToolPanic, got: {:?}", other.err()),
        }
    }

    #[tokio::test]
    async fn test_input_guardrail_denies() {
        let provider = Arc::new(MockProvider::new(vec![ChatResponse {
            message: ChatMessage::assistant("should not reach"),
            usage: None,
        }]));

        let guardrails = Guardrails::new().with_input_guardrail(|_msgs| {
            GuardrailResult::Deny("Blocked by policy".into())
        });

        let result = run(
            provider,
            vec![ChatMessage::user("Hello")],
            vec![],
            |_| Ok("".into()),
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            guardrails,
            |_| {},
        )
        .await;

        match result {
            Err(AgentError::Guardrailed(reason)) => {
                assert_eq!(reason, "Blocked by policy");
            }
            other => panic!("Expected Guardrailed, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_tool_guardrail_denies_specific_tool() {
        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![
                    ToolCall {
                        id: "c1".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "safe_tool".into(),
                            arguments: "{}".into(),
                        },
                    },
                    ToolCall {
                        id: "c2".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "dangerous_tool".into(),
                            arguments: "{}".into(),
                        },
                    },
                ]),
                usage: None,
            },
            ChatResponse {
                message: ChatMessage::assistant("Finished"),
                usage: None,
            },
        ]));

        let guardrails = Guardrails::new().with_tool_guardrail(|tc| {
            if tc.function.name == "dangerous_tool" {
                GuardrailResult::Deny("Tool not permitted".into())
            } else {
                GuardrailResult::Allow
            }
        });

        let result = run(
            provider,
            vec![ChatMessage::user("do it")],
            vec![
                Tool::function("safe_tool", "safe", serde_json::json!({})),
                Tool::function("dangerous_tool", "dangerous", serde_json::json!({})),
            ],
            |_| Ok("safe result".into()),
            RunnerConfig {
                parallel_tool_calls: false, // sequential so we can test order
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            guardrails,
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "Finished");
        // The dangerous tool should have a denial message, not a real result
        let tool_msgs: Vec<_> = result.messages.iter().filter(|m| m.role == "tool").collect();
        assert_eq!(tool_msgs.len(), 2);
        let denied = tool_msgs
            .iter()
            .find(|m| m.text().unwrap_or("").contains("denied by guardrail"))
            .expect("Should have a denied tool result");
        assert!(denied.text().unwrap().contains("not permitted"));
    }

    #[tokio::test]
    async fn test_output_guardrail_denies() {
        let provider = Arc::new(MockProvider::new(vec![ChatResponse {
            message: ChatMessage::assistant("Here is the SECRET_KEY: abc123"),
            usage: None,
        }]));

        let guardrails = Guardrails::new().with_output_guardrail(|msg| {
            if let Some(text) = msg.text() {
                if text.contains("SECRET_KEY") {
                    return GuardrailResult::Deny("Output contains secrets".into());
                }
            }
            GuardrailResult::Allow
        });

        let result = run(
            provider,
            vec![ChatMessage::user("give me the key")],
            vec![],
            |_| Ok("".into()),
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            guardrails,
            |_| {},
        )
        .await;

        match result {
            Err(AgentError::Guardrailed(reason)) => {
                assert!(reason.contains("secrets"));
            }
            other => panic!("Expected Guardrailed, got: {:?}", other),
        }
    }
}
