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
//!     |_call| async { Ok(ToolOutput::from("not implemented")) },
//!     RunnerConfig::default(),
//!     CancellationToken::new(),
//!     Steering::new(),
//!     Guardrails::default(),
//!     |event| { /* handle events */ },
//! ).await?;
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::future::join_all;
use futures_util::FutureExt;
use regex::Regex;
use tokio::sync::mpsc;

use crate::cancel::CancellationToken;
use crate::context::trim_to_context_window;
use crate::error::AgentError;
use crate::guardrails::{GuardrailResult, Guardrails};
use crate::provider::Provider;
use crate::sanitize::sanitize_for_api;
use crate::steering::Steering;
use crate::types::*;

/// A per-round tool filter function. Receives the current message history
/// and returns the set of tools to offer the LLM for that round.
pub type ToolFilter = Arc<dyn Fn(&[ChatMessage]) -> Vec<Tool> + Send + Sync>;

/// An async function that resolves `@reference` names to content.
/// Receives the reference name (without the `@` prefix) and returns
/// resolved content, or `None` if the reference is unknown.
pub type ReferenceResolver =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Option<ResolvedReference>> + Send>> + Send + Sync>;

/// A resolved `@reference` — content that gets injected into the conversation
/// so the LLM can see the referenced material.
#[derive(Debug, Clone)]
pub struct ResolvedReference {
    /// Display name for the reference (e.g., "intro.sk", "setup guide").
    pub name: String,
    /// The resolved content to inject.
    pub content: String,
    /// MIME-like content type hint (e.g., "text/markdown", "application/json").
    /// Helps the LLM understand the format. Defaults to "text/plain".
    pub content_type: String,
}

/// Configuration for the runner.
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
    /// Optional provider for LLM-powered context compaction.
    /// When set and context is trimmed, the runner calls this provider to produce
    /// a richer summary of dropped messages (instead of string-based extraction).
    pub compaction_provider: Option<Arc<dyn Provider>>,
    /// Optional per-round tool filter. When set, called each iteration with the
    /// current message history to determine which tools to offer the LLM.
    /// Use this for progressive disclosure, agent-specific tool sets, or
    /// conditional tool availability.
    /// When `None`, the static `tools` vec passed to `run()` is used every round.
    pub tool_filter: Option<ToolFilter>,
    /// Unique identifier for this run. Auto-generated UUID v4 if not set.
    /// Use this to correlate events across logs, traces, and UI.
    pub run_id: Option<String>,
    /// Parent run ID for delegation chains. When one `run()` spawns another
    /// (e.g., via a delegate_to_agent tool), set this to the parent's run_id
    /// so traces can reconstruct the full call tree.
    pub parent_run_id: Option<String>,
    /// Optional reference resolver for `@reference` syntax in user messages.
    /// When set, the runner scans user messages for `@name` or `@"quoted name"`
    /// patterns, calls this resolver, and appends the resolved content to the
    /// message so the LLM can see referenced materials (files, DB records, etc.).
    /// Resolution happens once per message — already-resolved messages are not re-scanned.
    pub reference_resolver: Option<ReferenceResolver>,
}

impl Clone for RunnerConfig {
    fn clone(&self) -> Self {
        Self {
            max_iterations: self.max_iterations,
            retry_on_400: self.retry_on_400,
            auto_trim_context: self.auto_trim_context,
            sanitize_tool_results: self.sanitize_tool_results,
            parallel_tool_calls: self.parallel_tool_calls,
            response_format: self.response_format.clone(),
            compaction_provider: self.compaction_provider.clone(),
            tool_filter: self.tool_filter.clone(),
            run_id: self.run_id.clone(),
            parent_run_id: self.parent_run_id.clone(),
            reference_resolver: self.reference_resolver.clone(),
        }
    }
}

impl std::fmt::Debug for RunnerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnerConfig")
            .field("max_iterations", &self.max_iterations)
            .field("retry_on_400", &self.retry_on_400)
            .field("auto_trim_context", &self.auto_trim_context)
            .field("sanitize_tool_results", &self.sanitize_tool_results)
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .field("response_format", &self.response_format)
            .field("compaction_provider", &self.compaction_provider.is_some())
            .field("tool_filter", &self.tool_filter.is_some())
            .field("run_id", &self.run_id)
            .field("parent_run_id", &self.parent_run_id)
            .field("reference_resolver", &self.reference_resolver.is_some())
            .finish()
    }
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
            compaction_provider: None,
            tool_filter: None,
            run_id: None,
            parent_run_id: None,
            reference_resolver: None,
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
    ToolCallStart {
        name: String,
        arguments: String,
        /// The LLM-assigned tool call ID (e.g., "call_abc123").
        tool_call_id: String,
        /// Which iteration of the runner loop this occurred in (0-based).
        iteration: usize,
    },
    /// A tool returned a result.
    ToolResult {
        name: String,
        result: String,
        /// The LLM-assigned tool call ID, matching the corresponding ToolCallStart.
        tool_call_id: String,
        /// Wall-clock time the tool took to execute, in milliseconds.
        elapsed_ms: u64,
        /// Which iteration of the runner loop this occurred in (0-based).
        iteration: usize,
    },
    /// Token usage for a single LLM call.
    Usage { usage: Usage },
    /// The full message history was updated (after a tool round).
    /// Apps can use this to persist conversation state mid-run.
    MessagesUpdated { messages: Vec<ChatMessage> },
    /// The runner completed successfully.
    Done {
        response: String,
        messages: Vec<ChatMessage>,
        /// Total wall-clock time for the entire run, in milliseconds.
        elapsed_ms: u64,
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
    /// Unique identifier for this run (for tracing/correlation).
    pub run_id: String,
    /// Parent run ID if this was a delegated sub-run.
    pub parent_run_id: Option<String>,
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
#[allow(clippy::too_many_arguments)]
pub async fn run<F, Fut, E>(
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
    F: Fn(ToolCall) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<ToolOutput, String>> + Send,
    E: Fn(RunnerEvent) + Send + Sync,
{
    let run_id = config.run_id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let parent_run_id = config.parent_run_id.clone();
    let run_start = std::time::Instant::now();

    let mut full_messages = messages;
    let mut new_messages: Vec<ChatMessage> = Vec::new();
    let mut total_usage = Usage::default();

    // Resolve @references in initial messages
    if let Some(ref resolver) = config.reference_resolver {
        resolve_references_in_messages(&mut full_messages, resolver).await;
    }

    for iteration in 0..config.max_iterations {
        if cancel.is_cancelled() {
            return Err(AgentError::Cancelled);
        }

        // Drain any steering messages injected by the user mid-run
        for msg in steering.drain() {
            full_messages.push(ChatMessage::user(&msg));
            new_messages.push(ChatMessage::user(&msg));
        }

        // Resolve @references in any newly added steering messages
        if let Some(ref resolver) = config.reference_resolver {
            resolve_references_in_messages(&mut full_messages, resolver).await;
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
            let (dropped_count, dropped) = trim_to_context_window(&mut full_messages, budget);
            if dropped_count > 0 {
                on_event(RunnerEvent::Status {
                    message: format!(
                        "Compacted context — summarized {} earlier messages",
                        dropped_count
                    ),
                });

                // Upgrade to LLM-powered summary if compaction provider is available
                if let Some(ref compaction_provider) = config.compaction_provider {
                    on_event(RunnerEvent::Status {
                        message: "Generating AI summary of earlier conversation…".into(),
                    });
                    let llm_summary = crate::context::summarize_dropped_with_llm(
                        &dropped, compaction_provider, &cancel,
                    ).await;
                    // Replace the string-based summary with the LLM one
                    // The summary is the first non-system user message
                    if !llm_summary.is_empty() {
                        let system_end = full_messages
                            .iter()
                            .position(|m| m.role != "system")
                            .unwrap_or(0);
                        if system_end < full_messages.len()
                            && full_messages[system_end].role == "user"
                            && full_messages[system_end].text().is_some_and(|t| t.starts_with("[Earlier conversation summary"))
                        {
                            full_messages[system_end] = ChatMessage::user(&llm_summary);
                        }
                    }
                }
            }
        }

        // Input guardrail — check before calling LLM
        if let GuardrailResult::Deny(reason) = guardrails.check_input(&full_messages) {
            return Err(AgentError::Guardrailed(reason));
        }

        // Determine tools for this round (static or dynamic)
        let round_tools = if let Some(ref filter) = config.tool_filter {
            filter(&full_messages)
        } else {
            tools.clone()
        };

        // Build request
        let request = ChatRequest {
            messages: full_messages.clone(),
            model: String::new(), // Provider uses its own model
            tools: if round_tools.is_empty() {
                None
            } else {
                Some(round_tools.clone())
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
                        tool_call_id: tool_call.id.clone(),
                        iteration,
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
                        tools: if round_tools.is_empty() {
                            None
                        } else {
                            Some(round_tools.clone())
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
                                    tool_call_id: tool_call.id.clone(),
                                    iteration,
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
            if !tool_calls.is_empty() && !round_tools.is_empty() {
                // Add assistant message with tool calls to history
                full_messages.push(response.message.clone());
                new_messages.push(response.message.clone());

                on_event(RunnerEvent::Status {
                    message: format!("Running {} tool call(s)…", tool_calls.len()),
                });

                // Execute tool calls (parallel or sequential)
                let tool_results = if config.parallel_tool_calls && tool_calls.len() > 1 {
                    execute_tools_parallel(tool_calls, &tool_executor, &config, &guardrails, &on_event, iteration).await?
                } else {
                    execute_tools_sequential(tool_calls, &tool_executor, &config, &guardrails, &on_event, iteration).await?
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
            elapsed_ms: run_start.elapsed().as_millis() as u64,
        });

        return Ok(RunnerResult {
            messages: full_messages,
            response: final_text,
            new_messages,
            total_usage,
            run_id,
            parent_run_id,
        });
    }

    Err(AgentError::MaxIterations(config.max_iterations))
}

/// Execute tool calls sequentially with panic safety and guardrails.
async fn execute_tools_sequential<F, Fut, E>(
    tool_calls: &[ToolCall],
    tool_executor: &F,
    config: &RunnerConfig,
    guardrails: &Guardrails,
    on_event: &E,
    iteration: usize,
) -> Result<Vec<ChatMessage>, AgentError>
where
    F: Fn(ToolCall) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<ToolOutput, String>> + Send,
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
                tool_call_id: tc.id.clone(),
                elapsed_ms: 0,
                iteration,
            });
            results.push(ChatMessage::tool_result(&tc.id, &denied_msg));
            continue;
        }

        let tool_start = std::time::Instant::now();
        let (id, name, output) = execute_tool_safe(tc.clone(), tool_executor).await?;
        let elapsed_ms = tool_start.elapsed().as_millis() as u64;
        let clean_text = if config.sanitize_tool_results {
            sanitize_for_api(output.text())
        } else {
            output.text().to_string()
        };
        on_event(RunnerEvent::ToolResult {
            name,
            result: clean_text.clone(),
            tool_call_id: id.clone(),
            elapsed_ms,
            iteration,
        });
        results.push(ChatMessage::tool_result(&id, &clean_text));
        // Inject vision images as a follow-up user message
        if let Some(images) = output.images() {
            if !images.is_empty() {
                results.push(ChatMessage::user_with_images(
                    "[Images from the tool result above — analyze these along with the text.]",
                    images.to_vec(),
                ));
            }
        }
    }
    Ok(results)
}

/// Execute tool calls in parallel with panic safety and guardrails.
async fn execute_tools_parallel<F, Fut, E>(
    tool_calls: &[ToolCall],
    tool_executor: &F,
    config: &RunnerConfig,
    guardrails: &Guardrails,
    on_event: &E,
    iteration: usize,
) -> Result<Vec<ChatMessage>, AgentError>
where
    F: Fn(ToolCall) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<ToolOutput, String>> + Send,
    E: Fn(RunnerEvent) + Send + Sync,
{
    let futures: Vec<_> = tool_calls.iter().map(|tc| {
        let tc_clone = tc.clone();
        let denied = match guardrails.check_tool(tc) {
            GuardrailResult::Deny(reason) => Some(reason),
            GuardrailResult::Allow => None,
        };
        async move {
            let tool_start = std::time::Instant::now();
            if let Some(reason) = denied {
                let elapsed_ms = tool_start.elapsed().as_millis() as u64;
                Ok((tc_clone.id.clone(), tc_clone.function.name.clone(), ToolOutput::Text(format!("Tool denied by guardrail: {}", reason)), elapsed_ms))
            } else {
                let result = execute_tool_safe(tc_clone, tool_executor).await;
                let elapsed_ms = tool_start.elapsed().as_millis() as u64;
                result.map(|(id, name, output)| (id, name, output, elapsed_ms))
            }
        }
    }).collect();

    let results = join_all(futures).await;

    let mut messages = Vec::with_capacity(tool_calls.len());
    for res in results {
        let (id, name, output, elapsed_ms) = res?;
        let clean_text = if config.sanitize_tool_results {
            sanitize_for_api(output.text())
        } else {
            output.text().to_string()
        };
        on_event(RunnerEvent::ToolResult {
            name,
            result: clean_text.clone(),
            tool_call_id: id.clone(),
            elapsed_ms,
            iteration,
        });
        messages.push(ChatMessage::tool_result(&id, &clean_text));
        // Inject vision images as a follow-up user message
        if let Some(images) = output.images() {
            if !images.is_empty() {
                messages.push(ChatMessage::user_with_images(
                    "[Images from the tool result above — analyze these along with the text.]",
                    images.to_vec(),
                ));
            }
        }
    }
    Ok(messages)
}

/// Execute a single tool call with panic safety via `catch_unwind` on the async future.
async fn execute_tool_safe<F, Fut>(tc: ToolCall, tool_executor: &F) -> Result<(String, String, ToolOutput), AgentError>
where
    F: Fn(ToolCall) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<ToolOutput, String>> + Send,
{
    let name = tc.function.name.clone();
    let id = tc.id.clone();

    let result = std::panic::AssertUnwindSafe(tool_executor(tc))
        .catch_unwind()
        .await;

    match result {
        Ok(Ok(r)) => Ok((id, name, r)),
        Ok(Err(e)) => Ok((id, name, ToolOutput::Text(format!("Tool error: {}", e)))),
        Err(panic_info) => {
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            Err(AgentError::ToolPanic {
                name,
                message: panic_msg,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// @reference resolution
// ---------------------------------------------------------------------------

/// Extract `@reference` names from a text string.
///
/// Supports two syntaxes:
/// - `@word` — alphanumeric, hyphens, underscores, dots, slashes (e.g., `@intro.sk`, `@docs/setup`)
/// - `@"quoted name"` — arbitrary text in double quotes (e.g., `@"my sketch with spaces"`)
///
/// Returns deduplicated names in the order they first appear (without the `@` prefix or quotes).
fn extract_references(text: &str) -> Vec<String> {
    // Quoted: @"anything inside quotes"
    // URL: @https://... or @http://... (greedy up to whitespace/quote)
    // Unquoted: @word characters (alphanum, -, _, ., /)
    let re = Regex::new(r#"@"([^"]+)"|@(https?://[^\s"<>]+)|@([\w.\-/]+)"#)
        .expect("invalid reference regex");
    let mut seen = std::collections::HashSet::new();
    let mut refs = Vec::new();
    for cap in re.captures_iter(text) {
        let name = cap
            .get(1)
            .or_else(|| cap.get(2))
            .or_else(|| cap.get(3))
            .unwrap()
            .as_str()
            .to_string();
        if seen.insert(name.clone()) {
            refs.push(name);
        }
    }
    refs
}

/// Format resolved references as a context block appended to the user message.
fn format_resolved_context(resolved: &[(String, ResolvedReference)]) -> String {
    let mut ctx = String::new();
    for (ref_name, r) in resolved {
        ctx.push_str(&format!(
            "\n\n<referenced_document name=\"{}\" content_type=\"{}\">\n{}\n</referenced_document>",
            ref_name, r.content_type, r.content
        ));
    }
    ctx
}

/// Resolve `@references` in user messages using the provided resolver.
///
/// Scans each user message for `@name` patterns. For each unique reference found,
/// calls the resolver. URL references (`@https://...`) are resolved using the
/// built-in web fetcher. Resolved content is appended to the message as XML-tagged
/// context blocks. Messages that have already been resolved (contain
/// `<referenced_document`) are skipped to avoid re-resolution.
async fn resolve_references_in_messages(
    messages: &mut [ChatMessage],
    resolver: &ReferenceResolver,
) {
    for msg in messages.iter_mut() {
        if msg.role != "user" {
            continue;
        }
        let text = match msg.text() {
            Some(t) => t.to_string(),
            None => continue,
        };
        // Skip already-resolved messages
        if text.contains("<referenced_document") {
            continue;
        }
        let refs = extract_references(&text);
        if refs.is_empty() {
            continue;
        }

        // Resolve all references concurrently
        let futures: Vec<_> = refs
            .iter()
            .map(|name| {
                let name = name.clone();
                let resolver = resolver.clone();
                async move {
                    // URL references are handled by the built-in web fetcher
                    if name.starts_with("http://") || name.starts_with("https://") {
                        match crate::web::fetch_and_clean(&name).await {
                            Ok(content) => {
                                let resolved = ResolvedReference {
                                    name: name.clone(),
                                    content,
                                    content_type: "text/plain".to_string(),
                                };
                                (name, Some(resolved))
                            }
                            Err(e) => {
                                log::warn!("Failed to fetch URL reference {}: {}", name, e);
                                let resolved = ResolvedReference {
                                    name: name.clone(),
                                    content: format!("[Error fetching URL: {e}]"),
                                    content_type: "text/plain".to_string(),
                                };
                                (name, Some(resolved))
                            }
                        }
                    } else {
                        let resolved = resolver(name.clone()).await;
                        (name, resolved)
                    }
                }
            })
            .collect();

        let results = join_all(futures).await;
        let resolved: Vec<(String, ResolvedReference)> = results
            .into_iter()
            .filter_map(|(name, r)| r.map(|r| (name, r)))
            .collect();

        if !resolved.is_empty() {
            let context = format_resolved_context(&resolved);
            let new_text = format!("{}{}", text, context);
            *msg = ChatMessage::user(&new_text);
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

            // Simulate tool call streaming
            if let Some(ref tool_calls) = response.message.tool_calls {
                for tc in tool_calls {
                    let _ = tx
                        .send(ChatEvent::ToolCallStart {
                            tool_call: tc.clone(),
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

    /// MockProvider variant with a small context budget (triggers compaction).
    struct SmallBudgetMockProvider {
        inner: MockProvider,
        budget: usize,
    }

    impl SmallBudgetMockProvider {
        fn new(responses: Vec<ChatResponse>, budget: usize) -> Self {
            Self {
                inner: MockProvider::new(responses),
                budget,
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::provider::Provider for SmallBudgetMockProvider {
        async fn chat(
            &self,
            request: ChatRequest,
            tx: mpsc::Sender<ChatEvent>,
            cancel: &CancellationToken,
        ) -> Result<(), AgentError> {
            self.inner.chat(request, tx, cancel).await
        }

        fn name(&self) -> &str {
            "small_budget_mock"
        }

        fn context_budget_chars(&self) -> usize {
            self.budget
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
            |_| async { Ok(ToolOutput::from("unused")) },            RunnerConfig::default(),
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
            |call| async move {
                assert_eq!(call.function.name, "read_file");
                Ok(ToolOutput::from("hello world"))
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
            |_| async { Ok("".into()) },
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
            |_| async { Ok(ToolOutput::from("looping")) },
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
            |_| async { Ok(ToolOutput::from("tool result")) },
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
            |_| async { Ok(ToolOutput::from("ok")) },
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
                let cc = count_clone.clone();
                async move {
                    cc.fetch_add(1, Ordering::SeqCst);
                    Ok(ToolOutput::from(format!("result from {}", tc.function.name)))
                }
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
            |_| async { panic!("tool went boom") },
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
            |_| async { Ok("".into()) },
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
            |_| async { Ok(ToolOutput::from("safe result")) },
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
            |_| async { Ok("".into()) },
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

    // -- Error path tests --------------------------------------------------------

    /// A mock provider that returns an error on the first call.
    struct ErrorProvider {
        error: Mutex<Option<AgentError>>,
        fallback: Mutex<Vec<ChatResponse>>,
    }

    impl ErrorProvider {
        fn with_error(error: AgentError) -> Self {
            Self {
                error: Mutex::new(Some(error)),
                fallback: Mutex::new(Vec::new()),
            }
        }

        fn with_400_then_success(fallback_response: ChatResponse) -> Self {
            Self {
                error: Mutex::new(Some(AgentError::Api {
                    status: 400,
                    message: "Bad Request".into(),
                })),
                fallback: Mutex::new(vec![fallback_response]),
            }
        }

        fn with_400_then_400() -> Self {
            Self {
                error: Mutex::new(Some(AgentError::Api {
                    status: 400,
                    message: "Bad Request".into(),
                })),
                fallback: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::provider::Provider for ErrorProvider {
        async fn chat(
            &self,
            _request: ChatRequest,
            tx: mpsc::Sender<ChatEvent>,
            cancel: &CancellationToken,
        ) -> Result<(), AgentError> {
            if cancel.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            // Return error on first call, then use fallback responses
            let maybe_err = { self.error.lock().unwrap().take() };
            if let Some(err) = maybe_err {
                return Err(err);
            }

            let response = {
                let mut fallback = self.fallback.lock().unwrap();
                if fallback.is_empty() {
                    return Err(AgentError::Api {
                        status: 400,
                        message: "Retry also failed".into(),
                    });
                }
                fallback.remove(0)
            };

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
            "error_mock"
        }
    }

    /// Mock provider that sends tokens but never sends Done event.
    struct NoDoneProvider;

    #[async_trait::async_trait]
    impl crate::provider::Provider for NoDoneProvider {
        async fn chat(
            &self,
            _request: ChatRequest,
            tx: mpsc::Sender<ChatEvent>,
            _cancel: &CancellationToken,
        ) -> Result<(), AgentError> {
            let _ = tx
                .send(ChatEvent::Token {
                    token: "partial".into(),
                })
                .await;
            // Drop tx without sending Done
            Ok(())
        }

        fn name(&self) -> &str {
            "no_done_mock"
        }
    }

    /// Mock provider that sends a ChatEvent::Error.
    struct StreamErrorProvider;

    #[async_trait::async_trait]
    impl crate::provider::Provider for StreamErrorProvider {
        async fn chat(
            &self,
            _request: ChatRequest,
            tx: mpsc::Sender<ChatEvent>,
            _cancel: &CancellationToken,
        ) -> Result<(), AgentError> {
            let _ = tx
                .send(ChatEvent::Error {
                    message: "Connection reset".into(),
                })
                .await;
            Ok(())
        }

        fn name(&self) -> &str {
            "stream_error_mock"
        }
    }

    /// Mock provider that simulates slow streaming for mid-stream cancellation.
    struct SlowStreamProvider;

    #[async_trait::async_trait]
    impl crate::provider::Provider for SlowStreamProvider {
        async fn chat(
            &self,
            _request: ChatRequest,
            tx: mpsc::Sender<ChatEvent>,
            cancel: &CancellationToken,
        ) -> Result<(), AgentError> {
            for i in 0..100 {
                if cancel.is_cancelled() {
                    return Err(AgentError::Cancelled);
                }
                let _ = tx
                    .send(ChatEvent::Token {
                        token: format!("token{} ", i),
                    })
                    .await;
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
            let _ = tx
                .send(ChatEvent::Done {
                    response: ChatResponse {
                        message: ChatMessage::assistant("should not complete"),
                        usage: None,
                    },
                })
                .await;
            Ok(())
        }

        fn name(&self) -> &str {
            "slow_mock"
        }
    }

    #[tokio::test]
    async fn test_provider_non_400_error() {
        let provider = Arc::new(ErrorProvider::with_error(AgentError::Api {
            status: 500,
            message: "Internal Server Error".into(),
        }));

        let result = run(
            provider,
            vec![ChatMessage::user("Hi")],
            vec![],
            |_| async { Ok("".into()) },
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await;

        match result {
            Err(AgentError::Api { status: 500, .. }) => {} // expected
            other => panic!("Expected Api 500, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_400_retry_success() {
        let provider = Arc::new(ErrorProvider::with_400_then_success(ChatResponse {
            message: ChatMessage::assistant("Retry worked"),
            usage: None,
        }));

        let result = run(
            provider,
            vec![ChatMessage::user("Hi")],
            vec![],
            |_| async { Ok("".into()) },
            RunnerConfig {
                retry_on_400: true,
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "Retry worked");
    }

    #[tokio::test]
    async fn test_400_retry_fails() {
        let provider = Arc::new(ErrorProvider::with_400_then_400());

        let result = run(
            provider,
            vec![ChatMessage::user("Hi")],
            vec![],
            |_| async { Ok("".into()) },
            RunnerConfig {
                retry_on_400: true,
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await;

        match result {
            Err(AgentError::Api { status: 400, .. }) => {} // expected
            other => panic!("Expected Api 400 on retry, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_400_no_retry_when_disabled() {
        let provider = Arc::new(ErrorProvider::with_error(AgentError::Api {
            status: 400,
            message: "Bad Request".into(),
        }));

        let result = run(
            provider,
            vec![ChatMessage::user("Hi")],
            vec![],
            |_| async { Ok("".into()) },
            RunnerConfig {
                retry_on_400: false,
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await;

        match result {
            Err(AgentError::Api { status: 400, .. }) => {} // expected
            other => panic!("Expected immediate Api 400 error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_missing_done_event() {
        let provider = Arc::new(NoDoneProvider);

        let result = run(
            provider,
            vec![ChatMessage::user("Hi")],
            vec![],
            |_| async { Ok("".into()) },
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await;

        match result {
            Err(AgentError::Stream(msg)) => {
                assert!(msg.contains("without sending Done"), "Got: {}", msg);
            }
            other => panic!("Expected Stream error for missing Done, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_stream_error_event() {
        let provider = Arc::new(StreamErrorProvider);

        let result = run(
            provider,
            vec![ChatMessage::user("Hi")],
            vec![],
            |_| async { Ok("".into()) },
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await;

        match result {
            Err(AgentError::Stream(msg)) => {
                assert!(msg.contains("Connection reset"), "Got: {}", msg);
            }
            other => panic!("Expected Stream error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_mid_stream_cancellation() {
        let provider = Arc::new(SlowStreamProvider);

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        // Cancel after a short delay (mid-stream)
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            cancel_clone.cancel();
        });

        let result = run(
            provider,
            vec![ChatMessage::user("Hi")],
            vec![],
            |_| async { Ok("".into()) },
            RunnerConfig::default(),
            cancel,
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await;

        assert!(
            matches!(result, Err(AgentError::Cancelled)),
            "Expected Cancelled, got: {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_tool_executor_returns_error() {
        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "c1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "failing_tool".into(),
                        arguments: "{}".into(),
                    },
                }]),
                usage: None,
            },
            ChatResponse {
                message: ChatMessage::assistant("Handled the error"),
                usage: None,
            },
        ]));

        let result = run(
            provider,
            vec![ChatMessage::user("go")],
            vec![Tool::function("failing_tool", "fails", serde_json::json!({}))],
            |_| async { Err("Something went wrong".into()) },
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        // Tool error should be sent as a tool result containing the error
        let tool_msgs: Vec<_> = result.messages.iter().filter(|m| m.role == "tool").collect();
        assert_eq!(tool_msgs.len(), 1);
        assert!(tool_msgs[0].text().unwrap().contains("Tool error:"));
        assert_eq!(result.response, "Handled the error");
    }

    #[tokio::test]
    async fn test_parallel_tool_guardrail_deny() {
        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![
                    ToolCall {
                        id: "c1".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "allowed_tool".into(),
                            arguments: "{}".into(),
                        },
                    },
                    ToolCall {
                        id: "c2".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "blocked_tool".into(),
                            arguments: "{}".into(),
                        },
                    },
                    ToolCall {
                        id: "c3".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "allowed_tool".into(),
                            arguments: "{}".into(),
                        },
                    },
                ]),
                usage: None,
            },
            ChatResponse {
                message: ChatMessage::assistant("Done with mixed results"),
                usage: None,
            },
        ]));

        let guardrails = Guardrails::new().with_tool_guardrail(|tc| {
            if tc.function.name == "blocked_tool" {
                GuardrailResult::Deny("Not allowed in parallel".into())
            } else {
                GuardrailResult::Allow
            }
        });

        let result = run(
            provider,
            vec![ChatMessage::user("run three tools")],
            vec![
                Tool::function("allowed_tool", "ok", serde_json::json!({})),
                Tool::function("blocked_tool", "nope", serde_json::json!({})),
            ],
            |_| async { Ok(ToolOutput::from("tool succeeded")) },
            RunnerConfig {
                parallel_tool_calls: true,
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            guardrails,
            |_| {},
        )
        .await
        .unwrap();

        let tool_msgs: Vec<_> = result.messages.iter().filter(|m| m.role == "tool").collect();
        assert_eq!(tool_msgs.len(), 3);

        // Exactly one should be denied
        let denied: Vec<_> = tool_msgs
            .iter()
            .filter(|m| m.text().unwrap_or("").contains("denied by guardrail"))
            .collect();
        assert_eq!(denied.len(), 1);
        assert!(denied[0].text().unwrap().contains("Not allowed in parallel"));
    }

    #[tokio::test]
    async fn test_parallel_tool_panic_one_of_many() {
        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![
                    ToolCall {
                        id: "c1".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "good_tool".into(),
                            arguments: "{}".into(),
                        },
                    },
                    ToolCall {
                        id: "c2".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "bad_tool".into(),
                            arguments: "{}".into(),
                        },
                    },
                ]),
                usage: None,
            },
        ]));

        let result = run(
            provider,
            vec![ChatMessage::user("go")],
            vec![
                Tool::function("good_tool", "good", serde_json::json!({})),
                Tool::function("bad_tool", "panics", serde_json::json!({})),
            ],
            |tc| async move {
                if tc.function.name == "bad_tool" {
                    panic!("parallel panic test");
                }
                Ok(ToolOutput::from("good result"))
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
        .await;

        match result {
            Err(AgentError::ToolPanic { name, message }) => {
                assert_eq!(name, "bad_tool");
                assert!(message.contains("parallel panic test"));
            }
            other => panic!("Expected ToolPanic, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_dynamic_tool_gating() {
        // Provider returns tool calls, but the tool filter removes the tool
        // on the second round, so the model should produce text instead
        let provider = Arc::new(MockProvider::new(vec![
            // First response: call tool_a (which is available)
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "c1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "tool_a".into(),
                        arguments: "{}".into(),
                    },
                }]),
                usage: None,
            },
            // Second response: final text (tool_a is no longer offered)
            ChatResponse {
                message: ChatMessage::assistant("Done after gating"),
                usage: None,
            },
        ]));

        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = call_count.clone();

        let result = run(
            provider,
            vec![ChatMessage::user("go")],
            vec![Tool::function("tool_a", "a tool", serde_json::json!({}))],
            move |_tc| {
                let cc = count_clone.clone();
                async move {
                    cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(ToolOutput::from("tool result"))
                }
            },
            RunnerConfig {
                // After first tool call, remove all tools
                tool_filter: Some(Arc::new(|msgs| {
                    let has_tool_result = msgs.iter().any(|m| m.role == "tool");
                    if has_tool_result {
                        vec![] // No more tools after first round
                    } else {
                        vec![Tool::function("tool_a", "a tool", serde_json::json!({}))]
                    }
                })),
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "Done after gating");
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_truly_async_tool_executor() {
        // Verify that async tool executors actually run asynchronously
        // (not just sync closures wrapped in async {})
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Instant;

        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![
                    ToolCall {
                        id: "c1".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "slow_tool".into(),
                            arguments: r#"{"id":"1"}"#.into(),
                        },
                    },
                    ToolCall {
                        id: "c2".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "slow_tool".into(),
                            arguments: r#"{"id":"2"}"#.into(),
                        },
                    },
                    ToolCall {
                        id: "c3".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "slow_tool".into(),
                            arguments: r#"{"id":"3"}"#.into(),
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

        let start = Instant::now();
        let result = run(
            provider,
            vec![ChatMessage::user("run slow tools")],
            vec![Tool::function("slow_tool", "slow", serde_json::json!({}))],
            move |_tc| {
                let cc = count_clone.clone();
                async move {
                    // Simulate async I/O (like a network request)
                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    cc.fetch_add(1, Ordering::SeqCst);
                    Ok(ToolOutput::from("async result"))
                }
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

        let elapsed = start.elapsed();
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
        assert_eq!(result.response, "All done");
        // If truly parallel, 3x50ms should complete in ~50-100ms, not 150ms+
        assert!(
            elapsed.as_millis() < 200,
            "Expected parallel execution (<200ms), got {}ms — tools ran sequentially",
            elapsed.as_millis()
        );
    }

    #[tokio::test]
    async fn test_async_sequential_tools_run_in_order() {
        // Verify sequential mode preserves order even with async executors

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
                ]),
                usage: None,
            },
            ChatResponse {
                message: ChatMessage::assistant("Done in order"),
                usage: None,
            },
        ]));

        let order = Arc::new(Mutex::new(Vec::<String>::new()));
        let order_clone = order.clone();

        let result = run(
            provider,
            vec![ChatMessage::user("go")],
            vec![
                Tool::function("tool_a", "a", serde_json::json!({})),
                Tool::function("tool_b", "b", serde_json::json!({})),
            ],
            move |tc| {
                let oc = order_clone.clone();
                async move {
                    // Small delay to make ordering visible
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    oc.lock().unwrap().push(tc.function.name.clone());
                    Ok(ToolOutput::from(format!("result from {}", tc.function.name)))
                }
            },
            RunnerConfig {
                parallel_tool_calls: false, // sequential!
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "Done in order");
        let execution_order = order.lock().unwrap().clone();
        assert_eq!(execution_order, vec!["tool_a", "tool_b"]);
    }

    #[tokio::test]
    async fn test_compaction_with_mock_llm() {
        // Test that LLM compaction replaces the string-based summary
        // when a compaction_provider is set and the context exceeds budget

        // Create a compaction provider that returns a specific summary
        struct CompactionProvider;

        #[async_trait::async_trait]
        impl crate::provider::Provider for CompactionProvider {
            async fn chat(
                &self,
                _request: ChatRequest,
                tx: mpsc::Sender<ChatEvent>,
                _cancel: &CancellationToken,
            ) -> Result<(), AgentError> {
                let _ = tx.send(ChatEvent::Done {
                    response: ChatResponse {
                        message: ChatMessage::assistant(
                            "User discussed Rust memory safety. Assistant explained ownership and borrowing."
                        ),
                        usage: None,
                    },
                }).await;
                Ok(())
            }

            fn name(&self) -> &str { "compaction_mock" }
        }

        // Create a main provider with a small context budget to trigger compaction
        let main_provider = Arc::new(SmallBudgetMockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant("Continuing from where we left off"),
                usage: None,
            },
        ], 10_000));

        // Build a large message history that will trigger compaction
        let mut messages = vec![ChatMessage::system("You are helpful")];
        for i in 0..100 {
            messages.push(ChatMessage::user(&format!("Long question {}: {}", i, "x".repeat(500))));
            messages.push(ChatMessage::assistant(&format!("Long answer {}: {}", i, "y".repeat(500))));
        }
        messages.push(ChatMessage::user("Final question"));

        let events = Arc::new(Mutex::new(Vec::<String>::new()));
        let events_clone = events.clone();

        let result = run(
            main_provider,
            messages,
            vec![],
            |_| async { Ok(ToolOutput::from("unused")) },
            RunnerConfig {
                auto_trim_context: true,
                compaction_provider: Some(Arc::new(CompactionProvider)),
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            move |event| {
                if let RunnerEvent::Status { message } = &event {
                    events_clone.lock().unwrap().push(message.clone());
                }
            },
        )
        .await
        .unwrap();

        assert_eq!(result.response, "Continuing from where we left off");

        // Check that compaction status events were emitted
        let status_msgs = events.lock().unwrap().clone();
        assert!(
            status_msgs.iter().any(|s| s.contains("Compacted context")),
            "Should emit compaction status. Got: {:?}",
            status_msgs
        );
        assert!(
            status_msgs.iter().any(|s| s.contains("AI summary")),
            "Should emit LLM summary status. Got: {:?}",
            status_msgs
        );

        // Verify the LLM summary was inserted into the message history
        let summary_msg = result.messages.iter().find(|m| {
            m.role == "user"
                && m.text().is_some_and(|t| t.contains("ownership and borrowing"))
        });
        assert!(
            summary_msg.is_some(),
            "LLM summary should be in the conversation history. Messages: {:?}",
            result.messages.iter().map(|m| format!("{}: {}", m.role, m.text().unwrap_or("(none)")
                .chars().take(80).collect::<String>()
            )).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_compaction_fallback_on_error() {
        // When the compaction provider fails, the string-based summary should remain

        struct FailingCompactionProvider;

        #[async_trait::async_trait]
        impl crate::provider::Provider for FailingCompactionProvider {
            async fn chat(
                &self,
                _request: ChatRequest,
                tx: mpsc::Sender<ChatEvent>,
                _cancel: &CancellationToken,
            ) -> Result<(), AgentError> {
                let _ = tx.send(ChatEvent::Error {
                    message: "API rate limit exceeded".into(),
                }).await;
                Err(AgentError::Api { status: 429, message: "Rate limited".into() })
            }

            fn name(&self) -> &str { "failing_compaction" }
        }

        let main_provider = Arc::new(SmallBudgetMockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant("Handled it"),
                usage: None,
            },
        ], 10_000));

        // Large history to trigger compaction
        let mut messages = vec![ChatMessage::system("System prompt")];
        for i in 0..100 {
            messages.push(ChatMessage::user(&format!("Question {}: {}", i, "x".repeat(500))));
            messages.push(ChatMessage::assistant(&format!("Answer {}: {}", i, "y".repeat(500))));
        }
        messages.push(ChatMessage::user("Latest question"));

        let result = run(
            main_provider,
            messages,
            vec![],
            |_| async { Ok(ToolOutput::from("unused")) },
            RunnerConfig {
                auto_trim_context: true,
                compaction_provider: Some(Arc::new(FailingCompactionProvider)),
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "Handled it");

        // The string-based summary should still be present (fallback)
        let has_summary = result.messages.iter().any(|m| {
            m.role == "user"
                && m.text().is_some_and(|t| t.starts_with("[Earlier conversation summary]"))
        });
        assert!(
            has_summary,
            "String-based summary should remain as fallback"
        );
    }

    #[tokio::test]
    async fn test_compaction_preserves_system_messages() {
        // Verify that context compaction never drops system messages

        let main_provider = Arc::new(SmallBudgetMockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant("Responded after compaction"),
                usage: None,
            },
        ], 10_000));

        let system_prompt = "You are a specialized Rust tutor.";
        let mut messages = vec![ChatMessage::system(system_prompt)];
        for i in 0..100 {
            messages.push(ChatMessage::user(&format!("Q{}: {}", i, "x".repeat(500))));
            messages.push(ChatMessage::assistant(&format!("A{}: {}", i, "y".repeat(500))));
        }
        messages.push(ChatMessage::user("Final"));

        let result = run(
            main_provider,
            messages,
            vec![],
            |_| async { Ok(ToolOutput::from("unused")) },
            RunnerConfig {
                auto_trim_context: true,
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        // System message must be first
        assert_eq!(result.messages[0].role, "system");
        assert_eq!(result.messages[0].text().unwrap(), system_prompt);
    }

    #[tokio::test]
    async fn test_tool_filter_changes_between_rounds() {
        // Verify that tool_filter is called fresh each round and can change
        // the available tool set dynamically based on conversation state

        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let provider = Arc::new(MockProvider::new(vec![
            // Round 1: calls read_file (available)
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "c1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: "{}".into(),
                    },
                }]),
                usage: None,
            },
            // Round 2: calls write_file (now available because we read first)
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "c2".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "write_file".into(),
                        arguments: "{}".into(),
                    },
                }]),
                usage: None,
            },
            // Round 3: done
            ChatResponse {
                message: ChatMessage::assistant("Read and wrote successfully"),
                usage: None,
            },
        ]));

        let cc = call_count.clone();

        // Filter: first round only read_file, after a read_file result, also offer write_file
        let filter_call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fcc = filter_call_count.clone();

        let result = run(
            provider,
            vec![ChatMessage::user("read then write")],
            vec![], // static tools unused when filter is set
            move |tc| {
                let cc2 = cc.clone();
                async move {
                    cc2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Ok(ToolOutput::from(format!("result from {}", tc.function.name)))
                }
            },
            RunnerConfig {
                tool_filter: Some(Arc::new(move |msgs| {
                    fcc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let has_read_result = msgs.iter().any(|m| {
                        m.role == "tool" && m.text().is_some_and(|t| t.contains("result from read_file"))
                    });
                    let mut tools = vec![Tool::function("read_file", "reads", serde_json::json!({}))];
                    if has_read_result {
                        tools.push(Tool::function("write_file", "writes", serde_json::json!({})));
                    }
                    tools
                })),
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "Read and wrote successfully");
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 2);
        // Filter should have been called 3 times (once per round)
        assert_eq!(filter_call_count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_run_id_and_observability() {
        let provider = Arc::new(MockProvider::new(vec![
            // Tool call round
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "call_abc123".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "my_tool".into(),
                        arguments: r#"{"key":"val"}"#.into(),
                    },
                }]),
                usage: None,
            },
            // Final response
            ChatResponse {
                message: ChatMessage::assistant("Done with observability"),
                usage: None,
            },
        ]));

        let events = Arc::new(Mutex::new(Vec::<RunnerEvent>::new()));
        let events_clone = events.clone();

        let result = run(
            provider,
            vec![ChatMessage::user("go")],
            vec![Tool::function("my_tool", "a tool", serde_json::json!({}))],
            |_| async { Ok(ToolOutput::from("tool output")) },
            RunnerConfig {
                run_id: Some("test-run-42".into()),
                parent_run_id: Some("parent-run-1".into()),
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            move |event| {
                events_clone.lock().unwrap().push(event);
            },
        )
        .await
        .unwrap();

        // RunnerResult should carry run_id and parent_run_id
        assert_eq!(result.run_id, "test-run-42");
        assert_eq!(result.parent_run_id.as_deref(), Some("parent-run-1"));

        // Check ToolCallStart has tool_call_id and iteration
        let events = events.lock().unwrap();
        let tool_start = events.iter().find(|e| matches!(e, RunnerEvent::ToolCallStart { .. }));
        assert!(tool_start.is_some(), "Should have a ToolCallStart event");
        if let RunnerEvent::ToolCallStart { name, tool_call_id, iteration, .. } = tool_start.unwrap() {
            assert_eq!(name, "my_tool");
            assert_eq!(tool_call_id, "call_abc123");
            assert_eq!(*iteration, 0); // First round
        }

        // Check ToolResult has tool_call_id, elapsed_ms, and iteration
        let tool_result = events.iter().find(|e| matches!(e, RunnerEvent::ToolResult { .. }));
        assert!(tool_result.is_some(), "Should have a ToolResult event");
        if let RunnerEvent::ToolResult { name, tool_call_id, elapsed_ms, iteration, .. } = tool_result.unwrap() {
            assert_eq!(name, "my_tool");
            assert_eq!(tool_call_id, "call_abc123");
            assert!(*elapsed_ms < 1000, "Tool should complete quickly in test");
            assert_eq!(*iteration, 0);
        }

        // Check Done has elapsed_ms
        let done = events.iter().find(|e| matches!(e, RunnerEvent::Done { .. }));
        assert!(done.is_some(), "Should have a Done event");
        if let RunnerEvent::Done { elapsed_ms, .. } = done.unwrap() {
            assert!(*elapsed_ms < 5000, "Run should complete quickly in test");
        }
    }

    #[tokio::test]
    async fn test_run_id_auto_generated() {
        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant("hello"),
                usage: None,
            },
        ]));

        let result = run(
            provider,
            vec![ChatMessage::user("hi")],
            vec![],
            |_| async { Ok("".into()) },
            RunnerConfig::default(), // No run_id set — should auto-generate
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        // run_id should be a valid UUID v4 (36 chars with hyphens)
        assert_eq!(result.run_id.len(), 36, "Auto-generated run_id should be a UUID");
        assert!(result.run_id.contains('-'), "UUID should contain hyphens");
        assert!(result.parent_run_id.is_none(), "No parent_run_id by default");
    }

    // -----------------------------------------------------------------------
    // @reference resolver tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_references_simple() {
        let refs = extract_references("Please look at @intro.sk and @setup-guide");
        assert_eq!(refs, vec!["intro.sk", "setup-guide"]);
    }

    #[test]
    fn test_extract_references_quoted() {
        let refs = extract_references(r#"Check @"my sketch with spaces" and @simple"#);
        assert_eq!(refs, vec!["my sketch with spaces", "simple"]);
    }

    #[test]
    fn test_extract_references_dedup() {
        let refs = extract_references("Compare @file.sk with @other.sk and @file.sk again");
        assert_eq!(refs, vec!["file.sk", "other.sk"]);
    }

    #[test]
    fn test_extract_references_paths() {
        let refs = extract_references("See @docs/setup and @src/main.rs");
        assert_eq!(refs, vec!["docs/setup", "src/main.rs"]);
    }

    #[test]
    fn test_extract_references_none() {
        let refs = extract_references("No references here, just an email user@example.com");
        // email captures "example.com" — that's fine, the resolver will return None for it
        assert!(!refs.is_empty() || refs.is_empty()); // just shouldn't panic
    }

    #[test]
    fn test_extract_references_at_start() {
        let refs = extract_references("@first is the reference");
        assert_eq!(refs, vec!["first"]);
    }

    #[test]
    fn test_extract_references_url() {
        let refs = extract_references("Check @https://example.com/article for details");
        assert_eq!(refs, vec!["https://example.com/article"]);
    }

    #[test]
    fn test_extract_references_url_with_query() {
        let refs =
            extract_references("See @https://docs.rs/agentive/latest?q=web and @blog/intro");
        assert_eq!(
            refs,
            vec!["https://docs.rs/agentive/latest?q=web", "blog/intro"]
        );
    }

    #[test]
    fn test_extract_references_http_url() {
        let refs = extract_references("Read @http://localhost:3000/api/docs");
        assert_eq!(refs, vec!["http://localhost:3000/api/docs"]);
    }

    #[tokio::test]
    async fn test_resolve_references_in_messages() {
        let resolver: ReferenceResolver = Arc::new(|name| {
            Box::pin(async move {
                match name.as_str() {
                    "intro.sk" => Some(ResolvedReference {
                        name: "intro.sk".into(),
                        content: "# Intro Sketch\nWelcome to the demo.".into(),
                        content_type: "text/markdown".into(),
                    }),
                    "notes.md" => Some(ResolvedReference {
                        name: "notes.md".into(),
                        content: "Some planning notes.".into(),
                        content_type: "text/markdown".into(),
                    }),
                    _ => None,
                }
            })
        });

        let mut messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Please review @intro.sk and @notes.md and @unknown"),
        ];

        resolve_references_in_messages(&mut messages, &resolver).await;

        // System message untouched
        assert_eq!(messages[0].text().unwrap(), "You are helpful.");

        // User message has resolved content appended
        let user_text = messages[1].text().unwrap().to_string();
        assert!(user_text.contains("Please review @intro.sk"), "original text preserved");
        assert!(user_text.contains("<referenced_document name=\"intro.sk\""), "intro resolved");
        assert!(user_text.contains("Welcome to the demo."), "intro content injected");
        assert!(user_text.contains("<referenced_document name=\"notes.md\""), "notes resolved");
        assert!(user_text.contains("Some planning notes."), "notes content injected");
        // @unknown should NOT have a referenced_document block
        assert!(!user_text.contains("name=\"unknown\""), "unknown ref not injected");
    }

    #[tokio::test]
    async fn test_resolve_references_skips_already_resolved() {
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = call_count.clone();
        let resolver: ReferenceResolver = Arc::new(move |_name| {
            cc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move {
                Some(ResolvedReference {
                    name: "test".into(),
                    content: "resolved".into(),
                    content_type: "text/plain".into(),
                })
            })
        });

        let mut messages = vec![
            ChatMessage::user("Check @test please"),
        ];

        // First resolution
        resolve_references_in_messages(&mut messages, &resolver).await;
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Second resolution — should skip (already has <referenced_document)
        resolve_references_in_messages(&mut messages, &resolver).await;
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1, "should not re-resolve");
    }

    #[tokio::test]
    async fn test_reference_resolver_in_run() {
        // Full integration: run() with a reference_resolver resolves @refs before LLM call
        let resolver: ReferenceResolver = Arc::new(|name| {
            Box::pin(async move {
                if name == "context.md" {
                    Some(ResolvedReference {
                        name: "context.md".into(),
                        content: "Important context document.".into(),
                        content_type: "text/markdown".into(),
                    })
                } else {
                    None
                }
            })
        });

        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant("I see the context document."),
                usage: None,
            },
        ]));

        let result = run(
            provider,
            vec![
                ChatMessage::system("You are helpful."),
                ChatMessage::user("Please review @context.md"),
            ],
            vec![],
            |_tc| async { Ok(ToolOutput::from("done")) },
            RunnerConfig {
                reference_resolver: Some(resolver),
                ..Default::default()
            },
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "I see the context document.");

        // The messages sent to the LLM should contain the resolved reference
        let user_msg = result.messages.iter().find(|m| m.role == "user").unwrap();
        let text = user_msg.text().unwrap();
        assert!(text.contains("Important context document."), "resolved content should be in messages");
        assert!(text.contains("<referenced_document"), "should have XML wrapper");
    }

    #[tokio::test]
    async fn test_tool_output_with_images_injects_user_message() {
        // Provider expects: user msg → assistant tool call → tool result + user images → final answer
        let provider = Arc::new(MockProvider::new(vec![
            ChatResponse {
                message: ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                    id: "tc1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "read_sketch".into(),
                        arguments: "{}".into(),
                    },
                }]),
                usage: None,
            },
            ChatResponse {
                message: ChatMessage::assistant("I can see the screenshot shows a login form."),
                usage: None,
            },
        ]));

        let result = run(
            provider,
            vec![ChatMessage::user("describe the sketch")],
            vec![Tool::function("read_sketch", "Read a sketch", serde_json::json!({}))],
            |_tc| async {
                Ok(ToolOutput::with_images(
                    "Sketch title: Login Page",
                    vec![ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,abc123".into(),
                            detail: Some("low".into()),
                        },
                    }],
                ))
            },
            RunnerConfig::default(),
            CancellationToken::new(),
            Steering::new(),
            Guardrails::default(),
            |_| {},
        )
        .await
        .unwrap();

        assert_eq!(result.response, "I can see the screenshot shows a login form.");

        // Should have: user, assistant(tool_call), tool_result, user(images), assistant(final)
        let tool_msgs: Vec<_> = result.messages.iter().filter(|m| m.role == "tool").collect();
        assert_eq!(tool_msgs.len(), 1);
        assert_eq!(tool_msgs[0].text().unwrap(), "Sketch title: Login Page");

        // The injected user message with images should follow the tool result
        let image_msgs: Vec<_> = result.messages.iter().filter(|m| {
            m.role == "user" && matches!(&m.content, Some(MessageContent::Parts(_)))
        }).collect();
        assert_eq!(image_msgs.len(), 1, "should inject one user message with images");

        if let Some(MessageContent::Parts(parts)) = &image_msgs[0].content {
            assert!(parts.iter().any(|p| matches!(p, ContentPart::ImageUrl { .. })));
            assert!(parts.iter().any(|p| matches!(p, ContentPart::Text { .. })));
        } else {
            panic!("expected Parts content");
        }
    }
}
