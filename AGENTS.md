# Agentive — Agent Documentation

> This file is designed for AI coding agents to quickly understand how to use
> and contribute to the agentive crate.

## Purpose

Agentive is a shared Rust crate that provides the core infrastructure for
building agentic LLM applications. It handles streaming SSE parsing, multi-turn
tool calling loops, context window management, and provider abstraction so that
consuming apps only need to define their tools and wire up their UI.

## Crate structure

```
lib/src/
├── lib.rs                  # Re-exports all public API
├── types.rs                # Core types: ChatMessage, ToolCall, Tool, ChatEvent, etc.
├── error.rs                # AgentError enum
├── cancel.rs               # CancellationToken (Arc<AtomicBool> wrapper)
├── provider.rs             # Provider trait definition
├── steering.rs             # Steering — inject user messages mid-run
├── parse.rs                # Robust JSON argument parsing for LLM output
├── guardrails.rs           # Input/output/tool guardrails (validation hooks)
├── providers/
│   ├── mod.rs              # Module declarations
│   ├── openai.rs           # OpenAI-compatible provider (OpenAI, Azure, Microsoft Foundry)
│   ├── anthropic.rs        # Anthropic Messages API provider
│   └── sse.rs              # Shared SSE line parser
├── runner.rs               # Agentic loop: run() function
├── context.rs              # Context window trimming + summarization
└── sanitize.rs             # Tool result sanitization
```

## Key types

### ChatMessage
Messages support multimodal content via `MessageContent`:
```rust
// Simple text
ChatMessage::user("Hello")
ChatMessage::assistant("Hi there")
ChatMessage::system("You are helpful")
ChatMessage::tool_result("call_id", "result text")

// With tool calls
ChatMessage::assistant_with_tool_calls(vec![tool_call])

// With images (multimodal)
ChatMessage::user_with_images("What's this?", vec![
    ContentPart::ImageUrl { image_url: ImageUrl { url: "...".into(), detail: None } }
])

// Access text content
msg.text() // -> Option<&str>
```

The `content` field is `Option<MessageContent>` where `MessageContent` is either
`Text(String)` or `Parts(Vec<ContentPart>)`. The `text()` method extracts text
from either variant.

### Tool / ToolCall
```rust
// Define a tool for the LLM
Tool::function("tool_name", "description", serde_json::json!({
    "type": "object",
    "properties": { "param": { "type": "string" } },
    "required": ["param"]
}))

// ToolCall is what the LLM returns when it wants to use a tool
// Fields: id, call_type, function (FunctionCall { name, arguments })
// arguments is a JSON string that needs to be parsed by the executor
```

### ChatEvent (provider → runner)
Events emitted by providers during streaming:
- `Token { token }` — text delta
- `Thinking { token }` — reasoning/chain-of-thought delta (o-series, Claude thinking)
- `ToolCallStart { tool_call }` — tool call detected (name known)
- `Done { response: ChatResponse }` — streaming complete
- `Error { message }` — stream error

### RunnerEvent (runner → app)
Events emitted by the runner to the consuming app:
- `Token { token }` — forwarded from provider
- `Thinking { token }` — forwarded from provider
- `Status { message }` — "Thinking…", "Running 3 tool calls…", "Compacting context…"
- `ToolCallStart { name, arguments }` — tool being invoked
- `ToolResult { name, result }` — tool returned a result
- `MessagesUpdated { messages }` — full history after a tool round (for persistence)
- `Done { response, messages }` — final text + full history
- `Error { message }` — error description

### AgentError
```rust
AgentError::NotConfigured(String)    // missing API key/endpoint
AgentError::Http(reqwest::Error)     // transport error
AgentError::Stream(String)           // SSE processing error
AgentError::Json(serde_json::Error)  // serialization error
AgentError::Tool(String)             // tool execution error
AgentError::Storage(String)          // persistence error
AgentError::Cancelled                // user cancelled
AgentError::Api { status, message }  // non-2xx API response
AgentError::MaxIterations(usize)     // exceeded tool loop limit
```

## Core function: `agentive::run()`

This is the main entry point. It runs the agentic loop:

```rust
pub async fn run<F, E>(
    provider: Arc<dyn Provider>,     // LLM provider
    messages: Vec<ChatMessage>,      // initial conversation (include system prompt)
    tools: Vec<Tool>,                // tool definitions for the LLM
    tool_executor: F,                // closure: &ToolCall -> Result<String, String>
    config: RunnerConfig,            // max_iterations, retry, trimming, sanitization
    cancel: CancellationToken,       // for user-initiated stop
    steering: Steering,              // inject user messages mid-loop
    guardrails: Guardrails,          // input/output/tool validation hooks
    on_event: E,                     // callback for RunnerEvent
) -> Result<RunnerResult, AgentError>
```

### RunnerConfig defaults
- `max_iterations: 10` — max tool-call rounds before giving up
- `retry_on_400: true` — retry once on HTTP 400
- `auto_trim_context: true` — trim old messages when over budget
- `sanitize_tool_results: true` — strip control chars and base64
- `parallel_tool_calls: true` — execute multiple tool calls concurrently

### RunnerResult
- `messages: Vec<ChatMessage>` — full conversation history
- `response: String` — final assistant text
- `new_messages: Vec<ChatMessage>` — only messages generated during this run
- `total_usage: Usage` — accumulated token usage across all LLM calls

## Provider trait

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    async fn chat(
        &self,
        request: ChatRequest,
        tx: mpsc::Sender<ChatEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), AgentError>;

    fn name(&self) -> &str;
    fn context_budget_chars(&self) -> usize { 200_000 }
    fn supports_vision(&self) -> bool { false }
}
```

The provider runs in a `tokio::spawn` task. It writes `ChatEvent`s to the
mpsc channel. The runner reads from the channel concurrently. This avoids
the deadlock that would occur if the provider blocked while nobody reads events.

## Built-in providers

### OpenAiProvider
Works with any OpenAI-compatible endpoint:
```rust
// Auto-detects Azure by endpoint URL (uses api-key header vs Bearer)
OpenAiProvider::new(endpoint, api_key, model)
    .with_context_budget(128_000)  // optional
    .with_vision(true)             // optional
```

### AnthropicProvider
Anthropic Messages API with content block streaming:
```rust
AnthropicProvider::new(api_key, model)
    .with_context_budget(200_000)  // optional
```

Handles Anthropic-specific concerns:
- System prompt extracted to top-level `system` field
- Tool calls sent as `tool_use` content blocks
- Tool results sent as `tool_result` content blocks in user messages
- `thinking_delta` events for extended thinking models

## Context trimming (context.rs)

When conversations exceed the provider's `context_budget_chars()`:
1. System messages are preserved at the front
2. Oldest non-system messages are dropped
3. Dropped messages are summarized into a compact user message
4. Summary is inserted after system messages, before recent conversation

The summary is string-based (no LLM call) — extracts user requests, assistant
decisions, and tool call names. Apps can upgrade to LLM-powered summarization
by handling `RunnerEvent::MessagesUpdated` and replacing the summary.

## Steering (steering.rs)

Steering allows users to inject additional messages while the agent is in its
tool-call loop. This is useful for "redirect" scenarios — the user sees the agent
working and wants to nudge it in a different direction before it finishes.

```rust
let steering = Steering::new();
let handle = steering.clone(); // Arc-based — cheap clone

// From UI thread (can be called any time, including during run):
handle.send("Also check the error handling path");

// Pass into run() — runner drains queued messages before each LLM call:
let result = run(provider, messages, tools, executor, config, cancel, steering, |_| {}).await?;
```

- `Steering::new()` — creates an empty queue
- `steering.clone()` — shares the queue (uses `Arc<Mutex<Vec<String>>>`)
- `steering.send(msg)` — enqueue a user message (pub)
- Messages are drained and appended as `ChatMessage::user()` at the top of each
  iteration, before the LLM call. If no messages are queued, nothing happens.

## Tool execution pattern

Tools are app-specific. The runner accepts a closure:
```rust
|call: &ToolCall| -> Result<String, String> {
    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
        .map_err(|e| e.to_string())?;
    match call.function.name.as_str() {
        "read_file" => {
            let path = args["path"].as_str().ok_or("missing path")?;
            std::fs::read_to_string(path).map_err(|e| e.to_string())
        }
        _ => Err(format!("Unknown tool: {}", call.function.name))
    }
}
```

## Guardrails (guardrails.rs)

Guardrails are optional validation hooks at three points in the runner loop:

```rust
use agentive::{Guardrails, GuardrailResult};

let guardrails = Guardrails::new()
    .with_input_guardrail(|messages| {
        // Check messages before each LLM call
        GuardrailResult::Allow
    })
    .with_output_guardrail(|assistant_msg| {
        // Check LLM response before processing
        if assistant_msg.text().unwrap_or("").contains("SECRET") {
            GuardrailResult::Deny("Output contains secrets".into())
        } else {
            GuardrailResult::Allow
        }
    })
    .with_tool_guardrail(|tool_call| {
        // Check before each tool execution
        if tool_call.function.name == "dangerous_tool" {
            GuardrailResult::Deny("Tool not permitted".into())
        } else {
            GuardrailResult::Allow
        }
    });
```

- `GuardrailResult::Allow` — proceed normally
- `GuardrailResult::Deny(reason)` — for input/output: abort with `AgentError::Guardrailed`;
  for tools: returns the denial message as the tool result (loop continues)
- Pass `Guardrails::default()` for no guardrails

## Robust argument parsing (parse.rs)

LLMs sometimes produce malformed JSON in tool call arguments. Use
`parse_tool_args()` instead of raw `serde_json::from_str()`:

```rust
use agentive::parse_tool_args;

|call: &ToolCall| -> Result<String, String> {
    let args = parse_tool_args(&call.function.arguments)
        .map_err(|e| e.to_string())?;
    // ... use args
}
```

Strategies tried in order:
1. Direct `serde_json::from_str`
2. Strip markdown code fences (`\`\`\`json ... \`\`\``)
3. Extract first `{...}` block with brace matching
4. Strip trailing commas before `}` or `]`

## Tool panic safety

Tool closures are user code — if one panics, agentive catches it via
`catch_unwind` and returns `AgentError::ToolPanic { name, message }`
instead of crashing the whole runtime. No special handling needed from the
consuming app; the error propagates normally from `run()`.

## Parallel tool execution

When the LLM returns multiple tool calls in a single response and
`RunnerConfig::parallel_tool_calls` is `true` (the default), tool calls
execute concurrently using scoped threads. Set to `false` for sequential
execution if your tools share mutable state.

## Usage tracking

`RunnerResult::total_usage` accumulates `prompt_tokens`,
`completion_tokens`, and `total_tokens` across all LLM calls in a run.
Each individual call also emits `RunnerEvent::Usage { usage }`.

## Persistence pattern

Agentive does NOT own persistence. Apps handle it via events:
```rust
|event| match event {
    RunnerEvent::MessagesUpdated { messages } => {
        // Save to DB mid-run (after each tool round)
        db.save_messages(&conversation_id, &messages);
    }
    RunnerEvent::Done { messages, .. } => {
        // Save final state
        db.save_messages(&conversation_id, &messages);
    }
    _ => {}
}
```

## Building and testing

```bash
cd lib
cargo build          # build the crate
cargo test           # run all 63 tests
cargo doc --no-deps  # generate documentation
```

## Consuming from another project

In the consuming project's Cargo.toml:
```toml
# Via git
agentive = { git = "https://github.com/sethjuarez/agentive", path = "lib" }

# For local development, add to .cargo/config.toml:
# [patch."https://github.com/sethjuarez/agentive"]
# agentive = { path = "../../agentive/lib" }
```

## Design decisions

1. **mpsc channel for streaming** — provider writes events to a channel, runner
   reads concurrently. Avoids deadlocks from buffered streams.
2. **Trait-based providers** — extensible; apps can implement custom providers.
3. **Closure-based tool execution** — no trait to implement, just a function.
4. **Events-based persistence injection** — no ChatStore trait; apps persist
   via event callbacks however they want (or don't).
5. **Multimodal from the start** — MessageContent supports text + image parts.
6. **Context trimming built-in** — automatic, with fast string summarization.
7. **Sanitization built-in** — strips control chars and base64 from tool results.
