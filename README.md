# agentive

A Rust crate for building agentic LLM applications with streaming, tool calling, and multi-turn conversation loops.

## What it does

**agentive** handles the hard parts of building LLM-powered tools:

- **Streaming SSE parsing** — OpenAI and Anthropic protocols, with correct tool call accumulation across chunks
- **Agentic tool loop** — stream → detect tool calls → execute → feed results back → repeat
- **Context window management** — automatic trimming and summarization when conversations get too long
- **Cancellation** — cooperative cancellation token for user-initiated stop
- **Provider abstraction** — trait-based, ships with OpenAI-compatible and Anthropic providers
- **Multimodal messages** — text and image content parts

You bring: your tools, your system prompt, your UI. Agentive handles the LLM plumbing.

## Quick start

```rust
use agentive::{OpenAiProvider, RunnerConfig, CancellationToken, ChatMessage, Tool};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), agentive::AgentError> {
    let provider = Arc::new(OpenAiProvider::new(
        "https://api.openai.com/v1",
        "sk-...",
        "gpt-4o",
    ));

    let tools = vec![
        Tool::function("read_file", "Read a file from disk", serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        })),
    ];

    let result = agentive::run(
        provider,
        vec![ChatMessage::user("What's in README.md?")],
        tools,
        |call| match call.function.name.as_str() {
            "read_file" => Ok(std::fs::read_to_string("README.md").unwrap_or_default()),
            _ => Err(format!("Unknown tool: {}", call.function.name)),
        },
        RunnerConfig::default(),
        CancellationToken::new(),
        |event| {
            if let agentive::RunnerEvent::Token { token } = event {
                print!("{}", token);
            }
        },
    ).await?;

    println!("\n\nFinal: {}", result.response);
    Ok(())
}
```

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────────┐
│  Your App   │────▶│  Runner  │────▶│   Provider   │
│             │     │          │     │ (OpenAI/etc)  │
│ • tools     │◀────│ • stream │◀────│ • SSE parse  │
│ • UI events │     │ • tools  │     │ • tool accum  │
│ • storage   │     │ • trim   │     │ • auth        │
└─────────────┘     └──────────┘     └──────────────┘
```

**Your app** defines tools and handles events. **The runner** orchestrates the stream→tool→loop cycle. **Providers** handle HTTP, SSE parsing, and API-specific formats.

## Modules

| Module | Description |
|--------|-------------|
| `types` | `ChatMessage`, `ToolCall`, `Tool`, `ChatRequest/Response`, `MessageContent` (multimodal) |
| `provider` | `Provider` trait — implement this for custom LLM backends |
| `providers::openai` | OpenAI, Azure OpenAI, Microsoft Foundry — any OpenAI-compatible endpoint |
| `providers::responses` | OpenAI Responses API (`/v1/responses`) — newer endpoint format |
| `providers::anthropic` | Anthropic Messages API with content block streaming |
| `providers::sse` | Shared SSE line parser |
| `runner` | The agentic loop — `run()` function with `RunnerConfig` and `RunnerEvent` |
| `context` | Context window trimming and conversation summarization |
| `steering` | [`Steering`] — inject user messages into a running agent loop |
| `parse` | [`parse_tool_args`] — robust JSON parsing for LLM-generated tool arguments |
| `guardrails` | [`Guardrails`] — input/output/tool validation hooks with `Allow`/`Deny` |
| `cancel` | `CancellationToken` for cooperative cancellation |
| `error` | `AgentError` — unified error type |

## Providers

### OpenAI-compatible

```rust
// OpenAI
let p = OpenAiProvider::new("https://api.openai.com/v1", "sk-...", "gpt-4o");

// Azure OpenAI (auto-detected by endpoint)
let p = OpenAiProvider::new("https://my-resource.openai.azure.com/...", "key", "gpt-4o");

// Microsoft Foundry
let p = OpenAiProvider::new("https://my-project.services.ai.azure.com/...", "key", "gpt-4o");

// With options
let p = OpenAiProvider::new("https://api.openai.com/v1", "sk-...", "gpt-4o")
    .with_context_budget(128_000)
    .with_vision(true);
```

### Anthropic

```rust
let p = AnthropicProvider::new("sk-ant-...", "claude-sonnet-4-20250514")
    .with_context_budget(200_000);
```

### Responses API

```rust
// OpenAI Responses API (newer endpoint)
let p = ResponsesProvider::new("https://api.openai.com", "sk-...", "gpt-4o");

// Azure / Microsoft Foundry
let p = ResponsesProvider::new("https://my-resource.openai.azure.com", "key", "gpt-4o");
```

## Steering

Steering lets users inject messages while the agent is mid-loop (e.g., "actually focus on the error case"). The runner drains queued messages before each LLM call:

```rust
let steering = Steering::new();
let handle = steering.clone(); // give this to your UI thread

// UI thread can call at any time:
handle.send("Please also consider edge cases");

// Pass steering into the runner
let result = run(provider, messages, tools, executor, config, cancel, steering, |_| {}).await?;
```

## Runner events

The runner emits events via a callback — handle what you need, ignore the rest:

```rust
|event| match event {
    RunnerEvent::Token { token } => { /* stream to UI */ },
    RunnerEvent::Thinking { token } => { /* show reasoning */ },
    RunnerEvent::ToolCallStart { name, .. } => { /* show tool activity */ },
    RunnerEvent::ToolResult { name, result } => { /* show result */ },
    RunnerEvent::Status { message } => { /* "Thinking…", "Running 3 tools…" */ },
    RunnerEvent::MessagesUpdated { messages } => { /* persist mid-run */ },
    RunnerEvent::Done { response, messages } => { /* final result */ },
    RunnerEvent::Error { message } => { /* handle error */ },
}
```

## License

MIT

