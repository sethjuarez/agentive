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
├── factory.rs              # Provider factory — auto-routing from model name
├── steering.rs             # Steering — inject user messages mid-run
├── parse.rs                # Robust JSON argument parsing for LLM output
├── guardrails.rs           # Input/output/tool guardrails (validation hooks)
├── providers/
│   ├── mod.rs              # Module declarations
│   ├── openai.rs           # OpenAI-compatible provider (OpenAI, Azure, Microsoft Foundry)
│   ├── responses.rs        # OpenAI Responses API provider (/v1/responses)
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
- `ToolCallStart { name, arguments, tool_call_id, iteration }` — tool being invoked, with LLM-assigned call ID and loop round
- `ToolResult { name, result, tool_call_id, elapsed_ms, iteration }` — tool returned, with timing and correlation
- `MessagesUpdated { messages }` — full history after a tool round (for persistence)
- `Done { response, messages, elapsed_ms }` — final text + full history + total run time
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
pub async fn run<F, Fut, E>(
    provider: Arc<dyn Provider>,     // LLM provider
    messages: Vec<ChatMessage>,      // initial conversation (include system prompt)
    tools: Vec<Tool>,                // tool definitions for the LLM
    tool_executor: F,                // async closure: ToolCall -> Result<String, String>
    config: RunnerConfig,            // max_iterations, retry, trimming, sanitization, compaction
    cancel: CancellationToken,       // for user-initiated stop
    steering: Steering,              // inject user messages mid-loop
    guardrails: Guardrails,          // input/output/tool validation hooks
    on_event: E,                     // callback for RunnerEvent
) -> Result<RunnerResult, AgentError>
where
    F: Fn(ToolCall) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, String>> + Send,
```

### RunnerConfig defaults
- `max_iterations: 10` — max tool-call rounds before giving up
- `retry_on_400: true` — retry once on HTTP 400
- `auto_trim_context: true` — trim old messages when over budget
- `sanitize_tool_results: true` — strip control chars and base64
- `parallel_tool_calls: true` — execute multiple tool calls concurrently
- `response_format: None` — optional structured output (JSON mode or JSON schema)
- `compaction_provider: None` — optional LLM provider for richer context compaction
- `tool_filter: None` — optional per-round tool filter for dynamic tool gating
- `run_id: None` — auto-generates UUID v4 if not set; use for trace correlation
- `parent_run_id: None` — set when delegating to link child runs to parent

### RunnerResult
- `messages: Vec<ChatMessage>` — full conversation history
- `response: String` — final assistant text
- `new_messages: Vec<ChatMessage>` — only messages generated during this run
- `total_usage: Usage` — accumulated token usage across all LLM calls
- `run_id: String` — unique identifier for this run (auto-generated or from config)
- `parent_run_id: Option<String>` — parent run ID if this was a delegated sub-run

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

### ResponsesProvider
OpenAI Responses API (`/v1/responses`) — newer endpoint with different format:
```rust
ResponsesProvider::new(endpoint, api_key, model)
    .with_context_budget(128_000)  // optional
    .with_vision(true)             // optional
```

Key differences from Chat Completions:
- System messages → `role: "developer"` input items
- Tool calls → separate `function_call` input items (not nested in assistant)
- Tool results → `function_call_output` input items
- SSE events: `response.output_text.delta`, `response.output_item.added`,
  `response.function_call_arguments.delta`, `response.completed`
- Tool defs are flattened (no `function` wrapper)

## Provider factory (factory.rs)

Auto-routing from endpoint + model name to the right provider:

```rust
use agentive::factory::{build_provider, build_provider_with_auth};
use agentive::AuthStrategy;

// Simple: auto-detects provider type, auth, vision, context budget
let provider = build_provider("https://api.openai.com/v1", "sk-...", "gpt-4o");

// With explicit auth (e.g., Entra tokens for Azure)
let provider = build_provider_with_auth(
    "https://my-resource.openai.azure.com",
    AuthStrategy::Dynamic(Arc::new(|| get_fresh_token())),
    "gpt-5.1-codex",
);
```

Auto-detection rules:
- **Provider type**: `ResponsesProvider` for codex/pro models, `OpenAiProvider` for everything else
- **Auth**: Azure endpoints (`azure.com`) use `api-key` header, others use `Bearer`
- **Vision**: Enabled for gpt-4o, gpt-4.1, gpt-5, claude-3.5/4, o-series, gemini
- **Context budget**: Model family heuristics (128k tokens for gpt-4o/5, 200k for Claude, etc.)

Helper functions:
- `needs_responses_api(model)` — check if model needs Responses API
- `supports_vision(model)` — check if model supports image inputs
- `default_context_budget(model)` — estimate context budget in characters

## Context trimming (context.rs)

When conversations exceed the provider's `context_budget_chars()`:
1. System messages are preserved at the front
2. Oldest non-system messages are dropped
3. Dropped messages are summarized into a compact user message
4. Summary is inserted after system messages, before recent conversation

The default summary is string-based (no LLM call) — extracts user requests,
assistant decisions, and tool call names.

### LLM-powered context compaction

Set `RunnerConfig::compaction_provider` to upgrade the string summary with an
LLM-generated one. When a compaction provider is configured and context is
trimmed, the runner:

1. First generates the fast string-based summary (as a fallback)
2. Sends the summary to the compaction provider with a prompt to compress it
3. Replaces the string summary with the LLM-generated one
4. Falls back to the string summary if the LLM call fails

```rust
let config = RunnerConfig {
    compaction_provider: Some(Arc::new(OpenAiProvider::new(
        "https://api.openai.com/v1",
        "sk-...",
        "gpt-4o-mini", // cheap model for summarization
    ))),
    ..Default::default()
};
```

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
let result = run(provider, messages, tools, executor, config, cancel, steering, Guardrails::default(), |_| {}).await?;
```

- `Steering::new()` — creates an empty queue
- `steering.clone()` — shares the queue (uses `Arc<Mutex<Vec<String>>>`)
- `steering.send(msg)` — enqueue a user message (pub)
- Messages are drained and appended as `ChatMessage::user()` at the top of each
  iteration, before the LLM call. If no messages are queued, nothing happens.

## Tool execution pattern

Tools are app-specific. The runner accepts an async closure that takes an
owned `ToolCall` (not a reference, since async closures need owned data):
```rust
|call: ToolCall| async move {
    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
        .map_err(|e| e.to_string())?;
    match call.function.name.as_str() {
        "read_file" => {
            let path = args["path"].as_str().ok_or("missing path")?;
            tokio::fs::read_to_string(path).await.map_err(|e| e.to_string())
        }
        _ => Err(format!("Unknown tool: {}", call.function.name))
    }
}
```

## Dynamic tool gating

Change which tools are available to the LLM each round:

```rust
use agentive::{RunnerConfig, Tool, ChatMessage};
use std::sync::Arc;

let config = RunnerConfig {
    tool_filter: Some(Arc::new(|messages: &[ChatMessage]| {
        let has_plan = messages.iter().any(|m| m.text().map_or(false, |t| t.contains("PLAN:")));
        if has_plan {
            // After planning, enable write tools
            vec![
                Tool::function("read_file", "Read a file", serde_json::json!({})),
                Tool::function("write_file", "Write a file", serde_json::json!({})),
            ]
        } else {
            // Before planning, only read tools
            vec![Tool::function("read_file", "Read a file", serde_json::json!({}))]
        }
    })),
    ..Default::default()
};
```

When `tool_filter` is `None` (the default), the static `tools` vec passed to `run()` is used every round.

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
`catch_unwind` on the async future (using `FutureExt::catch_unwind()`) and
returns `AgentError::ToolPanic { name, message }` instead of crashing the
whole runtime. No special handling needed from the consuming app; the error
propagates normally from `run()`.

## Parallel tool execution

When the LLM returns multiple tool calls in a single response and
`RunnerConfig::parallel_tool_calls` is `true` (the default), tool calls
execute concurrently using `futures_util::future::join_all`. Set to `false`
for sequential execution if your tools share mutable state.

## Structured output (types.rs)

Force the LLM to return structured JSON instead of free-form text:

```rust
use agentive::{RunnerConfig, ResponseFormat, JsonSchemaSpec};

// Simple JSON mode (no specific schema)
let config = RunnerConfig {
    response_format: Some(ResponseFormat::JsonObject),
    ..Default::default()
};

// Strict JSON schema
let config = RunnerConfig {
    response_format: Some(ResponseFormat::JsonSchema {
        json_schema: JsonSchemaSpec {
            name: "extraction".into(),
            strict: true,
            schema: serde_json::json!({
                "type": "object",
                "properties": { "answer": { "type": "string" } },
                "required": ["answer"]
            }),
        },
    }),
    ..Default::default()
};
```

OpenAI/Microsoft Foundry: sent natively as `response_format` in the request body.
Anthropic: pass-through (Anthropic doesn't have native `response_format` yet — use
system prompt instructions or tool-use patterns for structured output).

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
cargo test           # run all 107 tests (unit + doc + integration stubs)
cargo doc --no-deps  # generate documentation
```

### Integration tests (real providers)

Integration tests live in `lib/tests/integration.rs` and are **skipped by default**.
They run end-to-end against real LLM APIs when environment variables are set:

```bash
cp .env.example .env
# Edit .env with real API keys
cargo test --manifest-path lib/Cargo.toml --test integration
```

Supported providers: OpenAI, Azure OpenAI, Anthropic.

## Consuming from another project

In the consuming project's Cargo.toml:
```toml
# Via git
agentive = { git = "https://github.com/sethjuarez/agentive", path = "lib" }

# For local development, add to .cargo/config.toml:
# [patch."https://github.com/sethjuarez/agentive"]
# agentive = { path = "../../agentive/lib" }
```

## Azure Discovery & Endpoint Patterns

### Model discovery (discovery.rs)

`list_models()` auto-detects the endpoint type and routes accordingly:

| Endpoint pattern | Detection | Listing URL |
|---|---|---|
| `*.services.ai.azure.com/api/projects/*` | Foundry project | `{project}/deployments?api-version=v1` (then fallback to catalog) |
| `*.services.ai.azure.com` | Foundry resource | `{base}/openai/models?api-version=2024-10-21` |
| `*.openai.azure.com` | Azure OpenAI | `{base}/openai/deployments?api-version=2024-10-21` |
| Everything else | OpenAI-compatible | `{base}/models` |

### Chat URL construction (providers/openai.rs)

`OpenAiProvider::chat_url()` handles three cases:

| Endpoint type | Chat URL built |
|---|---|
| Already contains `/chat/completions` | Used as-is |
| Foundry project (`/api/projects/`) | **Strips** `/api/projects/...` → `{resource_base}/openai/deployments/{model}/chat/completions?api-version=2024-10-21` |
| Everything else | `{endpoint}/chat/completions` |

**Critical**: For Foundry projects, the `/api/projects/{name}` path is stripped
to use the resource-level endpoint. The project-scoped URL does NOT support the
OpenAI chat completions path. The model/deployment name goes into the URL
(deployment-based routing), NOT model routing via the request body. This matches
CutReady's proven production pattern.

### ARM resource discovery (arm_discovery.rs)

Discovers Azure AI resources and their Foundry projects via Azure Resource Manager:

```rust
// List AI resources in a subscription
let resources = list_ai_resources(&token, subscription_id).await?;
// Returns Vec<AiResource> with name, kind, endpoint, resource_group, foundry_url

// List Foundry projects for a resource
let projects = list_foundry_projects(&token, subscription_id, &resource).await?;
// Returns Vec<FoundryProject> with name, display_name, endpoint
```

**AiResource** fields:
- `name` — resource name
- `kind` — `"AIServices"`, `"OpenAI"`, `"CognitiveServices"`, etc.
- `endpoint` — from ARM `properties.endpoint` (e.g., `*.cognitiveservices.azure.com`)
- `resource_group` — extracted from ARM resource ID
- `foundry_url` — derived for AIServices: `https://{name}.services.ai.azure.com`

**Dual-strategy project discovery** (`list_foundry_projects`):
1. **Strategy 1** — CognitiveServices subresource API (`2025-04-01-preview`):
   `GET .../Microsoft.CognitiveServices/accounts/{name}/projects`
   Returns projects directly scoped to the resource. ARM returns names as
   `parent/child` — the code strips to just `child`.
2. **Strategy 2** (fallback, only if Strategy 1 returns empty) — ML workspaces:
   `GET .../Microsoft.MachineLearningServices/workspaces?$filter=kind eq 'Project'`
   Subscription-wide query for classic hub-based projects.

Project endpoint format: `https://{resource}.services.ai.azure.com/api/projects/{project_name}`

### Azure OAuth (azure_oauth.rs)

Browser-based OAuth flow for desktop apps:

```rust
// Start OAuth flow (opens local HTTP server, returns auth URL)
let (auth_url, port) = start_auth_code_flow(tenant_id, scope, client_id).await?;
// Open auth_url in browser, user signs in
// Wait for callback
let tokens = wait_for_auth_code(port, tenant_id, scope, client_id).await?;

// Refresh token later
let new_tokens = refresh_access_token(tenant_id, refresh_token, scope, client_id).await?;
```

**Scope constants:**
- `AZURE_MANAGEMENT_SCOPE` — for ARM API calls (resource/project discovery)
- `AZURE_AI_SCOPE` — for AI inference calls (chat completions)

### Auth strategy (auth.rs)

```rust
AuthStrategy::ApiKey(key)        // api-key header (Azure) or Bearer (OpenAI)
AuthStrategy::Bearer(token)      // explicit Bearer token
AuthStrategy::Dynamic(Arc<fn>)   // closure for Entra token refresh
```

`OpenAiProvider::new()` auto-detects: Azure endpoints (`azure.com`) get `ApiKey`,
others get `Bearer`. Use `with_auth()` for explicit control (e.g., Entra tokens).

## Design decisions

1. **mpsc channel for streaming** — provider writes events to a channel, runner
   reads concurrently. Avoids deadlocks from buffered streams.
2. **Trait-based providers** — extensible; apps can implement custom providers.
3. **Async closure-based tool execution** — no trait to implement, just an async function.
4. **Events-based persistence injection** — no ChatStore trait; apps persist
   via event callbacks however they want (or don't).
5. **Multimodal from the start** — MessageContent supports text + image parts.
6. **Context trimming built-in** — automatic, with fast string summarization.
7. **Sanitization built-in** — strips control chars and base64 from tool results.
8. **Foundry URL stripping** — project path is stripped for chat because the
   resource-level endpoint handles deployment routing. Project path is kept
   for model listing where it provides project-scoped results.
