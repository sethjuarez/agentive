# Agentive — Agent Documentation

> This file is designed for AI coding agents to quickly understand how to use
> and contribute to the agentive crate.

## Purpose

Agentive is a shared Rust crate that provides the core infrastructure for
building agentic LLM applications. It handles streaming SSE parsing, multi-turn
tool calling loops, context window management, and provider abstraction so that
consuming apps only need to define their tools and wire up their UI.

## Crate structure

```text
lib/src/
├── lib.rs                  # Re-exports all public API
├── types.rs                # Core types: ChatMessage, ToolCall, Tool, ToolOutput, ChatEvent, Usage, etc.
├── error.rs                # AgentError enum
├── cancel.rs               # CancellationToken (Arc<AtomicBool> wrapper)
├── auth.rs                 # AuthStrategy enum (ApiKey, Bearer, Dynamic)
├── chat.rs                 # simple_chat() — one-shot non-streaming helper
├── provider.rs             # Provider trait definition
├── factory.rs              # Provider factory — auto-routing, model heuristics, context budgets
├── steering.rs             # Steering — inject user messages mid-run
├── parse.rs                # Robust JSON argument parsing for LLM output
├── guardrails.rs           # Input/output/tool guardrails (validation hooks)
├── providers/
│   ├── mod.rs              # Module declarations
│   ├── openai.rs           # OpenAI-compatible provider (OpenAI, Azure, Microsoft Foundry)
│   ├── responses.rs        # OpenAI Responses API provider (/v1/responses)
│   ├── anthropic.rs        # Anthropic Messages API provider
│   └── sse.rs              # Shared SSE line parser
├── runner.rs               # Agentic loop: run(), RunnerConfig, RunnerEvent, @reference resolver
├── context.rs              # Context window trimming + LLM-powered summarization
├── context_index.rs        # Typed context items, budgeted packing, local context index trait
├── sanitize.rs             # Tool result sanitization
├── memory.rs               # Agent memory: MemoryStore, recall, system prompt injection, tool defs
├── discovery.rs            # Model listing across endpoint types
├── arm_discovery.rs        # Azure subscription/resource/project discovery via ARM
└── azure_oauth.rs          # OAuth PKCE + device code flows for Entra ID
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

### Supporting types

```rust
// MessageContent — text or multimodal parts
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

// ContentPart — text or image in multimodal messages
enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

// ImageUrl — for vision-capable models
struct ImageUrl {
    url: String,          // URL or base64 data URI
    detail: Option<String>, // "low", "high", or "auto"
}

// ToolOutput — returned by tool executors (implements From<String>)
enum ToolOutput {
    Text(String),
    WithImages { text: String, images: Vec<ContentPart> },
}

// ToolFunction — definition inside a Tool
struct ToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value, // JSON Schema
}

// Usage — token consumption per LLM call
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// ChatRequest — sent to providers (usually constructed by the runner)
struct ChatRequest {
    messages: Vec<ChatMessage>,
    model: String,
    tools: Option<Vec<Tool>>,
    stream: bool,
    response_format: Option<ResponseFormat>,
}

// ChatResponse — returned by providers after streaming completes
struct ChatResponse {
    message: ChatMessage,
    usage: Option<Usage>,
}
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
- `Usage { usage }` — token usage after each LLM call (`Usage { prompt_tokens, completion_tokens, total_tokens }`)
- `ContextPacked { run_id, iteration, selected_count, dropped_count, total_bytes, budget_bytes, decisions }` — budgeted context selection observability
- `MessagesUpdated { messages }` — full history after a tool round (for persistence)
- `Done { response, messages, elapsed_ms }` — final text + full history + total run time
- `Error { message }` — error description

### AgentError
```rust
AgentError::NotConfigured(String)         // missing API key/endpoint
AgentError::Http(reqwest::Error)          // transport error
AgentError::Stream(String)                // SSE processing error
AgentError::Json(serde_json::Error)       // serialization error
AgentError::Tool(String)                  // tool execution error
AgentError::ToolPanic { name, message }   // tool panicked during execution
AgentError::Storage(String)               // persistence error
AgentError::Cancelled                     // user cancelled
AgentError::Api { status, message }       // non-2xx API response
AgentError::MaxIterations(usize)          // exceeded tool loop limit
AgentError::Guardrailed(String)           // blocked by input/output guardrail
```

## Core function: `agentive::run()`

This is the main entry point. It runs the agentic loop:

```rust
pub async fn run<F, Fut, E>(
    provider: Arc<dyn Provider>,     // LLM provider
    messages: Vec<ChatMessage>,      // initial conversation (include system prompt)
    tools: Vec<Tool>,                // tool definitions for the LLM
    tool_executor: F,                // async closure: ToolCall -> Result<ToolOutput, String>
    config: RunnerConfig,            // max_iterations, retry, trimming, sanitization, compaction
    cancel: CancellationToken,       // for user-initiated stop
    steering: Steering,              // inject user messages mid-loop
    guardrails: Guardrails,          // input/output/tool validation hooks
    on_event: E,                     // callback for RunnerEvent
) -> Result<RunnerResult, AgentError>
where
    F: Fn(ToolCall) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput, String>> + Send,
```

### RunnerConfig defaults
- `max_iterations: 10` — max tool-call rounds before giving up
- `retry_on_400: true` — retry once on HTTP 400
- `auto_trim_context: true` — trim old messages when over budget
- `sanitize_tool_results: true` — strip control chars and base64
- `tool_result_budget: Some(ToolResultBudget::default())` — bound oversized tool results before they enter history
- `parallel_tool_calls: true` — execute multiple tool calls concurrently
- `response_format: None` — optional structured output (JSON mode or JSON schema)
- `compaction_provider: None` — optional LLM provider for richer context compaction
- `tool_filter: None` — optional per-round tool filter for dynamic tool gating
- `run_id: None` — auto-generates UUID v4 if not set; use for trace correlation
- `parent_run_id: None` — set when delegating to link child runs to parent
- `reference_resolver: None` — optional async resolver for `@reference` syntax in user messages
- `context_items: Vec::new()` — optional typed context available for per-round packing
- `context_packing: None` — opt-in policy for transient `<context_pack>` injection

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
    fn model(&self) -> Option<&str> { None }
    fn context_budget_chars(&self) -> usize { 200_000 }
    fn request_budget_bytes(&self) -> Option<usize> { None }
    fn estimate_request_bytes(&self, _request: &ChatRequest) -> Result<Option<usize>, AgentError> { Ok(None) }
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

Request byte-budget behavior:
- OpenAI-compatible non-Azure endpoints default to no byte cap (`usize::MAX`).
- Azure/OpenAI-compatible endpoints default to a 64KB serialized request cap with
  reserved overhead before provider serialization.
- If compaction cannot fit the serialized body, the provider returns an actionable
  "reduce attached files, references, web content, or earlier context" error.
- Keep Azure-specific request margins provider-aware; do not apply them globally
  to Anthropic or unknown providers.

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
- No default serialized byte cap; callers can opt in with `with_max_request_bytes`.

### ResponsesProvider

OpenAI Responses API (`/v1/responses`) — newer endpoint with different format:

```rust
ResponsesProvider::new(endpoint, api_key, model)
    .with_context_budget(128_000)    // optional
    .with_vision(true)               // optional
    .with_max_request_bytes(64_000)  // optional (64KB default)
```

Key differences from Chat Completions:

- System messages → `role: "developer"` input items
- Tool calls → separate `function_call` input items (not nested in assistant)
- Tool results → `function_call_output` input items
- SSE events: `response.output_text.delta`, `response.output_item.added`,
  `response.function_call_arguments.delta`, `response.completed`
- Tool defs are flattened (no `function` wrapper)
- **Request body guard**: The Azure Responses API silently truncates bodies at
  ~79KB. The provider measures the actual serialized JSON byte count and drops
  the oldest non-system input items until the body fits within the effective safe
  budget: `max_request_bytes` (default 64KB for Azure) minus reserved overhead.
  `function_call` / `function_call_output` pairs are dropped together to avoid
  orphaned tool results. Set `with_max_request_bytes(usize::MAX)` to disable.

## One-shot chat: `simple_chat()`

For quick LLM calls that don't need the agentic loop (no tools, no streaming,
no retries), use `simple_chat`:

```rust
use agentive::{simple_chat, ChatMessage};
use std::sync::Arc;

let provider = agentive::build_provider(
    "https://my-resource.openai.azure.com",
    "my-api-key",
    "gpt-4o",
);

let response = simple_chat(
    provider,
    vec![
        ChatMessage::system("You are a concise assistant."),
        ChatMessage::user("Summarize this text: ..."),
    ],
).await?;

println!("{}", response.text().unwrap_or("no response"));
```

Returns `Result<ChatMessage, AgentError>`. Use cases: text reformatting, sparkle
fills, summarization, classification — anything that's a single request/response.

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
- `default_context_budget(model)` — estimate context budget in characters from model name
- `context_budget(model, reported_context)` — budget with optional API-reported context length override

### Model families recognized

| Family | Vision | Context | Notes |
| --- | --- | --- | --- |
| gpt-4o, gpt-4.1, gpt-5 | ✅ | 384K chars | 128K tokens × 75% × 4 chars |
| gpt-4-turbo, gpt-4-1106, gpt-4-0125 | ✅ | 384K chars | 128K token variants |
| gpt-35-turbo-16k, gpt-3.5-turbo-16k | ❌ | 49K chars | 16K tokens |
| gpt-35-turbo, gpt-3.5-turbo | ❌ | 12K chars | 4K tokens |
| gpt-4 (base) | ❌ | 24K chars | 8K tokens |
| claude-3.5, claude-3-5, claude-4 | ✅ | 600K chars | 200K tokens |
| claude (other) | ❌ | 300K chars | 100K tokens |
| o1, o3, o4 series | ✅ | 384K chars | 128K tokens |
| gemini | ✅ | 384K chars | 128K tokens |
| deepseek | ❌ | 192K chars | 64K tokens |
| phi-3, phi-4 | ❌ | 48K chars | 16K tokens |
| mistral-large, mistral-medium | ❌ | 96K chars | 32K tokens |
| mistral (small) | ❌ | 24K chars | 8K tokens |
| codex, -pro models | ✅ | 48K chars | Responses API, body-size limited |
| unknown | ❌ | 96K chars | Conservative 32K token default |

### context_budget with reported override

When the API reports a model's actual context window (e.g., from `context_length`
in the discovery response), pass it to `context_budget()` to get a more accurate
budget:

```rust
use agentive::context_budget;

// Unknown deployment — uses heuristic (96K chars)
let budget = context_budget("my-custom-deployment", None);

// API reports 200K token context — overrides heuristic (600K chars)
let budget = context_budget("my-custom-deployment", Some(200_000));
```

Formula: `reported_tokens × 75% (usable) × 4 (chars/token)`.

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

## Context harness (context_index.rs + runner.rs)

Agentive 0.6.0 adds typed, budgeted context orchestration. The intended mental
model is:

```text
raw host context -> ContextItem records -> retrieval/ranking -> ContextPacker
  -> transient <context_pack> -> provider request -> provider byte safety rails
```

Context compaction is the emergency brake. Do not use compaction as the primary
strategy for large web pages, files, logs, memories, or tool outputs. Hosts should
prefer bounded context items, local retrieval, large payload references, and
budgeted packing before the provider sees the request.

### Core types

- `ContextItem` — host-supplied context candidate with `source`, `kind`,
  `priority`, `sensitivity`, optional `content`, optional `large_ref`, and
  metadata.
- `ContextKind` — budget category (`RecentTurn`, `MemoryFact`, `ReferenceDoc`,
  `ToolObservation`, `FileExcerpt`, `WebExcerpt`, `ErrorTrace`,
  `MediaSummary`, `Other`).
- `ContextSensitivity` — `Secret` is excluded by default; `Private` is included
  unless `include_private` is false.
- `LargeContextRef` — stable pointer to full payload content stored outside the
  prompt, with an `expand_tool` the host can expose later.
- `ContextPackingConfig` and `ContextKindBudget` — total, per-kind, per-item,
  and preview budgets.
- `ContextPacker` — deterministic ranking and rendered-byte-aware packing.
- `LocalContextIndex` — trait for host-owned search/vector stores; the crate
  includes deterministic `InMemoryContextIndex` for lexical tests and simple
  local use.

### Runner integration

Opt in with:

```rust
let config = RunnerConfig {
    context_items,
    context_packing: Some(ContextPackingConfig::default()),
    ..Default::default()
};
```

When enabled:
1. The runner packs context each model-call round using the latest user query.
2. It emits `RunnerEvent::ContextPacked` with all selected/dropped/redacted
   decisions for observability.
3. It renders a `<context_pack>` block labeled as untrusted reference material.
4. It inserts that block before the latest user message, so the user's final
   request remains the most recent instruction.
5. It sends the packed block only in the provider request. It is never appended
   to `RunnerResult.messages`, `new_messages`, or `MessagesUpdated` history.
6. If the packed block makes the provider request exceed the request budget, the
   runner drops the transient pack before compacting durable conversation
   history. Durable history must not be sacrificed to keep transient context.
7. 400 retries reuse the same prepared request messages, preserving packed
   context when it fit the first attempt.

### Packing rules and invariants

- Use rendered prompt bytes, not raw content bytes. XML escaping, attributes,
  content type tags, and large-ref tags all count against budget.
- Prefer large refs for big payloads. A `LargeContextRef` lets the model see a
  preview plus an expansion handle without paying for the full blob.
- Duplicate `large_ref.id` values are packed only once.
- Secret context is redacted by default.
- Packed context is untrusted. Never allow text inside `<context_pack>` to
  override the latest user request, system prompt, or tool policies.
- Public context structs are `#[non_exhaustive]`; prefer builder methods
  (`ContextItem::new().with_kind(...).with_content(...)`) over struct literals.

### Host guidance

Apps like CutReady should not inline full fetched web pages or unbounded
reference text into the final user message. They should:

1. Store full payloads in a host-owned blob/file/database store.
2. Create `ContextItem`s with short summaries or high-signal excerpts.
3. Attach `LargeContextRef` handles for expansion.
4. Use local retrieval (lexical, vector, or hybrid) to select candidates.
5. Let `ContextPacker` assemble a bounded prompt pack.
6. Treat Agentive provider byte guards as final safety rails, not the first
   filter.

### Common footguns for agents

- Do not persist `<context_pack>` into conversation history.
- Do not append packed context after the latest user message.
- Do not compact durable history just to keep transient packed context.
- Do not use `estimated_bytes` alone to decide prompt fit; rendered XML must fit.
- Do not add broad fallback behavior that silently drops user messages.
- Do not make Azure/OpenAI byte margins global across all providers.
- Do not inline huge web/file/tool payloads when a `LargeContextRef` would work.

### Required tests when changing context or request budgeting

Run the smallest focused tests first, then the full suite:

```bash
cd lib
cargo test --lib context_index
cargo test --lib context_packing
cargo test --lib request_budget_defaults
cargo test --lib tool_heavy
cargo test --lib drops_pack_before_compacting
cargo test
```

These cover CutReady-like web/reference floods, rendered XML byte budgets,
provider-aware request caps, opt-in compatibility, tool-heavy long sessions, and
the invariant that packed context is dropped before durable history compaction.

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
owned `ToolCall` and returns `Result<ToolOutput, String>`:

```rust
|call: ToolCall| async move {
    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
        .map_err(|e| e.to_string())?;
    match call.function.name.as_str() {
        "read_file" => {
            let path = args["path"].as_str().ok_or("missing path")?;
            let content = tokio::fs::read_to_string(path).await.map_err(|e| e.to_string())?;
            Ok(ToolOutput::Text(content))
        }
        _ => Err(format!("Unknown tool: {}", call.function.name))
    }
}
```

### ToolOutput — text or multimodal results

Tool executors return `ToolOutput` instead of plain `String`:

```rust
// Plain text result (most tools)
ToolOutput::Text("file contents here".into())

// Text + images (vision tools — e.g., screenshot, chart rendering)
ToolOutput::WithImages {
    text: "Screenshot of the current page".into(),
    images: vec![
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "data:image/png;base64,iVBOR...".into(),
                detail: Some("low".into()),
            },
        },
    ],
}
```

When a tool returns `WithImages`, the runner:

1. Sends the `text` as a normal tool result message
2. Appends a follow-up `ChatMessage::user_with_images()` containing the images
   so the LLM can see them on the next turn

**Backward compatibility**: `ToolOutput` implements `From<String>` and
`From<&str>`, so existing closures that return `Ok("result".into())` work
without changes.

## Dynamic tool gating

Change which tools are available to the LLM each round:

```rust
use agentive::{RunnerConfig, Tool, ChatMessage};
use std::sync::Arc;

let config = RunnerConfig {
    tool_filter: Some(Arc::new(|messages: &[ChatMessage]| {
        let has_plan = messages.iter().any(|m| m.text().is_some_and(|t| t.contains("PLAN:")));
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

## @reference resolver

Resolve `@name` references in user messages to inject contextual content before LLM calls:

```rust
use agentive::{RunnerConfig, ReferenceResolver, ResolvedReference};
use std::sync::Arc;

let resolver: ReferenceResolver = Arc::new(|name| {
    Box::pin(async move {
        // App-specific resolution: files, DB records, API responses, etc.
        match name.as_str() {
            "intro.sk" => Some(ResolvedReference {
                name: "intro.sk".into(),
                content: "# Intro\nWelcome to the demo.".into(),
                content_type: "text/markdown".into(),
            }),
            _ => None, // Unknown references are silently ignored
        }
    })
});

let config = RunnerConfig {
    reference_resolver: Some(resolver),
    ..Default::default()
};
```

### How it works

1. Before the first LLM call, the runner scans all user messages for `@name` or `@"quoted name"` patterns
2. For each unique reference found, calls the resolver asynchronously (all references resolved concurrently)
3. Resolved content is appended to the user message as XML-tagged context blocks
4. After steering injects new messages mid-run, those are also scanned and resolved
5. Messages that already contain `<referenced_document` are skipped (no re-resolution)

### Reference syntax

- `@word` — alphanumeric, hyphens, underscores, dots, slashes (e.g., `@intro.sk`, `@docs/setup`)
- `@"quoted name"` — arbitrary text in double quotes (e.g., `@"my sketch with spaces"`)

### Injected format

```xml
<referenced_document name="intro.sk" content_type="text/markdown">
# Intro
Welcome to the demo.
</referenced_document>
```

### Use cases

- **CutReady**: `@sketch.sk` resolves to sketch JSON, `@notes.md` to markdown content
- **sethjuarez.com**: `@blog-post` resolves to content by slug
- **Any app**: files, database records, API responses, embeddings — the resolver is fully app-defined

## Memory (memory.rs)

Agent memory for persistent knowledge across conversations. Based on a five-layer
cognitive model:

| Layer | Purpose | Managed by |
| --- | --- | --- |
| **Working** | Active conversation context | `run()` + context trimming |
| **Procedural** | Tool definitions & system prompts | Static per session |
| **Core** | Persistent project/user facts | `save_memory` tool |
| **Archival** | Compressed session summaries | Auto-saved on session end |
| **Recall** | On-demand memory search | `recall_memory` tool |

### Core types

```rust
use agentive::memory::{MemoryStore, MemoryCategory, MemoryEntry, MemoryBackend};

let mut store = MemoryStore::default();

// Save memories (core memories dedup by tags)
store.save(MemoryCategory::Core, "User prefers concise output", vec!["style".into()]);
store.save(MemoryCategory::Insight, "Dashboard needs chart builder", vec!["dashboard".into()]);
store.archive_session("Discussed login flow", "session-42");

// Search with keyword scoring (+3 tag match, +2 content match, +1 core boost)
let results = store.recall("dashboard");

// Inject core memories into system prompt
let prompt_block = store.format_for_system_prompt();
// → "\n[Memories about this project and user]\n• User prefers concise output\n"

// Format search results for LLM
let formatted = MemoryStore::format_recall_results(&results);
```

### Capacity management

- **Max 200 entries**. When exceeded, oldest archival entries are evicted first.
- **Core dedup**: saving a core memory with the same tags replaces the previous one.

### Persistence via `MemoryBackend`

The module provides the in-memory data model and operations. Persistence is
pluggable — apps implement `MemoryBackend` to choose their storage:

```rust
pub trait MemoryBackend: Send + Sync {
    fn load(&self) -> MemoryStore;
    fn save(&self, store: &MemoryStore) -> Result<(), String>;
}
```

Example file backend:

```rust
struct FileMemory { path: PathBuf }

impl MemoryBackend for FileMemory {
    fn load(&self) -> MemoryStore {
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|data| serde_json::from_str(&data).ok())
            .unwrap_or_default()
    }
    fn save(&self, store: &MemoryStore) -> Result<(), String> {
        let json = serde_json::to_string_pretty(store).map_err(|e| e.to_string())?;
        std::fs::write(&self.path, json).map_err(|e| e.to_string())
    }
}
```

### Tool definitions

Two standard tool definitions for LLM function calling:

```rust
use agentive::memory;

let tools = vec![
    memory::recall_memory_tool(),  // "recall_memory" — keyword search
    memory::save_memory_tool(),    // "save_memory" — persist core/insight
    // ... other app tools
];
```

These return `Tool` structs ready to pass to `run()`. The app's tool executor
handles the actual call by delegating to `MemoryStore` methods.

### Integration pattern

A typical app wires memory like this:

1. On startup: `let store = backend.load();`
2. Before `run()`: inject `store.format_for_system_prompt()` into system message
3. Tool executor: `"recall_memory" => store.recall(query)`,
   `"save_memory" => store.save(cat, content, tags)`
4. After each tool mutation: `backend.save(&store)`
5. On session end: `store.archive_session(summary, session_id)`

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
cargo test           # run unit, integration, and doc tests
cargo doc --no-deps  # generate documentation
```

## Release and publishing

Publishing to crates.io must be deliberate. Do **not** publish automatically on
every merge to `main`.

Release flow:

1. Merge the fix or feature PR to `main`.
2. Decide the next semver version. Use a patch version for bug fixes.
3. Update `lib/Cargo.toml` and `lib/Cargo.lock` to that version.
4. Run:
   ```bash
   cd lib
   cargo test
   cargo publish --dry-run
   ```
5. Commit the version bump on `main` with `chore(release): prepare agentive X.Y.Z`.
6. Publish to crates.io only after the user explicitly confirms. Use the
   manual GitHub Actions workflow **Publish crate** from `main`, enter the
   `version` as `X.Y.Z`, and type `publish` in the `confirm_publish` input.
   The workflow runs tests, verifies `lib/Cargo.toml` and `lib/Cargo.lock` for
   newly published versions, runs `cargo publish --dry-run` for newly published
   versions, publishes only if crates.io does not already have the version, then
   reconciles the annotated `vX.Y.Z` Git tag and GitHub release notes.
7. Prefer the workflow over local `cargo publish`. If local publishing is
   unavoidable, also create or reconcile the matching GitHub tag/release.

If the crate version already exists on crates.io, the workflow skips publishing
and still verifies/reconciles the GitHub tag and release metadata.

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

`OpenAiProvider::chat_url()` handles four cases:

| Endpoint type | Chat URL built |
| --- | --- |
| Already contains `/chat/completions` | Used as-is |
| Foundry project (`/api/projects/`) | **Strips** `/api/projects/...` → `{resource_base}/openai/deployments/{model}/chat/completions?api-version=2024-10-21` |
| Plain Azure (`*.azure.com`) | `{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-10-21` |
| Everything else (OpenAI, local) | `{endpoint}/chat/completions` |

**Critical**: For Azure endpoints (both Foundry and plain), the model name is
used as the deployment name in the URL path. For non-Azure endpoints, the model
is sent in the request body only.

### ARM resource discovery (arm_discovery.rs)

Discovers Azure AI resources and their Foundry projects via Azure Resource Manager:

```rust
use agentive::arm_discovery::*;

// List Azure subscriptions
let subs = list_subscriptions(&token).await?;
// Returns Vec<Subscription> with subscription_id, display_name, state

// List AI resources in a subscription
let resources = list_ai_resources(&token, &subs[0].subscription_id).await?;
// Returns Vec<AiResource> with name, kind, endpoint, resource_group, foundry_url

// List Foundry projects for a specific resource
let projects = list_foundry_projects(
    &token,
    &subs[0].subscription_id,
    &resources[0].resource_group,
    &resources[0].name,
).await?;
// Returns Vec<FoundryProject> with name, display_name, endpoint
```

**Types:**

- `Subscription` — `subscription_id`, `display_name`, `state`
- `AiResource` — `name`, `kind`, `endpoint`, `location`, `resource_group`, `foundry_url`
- `FoundryProject` — `name`, `display_name`, `endpoint`

**AiResource details:**
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

### Model discovery types (discovery.rs)

```rust
pub struct ModelInfo {
    pub id: String,
    pub owned_by: Option<String>,
    pub capabilities: Option<HashMap<String, String>>,
    pub context_length: Option<usize>,
}
```

### Azure OAuth (azure_oauth.rs)

Two authentication flows for desktop apps:

**Browser-based (auth code + PKCE):**

```rust
use agentive::azure_oauth::*;

// 1. Start flow — opens local HTTP server, returns auth URL + PKCE verifier
let (init, code_verifier) = start_auth_code_flow(tenant_id, None, None).await?;
// init.auth_url — open in browser for user sign-in
// init.port — local callback port

// 2. Wait for browser callback
let auth_code = wait_for_auth_code(init.port, 120, "MyApp").await?;

// 3. Exchange code for tokens
let tokens = exchange_code_for_token(
    tenant_id, &auth_code,
    &format!("http://localhost:{}", init.port),
    &code_verifier, None, None,
).await?;
// tokens.access_token, tokens.refresh_token

// 4. Refresh later
let new_tokens = refresh_token(tenant_id, &tokens.refresh_token.unwrap(), None, None).await?;
```

**Device code (headless/CLI):**

```rust
use agentive::azure_oauth::*;

// 1. Request device code — show user_code + verification_uri to user
let device = request_device_code(tenant_id, None, None).await?;
println!("{}", device.message); // "Go to https://... and enter code ABC-123"

// 2. Poll until user completes sign-in
let tokens = poll_for_token(
    tenant_id, &device.device_code, device.interval, device.expires_in, None,
).await?;
```

**Types:**
- `TokenResponse` — `access_token`, `token_type`, `expires_in`, `refresh_token`, `scope`
- `AuthCodeFlowInit` — `auth_url`, `port`
- `DeviceCodeResponse` — `device_code`, `user_code`, `verification_uri`, `expires_in`, `interval`, `message`

**Scope constants:**
- `AZURE_OPENAI_SCOPE` — for AI inference calls (`https://ai.azure.com/.default offline_access`)
- `AZURE_MANAGEMENT_SCOPE` — for ARM API calls (`https://management.azure.com/.default offline_access`)
- `DEFAULT_CLIENT_ID` — Microsoft's public client ID (works for most Entra tenants)

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
9. **App-defined @reference resolution** — agentive parses `@name` syntax but
   delegates resolution to the app. What `@thing` means (file, DB record, API
   response) is entirely app-specific.
10. **Observability via events** — run_id, tool_call_id, elapsed_ms, and iteration
    are embedded in RunnerEvent variants. No external tracing dependency required.
