use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::cancel::CancellationToken;
use crate::error::AgentError;
use crate::types::{ChatEvent, ChatRequest};

/// Unified interface for LLM providers.
///
/// Implementations handle HTTP requests, SSE parsing, and event emission.
/// The crate ships with [`OpenAiProvider`](crate::OpenAiProvider) and
/// [`AnthropicProvider`](crate::AnthropicProvider), but you can implement
/// this trait for any LLM backend.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Stream a chat completion. Tokens and events are sent via `tx`.
    /// Implementations should check `cancel` between SSE events to support
    /// user-initiated stop.
    async fn chat(
        &self,
        request: ChatRequest,
        tx: mpsc::Sender<ChatEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), AgentError>;

    /// Human-readable provider name (e.g. "openai", "anthropic").
    fn name(&self) -> &str;

    /// Approximate character budget for the model's context window.
    /// Used by the runner for context trimming. Default is 200k chars
    /// (~50k tokens for most models).
    fn context_budget_chars(&self) -> usize {
        200_000
    }

    /// Whether this provider/model supports vision (image content parts).
    fn supports_vision(&self) -> bool {
        false
    }
}
