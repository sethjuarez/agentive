use thiserror::Error;

/// Unified error type for all agentive operations.
#[derive(Debug, Error)]
pub enum AgentError {
    /// Provider is not configured (missing API key, endpoint, etc.)
    #[error("Provider not configured: {0}")]
    NotConfigured(String),

    /// HTTP transport error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Error during SSE stream processing.
    #[error("Stream error: {0}")]
    Stream(String),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Tool execution error.
    #[error("Tool error: {0}")]
    Tool(String),

    /// Storage/persistence error.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Operation was cancelled by the user.
    #[error("Cancelled")]
    Cancelled,

    /// API returned a non-success status code.
    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    /// Exceeded maximum tool-call iterations.
    #[error("Exceeded maximum tool iterations ({0})")]
    MaxIterations(usize),

    /// A tool closure panicked during execution.
    #[error("Tool '{name}' panicked: {message}")]
    ToolPanic { name: String, message: String },

    /// A guardrail denied the operation.
    #[error("Guardrail denied: {0}")]
    Guardrailed(String),
}
