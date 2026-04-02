//! # Agentive
//!
//! A Rust crate for building agentic LLM applications with streaming,
//! tool calling, and multi-turn conversation loops.
//!
//! ## Quick Start
//! ```no_run
//! use agentive::{OpenAiProvider, RunnerConfig, CancellationToken, Steering, Guardrails, ChatMessage};
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), agentive::AgentError> {
//! let provider = Arc::new(OpenAiProvider::new(
//!     "https://api.openai.com/v1",
//!     "sk-...",
//!     "gpt-4o",
//! ));
//!
//! let result = agentive::run(
//!     provider,
//!     vec![ChatMessage::user("Hello!")],
//!     vec![],
//!     |_call| Ok("tool not implemented".into()),
//!     RunnerConfig::default(),
//!     CancellationToken::new(),
//!     Steering::new(),
//!     Guardrails::default(),
//!     |_event| {},
//! ).await?;
//! # Ok(())
//! # }
//! ```

pub mod cancel;
pub mod context;
pub mod error;
pub mod guardrails;
pub mod parse;
pub mod provider;
pub mod providers;
pub mod runner;
pub mod sanitize;
pub mod steering;
pub mod types;

pub use cancel::CancellationToken;
pub use error::AgentError;
pub use guardrails::{GuardrailResult, Guardrails};
pub use parse::parse_tool_args;
pub use provider::Provider;
pub use providers::anthropic::AnthropicProvider;
pub use providers::openai::OpenAiProvider;
pub use providers::responses::ResponsesProvider;
pub use runner::{run, RunnerConfig, RunnerEvent, RunnerResult};
pub use steering::Steering;
pub use types::*;
