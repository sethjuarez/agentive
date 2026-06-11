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
//!     |_call| async { Ok(agentive::ToolOutput::from("tool not implemented")) },
//!     RunnerConfig::default(),
//!     CancellationToken::new(),
//!     Steering::new(),
//!     Guardrails::default(),
//!     |_event| {},
//! ).await?;
//! # Ok(())
//! # }
//! ```

pub mod arm_discovery;
pub mod auth;
pub mod azure_oauth;
pub mod cancel;
pub mod chat;
pub mod checkpoint;
pub mod context;
pub mod context_index;
pub mod discovery;
pub mod error;
pub mod factory;
pub mod guardrails;
pub mod memory;
pub mod observability;
pub mod parse;
pub mod provider;
pub mod providers;
pub mod resume;
pub mod runner;
pub mod sanitize;
pub mod state;
pub mod steering;
pub mod trajectory;
pub mod types;
pub mod web;

pub use auth::AuthStrategy;
pub use cancel::CancellationToken;
pub use chat::simple_chat;
pub use checkpoint::Checkpoint;
pub use context_index::{ContextItem, ContextSource, ReferenceRecord, SearchIndexEntry};
pub use error::AgentError;
pub use factory::{
    build_provider, build_provider_with_auth, context_budget, default_context_budget,
    needs_responses_api, supports_vision,
};
pub use guardrails::{GuardrailResult, Guardrails};
pub use memory::{MemoryBackend, MemoryCategory, MemoryEntry, MemoryStore};
pub use observability::{CheckpointStore, ContextIndex, MemoryPromotionHook, TrajectorySink};
pub use parse::parse_tool_args;
pub use provider::Provider;
pub use providers::anthropic::AnthropicProvider;
pub use providers::openai::OpenAiProvider;
pub use providers::responses::ResponsesProvider;
pub use resume::ResumeContext;
pub use runner::{
    run, ReferenceResolver, ResolvedReference, RunnerConfig, RunnerEvent, RunnerResult, ToolFilter,
    ToolResultBudget,
};
pub use sanitize::{sanitize_for_api, sanitize_message};
pub use state::{
    ErrorKind, FailureRecord, MemoryPromotionCandidate, MemoryPromotionOutcome, Plan, PlanStatus,
    PlanStep, PlanStepStatus, ResourceOperation, TouchedResource, VerificationResult,
    VerificationStatus,
};
pub use steering::Steering;
pub use trajectory::{
    ArgumentSummary, ModelUsage, PermissionDecision, TrajectoryEvent, TrajectoryMetadata,
};
pub use types::*;
