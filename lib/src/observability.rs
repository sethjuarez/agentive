//! Optional observability and resumability extension traits.
//!
//! These traits let hosts persist or index agent-loop state without making
//! `agentive` database-opinionated. Implementations may write to SQLite, files,
//! telemetry systems, in-memory test buffers, or any host-owned store.

use crate::checkpoint::Checkpoint;
use crate::context_index::{ContextItem, ReferenceRecord, SearchIndexEntry};
use crate::state::{MemoryPromotionCandidate, MemoryPromotionOutcome};
use crate::trajectory::TrajectoryEvent;

/// Append-only sink for structured trajectory events.
pub trait TrajectorySink: Send + Sync {
    /// Record a trajectory event.
    fn record(&self, event: TrajectoryEvent) -> Result<(), String>;
}

/// Store for resumable checkpoints.
pub trait CheckpointStore: Send + Sync {
    /// Save or append a checkpoint.
    fn save_checkpoint(&self, checkpoint: Checkpoint) -> Result<(), String>;

    /// Return the latest checkpoint for a run/session, if any.
    fn latest_checkpoint(&self, run_id: &str) -> Result<Option<Checkpoint>, String>;

    /// List checkpoints for a run/session in host-defined order.
    fn list_checkpoints(&self, run_id: &str) -> Result<Vec<Checkpoint>, String>;
}

/// Index for context, references, and searchable resources.
pub trait ContextIndex: Send + Sync {
    /// Record context that can be recalled or injected later.
    fn record_context_item(&self, item: ContextItem) -> Result<(), String>;

    /// Record a reference encountered or resolved during a run.
    fn record_reference(&self, reference: ReferenceRecord) -> Result<(), String>;

    /// Record a searchable resource entry.
    fn record_search_entry(&self, entry: SearchIndexEntry) -> Result<(), String>;
}

/// Host-owned policy hook for memory promotion candidates.
///
/// Agentive can surface that a tool or loop step found a potentially durable
/// fact. The host decides whether to save it, reject it, defer it, redact it, or
/// route it to a domain-specific memory system.
pub trait MemoryPromotionHook: Send + Sync {
    /// Consider a promotion candidate and return the host's decision.
    fn consider(
        &self,
        candidate: MemoryPromotionCandidate,
    ) -> Result<MemoryPromotionOutcome, String>;
}
