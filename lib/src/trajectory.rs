//! Structured trajectory events for agent-loop observability.
//!
//! This module provides a serializable, host-neutral event model. It is designed
//! for high-signal loop state and diagnostics; hosts decide where and how to
//! persist or display these events.

pub use crate::state::ErrorKind;
use crate::state::{FailureRecord, MemoryPromotionCandidate, TouchedResource, VerificationResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Common metadata attached to trajectory events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrajectoryMetadata {
    /// When the event was created.
    pub timestamp: DateTime<Utc>,
    /// Stable event ID for building an event tree or span graph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    /// Parent event ID when this event is nested under another event/span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_event_id: Option<String>,
    /// Run ID for correlating events across a single agent run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Parent run ID when this run was delegated from another run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<String>,
    /// Turn ID, if the host tracks turns separately from runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Runner loop iteration, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iteration: Option<usize>,
}

impl Default for TrajectoryMetadata {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            event_id: None,
            parent_event_id: None,
            run_id: None,
            parent_run_id: None,
            turn_id: None,
            iteration: None,
        }
    }
}

impl TrajectoryMetadata {
    /// Create metadata using the current timestamp.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach an event ID.
    pub fn with_event_id(mut self, event_id: impl Into<String>) -> Self {
        self.event_id = Some(event_id.into());
        self
    }

    /// Attach a parent event ID.
    pub fn with_parent_event_id(mut self, parent_event_id: impl Into<String>) -> Self {
        self.parent_event_id = Some(parent_event_id.into());
        self
    }

    /// Attach a run ID.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Attach a parent run ID.
    pub fn with_parent_run_id(mut self, parent_run_id: impl Into<String>) -> Self {
        self.parent_run_id = Some(parent_run_id.into());
        self
    }

    /// Attach a turn ID.
    pub fn with_turn_id(mut self, turn_id: impl Into<String>) -> Self {
        self.turn_id = Some(turn_id.into());
        self
    }

    /// Attach a loop iteration.
    pub fn with_iteration(mut self, iteration: usize) -> Self {
        self.iteration = Some(iteration);
        self
    }
}

/// Redacted summary of tool-call arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArgumentSummary {
    /// Stable SHA-256 hash of the original argument string.
    pub sha256: String,
    /// Original argument byte length.
    pub byte_len: usize,
    /// Optional short preview. Keep absent when arguments may contain secrets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
    /// Whether the original arguments were intentionally omitted.
    pub redacted: bool,
}

impl ArgumentSummary {
    /// Create a redacted summary containing only length and hash.
    pub fn redacted(arguments: impl AsRef<str>) -> Self {
        Self::from_arguments(arguments, None)
    }

    /// Create a summary with an optional bounded preview.
    pub fn from_arguments(arguments: impl AsRef<str>, preview_chars: Option<usize>) -> Self {
        let arguments = arguments.as_ref();
        let mut hasher = Sha256::new();
        hasher.update(arguments.as_bytes());
        let sha256 = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let preview = preview_chars.map(|limit| arguments.chars().take(limit).collect());

        Self {
            sha256,
            byte_len: arguments.len(),
            preview,
            redacted: preview_chars.is_none(),
        }
    }
}

/// Model-call usage summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Outcome of a permission check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Requested,
    Granted,
    Denied,
    NotRequired,
}

/// Structured event emitted by an agent loop.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TrajectoryEvent {
    /// A user/agent turn started.
    TurnStarted {
        metadata: TrajectoryMetadata,
        goal: String,
    },
    /// A turn completed or stopped.
    TurnCompleted {
        metadata: TrajectoryMetadata,
        success: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        failure: Option<FailureRecord>,
    },
    /// A model call started.
    ModelCallStarted {
        metadata: TrajectoryMetadata,
        provider: String,
        model: String,
    },
    /// A model call completed.
    ModelCallCompleted {
        metadata: TrajectoryMetadata,
        provider: String,
        model: String,
        duration_ms: u64,
        success: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<ModelUsage>,
        #[serde(skip_serializing_if = "Option::is_none")]
        failure: Option<FailureRecord>,
    },
    /// A tool call started. Match with [`TrajectoryEvent::ToolCallCompleted`]
    /// by `call_id`.
    ToolCallStarted {
        metadata: TrajectoryMetadata,
        call_id: String,
        tool_name: String,
        arguments: ArgumentSummary,
    },
    /// A tool call completed. Match with [`TrajectoryEvent::ToolCallStarted`]
    /// by `call_id`.
    ToolCallCompleted {
        metadata: TrajectoryMetadata,
        call_id: String,
        tool_name: String,
        duration_ms: u64,
        success: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        failure: Option<FailureRecord>,
    },
    /// A permission decision was requested or resolved.
    Permission {
        metadata: TrajectoryMetadata,
        decision: PermissionDecision,
        scope: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// A retry was scheduled after a failure.
    RetryScheduled {
        metadata: TrajectoryMetadata,
        attempt: usize,
        max_attempts: usize,
        error_kind: ErrorKind,
        reason: String,
    },
    /// A verification result was recorded.
    VerificationRecorded {
        metadata: TrajectoryMetadata,
        result: VerificationResult,
    },
    /// A checkpoint was created.
    CheckpointCreated {
        metadata: TrajectoryMetadata,
        checkpoint_id: String,
        summary: String,
    },
    /// Conversation compaction started.
    CompactionStarted {
        metadata: TrajectoryMetadata,
        reason: String,
    },
    /// Conversation compaction completed.
    CompactionCompleted {
        metadata: TrajectoryMetadata,
        success: bool,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        failure: Option<FailureRecord>,
    },
    /// A memory was proposed for promotion.
    MemoryPromotionSuggested {
        metadata: TrajectoryMetadata,
        candidate: MemoryPromotionCandidate,
    },
    /// A memory promotion completed or was rejected.
    MemoryPromotionCompleted {
        metadata: TrajectoryMetadata,
        success: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        memory_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        failure: Option<FailureRecord>,
    },
    /// A resource was touched by the loop.
    ResourceTouched {
        metadata: TrajectoryMetadata,
        resource: TouchedResource,
    },
    /// Host-defined extension event.
    Custom {
        metadata: TrajectoryMetadata,
        name: String,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        fields: BTreeMap<String, String>,
    },
}

impl TrajectoryEvent {
    /// Convenience constructor for a tool start event.
    pub fn tool_call_started(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: ArgumentSummary,
        metadata: TrajectoryMetadata,
    ) -> Self {
        Self::ToolCallStarted {
            metadata,
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            arguments,
        }
    }

    /// Convenience constructor for a tool completion event.
    pub fn tool_call_completed(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        duration_ms: u64,
        success: bool,
        failure: Option<FailureRecord>,
        metadata: TrajectoryMetadata,
    ) -> Self {
        Self::ToolCallCompleted {
            metadata,
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            duration_ms,
            success,
            failure,
        }
    }

    /// Return the tool call ID for tool lifecycle events.
    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Self::ToolCallStarted { call_id, .. } | Self::ToolCallCompleted { call_id, .. } => {
                Some(call_id)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_tool_lifecycle_events_with_matching_call_id() {
        let metadata = TrajectoryMetadata::new()
            .with_event_id("event-start")
            .with_parent_event_id("event-turn")
            .with_run_id("run-1")
            .with_iteration(2);
        let start = TrajectoryEvent::tool_call_started(
            "call-123",
            "shell",
            ArgumentSummary::redacted("Get-ChildItem"),
            metadata.clone(),
        );
        let end =
            TrajectoryEvent::tool_call_completed("call-123", "shell", 42, true, None, metadata);

        assert_eq!(start.tool_call_id(), end.tool_call_id());
        let json = serde_json::to_string(&start).unwrap();
        assert!(json.contains("\"type\":\"tool_call_started\""));
        assert!(json.contains("\"call_id\":\"call-123\""));
        assert!(json.contains("\"event_id\":\"event-start\""));
        assert!(json.contains("\"parent_event_id\":\"event-turn\""));
        assert!(!json.contains("Get-ChildItem"));

        let round_trip: TrajectoryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip.tool_call_id(), Some("call-123"));

        let end_json = serde_json::to_string(&end).unwrap();
        assert!(!end_json.contains("\"failure\":null"));
    }

    #[test]
    fn argument_summary_can_include_bounded_preview() {
        let summary = ArgumentSummary::from_arguments("abcdef", Some(3));
        assert_eq!(summary.byte_len, 6);
        assert_eq!(summary.preview.as_deref(), Some("abc"));
        assert!(!summary.redacted);
        assert_eq!(summary.sha256.len(), 64);
    }
}
