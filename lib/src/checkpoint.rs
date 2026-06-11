//! Host-neutral checkpoint model for long-session compaction and resume.
//!
//! Checkpoints capture high-signal state that should survive noisy event logs:
//! goal, completed work, decisions, touched resources, failures, verification,
//! blockers, and the next step. Hosts choose when to create and persist them.

use crate::state::{FailureRecord, TouchedResource, VerificationResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A resumable summary of agent-loop state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Checkpoint {
    /// Stable host- or agent-generated checkpoint ID.
    pub id: String,
    /// When the checkpoint was created.
    pub created_at: DateTime<Utc>,
    /// Current goal or user request.
    pub goal: String,
    /// Completed high-level steps.
    #[serde(default)]
    pub completed_steps: Vec<String>,
    /// Important decisions made so far.
    #[serde(default)]
    pub decisions: Vec<String>,
    /// Resources touched since the prior checkpoint or run start.
    #[serde(default)]
    pub touched_resources: Vec<TouchedResource>,
    /// Failures encountered and worth preserving across compaction/resume.
    #[serde(default)]
    pub failures: Vec<FailureRecord>,
    /// Verification records supporting resume/review.
    #[serde(default)]
    pub verification_results: Vec<VerificationResult>,
    /// Questions or blockers that remain unresolved.
    #[serde(default)]
    pub unresolved_questions: Vec<String>,
    /// Recommended next step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_step: Option<String>,
    /// Optional host-specific metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl Checkpoint {
    /// Create a checkpoint with the current timestamp.
    pub fn new(id: impl Into<String>, goal: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            created_at: Utc::now(),
            goal: goal.into(),
            completed_steps: Vec::new(),
            decisions: Vec::new(),
            touched_resources: Vec::new(),
            failures: Vec::new(),
            verification_results: Vec::new(),
            unresolved_questions: Vec::new(),
            next_step: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Add a completed step.
    pub fn add_completed_step(mut self, step: impl Into<String>) -> Self {
        self.completed_steps.push(step.into());
        self
    }

    /// Add a decision.
    pub fn add_decision(mut self, decision: impl Into<String>) -> Self {
        self.decisions.push(decision.into());
        self
    }

    /// Add a touched resource.
    pub fn add_touched_resource(mut self, resource: TouchedResource) -> Self {
        self.touched_resources.push(resource);
        self
    }

    /// Add a failure.
    pub fn add_failure(mut self, failure: FailureRecord) -> Self {
        self.failures.push(failure);
        self
    }

    /// Add a verification result.
    pub fn add_verification_result(mut self, result: VerificationResult) -> Self {
        self.verification_results.push(result);
        self
    }

    /// Add an unresolved question or blocker.
    pub fn add_unresolved_question(mut self, question: impl Into<String>) -> Self {
        self.unresolved_questions.push(question.into());
        self
    }

    /// Set the recommended next step.
    pub fn with_next_step(mut self, next_step: impl Into<String>) -> Self {
        self.next_step = Some(next_step.into());
        self
    }

    /// Attach metadata while preserving builder-style construction.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ErrorKind, ResourceOperation, VerificationStatus};

    #[test]
    fn constructs_checkpoint_for_resume() {
        let checkpoint = Checkpoint::new("checkpoint-1", "Add telemetry primitives")
            .add_completed_step("Added trajectory module")
            .add_decision("Keep persistence host-owned")
            .add_touched_resource(TouchedResource::new(
                "file",
                "lib/src/trajectory.rs",
                ResourceOperation::Write,
            ))
            .add_failure(
                FailureRecord::new(
                    ErrorKind::ShellSyntax,
                    "Used Bash heredoc syntax in PowerShell",
                )
                .with_source("shell"),
            )
            .add_verification_result(VerificationResult::new(
                "cargo test",
                VerificationStatus::Passed,
                "All unit tests passed",
            ))
            .add_unresolved_question("When should hosts emit durable checkpoints?")
            .with_next_step("Wire trajectory events into runner callbacks");

        assert_eq!(checkpoint.completed_steps.len(), 1);
        assert_eq!(checkpoint.touched_resources[0].kind, "file");
        assert_eq!(checkpoint.failures[0].kind, ErrorKind::ShellSyntax);
        assert_eq!(
            checkpoint.next_step.as_deref(),
            Some("Wire trajectory events into runner callbacks")
        );
    }

    #[test]
    fn serializes_checkpoint() {
        let checkpoint =
            Checkpoint::new("checkpoint-1", "Resume safely").with_next_step("Run verification");
        let json = serde_json::to_string(&checkpoint).unwrap();
        assert!(json.contains("\"id\":\"checkpoint-1\""));
        assert!(json.contains("\"next_step\":\"Run verification\""));
        let round_trip: Checkpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip.id, checkpoint.id);
        assert_eq!(round_trip.goal, checkpoint.goal);
    }
}
