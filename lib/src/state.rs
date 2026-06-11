//! Host-neutral state primitives for agent-loop observability.
//!
//! These types intentionally avoid persistence, UI, or domain assumptions. Hosts
//! can serialize them into their own stores and attach domain-specific metadata
//! through the `metadata` fields.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Structured failure categories that hosts can use for retries, UI, and review.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// A shell command used syntax for the wrong shell or malformed syntax.
    ShellSyntax,
    /// An operation exceeded its time budget.
    Timeout,
    /// The host or user denied permission.
    Permission,
    /// Tool arguments, model output, or persisted state failed schema validation.
    SchemaError,
    /// Content could not be read because an exclusion policy blocked it.
    ContentExclusion,
    /// The requested command or executable was not found.
    CommandNotFound,
    /// A validation or verification criterion failed.
    ValidationFailed,
    /// A model provider returned an error.
    ProviderError,
    /// A tool failed while executing.
    ToolError,
    /// The run or operation was cancelled.
    Cancellation,
    /// The failure is known but does not fit a more specific category.
    Unknown,
}

impl ErrorKind {
    /// Stable snake_case label for telemetry attributes and storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ShellSyntax => "shell_syntax",
            Self::Timeout => "timeout",
            Self::Permission => "permission",
            Self::SchemaError => "schema_error",
            Self::ContentExclusion => "content_exclusion",
            Self::CommandNotFound => "command_not_found",
            Self::ValidationFailed => "validation_failed",
            Self::ProviderError => "provider_error",
            Self::ToolError => "tool_error",
            Self::Cancellation => "cancellation",
            Self::Unknown => "unknown",
        }
    }

    /// Best-effort classification for generic error messages.
    ///
    /// Hosts should prefer explicit error kinds when available. This helper is a
    /// dependency-light fallback for adapting unstructured tool/provider errors.
    pub fn classify(message: impl AsRef<str>) -> Self {
        let normalized = message.as_ref().to_ascii_lowercase();

        if normalized.contains("content exclusion")
            || normalized.contains("excluded by policy")
            || normalized.contains("restricted by policy")
        {
            Self::ContentExclusion
        } else if normalized.contains("timed out")
            || normalized.contains("timeout")
            || normalized.contains("deadline")
        {
            Self::Timeout
        } else if normalized.contains("permission denied")
            || normalized.contains("access is denied")
            || normalized.contains("not permitted")
            || normalized.contains("unauthorized")
            || normalized.contains("forbidden")
        {
            Self::Permission
        } else if normalized.contains("command not found")
            || normalized.contains("not recognized as")
            || normalized.contains("is not recognized")
            || normalized.contains("no such file or directory")
        {
            Self::CommandNotFound
        } else if normalized.contains("parsererror")
            || normalized.contains("syntax error")
            || normalized.contains("missing file specification")
            || normalized.contains("unexpected token")
            || normalized.contains("unexpected end of file")
        {
            Self::ShellSyntax
        } else if normalized.contains("schema")
            || normalized.contains("invalid type")
            || normalized.contains("missing required")
            || normalized.contains("deserialize")
        {
            Self::SchemaError
        } else if normalized.contains("validation failed")
            || normalized.contains("assertion failed")
            || normalized.contains("test failed")
            || normalized.contains("criterion failed")
        {
            Self::ValidationFailed
        } else if normalized.contains("provider")
            || normalized.contains("api error")
            || normalized.contains("http error")
            || normalized.contains("rate limit")
        {
            Self::ProviderError
        } else if normalized.contains("cancelled") || normalized.contains("canceled") {
            Self::Cancellation
        } else if normalized.contains("tool error") || normalized.contains("tool failed") {
            Self::ToolError
        } else {
            Self::Unknown
        }
    }
}

/// A resource operation observed during an agent run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResourceOperation {
    Read,
    Write,
    Create,
    Update,
    Delete,
    Execute,
    Search,
    Inspect,
    Reference,
    Custom,
}

/// A host-neutral resource touched by an agent run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TouchedResource {
    /// Host-defined resource kind, such as `file`, `url`, `database_row`, or `tool`.
    pub kind: String,
    /// Stable identifier, path, or URI for the resource.
    pub id: String,
    /// Operation performed against the resource.
    pub operation: ResourceOperation,
    /// Optional host-specific metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl TouchedResource {
    /// Create a touched-resource record.
    pub fn new(
        kind: impl Into<String>,
        id: impl Into<String>,
        operation: ResourceOperation,
    ) -> Self {
        Self {
            kind: kind.into(),
            id: id.into(),
            operation,
            metadata: BTreeMap::new(),
        }
    }

    /// Attach metadata while preserving builder-style construction.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Status for a verification criterion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    Passed,
    Failed,
    Skipped,
    Blocked,
    Unknown,
}

/// Evidence that a result was checked against a criterion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerificationResult {
    /// The criterion being verified.
    pub criterion: String,
    /// Verification outcome.
    pub status: VerificationStatus,
    /// Human-readable evidence summary, command output summary, or artifact ref.
    pub evidence_summary: String,
    /// Structured failure kind when status is failed or blocked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<ErrorKind>,
}

/// A host-neutral suggestion that some fact may be worth promoting to memory.
///
/// Agentive does not decide persistence policy. Hosts can inspect these
/// candidates and choose whether to save, reject, defer, redact, or transform
/// them into their own memory store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryPromotionCandidate {
    /// Concise, non-secret summary of the candidate memory.
    pub content_summary: String,
    /// Host-defined category such as `core`, `insight`, or `archival`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Tags that help hosts deduplicate, route, or recall the candidate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Optional confidence score from 0.0 to 1.0, encoded as basis points to
    /// avoid floating-point wire-format surprises.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_basis_points: Option<u16>,
    /// Optional host-specific metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl MemoryPromotionCandidate {
    /// Construct a memory promotion candidate from a safe summary.
    pub fn new(content_summary: impl Into<String>) -> Self {
        Self {
            content_summary: content_summary.into(),
            category: None,
            tags: Vec::new(),
            confidence_basis_points: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Attach a host-defined category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Attach a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Attach a confidence score, clamped to the range 0.0..=1.0.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        let clamped = confidence.clamp(0.0, 1.0);
        self.confidence_basis_points = Some((clamped * 10_000.0).round() as u16);
        self
    }

    /// Attach metadata while preserving builder-style construction.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Host decision after considering a memory promotion candidate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum MemoryPromotionOutcome {
    /// The host accepted and persisted/promoted the candidate.
    Accepted {
        #[serde(skip_serializing_if = "Option::is_none")]
        memory_id: Option<String>,
    },
    /// The host rejected the candidate.
    Rejected {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// The host deferred a decision.
    Deferred {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// The host hook failed while considering the candidate.
    Failed {
        failure_kind: ErrorKind,
        reason: String,
    },
}

impl VerificationResult {
    /// Construct a verification result.
    pub fn new(
        criterion: impl Into<String>,
        status: VerificationStatus,
        evidence_summary: impl Into<String>,
    ) -> Self {
        Self {
            criterion: criterion.into(),
            status,
            evidence_summary: evidence_summary.into(),
            failure_kind: None,
        }
    }

    /// Attach a structured failure kind.
    pub fn with_failure_kind(mut self, failure_kind: ErrorKind) -> Self {
        self.failure_kind = Some(failure_kind);
        self
    }
}

/// A failure captured in a trajectory or checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailureRecord {
    /// Structured failure category.
    pub kind: ErrorKind,
    /// Concise failure summary safe for user-facing review.
    pub summary: String,
    /// Optional operation, tool, or phase where the failure occurred.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl FailureRecord {
    /// Construct a failure record.
    pub fn new(kind: ErrorKind, summary: impl Into<String>) -> Self {
        Self {
            kind,
            summary: summary.into(),
            source: None,
        }
    }

    /// Attach the source operation, tool, or phase.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

/// Lightweight status for plan steps.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    Pending,
    InProgress,
    Done,
    Blocked,
    Skipped,
}

/// A lightweight structured plan step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanStep {
    /// Stable host- or agent-generated ID.
    pub id: String,
    /// Short action-oriented title.
    pub title: String,
    /// Current step status.
    pub status: PlanStepStatus,
    /// Step IDs that must complete first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
    /// Optional details or acceptance notes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl PlanStep {
    /// Create a pending plan step.
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            status: PlanStepStatus::Pending,
            depends_on: Vec::new(),
            description: None,
        }
    }

    /// Set the current status.
    pub fn with_status(mut self, status: PlanStepStatus) -> Self {
        self.status = status;
        self
    }

    /// Add a dependency on another step ID.
    pub fn depends_on(mut self, step_id: impl Into<String>) -> Self {
        self.depends_on.push(step_id.into());
        self
    }

    /// Attach descriptive details.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Lightweight plan status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Draft,
    Active,
    Completed,
    Blocked,
    Abandoned,
}

/// A host-neutral structured plan for agent-loop state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Plan {
    /// Stable host- or agent-generated plan ID.
    pub id: String,
    /// Plan goal.
    pub goal: String,
    /// Overall plan status.
    pub status: PlanStatus,
    /// Ordered plan steps.
    #[serde(default)]
    pub steps: Vec<PlanStep>,
}

impl Plan {
    /// Create a draft plan.
    pub fn new(id: impl Into<String>, goal: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            goal: goal.into(),
            status: PlanStatus::Draft,
            steps: Vec::new(),
        }
    }

    /// Set the current plan status.
    pub fn with_status(mut self, status: PlanStatus) -> Self {
        self.status = status;
        self
    }

    /// Append a step to the plan.
    pub fn add_step(mut self, step: PlanStep) -> Self {
        self.steps.push(step);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_common_error_messages() {
        assert_eq!(
            ErrorKind::classify(
                "ParserError: Missing file specification after redirection operator."
            ),
            ErrorKind::ShellSyntax
        );
        assert_eq!(
            ErrorKind::classify("operation timed out"),
            ErrorKind::Timeout
        );
        assert_eq!(
            ErrorKind::classify("content restricted by policy"),
            ErrorKind::ContentExclusion
        );
        assert_eq!(
            ErrorKind::classify("cargo: command not found"),
            ErrorKind::CommandNotFound
        );
    }

    #[test]
    fn serializes_plan_and_resource_primitives() {
        let plan = Plan::new("plan-1", "Add observability")
            .with_status(PlanStatus::Active)
            .add_step(
                PlanStep::new("step-1", "Add trajectory module")
                    .with_status(PlanStepStatus::InProgress)
                    .with_description("Define serializable events"),
            );

        let json = serde_json::to_string(&plan).unwrap();
        assert!(json.contains("\"status\":\"active\""));
        let round_trip: Plan = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip, plan);

        let resource = TouchedResource::new("file", "lib/src/state.rs", ResourceOperation::Write)
            .with_metadata("language", "rust");
        let resource_json = serde_json::to_string(&resource).unwrap();
        let resource_round_trip: TouchedResource = serde_json::from_str(&resource_json).unwrap();
        assert_eq!(resource_round_trip, resource);

        let custom_resource = TouchedResource::new("queue", "jobs:42", ResourceOperation::Custom);
        let custom_json = serde_json::to_string(&custom_resource).unwrap();
        assert!(custom_json.contains("\"operation\":\"custom\""));
    }

    #[test]
    fn serializes_memory_promotion_candidate() {
        let candidate = MemoryPromotionCandidate::new("User prefers concise summaries")
            .with_category("core")
            .with_tag("preference")
            .with_confidence(0.875)
            .with_metadata("source", "tool");

        let json = serde_json::to_string(&candidate).unwrap();
        assert!(json.contains("\"content_summary\":\"User prefers concise summaries\""));
        assert!(json.contains("\"confidence_basis_points\":8750"));
        let round_trip: MemoryPromotionCandidate = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip, candidate);
    }
}
