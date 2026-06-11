//! Host-neutral resume context helpers.
//!
//! These helpers turn structured checkpoints, plans, and recent messages into a
//! compact context block that hosts can inject after compaction or session
//! resume. They intentionally do not know how checkpoints are persisted.

use crate::checkpoint::Checkpoint;
use crate::state::{Plan, PlanStepStatus};
use crate::types::ChatMessage;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Structured input for resuming an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeContext {
    /// When this resume context was assembled.
    pub generated_at: DateTime<Utc>,
    /// Latest durable checkpoint.
    pub checkpoint: Checkpoint,
    /// Optional structured plan state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<Plan>,
    /// Recent messages retained as conversational continuity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent_messages: Vec<ChatMessage>,
}

impl ResumeContext {
    /// Create a resume context from a checkpoint.
    pub fn new(checkpoint: Checkpoint) -> Self {
        Self {
            generated_at: Utc::now(),
            checkpoint,
            plan: None,
            recent_messages: Vec::new(),
        }
    }

    /// Attach structured plan state.
    pub fn with_plan(mut self, plan: Plan) -> Self {
        self.plan = Some(plan);
        self
    }

    /// Attach the last `limit` messages from a conversation.
    pub fn with_recent_messages(mut self, messages: &[ChatMessage], limit: usize) -> Self {
        let start = messages.len().saturating_sub(limit);
        self.recent_messages = messages[start..].to_vec();
        self
    }

    /// Render a compact XML-like context block for injection into a prompt.
    pub fn to_prompt_block(&self) -> String {
        let mut out = String::from("<agentive_resume_context>\n");
        push_tag(&mut out, "goal", &self.checkpoint.goal);

        if !self.checkpoint.completed_steps.is_empty() {
            out.push_str("<completed_steps>\n");
            for step in &self.checkpoint.completed_steps {
                push_list_item(&mut out, step);
            }
            out.push_str("</completed_steps>\n");
        }

        if !self.checkpoint.decisions.is_empty() {
            out.push_str("<decisions>\n");
            for decision in &self.checkpoint.decisions {
                push_list_item(&mut out, decision);
            }
            out.push_str("</decisions>\n");
        }

        if !self.checkpoint.touched_resources.is_empty() {
            out.push_str("<touched_resources>\n");
            for resource in &self.checkpoint.touched_resources {
                out.push_str(&format!(
                    "- kind={} operation={:?} id={}\n",
                    escape_text(&resource.kind),
                    resource.operation,
                    escape_text(&resource.id)
                ));
            }
            out.push_str("</touched_resources>\n");
        }

        if !self.checkpoint.failures.is_empty() {
            out.push_str("<failures>\n");
            for failure in &self.checkpoint.failures {
                out.push_str(&format!(
                    "- kind={} summary={}\n",
                    failure.kind.as_str(),
                    escape_text(&failure.summary)
                ));
            }
            out.push_str("</failures>\n");
        }

        if !self.checkpoint.verification_results.is_empty() {
            out.push_str("<verification_results>\n");
            for result in &self.checkpoint.verification_results {
                out.push_str(&format!(
                    "- criterion={} status={:?} evidence={}\n",
                    escape_text(&result.criterion),
                    result.status,
                    escape_text(&result.evidence_summary)
                ));
            }
            out.push_str("</verification_results>\n");
        }

        if !self.checkpoint.unresolved_questions.is_empty() {
            out.push_str("<unresolved_questions>\n");
            for question in &self.checkpoint.unresolved_questions {
                push_list_item(&mut out, question);
            }
            out.push_str("</unresolved_questions>\n");
        }

        if let Some(next_step) = &self.checkpoint.next_step {
            push_tag(&mut out, "next_step", next_step);
        }

        if let Some(plan) = &self.plan {
            out.push_str("<plan>\n");
            out.push_str(&format!(
                "status={:?} goal={}\n",
                plan.status,
                escape_text(&plan.goal)
            ));
            for step in &plan.steps {
                let marker = match step.status {
                    PlanStepStatus::Pending => "pending",
                    PlanStepStatus::InProgress => "in_progress",
                    PlanStepStatus::Done => "done",
                    PlanStepStatus::Blocked => "blocked",
                    PlanStepStatus::Skipped => "skipped",
                };
                out.push_str(&format!(
                    "- [{}] {}: {}\n",
                    marker,
                    escape_text(&step.id),
                    escape_text(&step.title)
                ));
            }
            out.push_str("</plan>\n");
        }

        if !self.recent_messages.is_empty() {
            out.push_str("<recent_messages>\n");
            for message in &self.recent_messages {
                if let Some(text) = message.text() {
                    out.push_str(&format!(
                        "<message role=\"{}\">{}</message>\n",
                        escape_text(&message.role),
                        escape_text(text)
                    ));
                }
            }
            out.push_str("</recent_messages>\n");
        }

        out.push_str("</agentive_resume_context>");
        out
    }
}

fn push_tag(out: &mut String, tag: &str, value: &str) {
    out.push_str(&format!("<{tag}>{}</{tag}>\n", escape_text(value)));
}

fn push_list_item(out: &mut String, value: &str) {
    out.push_str(&format!("- {}\n", escape_text(value)));
}

fn escape_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{PlanStep, ResourceOperation, TouchedResource, VerificationResult};

    #[test]
    fn renders_resume_prompt_block_from_checkpoint_plan_and_recent_messages() {
        let checkpoint = Checkpoint::new("checkpoint-1", "Resume work")
            .add_completed_step("Added trajectory events")
            .add_decision("Keep SQLite host-owned")
            .add_touched_resource(TouchedResource::new(
                "file",
                "lib/src/runner.rs",
                ResourceOperation::Write,
            ))
            .add_verification_result(VerificationResult::new(
                "cargo test",
                crate::state::VerificationStatus::Passed,
                "All tests passed",
            ))
            .with_next_step("Continue with resource metadata");
        let plan = Plan::new("plan-1", "Add resume helpers").add_step(
            PlanStep::new("step-1", "Render resume context")
                .with_status(PlanStepStatus::InProgress),
        );
        let messages = vec![
            ChatMessage::user("older"),
            ChatMessage::assistant("recent & safe"),
        ];

        let block = ResumeContext::new(checkpoint)
            .with_plan(plan)
            .with_recent_messages(&messages, 1)
            .to_prompt_block();

        assert!(block.contains("<agentive_resume_context>"));
        assert!(block.contains("Keep SQLite host-owned"));
        assert!(block.contains("lib/src/runner.rs"));
        assert!(block.contains("recent &amp; safe"));
        assert!(!block.contains("older"));
    }
}
