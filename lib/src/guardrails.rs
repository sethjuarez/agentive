//! Guardrails — optional validation hooks for the agentic runner.
//!
//! Guardrails let consuming apps inject validation logic at three points in
//! the runner loop without modifying the runner itself:
//!
//! - **Input guardrail** — runs before each LLM call (can modify messages or abort)
//! - **Output guardrail** — runs after each LLM response (can modify or reject)
//! - **Tool guardrail** — runs before each tool execution (can approve, deny, or modify)
//!
//! All guardrails are optional. When not set, execution proceeds normally.
//!
//! # Example
//! ```
//! use agentive::{Guardrails, GuardrailResult, ChatMessage, ToolCall};
//!
//! let guardrails = Guardrails::new()
//!     .with_input_guardrail(|messages| {
//!         // Block if conversation is too long
//!         if messages.len() > 100 {
//!             GuardrailResult::Deny("Conversation too long".into())
//!         } else {
//!             GuardrailResult::Allow
//!         }
//!     })
//!     .with_tool_guardrail(|tc| {
//!         if tc.function.name == "dangerous_tool" {
//!             GuardrailResult::Deny("Tool not permitted".into())
//!         } else {
//!             GuardrailResult::Allow
//!         }
//!     });
//! ```

use crate::types::{ChatMessage, ToolCall};

/// Result of a guardrail check.
#[derive(Debug, Clone)]
pub enum GuardrailResult {
    /// Allow the operation to proceed unchanged.
    Allow,
    /// Deny the operation with a reason. For tool guardrails, the denial
    /// message is returned as the tool result. For input/output guardrails,
    /// the run is aborted with an error.
    Deny(String),
}

/// Optional validation hooks for the runner.
///
/// All fields default to `None` (no guardrails). Use the builder methods
/// (`with_input_guardrail`, etc.) to set them.
pub struct Guardrails {
    /// Called before each LLM call with the current message history.
    pub input_guardrail: Option<Box<dyn Fn(&[ChatMessage]) -> GuardrailResult + Send + Sync>>,
    /// Called after each LLM response with the assistant message.
    pub output_guardrail: Option<Box<dyn Fn(&ChatMessage) -> GuardrailResult + Send + Sync>>,
    /// Called before each tool execution with the tool call.
    pub tool_guardrail: Option<Box<dyn Fn(&ToolCall) -> GuardrailResult + Send + Sync>>,
}

impl Default for Guardrails {
    fn default() -> Self {
        Self::new()
    }
}

impl Guardrails {
    /// Create empty guardrails (all hooks are `None`).
    pub fn new() -> Self {
        Self {
            input_guardrail: None,
            output_guardrail: None,
            tool_guardrail: None,
        }
    }

    /// Set the input guardrail (runs before each LLM call).
    pub fn with_input_guardrail<F>(mut self, f: F) -> Self
    where
        F: Fn(&[ChatMessage]) -> GuardrailResult + Send + Sync + 'static,
    {
        self.input_guardrail = Some(Box::new(f));
        self
    }

    /// Set the output guardrail (runs after each LLM response).
    pub fn with_output_guardrail<F>(mut self, f: F) -> Self
    where
        F: Fn(&ChatMessage) -> GuardrailResult + Send + Sync + 'static,
    {
        self.output_guardrail = Some(Box::new(f));
        self
    }

    /// Set the tool guardrail (runs before each tool execution).
    pub fn with_tool_guardrail<F>(mut self, f: F) -> Self
    where
        F: Fn(&ToolCall) -> GuardrailResult + Send + Sync + 'static,
    {
        self.tool_guardrail = Some(Box::new(f));
        self
    }

    /// Check input guardrail. Returns `Allow` if no guardrail is set.
    pub(crate) fn check_input(&self, messages: &[ChatMessage]) -> GuardrailResult {
        match &self.input_guardrail {
            Some(f) => f(messages),
            None => GuardrailResult::Allow,
        }
    }

    /// Check output guardrail. Returns `Allow` if no guardrail is set.
    pub(crate) fn check_output(&self, message: &ChatMessage) -> GuardrailResult {
        match &self.output_guardrail {
            Some(f) => f(message),
            None => GuardrailResult::Allow,
        }
    }

    /// Check tool guardrail. Returns `Allow` if no guardrail is set.
    pub(crate) fn check_tool(&self, tool_call: &ToolCall) -> GuardrailResult {
        match &self.tool_guardrail {
            Some(f) => f(tool_call),
            None => GuardrailResult::Allow,
        }
    }
}

impl std::fmt::Debug for Guardrails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Guardrails")
            .field("input_guardrail", &self.input_guardrail.is_some())
            .field("output_guardrail", &self.output_guardrail.is_some())
            .field("tool_guardrail", &self.tool_guardrail.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionCall, ToolCall};

    #[test]
    fn test_default_guardrails_allow_all() {
        let g = Guardrails::new();
        assert!(matches!(g.check_input(&[]), GuardrailResult::Allow));
        assert!(matches!(
            g.check_output(&ChatMessage::assistant("hi")),
            GuardrailResult::Allow
        ));
        let tc = ToolCall {
            id: "1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "test".into(),
                arguments: "{}".into(),
            },
        };
        assert!(matches!(g.check_tool(&tc), GuardrailResult::Allow));
    }

    #[test]
    fn test_input_guardrail_deny() {
        let g = Guardrails::new().with_input_guardrail(|msgs| {
            if msgs.len() > 2 {
                GuardrailResult::Deny("Too many messages".into())
            } else {
                GuardrailResult::Allow
            }
        });

        assert!(matches!(
            g.check_input(&[ChatMessage::user("a"), ChatMessage::user("b")]),
            GuardrailResult::Allow
        ));
        assert!(matches!(
            g.check_input(&[
                ChatMessage::user("a"),
                ChatMessage::user("b"),
                ChatMessage::user("c")
            ]),
            GuardrailResult::Deny(_)
        ));
    }

    #[test]
    fn test_tool_guardrail_deny() {
        let g = Guardrails::new().with_tool_guardrail(|tc| {
            if tc.function.name == "blocked" {
                GuardrailResult::Deny("Not allowed".into())
            } else {
                GuardrailResult::Allow
            }
        });

        let allowed = ToolCall {
            id: "1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "allowed".into(),
                arguments: "{}".into(),
            },
        };
        let blocked = ToolCall {
            id: "2".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "blocked".into(),
                arguments: "{}".into(),
            },
        };
        assert!(matches!(g.check_tool(&allowed), GuardrailResult::Allow));
        assert!(matches!(g.check_tool(&blocked), GuardrailResult::Deny(_)));
    }

    #[test]
    fn test_output_guardrail_deny() {
        let g = Guardrails::new().with_output_guardrail(|msg| {
            if let Some(text) = msg.text() {
                if text.contains("secret") {
                    return GuardrailResult::Deny("Contains secret".into());
                }
            }
            GuardrailResult::Allow
        });

        assert!(matches!(
            g.check_output(&ChatMessage::assistant("hello")),
            GuardrailResult::Allow
        ));
        assert!(matches!(
            g.check_output(&ChatMessage::assistant("the secret is 42")),
            GuardrailResult::Deny(_)
        ));
    }
}
