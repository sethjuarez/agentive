//! Steering — inject user messages into a running agent loop.
//!
//! When the agent is in the middle of its tool-call loop (thinking, executing
//! tools, waiting for the LLM), the user may want to add context or redirect:
//!
//! > "Actually, use the other file"
//! > "Focus on the error handling part"
//!
//! The [`Steering`] handle lets the app push messages that the runner will
//! drain and append before the next LLM call.
//!
//! # Example
//! ```no_run
//! use agentive::Steering;
//!
//! let steering = Steering::new();
//!
//! // Give a clone to the runner (via run())
//! let runner_steering = steering.clone();
//!
//! // Meanwhile, from your UI thread:
//! steering.send("Actually, skip the tests and just fix the bug");
//! ```

use std::sync::{Arc, Mutex};

/// A thread-safe handle for injecting user messages into a running agent loop.
///
/// The runner drains pending messages at the top of each tool-call iteration,
/// appending them as user messages before calling the LLM again.
#[derive(Debug, Clone)]
pub struct Steering {
    queue: Arc<Mutex<Vec<String>>>,
}

impl Steering {
    /// Create a new empty steering handle.
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Send a message to the running agent. It will be appended as a user
    /// message before the next LLM call.
    pub fn send(&self, message: &str) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.push(message.to_string());
        }
    }

    /// Drain all pending messages (called internally by the runner).
    pub(crate) fn drain(&self) -> Vec<String> {
        if let Ok(mut queue) = self.queue.lock() {
            queue.drain(..).collect()
        } else {
            Vec::new()
        }
    }

    /// Check if there are pending messages without consuming them.
    pub fn has_pending(&self) -> bool {
        self.queue
            .lock()
            .map(|q| !q.is_empty())
            .unwrap_or(false)
    }
}

impl Default for Steering {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_and_drain() {
        let steering = Steering::new();
        assert!(!steering.has_pending());

        steering.send("redirect A");
        steering.send("redirect B");
        assert!(steering.has_pending());

        let msgs = steering.drain();
        assert_eq!(msgs, vec!["redirect A", "redirect B"]);
        assert!(!steering.has_pending());
    }

    #[test]
    fn test_clone_shares_queue() {
        let s1 = Steering::new();
        let s2 = s1.clone();

        s1.send("from s1");
        s2.send("from s2");

        let msgs = s1.drain();
        assert_eq!(msgs, vec!["from s1", "from s2"]);
    }

    #[test]
    fn test_drain_empty() {
        let steering = Steering::new();
        let msgs = steering.drain();
        assert!(msgs.is_empty());
    }
}
