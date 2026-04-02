use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cooperative cancellation token for stopping long-running operations.
///
/// Share the token between the caller (who cancels) and the runner/provider
/// (who checks). Based on `Arc<AtomicBool>` so cloning is cheap.
#[derive(Debug, Clone)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Create a new token in the non-cancelled state.
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Signal cancellation. All clones of this token will observe it.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Reset the token to non-cancelled state (for reuse).
    pub fn reset(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_token_lifecycle() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        let clone = token.clone();
        token.cancel();
        assert!(token.is_cancelled());
        assert!(clone.is_cancelled());

        token.reset();
        assert!(!token.is_cancelled());
        assert!(!clone.is_cancelled());
    }
}
