//! Cooperative cancellation shared by local and hosted training.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Read-only cancellation contract checked at batch and checkpoint boundaries.
pub trait Cancellation: Send + Sync {
    /// Returns true once the caller has requested cancellation.
    fn is_cancelled(&self) -> bool;

    /// Fails at a cooperative cancellation boundary.
    fn checkpoint(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }
}

/// A cooperative cancellation signal.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// Creates an unset token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Permanently marks the token cancelled.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
}

impl Cancellation for CancelToken {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// A cancellation source for synchronous local calls that do not install a
/// signal handler.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeverCancel;

impl Cancellation for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Marker returned when a cooperative boundary observes cancellation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("training cancelled at a cooperative boundary")]
pub struct Cancelled;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_is_monotonic() {
        let token = CancelToken::new();
        assert!(token.checkpoint().is_ok());
        token.cancel();
        assert_eq!(token.checkpoint(), Err(Cancelled));
        assert!(token.is_cancelled());
    }
}
