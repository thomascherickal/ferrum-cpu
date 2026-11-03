//! Cancel-on-disconnect: axum drops a handler's future when the client closes the connection,
//! so a guard held across the wait for the model sets the request's cancel flag exactly then.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Sets its flag when dropped, unless [`CancelOnDrop::disarm`]ed first.
pub(crate) struct CancelOnDrop(Option<Arc<AtomicBool>>);

impl CancelOnDrop {
    /// A fresh flag (hand it to the job) and the armed guard that owns the other end.
    pub(crate) fn new() -> (Self, Arc<AtomicBool>) {
        let flag = Arc::new(AtomicBool::new(false));
        (Self(Some(flag.clone())), flag)
    }

    /// The request finished: dropping the guard no longer cancels anything.
    pub(crate) fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(f) = &self.0 {
            f.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_before_completion_sets_the_flag_and_completion_does_not() {
        let (guard, flag) = CancelOnDrop::new();
        assert!(!flag.load(Ordering::Relaxed));
        drop(guard);
        assert!(flag.load(Ordering::Relaxed));
        let (guard, flag) = CancelOnDrop::new();
        guard.disarm();
        assert!(!flag.load(Ordering::Relaxed));
    }
}
