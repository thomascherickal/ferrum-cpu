//! A graceful-stop request for long runs: an optional deadline, an optional
//! stop file and a shared cancel flag. Trainers poll [`StopCondition::should_stop`]
//! once per optimizer step and, when it answers, finish the step, checkpoint and
//! return, so an interrupted run resumes instead of being lost.
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Why a run was asked to stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopCause {
    /// The deadline passed (a `--max-seconds` budget).
    TimeBudget,
    /// The stop file exists.
    StopFile,
    /// The cancel flag was set (for example by a GUI button).
    Cancelled,
}

impl fmt::Display for StopCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            StopCause::TimeBudget => "time budget reached",
            StopCause::StopFile => "stop file found",
            StopCause::Cancelled => "cancelled",
        })
    }
}

/// When to stop. `Default` never stops. Cloning shares the cancel flag and the
/// record of where the run stopped.
#[derive(Clone, Debug, Default)]
pub struct StopCondition {
    deadline: Option<Instant>,
    stop_file: Option<PathBuf>,
    cancel: Arc<AtomicBool>,
    stopped: Arc<Mutex<Option<(StopCause, u64)>>>,
}

impl PartialEq for StopCondition {
    fn eq(&self, o: &Self) -> bool {
        self.deadline == o.deadline
            && self.stop_file == o.stop_file
            && Arc::ptr_eq(&self.cancel, &o.cancel)
    }
}

impl StopCondition {
    /// Never stops (same as `Default`).
    pub fn none() -> Self {
        Self::default()
    }

    /// Stop once `budget` has passed from now.
    pub fn with_deadline(mut self, budget: Duration) -> Self {
        self.deadline = Some(Instant::now() + budget);
        self
    }

    /// Stop while `path` exists.
    pub fn with_stop_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.stop_file = Some(path.into());
        self
    }

    /// Stop when `flag` becomes `true`.
    pub fn with_cancel(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel = flag;
        self
    }

    /// The shared cancel flag; set it to `true` from any thread to stop.
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// The stop file, if one is watched.
    pub fn stop_file(&self) -> Option<&Path> {
        self.stop_file.as_deref()
    }

    /// The first stop request that applies right now: cancel flag, then stop
    /// file, then deadline. Cheap (an atomic load, a `stat`, a clock read).
    pub fn should_stop(&self) -> Option<StopCause> {
        if self.cancel.load(Ordering::Relaxed) {
            Some(StopCause::Cancelled)
        } else if self.stop_file.as_deref().is_some_and(Path::exists) {
            Some(StopCause::StopFile)
        } else if self.deadline.is_some_and(|d| Instant::now() >= d) {
            Some(StopCause::TimeBudget)
        } else {
            None
        }
    }

    /// Remove the stop file (a missing file is fine).
    pub fn clear_stop_file(&self) {
        if let Some(p) = &self.stop_file {
            let _ = std::fs::remove_file(p);
        }
    }

    /// Record that a run stopped after `step` optimizer steps.
    pub(crate) fn note_stop(&self, cause: StopCause, step: u64) {
        *lock_mutex(&self.stopped) = Some((cause, step));
    }

    /// Forget a previous stop (a run does this when it starts).
    pub(crate) fn clear_record(&self) {
        *lock_mutex(&self.stopped) = None;
    }

    /// Why and at which step the last run using this condition stopped early,
    /// or `None` if it ran to completion.
    pub fn stopped(&self) -> Option<(StopCause, u64)> {
        *lock_mutex(&self.stopped)
    }
}

fn lock_mutex<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ferrum_stop_{name}_{}", std::process::id()))
    }

    #[test]
    fn default_never_stops() {
        assert_eq!(StopCondition::default().should_stop(), None);
        assert_eq!(StopCondition::none().stopped(), None);
    }

    #[test]
    fn deadline_fires_after_the_budget() {
        let long = StopCondition::none().with_deadline(Duration::from_secs(3600));
        assert_eq!(long.should_stop(), None);
        let zero = StopCondition::none().with_deadline(Duration::ZERO);
        assert_eq!(zero.should_stop(), Some(StopCause::TimeBudget));
        // Fractional budgets work.
        let tiny = StopCondition::none().with_deadline(Duration::from_secs_f64(0.02));
        assert_eq!(tiny.should_stop(), None);
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(tiny.should_stop(), Some(StopCause::TimeBudget));
    }

    #[test]
    fn stop_file_fires_while_it_exists_and_clears() {
        let p = tmp("file");
        let _ = std::fs::remove_file(&p);
        assert_eq!(StopCondition::none().stop_file(), None);
        let c = StopCondition::none().with_stop_file(&p);
        assert_eq!(c.stop_file(), Some(p.as_path()));
        assert_eq!(c.should_stop(), None);
        std::fs::write(&p, b"").unwrap();
        assert_eq!(c.should_stop(), Some(StopCause::StopFile));
        c.clear_stop_file();
        assert_eq!(c.should_stop(), None);
        c.clear_stop_file(); // already gone: no panic
    }

    #[test]
    fn cancel_flag_is_shared_across_clones_and_wins() {
        let c = StopCondition::none().with_deadline(Duration::ZERO);
        let c2 = c.clone();
        assert_eq!(c.should_stop(), Some(StopCause::TimeBudget));
        c2.cancel_flag().store(true, Ordering::Relaxed);
        assert_eq!(c.should_stop(), Some(StopCause::Cancelled));
        let own = Arc::new(AtomicBool::new(false));
        let d = StopCondition::none().with_cancel(own.clone());
        assert_eq!(d.should_stop(), None);
        own.store(true, Ordering::Relaxed);
        assert_eq!(d.should_stop(), Some(StopCause::Cancelled));
    }

    #[test]
    fn stop_record_is_shared_and_equality_follows_the_flag() {
        let c = StopCondition::none();
        let c2 = c.clone();
        c.note_stop(StopCause::StopFile, 9);
        assert_eq!(c2.stopped(), Some((StopCause::StopFile, 9)));
        c.clear_record();
        assert_eq!(c2.stopped(), None);
        c.note_stop(StopCause::StopFile, 9);
        assert_eq!(c, c2);
        assert_ne!(c, StopCondition::none());
        assert_eq!(StopCause::Cancelled.to_string(), "cancelled");

        let m = std::sync::Mutex::new(42);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = m.lock().unwrap();
            panic!("poisoning");
        }));
        assert_eq!(*lock_mutex(&m), 42);
    }
}
