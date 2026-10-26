//! Job control: one running slot per kind of long job, each with a cancel flag.
//!
//! A command takes its slot with [`Jobs::start`] before it spawns the blocking
//! task, so a second start of the same kind fails at once with a clear message.
//! The returned [`JobGuard`] frees the slot when the task ends (also on error or
//! panic) and carries the [`StopCondition`] that the engine polls once per
//! optimizer step. `cancel_job` sets the flag behind it.

use ferrum_core::StopCondition;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tauri::State;

use crate::AppState;

/// The kinds of job that can run, one at a time each.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Job {
    Train,
    Finetune,
    Export,
    Generate,
    Evaluate,
    Run,
}

impl Job {
    /// "a training job" in "a training job is already running".
    fn noun(self) -> &'static str {
        match self {
            Job::Train => "a training job",
            Job::Finetune => "a fine-tuning job",
            Job::Export => "an export job",
            Job::Generate => "a generation job",
            Job::Evaluate => "an evaluation job",
            Job::Run => "a GGUF run",
        }
    }
}

/// The slots. Cloning shares them (the state holds one, each running task another).
#[derive(Clone, Default)]
pub struct Jobs(Arc<Mutex<HashMap<Job, StopCondition>>>);

impl Jobs {
    /// Take the slot of `kind`, or fail if a job of that kind is already running.
    pub fn start(&self, kind: Job) -> Result<JobGuard, String> {
        let mut slots = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if slots.contains_key(&kind) {
            return Err(format!("{} is already running", kind.noun()));
        }
        let stop = StopCondition::none();
        slots.insert(kind, stop.clone());
        Ok(JobGuard {
            jobs: self.clone(),
            kind,
            stop,
        })
    }

    /// Ask the running job of `kind` to stop. `false` when none is running.
    pub fn cancel(&self, kind: Job) -> bool {
        let slots = self.0.lock().unwrap_or_else(|e| e.into_inner());
        slots.get(&kind).is_some_and(|stop| {
            stop.cancel_flag()
                .store(true, std::sync::atomic::Ordering::Relaxed);
            true
        })
    }
}

/// A taken slot. Dropping it frees the slot.
pub struct JobGuard {
    jobs: Jobs,
    kind: Job,
    stop: StopCondition,
}

impl JobGuard {
    /// The condition the engine polls (shares the cancel flag with `cancel_job`).
    pub fn stop(&self) -> &StopCondition {
        &self.stop
    }

    /// Has a cancel been requested?
    pub fn cancelled(&self) -> bool {
        self.stop.should_stop().is_some()
    }
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        self.jobs
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.kind);
    }
}

/// Stop the running job of `kind` (`"train"`, `"finetune"`, `"export"`,
/// `"generate"`, `"evaluate"` or `"run"`). Returns whether one was running.
#[tauri::command]
pub fn cancel_job(state: State<'_, AppState>, kind: Job) -> Result<bool, String> {
    Ok(state.jobs.cancel(kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_slot_per_kind_freed_on_drop() {
        let jobs = Jobs::default();
        let train = jobs.start(Job::Train).unwrap();
        let e = jobs.start(Job::Train).err().unwrap();
        assert_eq!(e, "a training job is already running");
        // Other kinds are independent.
        let gen = jobs.start(Job::Generate).unwrap();
        drop(gen);
        assert!(jobs.start(Job::Generate).is_ok());
        drop(train);
        assert!(jobs.start(Job::Train).is_ok());
    }

    #[test]
    fn cancel_sets_the_flag_of_the_running_job_only() {
        let jobs = Jobs::default();
        assert!(!jobs.cancel(Job::Export), "nothing running");
        let export = jobs.start(Job::Export).unwrap();
        let run = jobs.start(Job::Run).unwrap();
        assert!(!export.cancelled());
        assert!(jobs.cancel(Job::Export));
        assert!(export.cancelled());
        assert!(!run.cancelled(), "other kinds are untouched");
        // A new job of the same kind starts clean once the old one is gone.
        drop(export);
        assert!(!jobs.start(Job::Export).unwrap().cancelled());
    }

    #[test]
    fn a_panicking_task_still_frees_its_slot() {
        let jobs = Jobs::default();
        let j = jobs.clone();
        let r = std::thread::spawn(move || {
            let _guard = j.start(Job::Evaluate).unwrap();
            panic!("boom");
        })
        .join();
        assert!(r.is_err());
        assert!(jobs.start(Job::Evaluate).is_ok());
    }

    #[test]
    fn kinds_use_the_names_the_ui_sends() {
        for (name, kind) in [
            ("train", Job::Train),
            ("finetune", Job::Finetune),
            ("export", Job::Export),
            ("generate", Job::Generate),
            ("evaluate", Job::Evaluate),
            ("run", Job::Run),
        ] {
            let k: Job = serde_json::from_value(serde_json::json!(name)).unwrap();
            assert_eq!(k, kind);
        }
        assert!(serde_json::from_value::<Job>(serde_json::json!("nonsense")).is_err());
    }
}
