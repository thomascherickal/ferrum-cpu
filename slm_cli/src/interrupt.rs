//! Ctrl-C for `train` and `finetune-gguf`: the process-wide interrupt flag and its handler.
//! It knows nothing about models or training; [`crate::stop::condition`] hands the flag to
//! the run's [`ferrum_core::StopCondition`] as its cancel source, so the first Ctrl-C stops
//! the run the way the stop file does (finish the step, write the checkpoint). A second
//! Ctrl-C exits at once. Either way the process exits with [`EXIT_STATUS`].
//!
//! Only the binary installs the handler (`main.rs`, for those two commands); the `slm_cli`
//! library never does, so a front end such as the GUI keeps its own signal handling.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

/// The exit status of an interrupted run: 128 + SIGINT, the shell convention.
pub(crate) const EXIT_STATUS: i32 = 130;

/// Set once, by [`install`]; read by every stop condition built afterwards.
static FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// The installed interrupt flag, or `None` when no handler was installed (every command
/// but `train` and `finetune-gguf`).
pub(crate) fn installed() -> Option<Arc<AtomicBool>> {
    FLAG.get().cloned()
}

/// Whether a Ctrl-C arrived since [`install`] (always `false` without a handler).
pub(crate) fn is_set() -> bool {
    installed().is_some_and(|f| f.load(Ordering::SeqCst))
}

/// Whether `cmd` (as typed, aliases included) installs the handler: `train` and
/// `finetune-gguf` only, the two commands that write a resumable checkpoint.
pub(crate) fn wants_handler(cmd: &str) -> bool {
    matches!(cmd, "train" | "finetune-gguf" | "finetune")
}

/// Install the Ctrl-C (SIGINT) handler and return its flag. Once per process: a second call
/// returns the same flag and installs nothing. SIGTERM and SIGHUP are not caught (`ctrlc`'s
/// `termination` feature is off by default and not enabled).
pub(crate) fn install() -> Result<Arc<AtomicBool>, String> {
    if let Some(f) = installed() {
        return Ok(f);
    }
    let flag = Arc::new(AtomicBool::new(false));
    let h = flag.clone();
    // ctrlc runs this on its own thread, not in signal context, so printing and exiting is safe.
    let _ = ctrlc::set_handler(move || {
        if handle_interrupt(&h) {
            std::process::exit(EXIT_STATUS);
        }
    });
    Ok(FLAG.get_or_init(|| flag).clone())
}

pub(crate) fn handle_interrupt(flag: &AtomicBool) -> bool {
    if on_signal(flag) {
        let _ = std::io::stdout().flush();
        true
    } else {
        eprintln!(
            "interrupt: finishing the current step, then stopping (Ctrl-C again to quit now)"
        );
        false
    }
}

/// What one signal does: set the flag. Returns `true` when it was already set (the second
/// Ctrl-C: exit now).
fn on_signal(flag: &AtomicBool) -> bool {
    flag.swap(true, Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_signal_sets_the_flag_and_the_second_asks_to_exit() {
        let f = AtomicBool::new(false);
        assert!(!handle_interrupt(&f));
        assert!(handle_interrupt(&f));
    }

    #[test]
    fn only_the_two_training_commands_want_the_handler() {
        for c in ["train", "finetune-gguf", "finetune"] {
            assert!(wants_handler(c), "{c}");
        }
        for c in [
            "run", "generate", "run-gguf", "eval", "prepare", "tokenize", "info", "",
        ] {
            assert!(!wants_handler(c), "{c}");
        }
    }

    // The one test that installs the real handler in this test process; it never sets the
    // flag, so the stop conditions other tests build stay unaffected.
    #[test]
    fn install_is_once_per_process_and_returns_the_same_clear_flag() {
        let a = install().unwrap();
        let b = install().unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert!(Arc::ptr_eq(&a, &installed().unwrap()));
        assert!(!is_set());
        assert_eq!(EXIT_STATUS, 128 + 2);
    }
}
