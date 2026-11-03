//! Ctrl-C for `train` and `finetune-gguf`: the process-wide interrupt flag. It knows
//! nothing about models or training; [`crate::stop::condition`] hands the flag to the
//! run's [`ferrum_core::StopCondition`] as its cancel source, so an interrupt stops the
//! run the same way the stop file does.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

/// Set once, by the handler install; read by every stop condition built afterwards.
static FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// The installed interrupt flag, or `None` when no handler was installed (every command
/// but `train` and `finetune-gguf`, and every unit test).
pub(crate) fn installed() -> Option<Arc<AtomicBool>> {
    FLAG.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_installed_unless_a_command_asks() {
        assert!(installed().is_none());
    }
}
