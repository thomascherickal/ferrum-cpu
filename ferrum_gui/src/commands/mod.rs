//! Tauri commands: the bridge between the webview and `ferrum_core`.
//!
//! Conventions:
//! - Every command returns `Result<T, CmdError>` (`{ code, message, hint }`, see
//!   [`crate::infra::error`]) so failures surface as a clear, human-readable message with
//!   a hint in the GUI; the inner functions keep returning `String`.
//! - Heavy/blocking work runs inside `spawn_blocking` so the UI thread stays
//!   responsive and events can stream while the command runs.
//!
//! One file per concern; this module re-exports them so the command paths
//! (`commands::train_slm`, ...) stay flat.

mod export;
mod finetune;
mod generate;
mod gguf;
mod gguf_stream;
mod guard;
mod system;
mod text;
mod train;

pub use export::*;
pub use finetune::*;
pub use generate::*;
pub use gguf::*;
pub(crate) use guard::*;
pub use system::*;
pub use text::*;
pub use train::*;

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE)
        | 1
}

pub(crate) fn file_name(path: &str) -> String {
    match Path::new(path).file_name() {
        Some(s) => s.to_string_lossy().into_owned(),
        None => path.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_name_extracts_basename() {
        assert_eq!(file_name("/a/b/model.gguf"), "model.gguf");
        assert_eq!(file_name("model.gguf"), "model.gguf");
        assert_eq!(file_name(""), "");
    }

    #[test]
    fn time_seed_is_nonzero() {
        assert_ne!(time_seed(), 0);
    }
}
