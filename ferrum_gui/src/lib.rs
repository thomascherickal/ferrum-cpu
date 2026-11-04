//! Ferrum SLM Studio — Tauri backend.
//!
//! The frontend (vanilla HTML/CSS/JS in `ui/`) talks to the `ferrum_core`
//! engine through the `#[tauri::command]` functions in [`commands`]. Long
//! operations run on a blocking task and stream progress/log/output back to the
//! webview as Tauri events:
//!
//! | event           | payload                              | source                |
//! |-----------------|--------------------------------------|-----------------------|
//! | `engine-log`    | `String` (one diagnostic line)       | `--verbose` sink      |
//! | `train-progress`| `{ epoch, total, loss }` (total null under a token budget) | training callback |
//! | `train-done`    | [`commands::TrainResult`] (`stopped`, `stoppedAtStep` after a cancel) | training completion or cancel |
//! | `finetune-progress` | `{ epoch, total, loss, ppl }`    | GGUF fine-tune epoch  |
//! | `finetune-done` | [`commands::FinetuneResult`]         | GGUF fine-tune done   |
//! | `gen-fragment`  | `String` (streamed text fragment)    | streaming generation  |
//! | `export-progress` | `String` (one phase line)          | GGUF export phase     |
//! | `tokenize-progress` | `{ file, files, path, bytesDone, bytesTotal, tokens }` | tokenizing a corpus |
//! | `serve-log`     | `String` (one startup or request line) | the Serve tab's server |
//! | `prepare-progress` | `{ file, files, path, bytesDone, bytesTotal, documents }` | dataset preparation |

#![forbid(unsafe_code)]

mod commands;
mod data;
mod evaluate;
mod infra;
mod run;
mod serving;
mod sizing;
#[cfg(test)]
mod test_support;

// The public modules keep their `ferrum_gui_lib::x` paths (the integration tests use them).
pub use sizing::{estimate, estimate_cmd, hardware};

pub use run::run;
use std::sync::Mutex;
use sysinfo::System;

/// Process-wide state shared across commands.
pub struct AppState {
    /// Reused `sysinfo` handle so CPU usage deltas are meaningful between polls.
    pub sys: Mutex<System>,
    /// Last Capable-tab benchmark (release builds only), used to scale estimates.
    pub bench: Mutex<Option<estimate::Throughput>>,
    /// The running long jobs, one slot per kind (see [`infra::jobs`]).
    pub jobs: infra::jobs::Jobs,
    /// The Serve tab's running HTTP server, if any.
    pub serve: serving::serve::Serve,
}

/// Shared state and the command registry. `run` and the mock-runtime tests
/// both build the app through this, so the tests exercise the real handler list.
fn configure<R: tauri::Runtime>(builder: tauri::Builder<R>) -> tauri::Builder<R> {
    builder
        .manage(AppState {
            sys: Mutex::new(System::new_all()),
            bench: Mutex::new(None),
            jobs: infra::jobs::Jobs::default(),
            serve: serving::serve::Serve::default(),
        })
        .invoke_handler(tauri::generate_handler![
            commands::download_text,
            commands::clean_text,
            commands::save_corpus,
            commands::read_text_file,
            commands::train_slm,
            commands::generate_slm,
            evaluate::evaluate_slm,
            data::prepare::prepare_dataset,
            serving::runtime_export::preview_runtime_export,
            serving::runtime_export::export_to_runtime,
            data::tokenize::tokenize_corpus,
            serving::serve::serve_start,
            serving::serve::serve_stop,
            serving::serve::serve_status,
            commands::model_info,
            commands::gguf_info,
            commands::run_gguf,
            commands::finetune_gguf,
            commands::export_gguf,
            infra::jobs::cancel_job,
            commands::system_stats,
            sizing::hardware::hardware_info,
            sizing::estimate_cmd::estimate_job,
            sizing::preflight::preflight_train,
            sizing::capable::capability_report,
            data::datasets::list_datasets,
            data::datasets::download_dataset,
            data::datasets::download_hf_file,
            data::datasets::download_kaggle_file,
        ])
}

#[cfg(test)]
mod ipc_errors;
#[cfg(test)]
mod ipc_tests;
