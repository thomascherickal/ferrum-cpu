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
//! | `prepare-progress` | `{ file, files, path, bytesDone, bytesTotal, documents }` | dataset preparation |

#![forbid(unsafe_code)]

mod capable;
mod commands;
mod datasets;
pub mod estimate;
pub mod estimate_cmd;
mod evaluate;
pub mod hardware;
mod jobs;
mod paths;
mod preflight;
mod prepare;
#[cfg(test)]
mod test_support;

use std::sync::Mutex;
use sysinfo::System;
use tauri::Emitter;

/// Process-wide state shared across commands.
pub struct AppState {
    /// Reused `sysinfo` handle so CPU usage deltas are meaningful between polls.
    pub sys: Mutex<System>,
    /// Last Capable-tab benchmark (release builds only), used to scale estimates.
    pub bench: Mutex<Option<estimate::Throughput>>,
    /// The running long jobs, one slot per kind (see [`jobs`]).
    pub jobs: jobs::Jobs,
}

/// Shared state and the command registry. `run` and the mock-runtime tests
/// both build the app through this, so the tests exercise the real handler list.
fn configure<R: tauri::Runtime>(builder: tauri::Builder<R>) -> tauri::Builder<R> {
    builder
        .manage(AppState {
            sys: Mutex::new(System::new_all()),
            bench: Mutex::new(None),
            jobs: jobs::Jobs::default(),
        })
        .invoke_handler(tauri::generate_handler![
            commands::download_text,
            commands::clean_text,
            commands::save_corpus,
            commands::read_text_file,
            commands::train_slm,
            commands::generate_slm,
            evaluate::evaluate_slm,
            prepare::prepare_dataset,
            commands::model_info,
            commands::gguf_info,
            commands::run_gguf,
            commands::finetune_gguf,
            commands::export_gguf,
            jobs::cancel_job,
            commands::system_stats,
            hardware::hardware_info,
            estimate_cmd::estimate_job,
            preflight::preflight_train,
            capable::capability_report,
            datasets::list_datasets,
            datasets::download_dataset,
            datasets::download_hf_file,
            datasets::download_kaggle_file,
        ])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    configure(tauri::Builder::default())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // Mirror every ferrum_core `--verbose` line into the GUI terminal.
            let handle = app.handle().clone();
            ferrum_core::set_log_sink(move |line| {
                let _ = handle.emit("engine-log", line.to_string());
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Ferrum SLM Studio");
}

#[cfg(test)]
mod ipc_tests;
