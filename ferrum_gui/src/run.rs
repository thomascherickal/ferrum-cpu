//! The app entry point: opens the window and runs the event loop. It cannot run headless, so
//! `scripts/coverage.sh` excludes this file (like `main.rs`); everything it wires up is tested
//! through [`crate::configure`] on Tauri's mock runtime.

use tauri::Emitter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    crate::configure(tauri::Builder::default())
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
