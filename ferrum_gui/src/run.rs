//! The app entry point: opens the window and runs the event loop. It cannot run headless, so
//! `scripts/coverage.sh` excludes this file (like `main.rs`); everything it wires up is tested
//! through [`crate::configure`] on Tauri's mock runtime.

use tauri::Emitter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    crate::configure(tauri::Builder::default())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            forward_engine_log(app.handle().clone());
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Ferrum SLM Studio");
}

/// Mirror every ferrum_core `--verbose` line into the GUI terminal as an `engine-log` event.
fn forward_engine_log<R: tauri::Runtime>(handle: tauri::AppHandle<R>) {
    ferrum_core::set_log_sink(move |line| {
        let _ = handle.emit("engine-log", line.to_string());
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_tests::{app, collect, wait_for};

    #[test]
    fn engine_log_lines_reach_the_webview_as_events() {
        let (app, _w) = app();
        let got = collect(&app, "engine-log");
        forward_engine_log(app.handle().clone());
        let line = format!("run.rs engine-log probe {}", std::process::id());
        ferrum_core::log_line(&line);
        ferrum_core::clear_log_sink();
        wait_for(&got, 1);
        // The sink is process-wide, so a concurrent test's line may arrive too: look for ours.
        let want = serde_json::to_string(&line).unwrap();
        assert!(
            got.lock().unwrap().contains(&want),
            "{:?}",
            got.lock().unwrap()
        );
    }

    /// `build.rs` (`tauri_build::build()`) cannot hold an executed test of its own; its effect
    /// can: it sets the `desktop` / `mobile` cfg that `run`'s entry-point attribute reads.
    #[test]
    fn the_build_script_marks_this_a_desktop_build_and_run_is_the_entry_point() {
        // Checked when the test compiles: a build without the cfg fails to build this test.
        const { assert!(cfg!(desktop) && !cfg!(mobile)) };
        let _: fn() = run;
    }
}
