#![forbid(unsafe_code)]
// Prevents an extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    ferrum_gui_lib::run();
}

#[cfg(test)]
mod tests {
    /// The binary only starts the library's entry point; this pins that it stays a plain `fn()`.
    #[test]
    fn main_delegates_to_the_library_entry_point() {
        let _: fn() = ferrum_gui_lib::run;
        let _: fn() = super::main;
    }
}
