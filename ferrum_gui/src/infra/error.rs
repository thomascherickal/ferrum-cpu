//! The error every Tauri command returns: `{ code, message, hint }`.
//!
//! IPC contract: a failed command rejects the webview's `invoke()` with this JSON object
//! (`code` is one of [`CODES`], `message` is the text the commands always produced, `hint` says
//! what to try next). Inner functions keep returning plain `String` errors; the command wrapper
//! turns one into a `CmdError` with `?` / [`blocking`], and the code is derived from the
//! message by [`classify`]. The UI also accepts a plain string (a Tauri-level error such as a
//! malformed argument, or a page running outside Tauri), see `ui/core/errors.js`.

use serde::Serialize;

/// Every `code` a command can answer, with the hint shown under the message.
pub const CODES: [(&str, &str); 11] = [
    ("cancelled", "You stopped it. Nothing was lost; start it again when you are ready."),
    ("busy", "Wait for the running job to finish, or press its Stop button."),
    ("memory", "Pick a smaller model or output type, lower batch, context or width, close other programs, or tick the \"anyway\" option if you accept the risk."),
    ("not_found", "Check the path, or use Browse to pick the file."),
    ("path", "Use a regular file or folder you can access; devices and system folders are refused."),
    ("network", "Check your internet connection and the address. The host may be slow, down or need credentials."),
    ("invalid", "Fix the value named in the message and try again."),
    ("model", "The file may be damaged or not a model this tab supports. Inspect it on the Models or GGUF tab."),
    ("io", "Check that the folder exists, is writable and has free space."),
    ("internal", "This looks like a bug. Use Copy details and report it."),
    ("error", "If this is unexpected, use Copy details and report it."),
];

/// A command failure as the webview receives it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CmdError {
    pub code: &'static str,
    pub message: String,
    pub hint: &'static str,
}

/// The code for a message, by what it says (first match wins; `"error"` when nothing fits).
pub fn classify(message: &str) -> &'static str {
    let m = message.to_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|n| m.contains(n));
    if has(&["cancelled", "canceled"]) {
        "cancelled"
    } else if has(&["already running"]) {
        "busy"
    } else if has(&["task error"]) {
        "internal"
    } else if has(&["exceeds 90%", "would need about", "needs about"]) {
        "memory"
    } else if has(&["does not exist", "no such file"]) {
        "not_found"
    } else if has(&[
        "device or system path",
        "is a directory",
        "is not a directory",
        "not a regular file",
        "please provide a path",
        "path contains a nul",
        "path is longer",
    ]) {
        "path"
    } else if has(&[
        "download failed",
        "read failed",
        "timed out",
        "connection",
        "kaggle needs credentials",
    ]) {
        "network"
    } else if has(&[
        "must be",
        "must start",
        "is required",
        "provide ",
        "unknown ",
        "needs the",
        "is empty",
        "divisible",
        "nothing left",
    ]) {
        "invalid"
    } else if has(&[
        "not a valid model",
        "cannot load",
        "cannot open",
        "no tokenizer",
        "not runnable",
        "checkpoint",
    ]) {
        "model"
    } else if has(&[
        "cannot read",
        "cannot write",
        "cannot create",
        "cannot access",
        "cannot save",
        "cannot resolve",
        "write failed",
        "cannot move",
    ]) {
        "io"
    } else {
        "error"
    }
}

fn hint_for(code: &str) -> &'static str {
    CODES
        .iter()
        .find(|(c, _)| *c == code)
        .map_or(CODES[CODES.len() - 1].1, |(_, h)| h)
}

impl From<String> for CmdError {
    fn from(message: String) -> Self {
        let code = classify(&message);
        CmdError {
            code,
            hint: hint_for(code),
            message,
        }
    }
}

impl From<&str> for CmdError {
    fn from(message: &str) -> Self {
        message.to_string().into()
    }
}

impl std::fmt::Display for CmdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// A `CmdError` reads as its message, so `err.contains("...")` works on it as it did on the
/// `String` the commands used to return.
impl std::ops::Deref for CmdError {
    type Target = str;
    fn deref(&self) -> &str {
        &self.message
    }
}

/// Run `f` on a blocking task and flatten the two failure layers (the task itself, then `f`).
pub async fn blocking<T, F>(f: F) -> Result<T, CmdError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|e| CmdError::from(format!("task error: {e}")))?
        .map_err(CmdError::from)
}

/// [`blocking`] for work that cannot fail.
pub async fn blocking_ok<T, F>(f: F) -> Result<T, CmdError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    blocking(move || Ok(f())).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_has_a_hint_and_classify_only_answers_known_codes() {
        let mut seen = std::collections::HashSet::new();
        for (code, hint) in CODES {
            assert!(seen.insert(code), "duplicate {code}");
            assert!(hint.len() > 20 && hint.ends_with('.'), "{code}: {hint}");
        }
        for m in [
            "",
            "x",
            "cancelled",
            "does not exist",
            "download failed: x",
            "nothing like this",
        ] {
            assert!(seen.contains(classify(m)), "{m}");
        }
    }

    /// Real messages from the backend, one or more per code.
    #[test]
    fn classifies_the_messages_the_backend_produces() {
        for (msg, code) in [
            ("export cancelled; no file was written", "cancelled"),
            ("cancelled before any text was produced", "cancelled"),
            ("training is already running", "busy"),
            ("a GGUF run is already running", "busy"),
            ("task error: panicked", "internal"),
            ("estimated resident memory (3.10 GB) exceeds 90% of available (2.00 GB) — choose a smaller quant", "memory"),
            ("training would need about 20.0 GB (weights) but only 4.0 GB may be used; lower --batch", "memory"),
            ("This run needs about 5 GB but the engine allows 3 GB", "memory"),
            ("model path: /nope/x.gguf does not exist", "not_found"),
            ("cannot open /x.gguf: No such file or directory (os error 2)", "not_found"),
            ("file path: /dev/zero is a device or system path and is not allowed", "path"),
            ("model path: /tmp is a directory, expected a file", "path"),
            ("output file path: please provide a path", "path"),
            ("download failed: Connection Failed: timed out", "network"),
            ("Kaggle needs credentials: set KAGGLE_USERNAME", "network"),
            ("URL must start with http:// or https://", "invalid"),
            ("epochs must be ≥ 1", "invalid"),
            ("quant must be int4 | int8 | f32 (got 'x')", "invalid"),
            ("corpus /c.txt is empty", "invalid"),
            ("unknown dataset id 'zz'", "invalid"),
            ("not a valid model: old Ferrum .bin (FINF) model", "model"),
            ("cannot load model: tensor missing", "model"),
            ("this GGUF has no tokenizer; text fine-tuning needs one", "model"),
            ("cannot write /out/m.gguf: Permission denied", "io"),
            ("cannot read corpus /c.txt: invalid data", "io"),
            ("something odd happened", "error"),
        ] {
            assert_eq!(classify(msg), code, "{msg}");
        }
    }

    #[test]
    fn serializes_as_code_message_hint_and_reads_as_its_message() {
        let e = CmdError::from("model path: /x does not exist");
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            serde_json::json!({
                "code": "not_found",
                "message": "model path: /x does not exist",
                "hint": "Check the path, or use Browse to pick the file."
            })
        );
        assert!(e.contains("does not exist"));
        assert_eq!(e.to_string(), "model path: /x does not exist");
        assert_eq!(CmdError::from("x").code, "error");
    }
}
