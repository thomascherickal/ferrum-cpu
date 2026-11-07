//! `export-ollama`: Modelfile generation, preview, execution, and CLI entry point.

use super::find::{
    find_tool, home_fallback, run_tool, runnable_summary, show_command, GgufSummary,
};
use super::{Preview, CANCELLED};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

const USAGE_OLLAMA: &str =
    "usage: train_transformer export-ollama <model.gguf> <name> [--bin P] [--modelfile-only out.Modelfile]";

/// The Modelfile text. The path is double-quoted (spaces are fine); callers
/// reject paths containing `"` or newlines first.
pub fn ollama_modelfile(abs_model: &Path, s: &GgufSummary) -> String {
    format!(
        "FROM \"{}\"\nTEMPLATE \"{{{{ .Prompt }}}}\"\nPARAMETER stop \"<|endoftext|>\"\n\
         PARAMETER temperature 0.8\nPARAMETER num_ctx {}\n",
        abs_model.display(),
        s.context_len
    )
}

fn absolute_checked(model: &Path) -> Result<PathBuf, String> {
    let abs = std::path::absolute(model).map_err(|e| format!("{}: {e}", model.display()))?;
    let s = abs.to_string_lossy();
    if s.contains('"') || s.contains('\n') {
        return Err(format!(
            "{s}: path contains a double quote or newline, which a Modelfile cannot quote; \
             rename the file"
        ));
    }
    Ok(abs)
}

/// Absolute path + summary + Modelfile text (shared by export and `--modelfile-only`).
pub fn modelfile_for(model: &Path) -> Result<String, String> {
    let abs = absolute_checked(model)?;
    let s = runnable_summary(&abs)?;
    Ok(ollama_modelfile(&abs, &s))
}

fn ollama_tool(bin: Option<&str>) -> Result<PathBuf, String> {
    find_tool(
        bin,
        "FERRUM_OLLAMA",
        "ollama",
        home_fallback(".local/bin/ollama"),
    )
}

/// The Modelfile and the `ollama create` command for `model` (refused first when it is not
/// a runnable gpt2 GGUF). The Modelfile path in the command is a placeholder: a run writes it
/// to a temporary directory.
pub fn ollama_preview(model: &Path, name: &str, bin: Option<&str>) -> Result<Preview, String> {
    let modelfile = modelfile_for(model)?;
    let tool = ollama_tool(bin);
    let path = Path::new("<temporary Modelfile>");
    let shown = tool.as_deref().unwrap_or(Path::new("ollama"));
    let commands = vec![show_command(
        shown,
        &[
            "create".as_ref(),
            name.as_ref(),
            "-f".as_ref(),
            path.as_os_str(),
        ],
    )];
    Ok(Preview {
        modelfile: Some(modelfile),
        tool,
        commands,
    })
}

/// Write the Modelfile of `model` to `out` (the `--modelfile-only` mode) and return its text.
pub fn write_modelfile(model: &Path, out: &Path) -> Result<String, String> {
    let text = modelfile_for(model)?;
    std::fs::write(out, &text).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(text)
}

/// What a finished `ollama create` was given.
#[derive(Debug)]
pub struct OllamaRun {
    pub modelfile: String,
    /// The command that ran (its Modelfile was a temporary file, removed afterwards).
    pub command: String,
    /// The tool's own output.
    pub output: String,
}

pub fn export_ollama(model: &Path, name: &str, bin: Option<&str>) -> Result<(), String> {
    ollama_run(model, name, bin, &|| false).map(|_| ())
}

/// `ollama create <name> -f <Modelfile>` for `model`. `cancelled` is polled once, before the tool
/// starts (a running `ollama create` cannot be interrupted).
pub fn ollama_run(
    model: &Path,
    name: &str,
    bin: Option<&str>,
    cancelled: &dyn Fn() -> bool,
) -> Result<OllamaRun, String> {
    let modelfile = modelfile_for(model)?;
    let tool = ollama_tool(bin)?;
    if cancelled() {
        return Err(CANCELLED.into());
    }
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("ferrum_modelfile_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mf = dir.join(format!("{}.Modelfile", name.replace('/', "_")));
    std::fs::write(&mf, &modelfile).map_err(|e| format!("{}: {e}", mf.display()))?;
    let args: [&OsStr; 4] = [
        "create".as_ref(),
        name.as_ref(),
        "-f".as_ref(),
        mf.as_os_str(),
    ];
    let command = show_command(&tool, &args);
    let res = run_tool(&tool, &args);
    let _ = std::fs::remove_dir_all(&dir);
    let (ok, text) = res?;
    if ok {
        return Ok(OllamaRun {
            modelfile,
            command,
            output: text,
        });
    }
    let hint = if text.contains("connect") || text.contains("refused") {
        "\nthe Ollama server is not running: start it with `ollama serve`"
    } else {
        ""
    };
    Err(format!(
        "`ollama create {name}` failed:\n{}{hint}",
        text.trim()
    ))
}

pub fn cmd_export_ollama(
    positional: &[String],
    bin: Option<&str>,
    modelfile_only: Option<&str>,
) -> Result<(), String> {
    let [model, name] = positional else {
        return Err(USAGE_OLLAMA.into());
    };
    if let Some(out) = modelfile_only {
        write_modelfile(Path::new(model), Path::new(out))?;
        println!("wrote {out}");
        return Ok(());
    }
    export_ollama(Path::new(model), name, bin)?;
    println!("exported to Ollama as '{name}'  (try: ollama run {name})");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use crate::runtime_export::test_support::fake;
    use crate::runtime_export::test_support::{gguf, s, summary, tmp, v};

    #[test]
    fn modelfile_uses_absolute_quoted_path() {
        let m = ollama_modelfile(Path::new("/tmp/my model.gguf"), &summary(64));
        assert!(m.contains("FROM \"/tmp/my model.gguf\"\n"), "{m}");
        assert!(m.contains("TEMPLATE \"{{ .Prompt }}\"\n"), "{m}");
        assert!(m.contains("PARAMETER stop \"<|endoftext|>\"\n"), "{m}");
        assert!(m.contains("PARAMETER temperature 0.8\n"), "{m}");
        assert!(m.contains("PARAMETER num_ctx 64\n"), "{m}");
    }

    #[test]
    fn modelfile_for_makes_relative_paths_absolute_and_rejects_quotes() {
        let d = tmp("mf");
        let p = gguf(&d, "a b.gguf", Some("gpt2"), Some(32));
        let m = modelfile_for(&p).unwrap();
        assert!(
            m.contains(&format!("FROM \"{}\"", s(&p))) && m.contains("num_ctx 32"),
            "{m}"
        );
        let rel = Path::new("Cargo.toml");
        let e = modelfile_for(rel).err().unwrap();
        assert!(
            e.contains(std::env::current_dir().unwrap().to_str().unwrap()),
            "{e}"
        );
        for bad in ["q\"uote.gguf", "new\nline.gguf"] {
            let e = modelfile_for(&d.join(bad)).err().unwrap();
            assert!(e.contains("cannot quote"), "{e}");
        }
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_ollama_runs_create_with_a_modelfile() {
        let d = tmp("ol_ok");
        let model = gguf(&d, "my model.gguf", Some("gpt2"), Some(64));
        let tool = fake(
            &d,
            "ollama",
            &format!("echo \"$@\" > {0}/args; cp \"$4\" {0}/seen", s(&d)),
        );
        export_ollama(&model, "ferrum-x", Some(s(&tool))).unwrap();
        let args = std::fs::read_to_string(d.join("args")).unwrap();
        assert!(
            args.starts_with("create ferrum-x -f ") && args.contains("ferrum-x.Modelfile"),
            "{args}"
        );
        let seen = std::fs::read_to_string(d.join("seen")).unwrap();
        assert!(seen.contains(&format!("FROM \"{}\"", s(&model))), "{seen}");
        assert!(seen.contains("num_ctx 64"), "{seen}");
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_ollama_down_says_ollama_serve() {
        let d = tmp("ol_down");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = fake(&d, "ollama", "echo 'Error: connection refused' >&2; exit 1");
        let e = export_ollama(&model, "n", Some(s(&tool))).unwrap_err();
        assert!(
            e.contains("ollama serve") && e.contains("connection refused"),
            "{e}"
        );
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_ollama_other_failure_has_no_serve_hint() {
        let d = tmp("ol_fail");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = fake(&d, "ollama", "echo 'bad modelfile' >&2; exit 1");
        let e = export_ollama(&model, "n/with/slash", Some(s(&tool))).unwrap_err();
        assert!(
            e.contains("bad modelfile") && !e.contains("ollama serve"),
            "{e}"
        );
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_ollama_refuses_before_running_any_tool() {
        let d = tmp("ol_refuse");
        let marker = d.join("ran");
        let tool = fake(&d, "ollama", &format!("touch {}", s(&marker)));
        let bad = gguf(&d, "f.gguf", Some("ferrum"), Some(8));
        let e = export_ollama(&bad, "n", Some(s(&tool))).unwrap_err();
        assert!(e.contains("not runnable in llama.cpp"), "{e}");
        assert!(!marker.exists());
    }

    #[test]
    fn export_ollama_spawn_error() {
        let d = tmp("ol_tool");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let noexec = d.join("noexec");
        std::fs::write(&noexec, "x").unwrap();
        let e = export_ollama(&model, "n", Some(s(&noexec))).unwrap_err();
        assert!(e.contains("cannot run"), "{e}");
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn ollama_preview_shows_the_modelfile_and_command_and_runs_nothing() {
        let d = tmp("ol_prev");
        let marker = d.join("ran");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(32));
        let tool = fake(&d, "ollama", &format!("touch {}", s(&marker)));
        let p = ollama_preview(&model, "ferrum-x", Some(s(&tool))).unwrap();
        assert_eq!(
            p.modelfile.as_deref(),
            Some(modelfile_for(&model).unwrap().as_str())
        );
        assert_eq!(p.tool.as_deref().unwrap(), tool.as_path());
        assert_eq!(
            p.commands,
            vec![format!(
                "{} create ferrum-x -f \"<temporary Modelfile>\"",
                s(&tool)
            )]
        );
        assert!(!marker.exists(), "a preview never runs the tool");
        let p = ollama_preview(&model, "n", Some("/nonexistent/ollama")).unwrap();
        assert!(p.modelfile.is_some() && p.commands[0].contains(" create n -f "));
        let llama = gguf(&d, "l.gguf", Some("llama"), Some(8));
        assert!(ollama_preview(&llama, "n", None)
            .err()
            .unwrap()
            .contains("Manual 9.5"));
    }

    #[test]
    fn write_modelfile_returns_and_writes_the_text() {
        let d = tmp("mf_out");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(48));
        let out = d.join("o.Modelfile");
        let text = write_modelfile(&model, &out).unwrap();
        assert_eq!(std::fs::read_to_string(&out).unwrap(), text);
        assert!(text.contains("PARAMETER num_ctx 48"));
        let e = write_modelfile(&model, &d.join("no/such/dir/o")).unwrap_err();
        assert!(e.contains("o"), "{e}");
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn ollama_run_reports_details_and_honours_cancel() {
        let d = tmp("ol_run_cancel");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = fake(&d, "ollama", "echo created $2");
        let r = ollama_run(&model, "ferrum-x", Some(s(&tool)), &|| false).unwrap();
        assert!(r.modelfile.starts_with("FROM \""));
        assert!(r
            .command
            .starts_with(&format!("{} create ferrum-x -f ", s(&tool))));
        assert!(r.command.contains("ferrum-x.Modelfile") && r.output.trim() == "created ferrum-x");

        let marker = d.join("ran");
        let ollama = fake(&d, "ollama", &format!("touch {}", s(&marker)));
        assert_eq!(
            ollama_run(&model, "n", Some(s(&ollama)), &|| true)
                .err()
                .unwrap(),
            CANCELLED
        );
        assert!(!marker.exists(), "no tool was started");
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn cmd_export_ollama_paths() {
        let d = tmp("cmd_ol");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(16));
        let tool = fake(&d, "ollama", "exit 0");
        assert!(cmd_export_ollama(&v(&["only-one"]), None, None)
            .unwrap_err()
            .contains("usage:"));
        cmd_export_ollama(&v(&[s(&model), "n"]), Some(s(&tool)), None).unwrap();
        let out = d.join("out.Modelfile");
        cmd_export_ollama(&v(&[s(&model), "n"]), Some("/nonexistent"), Some(s(&out))).unwrap();
        assert!(std::fs::read_to_string(&out)
            .unwrap()
            .contains("num_ctx 16"));
        let e =
            cmd_export_ollama(&v(&[s(&model), "n"]), None, Some("/nonexistent/dir/o")).unwrap_err();
        assert!(e.contains("/nonexistent/dir/o"), "{e}");
        let bad = gguf(&d, "f.gguf", Some("ferrum"), Some(8));
        let e = cmd_export_ollama(&v(&[s(&bad), "n"]), None, Some(s(&out))).unwrap_err();
        assert!(e.contains("not runnable"), "{e}");
    }
}
