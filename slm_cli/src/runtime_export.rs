//! `export-ollama` / `export-lmstudio`: put a Ferrum GGUF into a local runtime
//! with one command. Only `gpt2`-architecture GGUF files run in llama.cpp,
//! so anything else is refused before any tool is touched.

use ferrum_core::Gguf;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct GgufSummary {
    pub context_len: u32,
}

const BPE_FIX: &str = "retrain with a BPE vocabulary (`train ... --vocab 1024`; \
                       `--vocab 0` is character-level) to get a 'gpt2' GGUF";

fn not_runnable(path: &str, arch: &str) -> String {
    let arch = if arch.is_empty() { "?" } else { arch };
    if arch == "llama" {
        return format!(
            "{path}: architecture 'llama': Ferrum exports only its own gpt2 models; for a llama \
             GGUF write a Modelfile by hand (Manual 9.5)"
        );
    }
    format!(
        "{path}: architecture '{arch}' is not runnable in llama.cpp (Ollama / LM Studio run only \
         'gpt2' GGUF files; character-level, generic and tabular Ferrum models have no \
         llama.cpp architecture). To run a Ferrum SLM there, {BPE_FIX}"
    )
}

/// One-line warning for a freshly written model file that llama.cpp / Ollama /
/// LM Studio cannot load (any architecture but `gpt2`); `None` when it is fine
/// or unreadable (the caller's own reload check reports that).
pub fn plain_gguf_warning(path: &Path) -> Option<String> {
    let g = Gguf::open(&path.to_string_lossy()).ok()?;
    let arch = g.architecture().unwrap_or("");
    (arch != "gpt2").then(|| {
        format!(
            "warning: {} is a plain Ferrum GGUF (architecture '{arch}'): only Ferrum can load it, \
             not llama.cpp / Ollama / LM Studio; {BPE_FIX}",
            path.display()
        )
    })
}

/// Errors unless `path` is a runnable (gpt2-architecture) GGUF file.
pub fn runnable_summary(path: &Path) -> Result<GgufSummary, String> {
    let p = path.to_string_lossy();
    let g = Gguf::open(&p).map_err(|e| format!("cannot read GGUF {p}: {e}"))?;
    let arch = g.architecture().unwrap_or("").to_string();
    if arch != "gpt2" {
        return Err(not_runnable(&p, &arch));
    }
    let context_len = g
        .meta("gpt2.context_length")
        .and_then(|v| v.as_usize())
        .ok_or_else(|| format!("{p}: missing gpt2.context_length"))? as u32;
    Ok(GgufSummary { context_len })
}

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

/// Tool lookup order: `--bin` flag, env var, `PATH`, default install location.
pub fn find_tool(
    flag: Option<&str>,
    env: &str,
    name: &str,
    fallback: Option<PathBuf>,
) -> Result<PathBuf, String> {
    find_tool_in(
        flag,
        std::env::var(env).ok().as_deref(),
        env,
        std::env::var_os("PATH").as_deref(),
        name,
        fallback,
    )
}

fn find_tool_in(
    flag: Option<&str>,
    env_val: Option<&str>,
    env: &str,
    path_var: Option<&OsStr>,
    name: &str,
    fallback: Option<PathBuf>,
) -> Result<PathBuf, String> {
    let from_flag = flag.map(PathBuf::from);
    let from_env = env_val.map(PathBuf::from);
    let from_path = path_var.and_then(|p| {
        std::env::split_paths(p)
            .map(|d| d.join(name))
            .find(|c| c.is_file())
    });
    let candidates = [
        from_flag.clone(),
        from_env.clone(),
        from_path,
        fallback.clone(),
    ];
    if let Some(found) = candidates.iter().flatten().find(|c| c.is_file()) {
        return Ok(found.clone());
    }
    let show = |p: &Option<PathBuf>| {
        p.as_ref()
            .map_or("not set".into(), |p| p.display().to_string())
    };
    Err(format!(
        "cannot find '{name}'. Checked: (1) --bin: {}; (2) ${env}: {}; (3) PATH; (4) default \
         location: {}",
        show(&from_flag),
        show(&from_env),
        show(&fallback)
    ))
}

fn home_fallback(rel: &str) -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| Path::new(&h).join(rel))
}

/// Run a tool; returns (success, stdout+stderr).
fn run_tool(bin: &Path, args: &[&OsStr]) -> Result<(bool, String), String> {
    // ETXTBSY (26): the file is momentarily open for writing (e.g. another thread just wrote
    // it and a concurrent fork still holds the descriptor); retry briefly instead of failing.
    let mut tries = 0;
    let out = loop {
        match Command::new(bin).args(args).output() {
            Err(e) if e.raw_os_error() == Some(26) && tries < 40 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            r => break r.map_err(|e| format!("cannot run {}: {e}", bin.display()))?,
        }
    };
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    Ok((out.status.success(), text))
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

/// The message of an export ended by the caller's cancel check.
pub const CANCELLED: &str = "export cancelled; nothing was run";

fn ollama_tool(bin: Option<&str>) -> Result<PathBuf, String> {
    find_tool(
        bin,
        "FERRUM_OLLAMA",
        "ollama",
        home_fallback(".local/bin/ollama"),
    )
}

fn lms_tool(bin: Option<&str>) -> Result<PathBuf, String> {
    find_tool(bin, "FERRUM_LMS", "lms", home_fallback(".lmstudio/bin/lms"))
}

/// A command line for display: `tool arg ...`, arguments with spaces or quotes in double quotes.
fn show_command(tool: &Path, args: &[&OsStr]) -> String {
    std::iter::once(tool.as_os_str())
        .chain(args.iter().copied())
        .map(|a| {
            let a = a.to_string_lossy();
            if a.is_empty() || a.contains([' ', '"', '\'', '\t']) {
                format!("\"{}\"", a.replace('"', "\\\""))
            } else {
                a.into_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// What an export would do, without doing it (nothing is run or contacted).
#[derive(Debug)]
pub struct Preview {
    /// The Modelfile text (Ollama only).
    pub modelfile: Option<String>,
    /// The tool that would run, or why it cannot be found.
    pub tool: Result<PathBuf, String>,
    /// The commands that would run, in order.
    pub commands: Vec<String>,
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

/// `path` ends with `suffix` on a `/` boundary (`xferrum/m` does not match `ferrum/m`).
fn path_matches(path: &str, suffix: &str) -> bool {
    path.strip_suffix(suffix)
        .is_some_and(|head| head.is_empty() || head.ends_with('/'))
}

/// `modelKey` of the `lms ls --json` entry whose `path` ends with `suffix`. The JSON may be
/// preceded by a banner line ("Waking up LM Studio service..."), so parsing starts at the first
/// `[` or `{`; trailing text is ignored.
fn model_key_for(json: &str, suffix: &str) -> Option<String> {
    let json = &json[json.find(['[', '{'])?..];
    let value: serde_json::Value = serde_json::Deserializer::from_str(json)
        .into_iter()
        .next()?
        .ok()?;
    let entries = match &value {
        serde_json::Value::Array(a) => a.as_slice(),
        one => std::slice::from_ref(one),
    };
    entries
        .iter()
        .find(|e| e["path"].as_str().is_some_and(|p| path_matches(p, suffix)))
        .and_then(|e| e["modelKey"].as_str())
        .map(String::from)
}

/// How long to wait for LM Studio to list a fresh import: a running desktop app indexes its
/// models directory asynchronously, so `lms ls --json` can lag the import by a few seconds.
const LISTING_ATTEMPTS: u32 = 30;
const LISTING_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

/// `ferrum/<file stem>`, the repo `export-lmstudio` uses when none is given.
pub fn default_lmstudio_repo(model: &Path) -> String {
    format!(
        "ferrum/{}",
        model.file_stem().unwrap_or_default().to_string_lossy()
    )
}

fn check_repo(repo: &str) -> Result<(), String> {
    let (publisher, rest) = repo.split_once('/').unwrap_or(("", ""));
    if publisher.is_empty() || rest.is_empty() || rest.contains('/') {
        return Err(format!("--repo '{repo}' must look like <publisher>/<name>"));
    }
    Ok(())
}

/// The `lms import` and `lms ls --json` commands for `model` (refused first when it is not a
/// runnable gpt2 GGUF); nothing is run.
pub fn lmstudio_preview(model: &Path, repo: &str, bin: Option<&str>) -> Result<Preview, String> {
    check_repo(repo)?;
    let abs = std::path::absolute(model).map_err(|e| format!("{}: {e}", model.display()))?;
    runnable_summary(&abs)?;
    let tool = lms_tool(bin);
    let shown = tool.as_deref().unwrap_or(Path::new("lms"));
    let commands = lms_commands(shown, repo, &abs);
    Ok(Preview {
        modelfile: None,
        tool,
        commands,
    })
}

fn lms_commands(tool: &Path, repo: &str, abs: &Path) -> Vec<String> {
    vec![
        show_command(
            tool,
            &[
                "import".as_ref(),
                "-y".as_ref(),
                "-l".as_ref(),
                "--user-repo".as_ref(),
                repo.as_ref(),
                abs.as_os_str(),
            ],
        ),
        show_command(tool, &["ls".as_ref(), "--json".as_ref()]),
    ]
}

/// What a finished LM Studio import found.
#[derive(Debug)]
pub struct LmsRun {
    /// The `modelKey` to `lms load`.
    pub key: String,
    /// The commands that ran.
    pub commands: Vec<String>,
}

pub fn export_lmstudio(model: &Path, repo: &str, bin: Option<&str>) -> Result<String, String> {
    export_lmstudio_with(model, repo, bin, LISTING_ATTEMPTS, LISTING_DELAY)
}

fn export_lmstudio_with(
    model: &Path,
    repo: &str,
    bin: Option<&str>,
    attempts: u32,
    delay: std::time::Duration,
) -> Result<String, String> {
    lmstudio_core(model, repo, bin, attempts, delay, &|| false).map(|r| r.key)
}

/// `lms import` then `lms ls --json`, as `export-lmstudio`. `cancelled` is polled before each
/// tool starts and while the listing is awaited.
pub fn lmstudio_run(
    model: &Path,
    repo: &str,
    bin: Option<&str>,
    cancelled: &dyn Fn() -> bool,
) -> Result<LmsRun, String> {
    lmstudio_core(model, repo, bin, LISTING_ATTEMPTS, LISTING_DELAY, cancelled)
}

fn lmstudio_core(
    model: &Path,
    repo: &str,
    bin: Option<&str>,
    attempts: u32,
    delay: std::time::Duration,
    cancelled: &dyn Fn() -> bool,
) -> Result<LmsRun, String> {
    check_repo(repo)?;
    let abs = std::path::absolute(model).map_err(|e| format!("{}: {e}", model.display()))?;
    runnable_summary(&abs)?;
    let tool = lms_tool(bin)?;
    let commands = lms_commands(&tool, repo, &abs);
    if cancelled() {
        return Err(CANCELLED.into());
    }
    let (ok, text) = run_tool(
        &tool,
        &[
            "import".as_ref(),
            "-y".as_ref(),
            "-l".as_ref(),
            "--user-repo".as_ref(),
            repo.as_ref(),
            abs.as_os_str(),
        ],
    )?;
    // Re-importing is not an error.
    if !ok && !text.to_lowercase().contains("already") {
        return Err(format!("`lms import` failed:\n{}", text.trim()));
    }
    let file = abs.file_name().unwrap_or_default().to_string_lossy();
    let suffix = format!("{repo}/{file}");
    for attempt in 0..attempts.max(1) {
        if attempt > 0 {
            std::thread::sleep(delay);
        }
        if cancelled() {
            return Err(format!(
                "{CANCELLED} after the import; the model is imported but no key was looked up"
            ));
        }
        let (ok, json) = run_tool(&tool, &["ls".as_ref(), "--json".as_ref()])?;
        if !ok {
            return Err(format!("`lms ls --json` failed:\n{}", json.trim()));
        }
        if let Some(key) = model_key_for(&json, &suffix) {
            return Ok(LmsRun { key, commands });
        }
    }
    Err(format!(
        "imported, but `lms ls --json` has no entry whose path ends with {suffix} \
         (if the LM Studio app is open, it may still be indexing; rerun in a moment)"
    ))
}

const USAGE_OLLAMA: &str =
    "usage: train_transformer export-ollama <model.gguf> <name> [--bin P] [--modelfile-only out.Modelfile]";
const USAGE_LMSTUDIO: &str =
    "usage: train_transformer export-lmstudio <model.gguf> [--repo ferrum/<stem>] [--bin P]";

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

pub fn cmd_export_lmstudio(
    positional: &[String],
    repo: Option<&str>,
    bin: Option<&str>,
) -> Result<(), String> {
    let [model] = positional else {
        return Err(USAGE_LMSTUDIO.into());
    };
    let default_repo;
    let repo = match repo {
        Some(r) => r,
        None => {
            default_repo = default_lmstudio_repo(Path::new(model));
            &default_repo
        }
    };
    let key = export_lmstudio(Path::new(model), repo, bin)?;
    println!("imported into LM Studio as modelKey '{key}'  (try: lms load {key})");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{GgufBuilder, MetaValue};
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    /// A fresh temp dir that is removed on drop (also when the test panics).
    struct Tmp(PathBuf);
    impl std::ops::Deref for Tmp {
        type Target = PathBuf;
        fn deref(&self) -> &PathBuf {
            &self.0
        }
    }
    impl AsRef<OsStr> for Tmp {
        fn as_ref(&self) -> &OsStr {
            self.0.as_os_str()
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tmp(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("ferrum_rx_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    fn gguf(dir: &Path, file: &str, arch: Option<&str>, ctx: Option<u32>) -> PathBuf {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.meta("general.architecture", MetaValue::String(a.into()));
        }
        if let Some(c) = ctx {
            b.meta("gpt2.context_length", MetaValue::U32(c));
        }
        let p = dir.join(file);
        std::fs::write(&p, b.into_bytes()).unwrap();
        p
    }

    /// An executable shell script named `name` in `dir`.
    #[cfg(unix)]
    fn fake(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    fn s(p: &Path) -> &str {
        p.to_str().unwrap()
    }

    fn summary(ctx: u32) -> GgufSummary {
        GgufSummary { context_len: ctx }
    }

    // ---- runnable_summary ----

    #[test]
    fn summary_accepts_gpt2() {
        let d = tmp("sum_ok");
        let p = gguf(&d, "tiny model.gguf", Some("gpt2"), Some(64));
        let g = runnable_summary(&p).unwrap();
        assert_eq!(g.context_len, 64);
    }

    #[test]
    fn refuses_non_runnable_gguf() {
        let d = tmp("sum_refuse");
        let p = gguf(&d, "m.gguf", Some("ferrum"), Some(64));
        let e = runnable_summary(&p).err().unwrap();
        assert!(e.contains("not runnable in llama.cpp"), "{e}");
        assert!(e.contains("'ferrum'"), "{e}");
        // A llama GGUF does run in llama.cpp; the refusal is Ferrum's own policy.
        let p = gguf(&d, "l.gguf", Some("llama"), Some(64));
        let e = runnable_summary(&p).err().unwrap();
        assert!(
            e.contains("'llama'") && e.contains("Modelfile by hand (Manual 9.5)"),
            "{e}"
        );
        assert!(!e.contains("not runnable") && !e.contains("retrain"), "{e}");
        // No architecture key at all.
        let e = runnable_summary(&gguf(&d, "n.gguf", None, None))
            .err()
            .unwrap();
        assert!(
            e.contains("not runnable in llama.cpp") && e.contains("'?'"),
            "{e}"
        );
    }

    #[test]
    fn refusal_names_the_bpe_fix() {
        let d = tmp("sum_fix");
        let e = runnable_summary(&gguf(&d, "m.gguf", Some("ferrum"), Some(8)))
            .err()
            .unwrap();
        assert!(e.contains("retrain with a BPE vocabulary"), "{e}");
        assert!(e.contains("--vocab 1024") && e.contains("--vocab 0"), "{e}");
    }

    #[test]
    fn plain_gguf_warning_only_for_non_gpt2() {
        let d = tmp("warn");
        let w = plain_gguf_warning(&gguf(&d, "f.gguf", Some("ferrum"), None)).unwrap();
        assert!(!w.contains('\n'), "must be one line: {w}");
        assert!(w.starts_with("warning:") && w.contains("'ferrum'"), "{w}");
        assert!(w.contains("llama.cpp") && w.contains("--vocab 1024"), "{w}");
        // No architecture key at all is still not runnable.
        assert!(plain_gguf_warning(&gguf(&d, "n.gguf", None, None)).is_some());
        // A gpt2 file is fine; an unreadable or missing file is left to the caller.
        assert!(plain_gguf_warning(&gguf(&d, "g.gguf", Some("gpt2"), Some(8))).is_none());
        let junk = d.join("junk.gguf");
        std::fs::write(&junk, b"nope").unwrap();
        assert!(plain_gguf_warning(&junk).is_none());
        assert!(plain_gguf_warning(&d.join("missing.gguf")).is_none());
    }

    #[test]
    fn summary_needs_context_length() {
        let d = tmp("sum_ctx");
        let e = runnable_summary(&gguf(&d, "m.gguf", Some("gpt2"), None))
            .err()
            .unwrap();
        assert!(e.contains("missing gpt2.context_length"), "{e}");
    }

    #[test]
    fn summary_rejects_unreadable_truncated_and_missing() {
        let d = tmp("sum_bad");
        let full = std::fs::read(gguf(&d, "ok.gguf", Some("gpt2"), Some(8))).unwrap();
        let cut = d.join("cut.gguf");
        std::fs::write(&cut, &full[..full.len() / 2]).unwrap();
        let junk = d.join("junk.gguf");
        std::fs::write(&junk, b"not a gguf at all").unwrap();
        for p in [cut, junk, d.join("missing.gguf")] {
            let e = runnable_summary(&p).err().unwrap();
            assert!(e.contains("cannot read GGUF"), "{e}");
        }
    }

    // ---- Modelfile ----

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
        // Relative input is made absolute against the cwd.
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

    // ---- find_tool ----

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn find_tool_order() {
        let d = tmp("find");
        let mk = |n: &str| fake(&d, n, "true");
        let (flag, env, path_tool, fb) = (mk("t_flag"), mk("t_env"), mk("mytool"), mk("t_fb"));
        let pv = std::ffi::OsString::from(&d);
        let go = |f: Option<&Path>,
                  e: Option<&Path>,
                  pv: Option<&std::ffi::OsStr>,
                  fb: Option<&Path>| {
            find_tool_in(
                f.map(s),
                e.map(s),
                "E",
                pv,
                "mytool",
                fb.map(Path::to_path_buf),
            )
        };
        let all = go(Some(&flag), Some(&env), Some(&pv), Some(&fb)).unwrap();
        assert_eq!(all, flag);
        assert_eq!(go(None, Some(&env), Some(&pv), Some(&fb)).unwrap(), env);
        assert_eq!(go(None, None, Some(&pv), Some(&fb)).unwrap(), path_tool);
        assert_eq!(go(None, None, None, Some(&fb)).unwrap(), fb);
        // A flag/env that points nowhere falls through to the next place.
        let gone = d.join("gone");
        assert_eq!(
            go(Some(&gone), Some(&gone), Some(&pv), None).unwrap(),
            path_tool
        );
    }

    #[test]
    fn find_tool_not_found_lists_all_four_places() {
        let d = tmp("find_none");
        let gone = d.join("gone");
        let e = find_tool_in(
            Some("/x/flagbin"),
            Some("/x/envbin"),
            "FERRUM_FOO",
            Some(d.as_os_str()),
            "mytool",
            Some(gone.clone()),
        )
        .unwrap_err();
        for want in [
            "'mytool'",
            "--bin: /x/flagbin",
            "$FERRUM_FOO: /x/envbin",
            "PATH",
            s(&gone),
        ] {
            assert!(e.contains(want), "{e}");
        }
        let e = find_tool_in(None, None, "FERRUM_FOO", None, "mytool", None).unwrap_err();
        assert!(
            e.contains("--bin: not set") && e.contains("$FERRUM_FOO: not set"),
            "{e}"
        );
        assert!(e.contains("default location: not set"), "{e}");
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn find_tool_reads_the_real_env_and_path() {
        // No set_var/remove_var: mutating the process environment races with other tests'
        // getenv calls. Use a variable cargo already provides, and one that is never set.
        let unset = "FERRUM_RX_DEFINITELY_UNSET";
        assert!(std::env::var_os(unset).is_none());
        let e = find_tool(None, unset, "nope-xyz", None).unwrap_err();
        assert!(e.contains("nope-xyz") && e.contains(unset), "{e}");
        // `$CARGO` is the cargo binary when run under cargo; it must be picked up from the env.
        if let Some(cargo) = std::env::var_os("CARGO")
            .map(PathBuf::from)
            .filter(|c| c.is_file())
        {
            assert_eq!(find_tool(None, "CARGO", "nope-xyz", None).unwrap(), cargo);
        }
        assert!(find_tool(None, unset, "sh", None).is_ok(), "sh is on PATH");
    }

    #[test]
    fn home_fallback_joins_home() {
        let h = std::env::var_os("HOME").map(PathBuf::from);
        assert_eq!(home_fallback("a/b"), h.map(|h| h.join("a/b")));
    }

    // ---- JSON scan ----

    #[test]
    fn model_key_scan() {
        let json = concat!(
            "Waking up LM Studio service...\n",
            r#"[{"type":"llm","modelKey":"a/one","path":"pub/repo-a/x.gguf","q":{"name":"Q8_0"},"v":["a{b"]},"#,
            r#"{"modelKey":"ferrum/m","note":"has \" quote and } brace","path":"ferrum/m/m.gguf"},"#,
            r#"{"path":"ferrum/nokey/z.gguf"}]"#
        );
        assert_eq!(
            model_key_for(json, "ferrum/m/m.gguf").as_deref(),
            Some("ferrum/m")
        );
        assert_eq!(
            model_key_for(json, "repo-a/x.gguf").as_deref(),
            Some("a/one")
        );
        assert_eq!(model_key_for(json, "ferrum/other/m.gguf"), None);
        assert_eq!(
            model_key_for(json, "ferrum/nokey/z.gguf"),
            None,
            "match without a modelKey"
        );
        // Pretty-printed JSON (space after the colon) and a non-boundary suffix.
        let pretty = "[{\n \"modelKey\": \"p/k\",\n \"path\": \"pub/repo/f.gguf\"\n}]";
        assert_eq!(model_key_for(pretty, "repo/f.gguf").as_deref(), Some("p/k"));
        assert_eq!(
            model_key_for(pretty, "po/f.gguf"),
            None,
            "not on a / boundary"
        );
        assert_eq!(
            model_key_for(pretty, "pub/repo/f.gguf").as_deref(),
            Some("p/k")
        );
        assert_eq!(model_key_for(r#"{"path": 5}"#, "x"), None);
        assert_eq!(model_key_for("", "x"), None);
        assert_eq!(model_key_for("[]", "x"), None);
        assert_eq!(model_key_for("} stray", "x"), None);
        assert_eq!(model_key_for(r#"{"path":"unterminated"#, "x"), None);
        // A top-level object, and trailing text after the JSON.
        assert_eq!(
            model_key_for(r#"{"path":"a/m.gguf","modelKey":"k"} done"#, "m.gguf").as_deref(),
            Some("k")
        );
    }

    /// Regression (deferred minor): the old string scan cut a value at an escaped quote
    /// and read keys out of nested objects; a real parser does neither.
    #[test]
    fn model_key_scan_handles_escapes_and_nesting() {
        let json = r#"[{"modelKey":"p/k\"q","path":"pub/r/f.gguf"}]"#;
        assert_eq!(model_key_for(json, "r/f.gguf").as_deref(), Some("p/k\"q"));
        let esc_path = r#"[{"modelKey":"e","path":"pub/we\"ird/f.gguf"}]"#;
        assert_eq!(
            model_key_for(esc_path, "we\"ird/f.gguf").as_deref(),
            Some("e")
        );
        // A nested object's "path" must not be mistaken for the entry's own.
        let nested = r#"[{"meta":{"path":"pub/r/f.gguf","modelKey":"wrong"},"path":"other/g.gguf","modelKey":"right"}]"#;
        assert_eq!(model_key_for(nested, "r/f.gguf"), None);
        assert_eq!(
            model_key_for(nested, "other/g.gguf").as_deref(),
            Some("right")
        );
    }

    // ---- export_ollama ----

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

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn run_tool_retries_while_the_script_is_still_open_for_writing() {
        let d = tmp("txtbsy");
        let t = fake(&d, "slowtool", "echo hi");
        let hold = std::fs::OpenOptions::new().append(true).open(&t).unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(150));
            drop(hold);
        });
        let (ok, text) = run_tool(&t, &[]).unwrap();
        release.join().unwrap();
        assert!(ok && text.trim() == "hi", "{text}");
    }

    #[test]
    fn export_ollama_spawn_error() {
        let d = tmp("ol_tool");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        // Present but not executable: spawn fails with a clear message.
        let noexec = d.join("noexec");
        std::fs::write(&noexec, "x").unwrap();
        let e = export_ollama(&model, "n", Some(s(&noexec))).unwrap_err();
        assert!(e.contains("cannot run"), "{e}");
    }

    // ---- previews, runs with details, cancel ----

    #[test]
    fn show_command_quotes_only_what_needs_it() {
        let c = show_command(
            Path::new("/bin/ollama"),
            &[
                "create".as_ref(),
                "my name".as_ref(),
                "a\"b".as_ref(),
                "".as_ref(),
            ],
        );
        assert_eq!(c, r#"/bin/ollama create "my name" "a\"b" """#);
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
        // The preview never depends on a tool being installed (a tool found on this machine
        // is only looked up, never started).
        let p = ollama_preview(&model, "n", Some("/nonexistent/ollama")).unwrap();
        assert!(p.modelfile.is_some() && p.commands[0].contains(" create n -f "));
        // Non-runnable models are refused, with the Manual pointer for a llama GGUF.
        let llama = gguf(&d, "l.gguf", Some("llama"), Some(8));
        assert!(ollama_preview(&llama, "n", None)
            .err()
            .unwrap()
            .contains("Manual 9.5"));
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn lmstudio_preview_lists_both_commands_and_validates_the_repo() {
        let d = tmp("lm_prev");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = lms(&d, "exit 0", "exit 0");
        let p = lmstudio_preview(&model, "ferrum/m", Some(s(&tool))).unwrap();
        assert!(p.modelfile.is_none());
        assert_eq!(p.commands.len(), 2);
        assert!(
            p.commands[0].starts_with(&format!("{} import -y -l --user-repo ferrum/m ", s(&tool)))
        );
        assert!(
            p.commands[0].ends_with("m.gguf") && p.commands[1] == format!("{} ls --json", s(&tool))
        );
        assert!(!d.join("args").exists(), "a preview never runs the tool");
        assert!(lmstudio_preview(&model, "nope", Some(s(&tool)))
            .err()
            .unwrap()
            .contains("<publisher>/<name>"));
        assert_eq!(default_lmstudio_repo(&model), "ferrum/m");
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
    fn runs_report_what_was_given_to_the_tool() {
        let d = tmp("runs");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = fake(&d, "ollama", "echo created $2");
        let r = ollama_run(&model, "ferrum-x", Some(s(&tool)), &|| false).unwrap();
        assert!(r.modelfile.starts_with("FROM \""));
        assert!(
            r.command
                .starts_with(&format!("{} create ferrum-x -f ", s(&tool))),
            "{}",
            r.command
        );
        assert!(r.command.contains("ferrum-x.Modelfile") && r.output.trim() == "created ferrum-x");
        let tool = lms(&d, "exit 0", &format!("cat {}/ls.json", s(&d)));
        let r = lmstudio_run(&model, "ferrum/m", Some(s(&tool)), &|| false).unwrap();
        assert_eq!(r.key, "ferrum/m");
        assert_eq!(r.commands.len(), 2);
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn cancel_stops_before_the_tool_starts_and_between_lmstudio_steps() {
        let d = tmp("cancel");
        let marker = d.join("ran");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let ollama = fake(&d, "ollama", &format!("touch {}", s(&marker)));
        assert_eq!(
            ollama_run(&model, "n", Some(s(&ollama)), &|| true)
                .err()
                .unwrap(),
            CANCELLED
        );
        let lms_bin = fake(&d, "lms", &format!("touch {}", s(&marker)));
        assert_eq!(
            lmstudio_run(&model, "ferrum/m", Some(s(&lms_bin)), &|| true)
                .err()
                .unwrap(),
            CANCELLED
        );
        assert!(!marker.exists(), "no tool was started");
        // Cancelled after the import, while waiting for the listing.
        let tool = lms(&d, "exit 0", "echo []");
        let calls = std::cell::Cell::new(0);
        let cancel_second = || {
            calls.set(calls.get() + 1);
            calls.get() >= 2
        };
        let e =
            lmstudio_core(&model, "ferrum/m", Some(s(&tool)), 5, ZERO, &cancel_second).unwrap_err();
        assert!(e.starts_with(CANCELLED) && e.contains("imported"), "{e}");
    }

    // ---- export_lmstudio ----

    #[cfg(unix)]
    const LS: &str = r#"[{"modelKey":"ferrum/m","path":"ferrum/m/m.gguf"}]"#;

    #[cfg(unix)]
    fn lms(dir: &Path, import: &str, ls: &str) -> PathBuf {
        std::fs::write(dir.join("ls.json"), LS).unwrap();
        fake(
            dir,
            "lms",
            &format!(
                "case \"$1\" in import) echo \"$@\" > {0}/args; {import};; ls) {ls};; esac",
                s(dir)
            ),
        )
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_lmstudio_imports_and_resolves_key() {
        let d = tmp("lm_ok");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = lms(&d, "exit 0", &format!("cat {}/ls.json", s(&d)));
        let key = export_lmstudio(&model, "ferrum/m", Some(s(&tool))).unwrap();
        assert_eq!(key, "ferrum/m");
        let args = std::fs::read_to_string(d.join("args")).unwrap();
        assert!(
            args.starts_with("import -y -l --user-repo ferrum/m ") && args.contains("m.gguf"),
            "{args}"
        );
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_lmstudio_already_imported_is_success() {
        let d = tmp("lm_again");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let ls = format!("cat {}/ls.json", s(&d));
        for import in [
            "echo 'Model already exists' >&2; exit 1",
            "echo Already imported; exit 0",
        ] {
            let tool = lms(&d, import, &ls);
            assert_eq!(
                export_lmstudio(&model, "ferrum/m", Some(s(&tool))).unwrap(),
                "ferrum/m"
            );
        }
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_lmstudio_failures() {
        let d = tmp("lm_fail");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let ls = format!("cat {}/ls.json", s(&d));
        let tool = lms(&d, "echo disk full >&2; exit 1", &ls);
        let e = export_lmstudio(&model, "ferrum/m", Some(s(&tool))).unwrap_err();
        assert!(
            e.contains("`lms import` failed") && e.contains("disk full"),
            "{e}"
        );
        let tool = lms(&d, "exit 0", "echo daemon down >&2; exit 1");
        let e = export_lmstudio(&model, "ferrum/m", Some(s(&tool))).unwrap_err();
        assert!(
            e.contains("`lms ls --json` failed") && e.contains("daemon down"),
            "{e}"
        );
        let tool = lms(&d, "exit 0", "echo '[]'");
        let e = export_lmstudio_with(&model, "ferrum/m", Some(s(&tool)), 2, ZERO).unwrap_err();
        assert!(
            e.contains("no entry whose path ends with ferrum/m/m.gguf"),
            "{e}"
        );
    }

    #[cfg(unix)]
    const ZERO: std::time::Duration = std::time::Duration::ZERO;

    /// Regression: with the LM Studio desktop app open, `lms ls --json` lags the import, so the
    /// export used to fail with "no entry whose path ends with ...". It now polls the listing.
    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_lmstudio_waits_for_a_lagging_listing() {
        let d = tmp("lm_lag");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        // `ls` prints an empty list for the first two calls, then the entry.
        let ls = format!(
            "n=$(cat {0}/n 2>/dev/null || echo 0); echo $((n+1)) > {0}/n; \
             if [ $n -ge 2 ]; then cat {0}/ls.json; else echo '[]'; fi",
            s(&d)
        );
        let tool = lms(&d, "exit 0", &ls);
        let key = export_lmstudio_with(&model, "ferrum/m", Some(s(&tool)), 5, ZERO).unwrap();
        assert_eq!(key, "ferrum/m");
        assert_eq!(std::fs::read_to_string(d.join("n")).unwrap().trim(), "3");
        // Too few attempts still reports the miss, mentioning the app.
        std::fs::remove_file(d.join("n")).unwrap();
        let e = export_lmstudio_with(&model, "ferrum/m", Some(s(&tool)), 2, ZERO).unwrap_err();
        assert!(e.contains("still be indexing"), "{e}");
        // Zero attempts is clamped to one (never skips the listing).
        std::fs::remove_file(d.join("n")).unwrap();
        assert!(export_lmstudio_with(&model, "ferrum/m", Some(s(&tool)), 0, ZERO).is_err());
        assert_eq!(std::fs::read_to_string(d.join("n")).unwrap().trim(), "1");
    }

    #[test]
    fn export_lmstudio_spawn_error() {
        let d = tmp("lm_spawn");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let noexec = d.join("noexec");
        std::fs::write(&noexec, "x").unwrap();
        let e = export_lmstudio(&model, "ferrum/m", Some(s(&noexec))).unwrap_err();
        assert!(e.contains("cannot run"), "{e}");
    }

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_lmstudio_validates_repo_and_model() {
        let d = tmp("lm_val");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let marker = d.join("ran");
        let tool = fake(&d, "lms", &format!("touch {}", s(&marker)));
        for repo in ["noslash", "/x", "x/", "a/b/c", ""] {
            let e = export_lmstudio(&model, repo, Some(s(&tool))).unwrap_err();
            assert!(e.contains("<publisher>/<name>"), "{repo}: {e}");
        }
        let bad = gguf(&d, "f.gguf", Some("ferrum"), Some(8));
        let e = export_lmstudio(&bad, "ferrum/f", Some(s(&tool))).unwrap_err();
        assert!(e.contains("not runnable in llama.cpp"), "{e}");
        assert!(!marker.exists());
    }

    // ---- CLI glue ----

    fn v(a: &[&str]) -> Vec<String> {
        a.iter().map(|x| x.to_string()).collect()
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
        // --modelfile-only writes the file and never needs the tool.
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

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn cmd_export_lmstudio_paths() {
        let d = tmp("cmd_lm");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(16));
        let ls = r#"echo '[{"modelKey":"K","path":"ferrum/m/m.gguf"}]'"#.to_string();
        let tool = lms(&d, "exit 0", &ls);
        assert!(cmd_export_lmstudio(&[], None, None)
            .unwrap_err()
            .contains("usage:"));
        // Default repo is ferrum/<stem>.
        cmd_export_lmstudio(&v(&[s(&model)]), None, Some(s(&tool))).unwrap();
        assert!(std::fs::read_to_string(d.join("args"))
            .unwrap()
            .contains("--user-repo ferrum/m "));
        // An explicit repo wins (and here matches no ls entry).
        let e =
            cmd_export_lmstudio(&v(&[s(&model)]), Some("me/other"), Some(s(&tool))).unwrap_err();
        assert!(e.contains("me/other/m.gguf"), "{e}");
    }
}
