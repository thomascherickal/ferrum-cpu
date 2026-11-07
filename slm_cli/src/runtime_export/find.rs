//! Tool discovery (`find_tool`), subprocess execution (`run_tool`), and `gpt2`
//! GGUF validation (`runnable_summary`, `plain_gguf_warning`).

use ferrum_core::Gguf;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn path_err<T>(res: std::io::Result<T>, p: &Path) -> Result<T, String> {
    match res {
        Ok(v) => Ok(v),
        Err(e) => Err(format!("{}: {e}", p.display())),
    }
}

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

/// File names tried per `PATH` directory: `name`, plus `name.exe` on Windows.
fn candidate_names(name: &str, windows: bool) -> Vec<String> {
    let mut v = vec![name.to_string()];
    if windows {
        v.push(format!("{name}.exe"));
    }
    v
}

fn scan_path(path_var: &OsStr, name: &str, windows: bool) -> Option<PathBuf> {
    let names = candidate_names(name, windows);
    std::env::split_paths(path_var)
        .flat_map(|d| names.iter().map(move |n| d.join(n)))
        .find(|c| c.is_file())
}

pub(super) fn find_tool_in(
    flag: Option<&str>,
    env_val: Option<&str>,
    env: &str,
    path_var: Option<&OsStr>,
    name: &str,
    fallback: Option<PathBuf>,
) -> Result<PathBuf, String> {
    let from_flag = flag.map(PathBuf::from);
    let from_env = env_val.map(PathBuf::from);
    let from_path = path_var.and_then(|p| scan_path(p, name, cfg!(windows)));
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

pub(super) fn home_fallback(rel: &str) -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| Path::new(&h).join(rel))
}

/// Run a tool; returns (success, stdout+stderr).
pub(super) fn run_tool(bin: &Path, args: &[&OsStr]) -> Result<(bool, String), String> {
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

/// A command line for display: `tool arg ...`, arguments with spaces or quotes in double quotes.
pub(super) fn show_command(tool: &Path, args: &[&OsStr]) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use crate::runtime_export::test_support::fake;
    use crate::runtime_export::test_support::{gguf, s, tmp};

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

    #[test]
    fn candidate_names_adds_exe_only_on_windows() {
        assert_eq!(candidate_names("lms", false), ["lms"]);
        assert_eq!(candidate_names("lms", true), ["lms", "lms.exe"]);
    }

    /// Regression (M16): a PATH holding only `ollama.exe` must be found on Windows.
    #[test]
    fn scan_path_finds_exe_when_windows() {
        let d = tmp("exe_scan");
        let exe = d.join("ollama.exe");
        std::fs::write(&exe, b"x").unwrap();
        let pv = std::ffi::OsString::from(&d);
        assert_eq!(scan_path(&pv, "ollama", true), Some(exe));
        assert_eq!(scan_path(&pv, "ollama", false), None);
    }

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
        let unset = "FERRUM_RX_DEFINITELY_UNSET";
        assert!(std::env::var_os(unset).is_none());
        let e = find_tool(None, unset, "nope-xyz", None).unwrap_err();
        assert!(e.contains("nope-xyz") && e.contains(unset), "{e}");
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
    fn show_command_quotes_only_what_needs_it() {
        assert_eq!(path_err(Ok(1), Path::new("p")), Ok(1));
        assert_eq!(
            path_err::<i32>(Err(std::io::Error::other("x")), Path::new("p")),
            Err("p: x".into())
        );
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
}
