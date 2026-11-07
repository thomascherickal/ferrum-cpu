//! `export-lmstudio`: `lms import` / `lms ls --json`, preview, execution, and CLI entry point.

use super::find::{find_tool, home_fallback, path_err, run_tool, runnable_summary, show_command};
use super::{Preview, CANCELLED};
use std::path::{Path, PathBuf};

const USAGE_LMSTUDIO: &str =
    "usage: train_transformer export-lmstudio <model.gguf> [--repo ferrum/<stem>] [--bin P]";

/// How long to wait for LM Studio to list a fresh import: a running desktop app indexes its
/// models directory asynchronously, so `lms ls --json` can lag the import by a few seconds.
const LISTING_ATTEMPTS: u32 = 30;
const LISTING_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

fn lms_tool(bin: Option<&str>) -> Result<PathBuf, String> {
    find_tool(bin, "FERRUM_LMS", "lms", home_fallback(".lmstudio/bin/lms"))
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
    let abs = path_err(std::path::absolute(model), model)?;
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
    let abs = path_err(std::path::absolute(model), model)?;
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
    #[cfg(unix)]
    use crate::runtime_export::test_support::fake;
    use crate::runtime_export::test_support::{gguf, s, tmp, v};

    #[cfg(unix)]
    const LS: &str = r#"[{"modelKey":"ferrum/m","path":"ferrum/m/m.gguf"}]"#;

    #[cfg(unix)]
    const ZERO: std::time::Duration = std::time::Duration::ZERO;

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
        assert_eq!(
            model_key_for(r#"{"path":"a/m.gguf","modelKey":"k"} done"#, "m.gguf").as_deref(),
            Some("k")
        );
    }

    #[test]
    fn model_key_scan_handles_escapes_and_nesting() {
        let json = r#"[{"modelKey":"p/k\"q","path":"pub/r/f.gguf"}]"#;
        assert_eq!(model_key_for(json, "r/f.gguf").as_deref(), Some("p/k\"q"));
        let esc_path = r#"[{"modelKey":"e","path":"pub/we\"ird/f.gguf"}]"#;
        assert_eq!(
            model_key_for(esc_path, "we\"ird/f.gguf").as_deref(),
            Some("e")
        );
        let nested = r#"[{"meta":{"path":"pub/r/f.gguf","modelKey":"wrong"},"path":"other/g.gguf","modelKey":"right"}]"#;
        assert_eq!(model_key_for(nested, "r/f.gguf"), None);
        assert_eq!(
            model_key_for(nested, "other/g.gguf").as_deref(),
            Some("right")
        );
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

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn lmstudio_run_reports_details_and_honours_cancel() {
        let d = tmp("lm_run_cancel");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let tool = lms(&d, "exit 0", &format!("cat {}/ls.json", s(&d)));
        let r = lmstudio_run(&model, "ferrum/m", Some(s(&tool)), &|| false).unwrap();
        assert_eq!(r.key, "ferrum/m");
        assert_eq!(r.commands.len(), 2);

        let marker = d.join("ran");
        let lms_bin = fake(&d, "lms_stop", &format!("touch {}", s(&marker)));
        assert_eq!(
            lmstudio_run(&model, "ferrum/m", Some(s(&lms_bin)), &|| true)
                .err()
                .unwrap(),
            CANCELLED
        );
        assert!(!marker.exists(), "no tool was started");

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

    #[cfg(unix)] // needs a /bin/sh fake tool; skipped on Windows
    #[test]
    fn export_lmstudio_waits_for_a_lagging_listing() {
        let d = tmp("lm_lag");
        let model = gguf(&d, "m.gguf", Some("gpt2"), Some(8));
        let ls = format!(
            "n=$(cat {0}/n 2>/dev/null || echo 0); echo $((n+1)) > {0}/n; \
             if [ $n -ge 2 ]; then cat {0}/ls.json; else echo '[]'; fi",
            s(&d)
        );
        let tool = lms(&d, "exit 0", &ls);
        let key = export_lmstudio_with(&model, "ferrum/m", Some(s(&tool)), 5, ZERO).unwrap();
        assert_eq!(key, "ferrum/m");
        assert_eq!(std::fs::read_to_string(d.join("n")).unwrap().trim(), "3");
        std::fs::remove_file(d.join("n")).unwrap();
        let e = export_lmstudio_with(&model, "ferrum/m", Some(s(&tool)), 2, ZERO).unwrap_err();
        assert!(e.contains("still be indexing"), "{e}");
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
        cmd_export_lmstudio(&v(&[s(&model)]), None, Some(s(&tool))).unwrap();
        assert!(std::fs::read_to_string(d.join("args"))
            .unwrap()
            .contains("--user-repo ferrum/m "));
        cmd_export_lmstudio(&v(&[s(&model)]), Some("ferrum/m"), Some(s(&tool))).unwrap();
        let e = cmd_export_lmstudio(&v(&[s(&model)]), Some("bad"), Some(s(&tool))).unwrap_err();
        assert!(e.contains("<publisher>/<name>"), "{e}");
    }
}
