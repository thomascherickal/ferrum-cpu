//! `scripts/coverage.sh` guards: unknown flags, no concurrent cargo (OOM on a 15 GB machine),
//! and always `--release` with the low-memory build variables. `cargo` and `pgrep` are fakes.

// These drive scripts/coverage.sh through bash and POSIX tools: developer tooling, Linux only.
#![cfg(target_os = "linux")]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fake(dir: &Path, name: &str, body: &str) {
    let p = dir.join(name);
    std::fs::write(&p, format!("#!/bin/bash\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// Runs the script with fakes first on PATH; `pgrep_out` is what the fake pgrep prints.
fn run(tag: &str, args: &[&str], pgrep_out: &str) -> (Output, PathBuf) {
    run_with(tag, args, pgrep_out, "", &[])
}

/// `cargo_extra` runs after the fake cargo logs its call (use it to fail); `node` installs a fake node.
fn run_with(
    tag: &str,
    args: &[&str],
    pgrep_out: &str,
    cargo_extra: &str,
    envs: &[(&str, &str)],
) -> (Output, PathBuf) {
    let d = std::env::temp_dir().join(format!("ferrum_cov_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let log = d.join("cargo.log");
    fake(
        &d,
        "cargo",
        &format!(
            "echo \"ARGS: $*\" >> {l}\nenv | grep '^CARGO_' >> {l}\n{x}",
            l = log.display(),
            x = cargo_extra
        ),
    );
    fake(&d, "pgrep", &format!("echo '{pgrep_out}'"));
    let path = format!("{}:{}", d.display(), std::env::var("PATH").unwrap());
    if let Some((_, body)) = envs.iter().find(|(k, _)| *k == "FAKE_NODE") {
        fake(&d, "node", body);
    }
    let o = Command::new("bash")
        .arg(root().join("scripts/coverage.sh"))
        .args(args)
        .current_dir(root())
        .env("PATH", path)
        .env("FERRUM_COVERAGE_OUT", d.join("out"))
        .envs(envs.iter().filter(|(k, _)| *k != "FAKE_NODE").copied())
        .output()
        .unwrap();
    (o, log)
}

#[test]
fn coverage_script_rejects_unknown_flag() {
    let (o, _) = run("flag", &["--bogus"], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("usage: scripts/coverage.sh"));
}

#[test]
fn coverage_script_refuses_concurrent_cargo() {
    let (o, _) = run("busy", &["--rust"], "4242");
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("another cargo process is running"));
}

#[test]
fn coverage_script_rejects_non_numeric_floor() {
    let (o, _) = run("nan", &["--rust", "--fail-under", "abc"], "");
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn coverage_script_passes_release_and_low_memory_env() {
    let (o, log) = run("env", &["--rust", "--fail-under", "42"], "");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let l = std::fs::read_to_string(log).unwrap();
    for want in [
        "llvm-cov",
        "--release",
        "--fail-under-lines 42",
        "--ignore-filename-regex",
        "llvm-cov report",
        "CARGO_BUILD_JOBS=3",
        "CARGO_PROFILE_RELEASE_LTO=off",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16",
    ] {
        assert!(l.contains(want), "missing {want}: {l}");
    }
}

#[test]
fn coverage_script_writes_missing_lines_even_below_the_floor() {
    // the first llvm-cov call (the gate) fails; the report call must still happen, and the script must fail
    let (o, log) = run_with(
        "below",
        &["--rust"],
        "",
        "case \"$*\" in *report*) ;; *) exit 1;; esac",
        &[],
    );
    assert!(!o.status.success());
    let l = std::fs::read_to_string(log).unwrap();
    assert!(l.contains("llvm-cov report"), "{l}");
}

#[test]
fn js_guard_is_not_satisfied_by_a_longer_file_name() {
    // fake node lists every ui file except app.js; testapp.js must not stand in for it
    let (o, _) = run_with(
        "js",
        &["--js", "--fail-under", "0"],
        "",
        "",
        &[
            ("FERRUM_JS_DIRS", "ferrum_gui/ui"),
            (
                "FAKE_NODE",
                "for f in *.js; do [ \"$f\" = app.js ] || echo \"i $f | 100\"; done",
            ),
        ],
    );
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("not in the coverage report: ferrum_gui/ui/app.js"),
        "{err}"
    );
    assert!(!err.contains("ui/testapp.js"), "{err}");
}
