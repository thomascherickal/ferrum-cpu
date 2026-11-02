//! `scripts/coverage.sh` guards: unknown flags, no concurrent cargo (OOM on a 15 GB machine),
//! and always `--release` with the low-memory build variables. `cargo` and `pgrep` are fakes.

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
    let d = std::env::temp_dir().join(format!("ferrum_cov_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let log = d.join("cargo.log");
    fake(
        &d,
        "cargo",
        &format!(
            "echo \"ARGS: $*\" >> {l}\nenv | grep '^CARGO_' >> {l}",
            l = log.display()
        ),
    );
    fake(&d, "pgrep", &format!("echo '{pgrep_out}'"));
    let path = format!("{}:{}", d.display(), std::env::var("PATH").unwrap());
    let o = Command::new("bash")
        .arg(root().join("scripts/coverage.sh"))
        .args(args)
        .current_dir(root())
        .env("PATH", path)
        .env("FERRUM_COVERAGE_OUT", d.join("out"))
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
fn coverage_script_passes_release_and_low_memory_env() {
    let (o, log) = run("env", &["--rust"], "");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let l = std::fs::read_to_string(log).unwrap();
    for want in [
        "llvm-cov",
        "--release",
        "--fail-under-lines 100",
        "CARGO_BUILD_JOBS=3",
    ] {
        assert!(l.contains(want), "missing {want}: {l}");
    }
}
