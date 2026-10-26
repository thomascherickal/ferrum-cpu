//! `scripts/bench_slm.sh` must accept a corpus path relative to the caller's directory (it
//! used to `cd` to the repo root first, so only repo-relative or absolute paths worked).

use std::path::Path;
use std::process::Command;

fn script() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scripts/bench_slm.sh")
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_bench_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn unreadable_corpus_is_named_before_any_build() {
    let d = scratch("missing");
    let o = Command::new("bash")
        .arg(script())
        .arg("nope.txt")
        .current_dir(&d)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success());
    assert!(
        err.contains("cannot read corpus") && err.contains("nope.txt"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn relative_corpus_path_is_resolved_from_the_callers_directory() {
    let d = scratch("rel");
    std::fs::write(
        d.join("big.txt"),
        "the cat sat on the mat. the dog sat on the log. ".repeat(40),
    )
    .unwrap();
    let o = Command::new("bash")
        .arg(script())
        .args(["big.txt", "2e3"])
        .current_dir(&d)
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "{out}{err}");
    assert!(
        out.contains("train: ") && out.contains("generate: "),
        "{out}"
    );
    let _ = std::fs::remove_dir_all(&d);
}
