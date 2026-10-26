//! D5/D16/D20: nonsense numeric settings are rejected up front, naming the flag.
mod common;

use common::{letter_vocab, tiny_llama_gguf_with_tokens};
use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_numf_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tt_env(args: &[&str], env: &[(&str, &str)]) -> (bool, String, String) {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .envs(env.iter().copied())
        .output()
        .unwrap();
    (
        o.status.success(),
        String::from_utf8_lossy(&o.stdout).into(),
        String::from_utf8_lossy(&o.stderr).into(),
    )
}

fn corpus(d: &Path) -> String {
    let c = d.join("c.txt");
    std::fs::write(&c, "the cat sat on the mat. ".repeat(40)).unwrap();
    c.to_str().unwrap().to_string()
}

const TINY: [&str; 10] = [
    "--context",
    "8",
    "--embed",
    "16",
    "--heads",
    "2",
    "--blocks",
    "1",
    "--hidden",
    "32",
];

#[test]
fn train_rejects_zero_epochs_zero_batch_and_absurd_threads() {
    let d = tmp("train");
    let c = corpus(&d);
    let m = d.join("m.gguf");
    let m = m.to_str().unwrap();
    for (extra, flag) in [
        (&["--epochs", "0"][..], "--epochs"),
        (&["--batch", "0"], "--batch"),
        (&["--threads", "5000"], "--threads"),
    ] {
        let mut a = vec!["train", &c, m, "--vocab", "300"];
        a.extend(TINY);
        a.extend_from_slice(extra);
        let (ok, so, se) = tt_env(&a, &[]);
        assert!(!ok && se.contains(flag), "{extra:?}: {so}{se}");
        assert!(!Path::new(m).exists(), "{extra:?} saved a model");
    }
    // A token budget makes --epochs irrelevant, so 0 is fine there.
    let mut a = vec![
        "train", &c, m, "--vocab", "300", "--tokens", "200", "--epochs", "0",
    ];
    a.extend(TINY);
    let (ok, so, se) = tt_env(&a, &[]);
    assert!(ok, "{so}{se}");
}

#[test]
fn finetune_rejects_zero_epochs_batch_and_threads() {
    let d = tmp("ft");
    let model = d.join("m.gguf");
    std::fs::write(&model, tiny_llama_gguf_with_tokens(&letter_vocab())).unwrap();
    let c = d.join("c.txt");
    std::fs::write(&c, "abc abc abc abc abc abc abc abc").unwrap();
    let o = d.join("o.flck");
    for (extra, flag) in [
        (["--epochs", "0"], "--epochs"),
        (["--batch", "0"], "--batch"),
        (["--threads", "5000"], "--threads"),
    ] {
        let mut a = vec![
            "finetune-gguf",
            model.to_str().unwrap(),
            c.to_str().unwrap(),
            o.to_str().unwrap(),
        ];
        a.extend(extra);
        let (ok, so, se) = tt_env(&a, &[]);
        assert!(!ok && se.contains(flag), "{extra:?}: {so}{se}");
        assert!(!so.contains("Opening"), "validated before opening: {so}");
        assert!(!o.exists());
    }
}

#[test]
fn bad_ferrum_num_threads_warns_once_on_stderr_and_training_still_runs() {
    let d = tmp("env");
    let c = corpus(&d);
    let m = d.join("m.gguf");
    let mut a = vec![
        "train",
        &c,
        m.to_str().unwrap(),
        "--vocab",
        "300",
        "--epochs",
        "1",
    ];
    a.extend(TINY);
    let (ok, so, se) = tt_env(&a, &[("FERRUM_NUM_THREADS", "abc")]);
    assert!(ok, "{so}{se}");
    assert_eq!(se.matches("FERRUM_NUM_THREADS").count(), 1, "{se}");
    assert!(se.contains("\"abc\""), "{se}");
    let m2 = d.join("m2.gguf");
    a[2] = m2.to_str().unwrap();
    let (ok, _, se) = tt_env(&a, &[("FERRUM_NUM_THREADS", "2")]);
    assert!(ok && !se.contains("FERRUM_NUM_THREADS"), "{se}");
}
