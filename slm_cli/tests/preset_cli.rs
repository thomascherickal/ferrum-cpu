//! `train --preset tiny` through the real binary: the preset sizes apply, explicit flags win,
//! an unknown name is an error that lists the valid ones.
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_train_transformer");

fn corpus() -> String {
    let mut state = 99u64;
    let words = [
        "the", "cat", "dog", "sat", "ran", "on", "a", "mat", "and", "then",
    ];
    (0..900)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            words[(state >> 33) as usize % words.len()]
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn preset_tiny_trains_a_tiny_model_and_flags_override_it() {
    let dir = std::env::temp_dir().join(format!("ferrum_preset_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (c, m, m2) = (dir.join("c.txt"), dir.join("m.gguf"), dir.join("m2.gguf"));
    std::fs::write(&c, corpus()).unwrap();
    let train = |model: &std::path::Path, extra: &[&str]| {
        let o = Command::new(BIN)
            .args(["train", c.to_str().unwrap(), model.to_str().unwrap()])
            .args(["--preset", "tiny", "--epochs", "1", "--threads", "1"])
            .args(extra)
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8(o.stdout).unwrap()
    };
    let out = train(&m, &[]);
    assert!(out.contains("Preset  : tiny"), "{out}");
    assert!(
        out.contains("Context : 64   Embed: 64   Hidden: 256"),
        "{out}"
    );
    let info = Command::new(BIN)
        .args(["info", m.to_str().unwrap()])
        .output()
        .unwrap();
    let info = String::from_utf8(info.stdout).unwrap();
    assert!(info.contains("Format    : GGUF v3 (gpt2, int8)"), "{info}");

    let out = train(&m2, &["--embed", "32", "--hidden", "64", "--heads", "2"]);
    assert!(
        out.contains("Context : 64   Embed: 32   Hidden: 64"),
        "{out}"
    );

    let bad = Command::new(BIN)
        .args([
            "train",
            c.to_str().unwrap(),
            m.to_str().unwrap(),
            "--preset",
            "huge",
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    let err = String::from_utf8_lossy(&bad.stderr);
    assert!(
        err.contains("unknown --preset 'huge' (valid: tiny, small, medium)"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
