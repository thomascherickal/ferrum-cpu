//! D19/D33/D6: train and fine-tune errors say what happened.
mod common;

use common::{letter_vocab, tiny_llama_gguf_with_tokens};
use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_terr_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tt(args: &[&str]) -> (bool, String) {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .unwrap();
    (
        o.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
    )
}

fn corpus(d: &Path) -> String {
    let c = d.join("c.txt");
    std::fs::write(&c, "the cat sat on the mat. ".repeat(40)).unwrap();
    c.to_str().unwrap().to_string()
}

fn arch(embed: &str) -> Vec<&str> {
    vec![
        "--vocab",
        "300",
        "--context",
        "8",
        "--embed",
        embed,
        "--heads",
        "2",
        "--blocks",
        "1",
        "--hidden",
        "32",
    ]
}

#[test]
fn finetune_divergence_says_training_diverged() {
    let d = tmp("div");
    let model = d.join("m.gguf");
    std::fs::write(&model, tiny_llama_gguf_with_tokens(&letter_vocab())).unwrap();
    let c = d.join("c.txt");
    std::fs::write(&c, "abc abc abc abc abc abc abc abc abc abc abc abc").unwrap();
    let o = d.join("o.flck");
    let (ok, out) = tt(&[
        "finetune-gguf",
        model.to_str().unwrap(),
        c.to_str().unwrap(),
        o.to_str().unwrap(),
        "--epochs",
        "5",
        "--batch",
        "2",
        "--seq",
        "8",
        "--lr",
        "1e30",
        "--clip",
        "0",
    ]);
    assert!(!ok, "{out}");
    assert!(
        out.contains("training diverged") && out.contains("lower --lr"),
        "{out}"
    );
    assert!(!out.contains("dimension mismatch"), "{out}");
    assert!(!o.exists(), "a diverged fine-tune wrote a checkpoint");
}

#[test]
fn checkpoint_or_tokenizer_equal_to_the_model_path_is_refused_before_training() {
    let d = tmp("same");
    let c = corpus(&d);
    let m = d.join("m.gguf");
    let m_s = m.to_str().unwrap();
    // Same spelling, and a different spelling of the same file (through a `..` detour).
    std::fs::create_dir_all(d.join("sub")).unwrap();
    let detour = format!("{}/sub/../m.gguf", d.display());
    for (flag, other) in [
        ("--checkpoint", m_s),
        ("--tokenizer", m_s),
        ("--checkpoint", detour.as_str()),
    ] {
        let mut a = vec!["train", &c, m_s, "--epochs", "3", flag, other];
        a.extend(arch("16"));
        let (ok, out) = tt(&a);
        assert!(!ok, "{flag} {other}: {out}");
        assert!(
            out.contains(&format!("the model output and {flag} are the same file")),
            "{out}"
        );
        assert!(!out.contains("Trained in") && !m.exists(), "{out}");
    }
    let mut a = vec!["train", &c, m_s, "--epochs", "3", "--resume", m_s];
    a.extend(arch("16"));
    // --resume needs an existing file, so its own error comes first; just check no model appears.
    let (ok, _) = tt(&a);
    assert!(!ok && !m.exists());
}

#[test]
fn finetune_refuses_an_output_that_is_the_base_model() {
    let d = tmp("ftsame");
    let model = d.join("m.gguf");
    std::fs::write(&model, tiny_llama_gguf_with_tokens(&letter_vocab())).unwrap();
    let before = std::fs::read(&model).unwrap();
    let c = d.join("c.txt");
    std::fs::write(&c, "abc abc abc abc abc abc abc abc").unwrap();
    let m = model.to_str().unwrap();
    let (ok, out) = tt(&["finetune-gguf", m, c.to_str().unwrap(), m, "--epochs", "1"]);
    assert!(
        !ok && out.contains("the output and the model are the same file"),
        "{out}"
    );
    assert_eq!(std::fs::read(&model).unwrap(), before);
}

#[test]
fn resume_reports_the_values_actually_used() {
    let d = tmp("resume");
    let c = corpus(&d);
    let (m1, m2) = (d.join("m1.gguf"), d.join("m2.gguf"));
    let ck = d.join("ck.fckp");
    let mut a = vec![
        "train",
        &c,
        m1.to_str().unwrap(),
        "--epochs",
        "1",
        "--checkpoint",
        ck.to_str().unwrap(),
        "--checkpoint-every",
        "1",
    ];
    a.extend(arch("16"));
    let (ok, out) = tt(&a);
    assert!(ok && ck.exists(), "{out}");
    // Resume with a conflicting --embed and an explicit --seed.
    let mut a = vec![
        "train",
        &c,
        m2.to_str().unwrap(),
        "--epochs",
        "2",
        "--resume",
        ck.to_str().unwrap(),
        "--seed",
        "5",
    ];
    a.extend(arch("64"));
    let (ok, out) = tt(&a);
    assert!(ok, "{out}");
    assert!(
        out.contains("note: --embed 64 ignored, using 16 from the checkpoint"),
        "{out}"
    );
    assert!(
        out.contains("Embed: 16") && !out.contains("Embed: 64"),
        "{out}"
    );
    assert!(
        out.contains("note: --seed 5 ignored") && out.contains("Seed: from checkpoint"),
        "{out}"
    );
    assert!(
        out.contains("corpus is not checked against the checkpoint"),
        "{out}"
    );
    // Flags that agree with the checkpoint get no note.
    let m3 = d.join("m3.gguf");
    let mut a = vec![
        "train",
        &c,
        m3.to_str().unwrap(),
        "--epochs",
        "3",
        "--resume",
        ck.to_str().unwrap(),
    ];
    a.extend(arch("16"));
    let (ok, out) = tt(&a);
    assert!(ok && !out.contains("ignored"), "{out}");
}
