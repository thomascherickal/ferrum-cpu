//! D3/D22: every output path is checked before any work starts, and the error names the path.
mod common;

use common::{letter_vocab, tiny_llama_gguf_with_tokens};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_outp_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tt(args: &[&str]) -> (bool, String, std::time::Duration) {
    let t = Instant::now();
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    (o.status.success(), text, t.elapsed())
}

/// A path whose parent is a regular file: unwritable for every user, root included.
fn blocked(d: &Path, leaf: &str) -> String {
    let f = d.join("file");
    std::fs::write(&f, b"x").unwrap();
    f.join("sub").join(leaf).to_str().unwrap().to_string()
}

fn corpus(d: &Path) -> String {
    let c = d.join("c.txt");
    std::fs::write(&c, "the cat sat on the mat. ".repeat(40)).unwrap();
    c.to_str().unwrap().to_string()
}

const TINY: [&str; 12] = [
    "--epochs",
    "2000",
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
fn train_checks_model_checkpoint_and_tokenizer_paths_before_training() {
    let d = tmp("train");
    let c = corpus(&d);
    for flag in ["model", "--checkpoint", "--tokenizer"] {
        let bad = blocked(&d, "x.out");
        let good = d.join("good.gguf");
        let mut a = vec!["train", &c];
        let mut extra: Vec<&str> = TINY.to_vec();
        extra.extend(["--vocab", "300"]);
        if flag == "model" {
            a.push(&bad);
        } else {
            a.push(good.to_str().unwrap());
            extra.extend([flag, &bad]);
        }
        a.extend(extra);
        let (ok, out, took) = tt(&a);
        assert!(!ok, "{flag}: {out}");
        assert!(
            out.contains(&format!("cannot write {bad}: ")),
            "{flag}: {out}"
        );
        assert!(!out.contains("Trained in"), "{flag}: trained first: {out}");
        assert!(took.as_secs_f32() < 5.0, "{flag}: slow failure {took:?}");
        assert!(!good.exists(), "{flag}: left a model behind");
    }
}

#[test]
fn train_creates_missing_checkpoint_and_tokenizer_parents() {
    let d = tmp("parents");
    let c = corpus(&d);
    let (m, ck, tk) = (
        d.join("m/a.gguf"),
        d.join("ck/dir/c.fckp"),
        d.join("tk/dir/t.gguf"),
    );
    let mut a = vec!["train", &c, m.to_str().unwrap()];
    let mut tiny: Vec<&str> = TINY.to_vec();
    tiny[1] = "1";
    a.extend(tiny);
    a.extend([
        "--vocab",
        "300",
        "--checkpoint",
        ck.to_str().unwrap(),
        "--checkpoint-every",
        "1",
        "--tokenizer",
        tk.to_str().unwrap(),
    ]);
    let (ok, out, _) = tt(&a);
    assert!(ok, "{out}");
    assert!(m.exists() && ck.exists() && tk.exists());
}

#[test]
fn finetune_and_export_check_their_output_up_front_and_create_parents() {
    let d = tmp("ft");
    let model = d.join("m.gguf");
    std::fs::write(&model, tiny_llama_gguf_with_tokens(&letter_vocab())).unwrap();
    let c = d.join("c.txt");
    std::fs::write(&c, "abc abc abc abc abc abc abc abc").unwrap();
    let (m, c) = (model.to_str().unwrap(), c.to_str().unwrap());
    let bad = blocked(&d, "o.flck");
    let (ok, out, _) = tt(&["finetune-gguf", m, c, &bad, "--epochs", "1"]);
    assert!(
        !ok && out.contains(&format!("cannot write {bad}: ")),
        "{out}"
    );
    assert!(
        !out.contains("Opening"),
        "checked before opening the model: {out}"
    );
    let bad = blocked(&d, "o.gguf");
    let (ok, out, _) = tt(&["export-gguf", m, &bad]);
    assert!(
        !ok && out.contains(&format!("cannot write {bad}: ")),
        "{out}"
    );
    assert!(!out.contains("Opening"), "{out}");
    // Missing parents are created, like `train` does.
    let ck = d.join("n/e/w/o.flck");
    let (ok, out, _) = tt(&[
        "finetune-gguf",
        m,
        c,
        ck.to_str().unwrap(),
        "--epochs",
        "1",
        "--batch",
        "2",
        "--seq",
        "8",
    ]);
    assert!(ok && ck.exists(), "{out}");
    let ex = d.join("x/y/o.gguf");
    let (ok, out, _) = tt(&["export-gguf", m, ex.to_str().unwrap()]);
    assert!(ok && ex.exists(), "{out}");
}

#[test]
fn eval_json_path_is_checked_before_evaluating() {
    let d = tmp("eval");
    let c = corpus(&d);
    let model = d.join("m.gguf");
    let mut a = vec!["train", &c, model.to_str().unwrap()];
    let mut tiny: Vec<&str> = TINY.to_vec();
    tiny[1] = "1";
    a.extend(tiny);
    a.extend(["--vocab", "300"]);
    assert!(tt(&a).0);
    let bad = blocked(&d, "x.json");
    let (ok, out, _) = tt(&["eval", model.to_str().unwrap(), &c, "--json", &bad]);
    assert!(
        !ok && out.contains(&format!("cannot write {bad}: ")),
        "{out}"
    );
    assert!(!out.contains("Perplexity"), "evaluated first: {out}");
}
