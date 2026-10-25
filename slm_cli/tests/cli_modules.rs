//! Per-subcommand integration tests for the modules split out of `main.rs`: usage errors,
//! the train-or-load paths, and every `run-gguf` / `export-gguf` / `finetune-gguf` branch
//! that needs a real (tiny, synthetic) `llama` GGUF.
mod common;

use common::{letter_vocab, tiny_llama_gguf};
use ferrum_core::MetaValue;
use std::path::PathBuf;
use std::process::Command;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_mods_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// (success, stdout + stderr)
fn tt(args: &[&str]) -> (bool, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .unwrap();
    (
        o.status.success(),
        String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr),
    )
}

fn s(p: &std::path::Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn every_subcommand_prints_its_usage_when_arguments_are_missing() {
    for (args, usage) in [
        (&["train", "c.txt"][..], "usage: train_transformer train "),
        (&["run", "c.txt", "m.gguf"], "usage: train_transformer run "),
        (
            &["generate", "m.gguf"],
            "usage: train_transformer generate ",
        ),
        (&["eval", "m.gguf"], "usage: train_transformer eval "),
        (&["info"], "usage: train_transformer info"),
        (&["run-gguf"], "usage: train_transformer run-gguf "),
        (
            &["finetune-gguf", "m.gguf", "c.txt"],
            "usage: train_transformer finetune-gguf ",
        ),
        (
            &["export-gguf", "in.gguf"],
            "usage: train_transformer export-gguf ",
        ),
    ] {
        let (ok, out) = tt(args);
        assert!(!ok && out.contains(usage), "{args:?}: {out}");
    }
}

/// `train` twice: the second call finds the model and skips training; `--sample` prints one;
/// `run` trains when the model is missing and reuses it after.
#[test]
fn train_skips_existing_models_samples_and_run_trains_when_missing() {
    let d = tmp("train");
    let corpus = d.join("c.txt");
    std::fs::write(&corpus, "the cat sat on the mat. ".repeat(30)).unwrap();
    let model = d.join("m.gguf");
    let small = [
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
        "--epochs",
        "1",
        "--vocab",
        "0",
    ];
    let args = |extra: &[&'static str]| -> Vec<String> {
        let mut v: Vec<String> = ["train", s(&corpus), s(&model)]
            .iter()
            .map(|x| x.to_string())
            .collect();
        v.extend(small.iter().chain(extra).map(|x| x.to_string()));
        v
    };
    let run = |v: Vec<String>| {
        let r: Vec<&str> = v.iter().map(String::as_str).collect();
        tt(&r)
    };
    let (ok, out) = run(args(&["--sample"]));
    assert!(ok, "{out}");
    assert!(out.contains("── sample ──"), "{out}");
    let (ok, out) = run(args(&["--sample"]));
    assert!(ok, "{out}");
    assert!(
        out.contains("already exists") && out.contains("Skipping training"),
        "{out}"
    );
    assert!(
        out.contains("── sample ──"),
        "sample also after a load: {out}"
    );

    // `run` with no model file trains first, then generates; the next call reuses it.
    let model2 = d.join("m2.gguf");
    let mut r = vec!["run", s(&corpus), s(&model2), "the", "cat"];
    r.extend(small);
    r.extend(["--chars", "10"]);
    let (ok, out) = tt(&r);
    assert!(ok, "{out}");
    assert!(model2.exists() && !out.contains("no retraining"), "{out}");
    let (ok, out) = tt(&r);
    assert!(ok && out.contains("no retraining needed"), "{out}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn eval_and_generate_name_a_missing_model_or_empty_corpus() {
    let d = tmp("eval_err");
    let empty = d.join("empty.txt");
    std::fs::write(&empty, "  \n").unwrap();
    let (ok, out) = tt(&["eval", s(&d.join("nope.gguf")), s(&empty)]);
    assert!(!ok && out.contains("cannot load model"), "{out}");
    let (ok, out) = tt(&["generate", s(&d.join("nope.gguf")), "hi"]);
    assert!(!ok && out.contains("cannot load model"), "{out}");
    let _ = std::fs::remove_dir_all(&d);
}

fn llama_file(d: &std::path::Path, name: &str, bytes: Vec<u8>) -> PathBuf {
    let p = d.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

#[test]
fn run_gguf_quant_choices_and_prompt_ids() {
    let d = tmp("rungguf");
    let model = llama_file(&d, "m.gguf", tiny_llama_gguf(Some(&letter_vocab()), &[]));
    // Every accepted --quant spelling loads and generates (raw text, no chat wrapping).
    for q in ["int4", "q4", "int8", "q8", "f32", "none"] {
        let (ok, out) = tt(&[
            "run-gguf",
            s(&model),
            "abc",
            "--raw",
            "--max",
            "3",
            "--temp",
            "0",
            "--quant",
            q,
        ]);
        assert!(ok && out.contains("── output ──"), "{q}: {out}");
        assert!(out.contains(&format!("--quant {q}")), "{q}: {out}");
    }
    // An unknown one is refused before anything loads.
    let (ok, out) = tt(&["run-gguf", s(&model), "abc", "--quant", "int3"]);
    assert!(
        !ok && out.contains("--quant must be int4|int8|f32 (got 'int3')"),
        "{out}"
    );
    // Text with a tokenizer but no characters of it: the prompt encodes to nothing.
    let (ok, out) = tt(&["run-gguf", s(&model), "--raw"]);
    assert!(!ok && out.contains("empty prompt"), "{out}");
    // --ids wins over text and prints token ids back when there is no tokenizer.
    let bare = llama_file(&d, "bare.gguf", tiny_llama_gguf(None, &[]));
    let (ok, out) = tt(&[
        "run-gguf",
        s(&bare),
        "--ids",
        "1 2 3 junk",
        "--max",
        "3",
        "--temp",
        "0",
    ]);
    assert!(ok, "{out}");
    assert!(
        out.contains("tokenizer = none in file") && out.contains("token ids: "),
        "{out}"
    );
    // No tokenizer and no ids: a clear error. Ids that all fail to parse: empty prompt.
    let (ok, out) = tt(&["run-gguf", s(&bare), "hello"]);
    assert!(!ok && out.contains("has no tokenizer; pass --ids"), "{out}");
    let (ok, out) = tt(&["run-gguf", s(&bare), "--ids", "x y"]);
    assert!(!ok && out.contains("empty prompt"), "{out}");
    // Not a GGUF at all.
    let junk = d.join("junk.gguf");
    std::fs::write(&junk, b"not gguf").unwrap();
    let (ok, out) = tt(&["run-gguf", s(&junk), "hi"]);
    assert!(!ok && out.contains("cannot open GGUF"), "{out}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn run_gguf_wraps_chat_templates_unless_raw_and_applies_a_resume_checkpoint() {
    let d = tmp("chat");
    let mut vocab = letter_vocab();
    vocab[28] = "<|im_start|>".into();
    vocab[29] = "<|im_end|>".into();
    let extra = [
        (
            "tokenizer.chat_template",
            MetaValue::String("{{ '<|im_start|>' }}".into()),
        ),
        ("llama.context_length", MetaValue::U32(64)),
        ("tokenizer.ggml.bos_token_id", MetaValue::U32(30)),
        ("tokenizer.ggml.add_bos_token", MetaValue::Bool(true)),
    ];
    let model = llama_file(&d, "chatml.gguf", tiny_llama_gguf(Some(&vocab), &extra));
    let (ok, out) = tt(&["run-gguf", s(&model), "abc", "--max", "4", "--temp", "0"]);
    assert!(ok, "{out}");
    assert!(out.contains("chat format = ChatMl"), "{out}");
    let (ok, out) = tt(&[
        "run-gguf",
        s(&model),
        "abc",
        "--raw",
        "--max",
        "4",
        "--temp",
        "0",
    ]);
    assert!(ok && !out.contains("chat format"), "{out}");
    // A fine-tune checkpoint forces f32 and says --quant is ignored (unless it is f32/int4).
    let corpus = d.join("c.txt");
    std::fs::write(&corpus, "abc abc abc abc abc abc abc abc").unwrap();
    let ck = d.join("o.flck");
    let (ok, out) = tt(&[
        "finetune-gguf",
        s(&model),
        s(&corpus),
        s(&ck),
        "--epochs",
        "1",
        "--batch",
        "2",
        "--seq",
        "8",
        "--threads",
        "1",
        "--sample",
        "--max",
        "3",
    ]);
    assert!(ok, "{out}");
    assert!(out.contains("── sample ──"), "{out}");
    let (ok, out) = tt(&[
        "run-gguf",
        s(&model),
        "abc",
        "--raw",
        "--max",
        "2",
        "--resume",
        s(&ck),
        "--quant",
        "int8",
    ]);
    assert!(ok, "{out}");
    assert!(
        out.contains("ignoring --quant int8") && out.contains("applied fine-tuned"),
        "{out}"
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn export_gguf_every_quant_and_unknown_quant() {
    let d = tmp("export");
    let model = llama_file(&d, "m.gguf", tiny_llama_gguf(Some(&letter_vocab()), &[]));
    for q in ["f16", "f32", "q8_0", "q4_0"] {
        let out = d.join(format!("{q}.gguf"));
        let (ok, text) = tt(&["export-gguf", s(&model), s(&out), "--quant", q]);
        assert!(ok && text.contains("tensors: "), "{q}: {text}");
        assert!(text.contains("estimated peak"), "{q}: {text}");
    }
    let (ok, text) = tt(&[
        "export-gguf",
        s(&model),
        s(&d.join("x.gguf")),
        "--quant",
        "q9",
    ]);
    assert!(!ok && text.contains("unknown --quant 'q9'"), "{text}");
    let (ok, text) = tt(&["export-gguf", s(&d.join("nope.gguf")), s(&d.join("y.gguf"))]);
    assert!(!ok && text.contains("cannot open GGUF"), "{text}");
    let _ = std::fs::remove_dir_all(&d);
}

/// `--checkpoint` writes one while training; `--resume` continues from it and says so; a corpus
/// too short for the context window (with or without a validation split) is a clean error.
#[test]
fn train_checkpoint_resume_and_short_corpus_errors() {
    let d = tmp("resume");
    let corpus = d.join("c.txt");
    std::fs::write(&corpus, "abcdefghijklmnopqrstuvwxyz ".repeat(2)).unwrap();
    let ck = d.join("ck.fckp");
    let small = [
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
        "--vocab",
        "0",
        "--force",
    ];
    let train = |model: &str, extra: &[&str]| {
        let mut v = vec!["train", s(&corpus), model];
        v.extend(small);
        v.extend(extra);
        tt(&v)
    };
    let m1 = s(&d.join("m1.gguf")).to_string();
    let (ok, out) = train(
        &m1,
        &[
            "--epochs",
            "2",
            "--checkpoint",
            s(&ck),
            "--checkpoint-every",
            "1",
        ],
    );
    assert!(ok && ck.exists(), "{out}");
    let m2 = s(&d.join("m2.gguf")).to_string();
    let (ok, out) = train(&m2, &["--epochs", "3", "--resume", s(&ck)]);
    assert!(
        ok && out.contains(&format!("Resume  : {}", s(&ck))),
        "{out}"
    );
    // Too short for the window.
    std::fs::write(&corpus, "abcdefgh").unwrap();
    for extra in [&["--epochs", "1"][..], &["--epochs", "1", "--val", "0.5"]] {
        let (ok, out) = train(&m1, extra);
        assert!(
            !ok && out.contains("longer than the context window"),
            "{extra:?}: {out}"
        );
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// A llama GGUF with no chat template is only wrapped when its vocabulary has the Llama-3
/// header token (the instruct signal); a plain vocabulary is a base model and stays raw.
#[test]
fn run_gguf_detects_llama3_instruct_from_the_vocabulary_alone() {
    let d = tmp("llama3");
    let mut vocab = letter_vocab();
    vocab[28] = "<|start_header_id|>".into();
    vocab[29] = "<|end_header_id|>".into();
    vocab[30] = "<|eot_id|>".into();
    let extra = [("llama.context_length", MetaValue::U32(64))];
    let model = llama_file(&d, "l3.gguf", tiny_llama_gguf(Some(&vocab), &extra));
    let (ok, out) = tt(&["run-gguf", s(&model), "abc", "--max", "3", "--temp", "0"]);
    assert!(ok && out.contains("chat format = Llama3"), "{out}");
    let base = llama_file(
        &d,
        "base.gguf",
        tiny_llama_gguf(Some(&letter_vocab()), &extra),
    );
    let (ok, out) = tt(&["run-gguf", s(&base), "abc", "--max", "3", "--temp", "0"]);
    assert!(ok && !out.contains("chat format"), "{out}");
    let _ = std::fs::remove_dir_all(&d);
}
