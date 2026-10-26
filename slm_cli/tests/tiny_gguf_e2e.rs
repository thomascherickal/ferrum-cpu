//! The imported-GGUF path through the real binary with a synthetic tiny llama (dim 8, 2 layers):
//! finetune-gguf, export-gguf --resume, run-gguf on the base plus checkpoint versus the exported
//! file, and the Ollama/LM Studio Modelfile export (never contacting either tool).
mod common;

use common::{letter_vocab, tiny_llama_gguf_with_tokens};
use ferrum_core::Gguf;
use std::path::PathBuf;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_train_transformer");

/// A unique temp dir, removed on drop (also when an assertion panics).
struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("ferrum_tiny_gguf_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    fn s(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_string()
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tt(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().unwrap()
}

fn ok(args: &[&str]) -> String {
    let o = tt(args);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(o.status.success(), "{args:?}: {text}");
    text
}

/// The generated text of a `run-gguf` run (between the output rules).
fn generated(out: &str) -> String {
    let after = out.split("── output ──\n").nth(1).expect("no output block");
    after.split("\n────────────").next().unwrap().to_string()
}

/// Fine-tune `base`, export it with the checkpoint, and check the exported file runs exactly
/// like base plus checkpoint.
fn round_trip(tag: &str, base_bytes: Vec<u8>) {
    let t = Tmp::new(tag);
    std::fs::write(t.s("base.gguf"), base_bytes).unwrap();
    std::fs::write(t.s("c.txt"), "abc abc abc abc abc abc abc abc").unwrap();
    let (base, corpus, ck, tuned) = (
        t.s("base.gguf"),
        t.s("c.txt"),
        t.s("o.flck"),
        t.s("ft.gguf"),
    );

    let out = ok(&[
        "finetune-gguf",
        &base,
        &corpus,
        &ck,
        "--epochs",
        "3",
        "--batch",
        "2",
        "--seq",
        "8",
        "--lr",
        "1e-2",
        "--threads",
        "1",
    ]);
    assert!(out.contains("Saved fine-tune checkpoint"), "{out}");
    ok(&[
        "export-gguf",
        &base,
        &tuned,
        "--resume",
        &ck,
        "--quant",
        "f32",
    ]);

    let run = |model: &str, extra: &[&str]| {
        let mut a = vec![
            "run-gguf", model, "abc", "--raw", "--max", "12", "--temp", "0", "--quant", "f32",
        ];
        a.extend(extra);
        generated(&ok(&a))
    };
    let resumed = run(&base, &["--resume", &ck]);
    let exported = run(&tuned, &[]);
    assert_eq!(resumed, exported);
    assert!(resumed.starts_with("abc"), "{resumed:?}");
    // The fine-tune changed behaviour, and re-running either side repeats exactly.
    assert_ne!(resumed, run(&base, &[]), "fine-tune had no effect");
    assert_eq!(exported, run(&tuned, &[]));

    // The exported file is a loadable llama GGUF with the same shape as the base.
    let (b, e) = (Gguf::open(&base).unwrap(), Gguf::open(&tuned).unwrap());
    assert_eq!(e.architecture(), Some("llama"));
    assert_eq!(e.tensors.len(), b.tensors.len());
    e.load_llama_prec(None).unwrap();
    // `info` is for Ferrum's own models and says so for a llama file.
    let i = tt(&["info", &tuned]);
    assert!(!i.status.success());
    assert!(String::from_utf8_lossy(&i.stderr).contains("need gpt2 or ferrum"));
}

#[test]
fn untied_llama_finetune_export_run_round_trip() {
    round_trip("untied", tiny_llama_gguf_with_tokens(&letter_vocab()));
}

#[test]
fn tied_embedding_llama_finetune_export_run_round_trip() {
    round_trip("tied", common::tiny_llama_gguf_tied(&letter_vocab()));
}

#[test]
fn runtime_exports_refuse_llama_and_write_a_modelfile_for_gpt2() {
    let t = Tmp::new("ollama");
    std::fs::write(
        t.s("llama.gguf"),
        tiny_llama_gguf_with_tokens(&letter_vocab()),
    )
    .unwrap();
    let mf = t.s("llama.Modelfile");
    for args in [
        vec![
            "export-ollama",
            &t.s("llama.gguf"),
            "x",
            "--modelfile-only",
            &mf,
        ],
        vec!["export-lmstudio", &t.s("llama.gguf")],
    ] {
        let o = tt(&args);
        let text = String::from_utf8_lossy(&o.stderr);
        assert!(!o.status.success(), "{args:?}");
        assert!(
            text.contains("architecture 'llama' is not runnable in llama.cpp"),
            "{text}"
        );
    }
    assert!(!std::path::Path::new(&mf).exists());

    // A trained BPE model is a gpt2 GGUF: --modelfile-only writes the Modelfile and runs nothing.
    std::fs::write(t.s("c.txt"), "the cat sat on the mat. ".repeat(30)).unwrap();
    ok(&[
        "train",
        &t.s("c.txt"),
        &t.s("m.gguf"),
        "--context",
        "16",
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
        "280",
        "--threads",
        "1",
    ]);
    let model = t.s("m.gguf");
    let out = t.s("m.Modelfile");
    // An empty PATH and a fake HOME make sure no real Ollama could be found even by accident.
    let o = Command::new(BIN)
        .args(["export-ollama", &model, "tiny", "--modelfile-only", &out])
        .env("PATH", "")
        .env("HOME", &t.0)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains(&format!("FROM \"{model}\"")), "{text}");
    assert!(text.contains("TEMPLATE \"{{ .Prompt }}\""), "{text}");
    assert!(text.contains("PARAMETER stop \"<|endoftext|>\""), "{text}");
    assert!(text.contains("PARAMETER num_ctx 16"), "{text}");
}
