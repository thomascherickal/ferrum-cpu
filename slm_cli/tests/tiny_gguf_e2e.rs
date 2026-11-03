//! The imported-GGUF path through the real binary with a synthetic tiny llama (dim 8, 2 layers):
//! finetune-gguf, export-gguf --resume, run-gguf on the base plus checkpoint versus the exported
//! file, and the Ollama/LM Studio Modelfile export (never contacting either tool).
mod common;

use common::{letter_vocab, tiny_llama_gguf_with_tokens};
use ferrum_core::{Gguf, GgufBuilder, MetaValue};
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

/// The generated text of a `run-gguf` run: its stdout (status chatter goes to stderr).
fn generated(args: &[&str]) -> String {
    let o = tt(args);
    assert!(
        o.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).unwrap().trim_end().to_string()
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
        generated(&a)
    };
    let resumed = run(&base, &["--resume", &ck]);
    let exported = run(&tuned, &[]);
    assert_eq!(resumed, exported);
    assert!(resumed.starts_with("abc"), "{resumed:?}");
    // The fine-tune changed behaviour, and re-running either side repeats exactly.
    assert_ne!(resumed, run(&base, &[]), "fine-tune had no effect");
    assert_eq!(exported, run(&tuned, &[]));

    // eval-gguf: the fine-tune lowers the loss on its own training text, and scoring base plus
    // checkpoint equals scoring the exported file.
    let eval = |model: &str, extra: &[&str]| -> serde_json::Value {
        let json = t.s("eval.json");
        let mut a = vec!["eval-gguf", model, &corpus, "--json", &json];
        a.extend(extra);
        let out = ok(&a);
        for key in [
            "Predictions",
            "Cross-entropy",
            "Bits/token",
            "Perplexity",
            "Bits/byte",
        ] {
            assert!(out.contains(key), "missing {key}: {out}");
        }
        serde_json::from_slice(&std::fs::read(json).unwrap()).unwrap()
    };
    let (before, after, via_export) = (
        eval(&base, &[]),
        eval(&base, &["--resume", &ck]),
        eval(&tuned, &[]),
    );
    let nats = |j: &serde_json::Value| j["cross_entropy_nats"].as_f64().unwrap();
    assert!(nats(&before).is_finite() && nats(&before) > 0.0);
    assert!(
        nats(&after) < nats(&before),
        "{} !< {}",
        nats(&after),
        nats(&before)
    );
    assert_eq!(after["tokens"], before["tokens"]);
    assert!((nats(&after) - nats(&via_export)).abs() < 1e-4);
    let (w, bad) = (
        eval(&base, &["--ctx", "8"]),
        tt(&["eval-gguf", &base, &corpus, "--ctx", "99"]),
    );
    assert!(w["tokens"].as_u64().unwrap() > 0);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("--ctx must be between 2 and 16"));

    // The exported file is a loadable llama GGUF with the same shape as the base.
    let (b, e) = (Gguf::open(&base).unwrap(), Gguf::open(&tuned).unwrap());
    assert_eq!(e.architecture(), Some("llama"));
    assert_eq!(e.tensors.len(), b.tensors.len());
    e.load_llama_prec(None).unwrap();
    // `info` describes an imported llama file from its header.
    let info = ok(&["info", &tuned]);
    let params: usize = e.tensors.iter().map(|t| t.num_elements()).sum();
    for want in [
        "Format    : GGUF v3 (llama, f32)".to_string(),
        format!("Params    : {params}"),
        "Layers    : 2".to_string(),
        "Dim       : 8".to_string(),
        "Heads     : 2 (2 KV)".to_string(),
        "Vocab     : 32".to_string(),
        "Context   : 16".to_string(),
    ] {
        assert!(info.contains(&want), "missing {want:?}: {info}");
    }
    assert!(info.contains("Tokenizer : "), "{info}");
    // A fine-tune checkpoint is not a model: say what it is instead of a format error.
    let i = ok(&["info", &ck]);
    assert!(i.contains("ferrum-llama-checkpoint"), "{i}");
}

/// A stop file ends `finetune-gguf` after a step with the checkpoint written; resuming for the
/// remaining epochs equals the uninterrupted run (one step per epoch, so the stop is on a boundary).
#[test]
fn finetune_stops_on_a_stop_file_and_resume_equals_the_full_run() {
    let t = Tmp::new("ftstop");
    std::fs::write(
        t.s("base.gguf"),
        tiny_llama_gguf_with_tokens(&letter_vocab()),
    )
    .unwrap();
    std::fs::write(t.s("c.txt"), "abc abc abc abc abc abc abc abc").unwrap();
    let (base, corpus) = (t.s("base.gguf"), t.s("c.txt"));
    let (full, part, rest) = (t.s("full.flck"), t.s("part.flck"), t.s("rest.flck"));
    let ft = |out: &str, epochs: &str, extra: &[&str]| {
        let mut a = vec![
            "finetune-gguf",
            &base,
            &corpus,
            out,
            "--epochs",
            epochs,
            "--batch",
            "1000",
            "--seq",
            "8",
            "--lr",
            "1e-2",
            "--threads",
            "1",
        ];
        a.extend(extra);
        ok(&a)
    };
    ft(&full, "3", &[]);

    std::fs::write(t.s("part.flck.stop"), b"").unwrap();
    let out = ft(&part, "3", &[]);
    assert!(
        out.contains(&format!(
            "stopped early at step 1: stop file found; resume with --resume {part}"
        )),
        "{out}"
    );
    assert!(std::path::Path::new(&part).is_file());
    assert!(!std::path::Path::new(&t.s("part.flck.stop")).exists());

    ft(&rest, "2", &["--resume", &part]);
    assert_eq!(std::fs::read(&full).unwrap(), std::fs::read(&rest).unwrap());

    let out = ft(&t.s("timed.flck"), "3", &["--max-seconds", "0.000001"]);
    assert!(
        out.contains("stopped early at step 1: time budget reached"),
        "{out}"
    );
}

/// A stop after step 1 of a 6-step epoch, then `--resume` for the 3 epochs not finished, writes
/// the same `.flck` (and export) as the uninterrupted run, serial and with two threads, with
/// dropout and a warmup schedule on. A resume on another corpus or batch exits 1 and writes nothing.
#[test]
fn finetune_resumed_mid_epoch_is_byte_identical_and_another_corpus_is_refused() {
    let t = Tmp::new("ftmid");
    std::fs::write(
        t.s("base.gguf"),
        tiny_llama_gguf_with_tokens(&letter_vocab()),
    )
    .unwrap();
    std::fs::write(t.s("c.txt"), "abc abc abc abc abc abc abc abc").unwrap();
    std::fs::write(t.s("d.txt"), "cba cba cba cba cba cba cba cba").unwrap();
    let base = t.s("base.gguf");
    let ft = |corpus: &str, out: &str, threads: &str, extra: &[&str]| {
        let corpus = t.s(corpus);
        let mut a = vec![
            "finetune-gguf",
            &base,
            &corpus,
            out,
            "--epochs",
            "3",
            "--batch",
            "4",
            "--seq",
            "8",
            "--lr",
            "1e-2",
            "--dropout",
            "0.1",
            "--warmup",
            "2",
            "--threads",
            threads,
        ];
        a.extend(extra);
        tt(&a)
    };
    for threads in ["1", "2"] {
        let full = t.s(&format!("full{threads}.flck"));
        let part = t.s(&format!("part{threads}.flck"));
        let rest = t.s(&format!("rest{threads}.flck"));
        let out = ft("c.txt", &full, threads, &[]);
        let text = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(
            out.status.success() && text.contains("steps/epoch"),
            "{text}"
        );
        assert!(
            !text.contains("· 1 steps/epoch"),
            "the stop must fall inside an epoch: {text}"
        );
        std::fs::write(format!("{part}.stop"), b"").unwrap();
        let out = ft("c.txt", &part, threads, &[]);
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("stopped early at step 1"),
            "{out:?}"
        );
        assert!(ft("c.txt", &rest, threads, &["--resume", &part])
            .status
            .success());
        assert!(
            std::fs::read(&full).unwrap() == std::fs::read(&rest).unwrap(),
            "threads {threads}: the resumed checkpoint differs from the uninterrupted one"
        );
        let export = |ck: &str, name: &str| {
            ok(&["export-gguf", &base, &t.s(name), "--resume", ck]);
            std::fs::read(t.s(name)).unwrap()
        };
        assert!(export(&full, "full.gguf") == export(&rest, "rest.gguf"));
    }

    // Another corpus: refused with both ids, exit 1, the checkpoint untouched, nothing written.
    let part = t.s("part1.flck");
    let before = std::fs::read(&part).unwrap();
    let out = ft("d.txt", &t.s("x.flck"), "1", &["--resume", &part]);
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    let (a, b) = (
        slm_cli::hash::file_id(&t.s("c.txt")).unwrap(),
        slm_cli::hash::file_id(&t.s("d.txt")).unwrap(),
    );
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(
        err.contains("corpus") && err.contains(&a) && err.contains(&b),
        "{err}"
    );
    let out = ft(
        "c.txt",
        &t.s("x.flck"),
        "1",
        &["--resume", &part, "--batch", "2"],
    );
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(
        err.contains("--batch is 2 but the checkpoint was trained with batch 4"),
        "{err}"
    );
    let out = ft(
        "c.txt",
        &t.s("x.flck"),
        "1",
        &["--resume", &part, "--seq", "6"],
    );
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("--seq is 6"), "{err}");
    assert_eq!(std::fs::read(&part).unwrap(), before);
    assert!(!std::path::Path::new(&t.s("x.flck")).exists());
    // A changed learning rate is only noted.
    let out = ft(
        "c.txt",
        &t.s("y.flck"),
        "1",
        &["--resume", &part, "--lr", "5e-3"],
    );
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "{out:?}");
    assert!(
        text.contains("note: training settings changed") && text.contains("lr 0.01 -> 0.005"),
        "{text}"
    );
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
            text.contains("architecture 'llama': Ferrum exports only its own gpt2 models")
                && text.contains("Modelfile by hand (Manual 9.5)"),
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

/// A stand-in for a model far bigger than any RAM, written as a sparse file: llama metadata, a
/// vocabulary, and one Q8_0 tensor of 2^36 elements (256 GiB as f32). The file only has to be as
/// long as the tensor claims, and a sparse file costs no disk. Commands that size a model from
/// its tensor directory refuse it before reading any weights. Returns `false` (and removes the
/// file) when the temp filesystem cannot make a sparse file that large, so the caller can skip.
#[cfg(target_os = "linux")]
fn write_over_budget_llama(path: &str) -> bool {
    const ELEMENTS: u64 = 1 << 36;
    const Q8_0: u32 = 8;
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("llama".into()));
    for (k, v) in [
        ("embedding_length", 8u32),
        ("block_count", 2),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 2),
        ("feed_forward_length", 16),
        ("context_length", 16),
    ] {
        b.meta(&format!("llama.{k}"), MetaValue::U32(v));
    }
    b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
    b.meta(
        "tokenizer.ggml.tokens",
        MetaValue::Array(letter_vocab().into_iter().map(MetaValue::String).collect()),
    );
    b.tensor("blk.0.huge.weight", &[ELEMENTS], Q8_0, vec![0; 34]);
    let bytes = b.into_bytes();
    std::fs::write(path, &bytes).unwrap();
    let f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    let len = bytes.len() as u64 + ELEMENTS / 32 * 34;
    // A ~73 GB sparse file needs a filesystem with sparse-file support and no file-size limit.
    // Windows (NTFS) would allocate it for real, so there the test is skipped without trying.
    let sparse = cfg!(unix) && f.set_len(len).is_ok() && is_sparse(&f);
    if !sparse {
        let _ = std::fs::remove_file(path);
    }
    sparse
}

/// Fewer than 1 MiB of disk blocks actually allocated.
#[cfg(unix)]
fn is_sparse(f: &std::fs::File) -> bool {
    std::os::unix::fs::MetadataExt::blocks(&f.metadata().unwrap()) * 512 < (1 << 20)
}
#[cfg(not(unix))]
fn is_sparse(_: &std::fs::File) -> bool {
    false
}

/// A GGUF with an unsupported architecture, no tokenizer, and one tensor of ggml type `ty`.
fn bare_gguf(arch: &str, ty: u32) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String(arch.into()));
    b.tensor("w", &[32], ty, vec![0; 128]);
    b.into_bytes()
}

// The RAM guard reads /proc/meminfo, so it is inactive off Linux (Status 7.0.13).
#[cfg(target_os = "linux")]
#[test]
fn every_loading_command_refuses_a_model_over_the_ram_budget() {
    let t = Tmp::new("budget");
    if !write_over_budget_llama(&t.s("big.gguf")) {
        eprintln!("SKIPPED: the temp filesystem cannot create a 73 GB sparse file");
        return;
    }
    std::fs::write(t.s("c.txt"), "abc abc abc abc abc abc abc abc").unwrap();
    let (big, corpus) = (t.s("big.gguf"), t.s("c.txt"));
    let (out, ck) = (t.s("out.gguf"), t.s("out.flck"));
    for (args, what) in [
        (vec!["eval-gguf", &big, &corpus], "resident memory"),
        (vec!["run-gguf", &big, "abc"], "resident memory"),
        (vec!["export-gguf", &big, &out], "peak memory"),
        (vec!["finetune-gguf", &big, &corpus, &ck], "memory"),
    ] {
        let o = tt(&args);
        let text = String::from_utf8_lossy(&o.stderr);
        assert!(!o.status.success(), "{args:?}: {text}");
        assert!(text.contains(what), "{args:?}: {text}");
        assert!(!std::path::Path::new(&out).exists() && !std::path::Path::new(&ck).exists());
    }
}

#[test]
fn eval_gguf_usage_options_and_short_text_are_refused() {
    let t = Tmp::new("evalerr");
    std::fs::write(t.s("m.gguf"), tiny_llama_gguf_with_tokens(&letter_vocab())).unwrap();
    std::fs::write(t.s("one.txt"), "a").unwrap();
    let (m, one) = (t.s("m.gguf"), t.s("one.txt"));
    for (args, want) in [
        (vec!["eval-gguf", &m], "usage: train_transformer eval-gguf"),
        (
            vec!["eval-gguf", &m, &one, "--quant", "fp8"],
            "--quant must be f32|int8|int4 (got 'fp8')",
        ),
        (vec!["eval-gguf", &m, &one], "at least two tokens"),
    ] {
        let o = tt(&args);
        let text = String::from_utf8_lossy(&o.stderr);
        assert!(
            !o.status.success() && text.contains(want),
            "{args:?}: {text}"
        );
    }
}

#[test]
fn eval_gguf_with_checkpoint_and_precision() {
    let t = Tmp::new("evalckpt");
    std::fs::write(t.s("m.gguf"), tiny_llama_gguf_with_tokens(&letter_vocab())).unwrap();
    std::fs::write(t.s("c.txt"), "abc abc abc abc abc abc abc abc").unwrap();
    let (m, c, ck) = (t.s("m.gguf"), t.s("c.txt"), t.s("o.flck"));
    ok(&[
        "finetune-gguf",
        &m,
        &c,
        &ck,
        "--epochs",
        "2",
        "--batch",
        "2",
        "--seq",
        "8",
        "--lr",
        "1e-2",
        "--threads",
        "1",
    ]);
    let base = ok(&["eval-gguf", &m, &c, "--quant", "int8"]);
    assert!(base.contains("Perplexity"), "{base}");
    // A checkpoint holds f32 weights: --quant is ignored with a note, and the scores change.
    let tuned = ok(&["eval-gguf", &m, &c, "--quant", "int8", "--resume", &ck]);
    assert!(tuned.contains("ignoring --quant int8"), "{tuned}");
    assert!(tuned.contains("applied fine-tuned checkpoint"), "{tuned}");
    let line = |s: &str| {
        s.lines()
            .find(|l| l.starts_with("Perplexity"))
            .unwrap()
            .to_string()
    };
    assert_ne!(line(&base), line(&tuned));
}

#[test]
fn info_describes_an_unrunnable_architecture_without_a_tokenizer() {
    let t = Tmp::new("infomamba");
    std::fs::write(t.s("m.gguf"), bare_gguf("mamba", 0)).unwrap();
    let text = ok(&["info", &t.s("m.gguf")]);
    assert!(text.contains("Tokenizer : none in file"), "{text}");
    assert!(
        text.contains("architecture 'mamba' is not runnable (llama / qwen2 only)"),
        "{text}"
    );
}

#[test]
fn info_of_a_token_file_without_an_eos_id_says_none() {
    let t = Tmp::new("infotok");
    let meta = ferrum_core::TokenFileMeta {
        vocab_size: 300,
        eos_id: None,
        tokenizer_sha256: "ab".repeat(32),
    };
    ferrum_core::write_token_file(&t.s("t.gguf"), &meta, |s| s.push(&[1, 2, 3])).unwrap();
    let text = ok(&["info", &t.s("t.gguf")]);
    assert!(text.contains("EOS id    : none"), "{text}");
}

#[test]
fn a_closed_stdout_ends_a_printing_command_quietly_with_exit_zero() {
    let t = Tmp::new("pipe");
    std::fs::write(t.s("m.gguf"), bare_gguf("mamba", 0)).unwrap();
    // The reader end is gone before the command starts, so its first println hits EPIPE.
    let (reader, writer) = std::io::pipe().unwrap();
    drop(reader);
    let o = Command::new(BIN)
        .args(["info", &t.s("m.gguf")])
        .stdout(writer)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        !err.contains("panicked") && !err.contains("Broken pipe"),
        "{err}"
    );
    assert_eq!(o.status.code(), Some(0), "{err}");
}

/// `run-gguf --stop`: the first stop string ends the text before itself, a stop that never
/// appears changes nothing (bit for bit), `--stop` repeats, and a file without a tokenizer
/// refuses it.
#[test]
fn run_gguf_stop_strings_cut_the_text_and_absent_ones_change_nothing() {
    let t = Tmp::new("stop");
    std::fs::write(t.s("m.gguf"), tiny_llama_gguf_with_tokens(&letter_vocab())).unwrap();
    let m = t.s("m.gguf");
    let run = |extra: &[&str]| {
        let mut a = vec!["run-gguf", &m, "ab", "--raw", "--temp", "0", "--max", "10"];
        a.extend_from_slice(extra);
        generated(&a)
    };
    let full = run(&[]);
    assert_eq!(
        run(&["--stop", "zzzz"]),
        full,
        "an absent stop changes nothing"
    );
    let cont = &full["ab".len()..];
    assert!(cont.chars().count() >= 3, "{full:?}");
    let ch: String = cont.chars().nth(2).unwrap().to_string();
    let cut = cont.find(&ch).unwrap();
    assert_eq!(run(&["--stop", &ch]), format!("ab{}", &cont[..cut]));
    // Repeated flags: the earliest match wins whatever the order.
    let first: String = cont.chars().next().unwrap().to_string();
    let want = format!("ab{}", &cont[..cont.find(&first).unwrap()]);
    assert_eq!(run(&["--stop", &ch, "--stop", &first]), want);
    assert_eq!(run(&["--stop", &first, "--stop", &ch]), want);

    // No tokenizer: a text stop cannot be matched, and the error says so before loading.
    std::fs::write(t.s("n.gguf"), common::tiny_llama_gguf(None, &[])).unwrap();
    let o = tt(&["run-gguf", &t.s("n.gguf"), "--ids", "1 2", "--stop", "a"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--stop needs a tokenizer"));
}
