//! Binary-spawn happy-path test for the `export-gguf` subcommand.
//!
//! The library round-trip (`tests/tests/test_gguf_export.rs` in the
//! `integration_tests` crate) covers `llama_gguf_bytes` directly, but the
//! actual `train_transformer export-gguf` subcommand — its arg parsing,
//! streamed open, and file write — was otherwise untested. This spawns the
//! real binary against a tiny synthetic `llama` GGUF and confirms the output
//! re-opens and re-loads as a runnable model.

use ferrum_core::{Gguf, GgufBuilder, MetaValue};

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    let mut o = Vec::new();
    for &x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

/// A tiny but complete `llama`-architecture GGUF (2 layers, dim 8, all F32
/// tensors). Mirrors `tiny_llama` in `tests/tests/test_gguf_export.rs`; kept
/// as a minimal duplicate here since that helper lives in a different crate.
fn tiny_llama_gguf_bytes() -> Vec<u8> {
    tiny_llama_gguf_with_tokens(&(0..32).map(|i| format!("t{i}")).collect::<Vec<_>>())
}

/// Same model with a caller-supplied (32-entry) vocabulary, so a corpus can
/// actually be tokenized.
fn tiny_llama_gguf_with_tokens(tokens: &[String]) -> Vec<u8> {
    const GGML_F32: u32 = 0;
    let (dim, n_layers, n_heads, ffn, vocab) = (8usize, 2usize, 2usize, 16usize, 32usize);
    let head_dim = dim / n_heads;
    let gen = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((seed * 131 + i * 17) % 101) as f32 / 500.0) - 0.1)
            .collect()
    };

    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("llama".into()));
    b.meta("llama.embedding_length", MetaValue::U32(dim as u32));
    b.meta("llama.block_count", MetaValue::U32(n_layers as u32));
    b.meta("llama.attention.head_count", MetaValue::U32(n_heads as u32));
    b.meta(
        "llama.attention.head_count_kv",
        MetaValue::U32(n_heads as u32),
    );
    b.meta("llama.feed_forward_length", MetaValue::U32(ffn as u32));
    b.meta("llama.context_length", MetaValue::U32(16));
    b.meta(
        "llama.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-5),
    );
    b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
    b.meta(
        "tokenizer.ggml.tokens",
        MetaValue::Array(
            tokens
                .iter()
                .map(|t| MetaValue::String(t.clone()))
                .collect(),
        ),
    );

    let t = |b: &mut GgufBuilder, name: &str, dims: &[u64], seed: usize| {
        let n: usize = dims.iter().product::<u64>() as usize;
        b.tensor(name, dims, GGML_F32, f32_bytes(&gen(seed, n)));
    };
    t(&mut b, "token_embd.weight", &[dim as u64, vocab as u64], 1);
    for i in 0..n_layers {
        let p = format!("blk.{i}");
        t(
            &mut b,
            &format!("{p}.attn_norm.weight"),
            &[dim as u64],
            10 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_q.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            20 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_k.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            30 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_v.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            40 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_output.weight"),
            &[(n_heads * head_dim) as u64, dim as u64],
            50 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_norm.weight"),
            &[dim as u64],
            60 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_gate.weight"),
            &[dim as u64, ffn as u64],
            70 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_up.weight"),
            &[dim as u64, ffn as u64],
            80 + i,
        );
        t(
            &mut b,
            &format!("{p}.ffn_down.weight"),
            &[ffn as u64, dim as u64],
            90 + i,
        );
    }
    t(&mut b, "output_norm.weight", &[dim as u64], 200);
    t(&mut b, "output.weight", &[dim as u64, vocab as u64], 201);
    b.into_bytes()
}

#[test]
fn export_gguf_cli_happy_path() {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let in_path = std::env::temp_dir().join(format!("ferrum_export_cli_in_{pid}_{nanos}.gguf"));
    let out_path = std::env::temp_dir().join(format!("ferrum_export_cli_out_{pid}_{nanos}.gguf"));

    std::fs::write(&in_path, tiny_llama_gguf_bytes()).expect("failed to write input GGUF fixture");

    let status = std::process::Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args([
            "export-gguf",
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
            "--quant",
            "q8_0",
        ])
        .status()
        .expect("failed to spawn train_transformer binary");
    assert!(
        status.success(),
        "export-gguf subcommand exited with failure status: {status:?}"
    );

    assert!(
        out_path.exists(),
        "export-gguf did not produce an output file at {out_path:?}"
    );
    let g = Gguf::open(out_path.to_str().unwrap()).expect("re-opening exported GGUF failed");
    assert_eq!(g.architecture(), Some("llama"));
    g.load_llama_prec(None)
        .expect("exported GGUF failed to load as a runnable model");

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_ft_cli_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tt(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .expect("failed to spawn train_transformer")
}

fn text(o: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

/// 26 letters, then `Ġ` (byte-level space) and 5 filler tokens.
fn letter_vocab() -> Vec<String> {
    let mut v: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
    v.push("\u{120}".into());
    v.extend((0..5).map(|i| format!("<f{i}>")));
    v
}

#[test]
fn finetune_gguf_cli_trains_resumes_and_exports() {
    let d = tmp("pipeline");
    let model = d.join("m.gguf");
    std::fs::write(&model, tiny_llama_gguf_with_tokens(&letter_vocab())).unwrap();
    let corpus = d.join("c.txt");
    std::fs::write(&corpus, "abc abc abc abc abc abc abc abc").unwrap();
    let (m, c) = (model.to_str().unwrap(), corpus.to_str().unwrap());
    let ck = d.join("o.flck");
    let ck_s = ck.to_str().unwrap();

    let o = tt(&[
        "finetune-gguf",
        m,
        c,
        ck_s,
        "--epochs",
        "2",
        "--batch",
        "2",
        "--seq",
        "8",
        "--warmup",
        "2",
        "--lr",
        "1e-2",
        "--threads",
        "2",
        "--sample",
        "--max",
        "4",
    ]);
    let out = text(&o);
    assert!(o.status.success(), "{out}");
    assert!(
        out.contains("epoch   2/2") && out.contains("Saved fine-tune checkpoint"),
        "{out}"
    );
    assert!(out.contains("── sample ──"), "{out}");
    assert!(ck.exists());

    // Continue from the checkpoint (resume path, no warmup).
    let ck2 = d.join("o2.flck");
    let o = tt(&[
        "finetune-gguf",
        m,
        c,
        ck2.to_str().unwrap(),
        "--epochs",
        "1",
        "--batch",
        "2",
        "--seq",
        "8",
        "--resume",
        ck_s,
        "--threads",
        "1",
        "--qat",
    ]);
    let out = text(&o);
    assert!(o.status.success(), "{out}");
    assert!(
        out.contains("resumed from") && out.contains("QAT int8"),
        "{out}"
    );

    // Apply it for generation and for export.
    let o = tt(&[
        "run-gguf", m, "abc", "--raw", "--max", "4", "--temp", "0", "--resume", ck_s,
    ]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(
        text(&o).contains("applied fine-tuned checkpoint"),
        "{}",
        text(&o)
    );
    let out_gguf = d.join("ft.gguf");
    let o = tt(&[
        "export-gguf",
        m,
        out_gguf.to_str().unwrap(),
        "--resume",
        ck_s,
        "--quant",
        "f32",
    ]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(
        text(&o).contains("Applying fine-tune checkpoint"),
        "{}",
        text(&o)
    );
    Gguf::open(out_gguf.to_str().unwrap())
        .unwrap()
        .load_llama_prec(None)
        .unwrap();
}

#[test]
fn finetune_gguf_cli_rejects_bad_input() {
    let d = tmp("errors");
    let tokenless = d.join("n.gguf");
    std::fs::write(&tokenless, tiny_llama_gguf_with_tokens(&[])).unwrap();
    let model = d.join("m.gguf");
    std::fs::write(&model, tiny_llama_gguf_with_tokens(&letter_vocab())).unwrap();
    let corpus = d.join("c.txt");
    std::fs::write(&corpus, "abc").unwrap();
    let (n, m, c) = (
        tokenless.to_str().unwrap(),
        model.to_str().unwrap(),
        corpus.to_str().unwrap(),
    );
    let o = d.join("o.flck");
    let o = o.to_str().unwrap();

    let usage = tt(&["finetune-gguf", m]);
    assert!(
        !usage.status.success() && text(&usage).contains("usage:"),
        "{}",
        text(&usage)
    );
    let missing = tt(&["finetune-gguf", "/no/such.gguf", c, o]);
    assert!(
        text(&missing).contains("cannot open GGUF"),
        "{}",
        text(&missing)
    );
    let no_tok = tt(&["finetune-gguf", n, c, o]);
    assert!(!no_tok.status.success(), "{}", text(&no_tok));
    let no_corpus = tt(&["finetune-gguf", m, "/no/such.txt", o]);
    assert!(
        text(&no_corpus).contains("cannot read corpus"),
        "{}",
        text(&no_corpus)
    );
    let short = tt(&["finetune-gguf", m, c, o, "--seq", "9"]);
    assert!(
        text(&short).contains("supply more text"),
        "{}",
        text(&short)
    );
    let lr = tt(&["finetune-gguf", m, c, o, "--lr", "-1"]);
    assert!(!lr.status.success(), "{}", text(&lr));
    let res = tt(&[
        "finetune-gguf",
        m,
        c,
        o,
        "--seq",
        "2",
        "--resume",
        "/no/such.flck",
    ]);
    assert!(
        text(&res).contains("checkpoint /no/such.flck does not exist"),
        "{}",
        text(&res)
    );
    assert!(!std::path::Path::new(o).exists());
}
