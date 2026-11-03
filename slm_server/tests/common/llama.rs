//! A tiny synthetic Llama GGUF (2 layers, dim 8, 32-token letter vocabulary, f32 weights), built
//! in memory: the stub model for the imported-model tests. Shared by the unit tests
//! (`src/test_util.rs` includes this file) and the integration tests.
#![allow(dead_code)] // each test crate uses a different subset

use ferrum_core::{GgufBuilder, MetaValue};

/// The 32-entry vocabulary: `a`..`z`, the byte-level BPE space `Ġ`, then `<f0>`..`<f4>`.
pub fn letter_vocab() -> Vec<String> {
    let mut v: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
    v.push("\u{120}".into());
    v.extend((0..5).map(|i| format!("<f{i}>")));
    v
}

/// Id of `<f0>`, used as BOS and `<f1>`, used as EOS when a test asks for them.
pub const BOS: u32 = 27;
pub const EOS: u32 = 28;

/// The tiny llama with the letter vocabulary, a 16-token context and `extra` metadata (BOS/EOS
/// ids, `add_bos_token`, another architecture name ...). An `extra` key overrides the default.
pub fn tiny_llama(extra: &[(&str, MetaValue)]) -> Vec<u8> {
    const GGML_F32: u32 = 0;
    let (dim, layers, ffn, vocab) = (8u64, 2usize, 16u64, 32u64);
    let arch = extra
        .iter()
        .find(|(k, _)| *k == "general.architecture")
        .and_then(|(_, v)| v.as_str().map(String::from))
        .unwrap_or_else(|| "llama".into());
    let mut b = GgufBuilder::new();
    let mut meta = vec![
        (
            "general.architecture".to_string(),
            MetaValue::String(arch.clone()),
        ),
        (format!("{arch}.embedding_length"), MetaValue::U32(8)),
        (format!("{arch}.block_count"), MetaValue::U32(2)),
        (format!("{arch}.attention.head_count"), MetaValue::U32(2)),
        (format!("{arch}.attention.head_count_kv"), MetaValue::U32(2)),
        (format!("{arch}.feed_forward_length"), MetaValue::U32(16)),
        (format!("{arch}.context_length"), MetaValue::U32(16)),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            MetaValue::F32(1e-5),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            MetaValue::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            MetaValue::Array(letter_vocab().into_iter().map(MetaValue::String).collect()),
        ),
    ];
    for (k, v) in extra {
        match meta.iter_mut().find(|(m, _)| m == k) {
            Some(slot) => slot.1 = v.clone(),
            None => meta.push((k.to_string(), v.clone())),
        }
    }
    for (k, v) in meta {
        b.meta(&k, v);
    }
    let mut seed = 0usize;
    let mut t = |b: &mut GgufBuilder, name: &str, dims: &[u64]| {
        seed += 1;
        let n = dims.iter().product::<u64>() as usize;
        let bytes = (0..n)
            .flat_map(|i| ((((seed * 131 + i * 17) % 101) as f32 / 50.0) - 1.0).to_le_bytes())
            .collect();
        b.tensor(name, dims, GGML_F32, bytes);
    };
    t(&mut b, "token_embd.weight", &[dim, vocab]);
    for i in 0..layers {
        for (name, dims) in [
            ("attn_norm", vec![dim]),
            ("attn_q", vec![dim, dim]),
            ("attn_k", vec![dim, dim]),
            ("attn_v", vec![dim, dim]),
            ("attn_output", vec![dim, dim]),
            ("ffn_norm", vec![dim]),
            ("ffn_gate", vec![dim, ffn]),
            ("ffn_up", vec![dim, ffn]),
            ("ffn_down", vec![ffn, dim]),
        ] {
            t(&mut b, &format!("blk.{i}.{name}.weight"), &dims);
        }
    }
    t(&mut b, "output_norm.weight", &[dim]);
    t(&mut b, "output.weight", &[dim, vocab]);
    b.into_bytes()
}

/// [`tiny_llama`] with BOS (`<f0>`, prepended to prompts) and EOS (`<f1>`) ids.
pub fn tiny_llama_bos_eos() -> Vec<u8> {
    tiny_llama(&[
        ("tokenizer.ggml.bos_token_id", MetaValue::U32(BOS)),
        ("tokenizer.ggml.eos_token_id", MetaValue::U32(EOS)),
        ("tokenizer.ggml.add_bos_token", MetaValue::Bool(true)),
    ])
}

/// [`tiny_llama`] with a SentencePiece tokenizer (`tokenizer.ggml.model = "llama"`):
/// `a`..`z`, `▁`, `▁a`, `<f0>`..`<f3>`.
pub fn tiny_llama_spm() -> Vec<u8> {
    let mut v: Vec<String> = ('a'..='z').map(|c| c.to_string()).collect();
    v.push("\u{2581}".into());
    v.push("\u{2581}a".into());
    v.extend((0..4).map(|i| format!("<f{i}>")));
    tiny_llama(&[
        ("tokenizer.ggml.model", MetaValue::String("llama".into())),
        (
            "tokenizer.ggml.tokens",
            MetaValue::Array(v.into_iter().map(MetaValue::String).collect()),
        ),
    ])
}
