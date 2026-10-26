//! Shared test fixtures (compiled only for tests).

/// A valid `llama` GGUF (gpt2-style tokenizer metadata, F32 weights,
/// deterministic values, context length 128) of the given size. Returns the
/// file's bytes. The default test size is `(8, 2, 2, 16, 32)`.
pub fn tiny_llama_gguf(
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    ffn: usize,
    vocab: usize,
) -> Vec<u8> {
    use ferrum_core::{GgufBuilder, MetaValue};
    const GGML_F32: u32 = 0;
    let head_dim = dim / n_heads;
    let gen = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((seed * 131 + i * 17) % 101) as f32 / 500.0) - 0.1)
            .collect()
    };
    let f32_bytes = |xs: &[f32]| -> Vec<u8> {
        let mut o = Vec::with_capacity(xs.len() * 4);
        for &x in xs {
            o.extend_from_slice(&x.to_le_bytes());
        }
        o
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
    b.meta("llama.context_length", MetaValue::U32(128));
    b.meta(
        "llama.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-5),
    );
    b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
    // A usable tokenizer: single-character tokens for "hi abcdefg" first, filler after,
    // no merges, BOS = 0, EOS = 9 (so text fine-tuning and prompts work on the fixture).
    let chars = ["h", "i", " ", "a", "b", "c", "d", "e", "f", "g"];
    b.meta(
        "tokenizer.ggml.tokens",
        MetaValue::Array(
            (0..vocab)
                .map(|i| MetaValue::String(chars.get(i).map_or(format!("t{i}"), |c| c.to_string())))
                .collect(),
        ),
    );
    b.meta("tokenizer.ggml.merges", MetaValue::Array(vec![]));
    b.meta("tokenizer.ggml.bos_token_id", MetaValue::U32(0));
    b.meta("tokenizer.ggml.eos_token_id", MetaValue::U32(9));
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
        let qkv = (n_heads * head_dim) as u64;
        t(
            &mut b,
            &format!("{p}.attn_q.weight"),
            &[dim as u64, qkv],
            20 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_k.weight"),
            &[dim as u64, qkv],
            30 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_v.weight"),
            &[dim as u64, qkv],
            40 + i,
        );
        t(
            &mut b,
            &format!("{p}.attn_output.weight"),
            &[qkv, dim as u64],
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
