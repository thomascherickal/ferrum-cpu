//! Shared fixtures for the unit tests of this module tree.
use crate::gguf_write::blocks::encode_tensor;
use crate::gguf_write::builder::GgufBuilder;

// Test-only helper: little-endian f32 bytes.
pub(super) fn f32s_to_le_bytes(xs: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

// Round-trip a tensor through the real writer+reader stack and return the
// reconstructed f32 values. This is the encoder correctness oracle.
pub(crate) fn dequant_via_reader(data: &[f32], ggml_type: u32, dims: &[u64]) -> Vec<f32> {
    use crate::gguf::{Gguf, MetaValue};
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("llama".into()));
    b.tensor(
        "t",
        dims,
        ggml_type,
        encode_tensor(data, ggml_type).unwrap(),
    );
    let g = Gguf::parse(b.into_bytes()).unwrap();
    g.dequantize(g.tensor("t").unwrap()).unwrap()
}

pub(super) fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// Deterministic filler in [-0.1, 0.1], shared by fixture builders below and
// by tests that need to know a fixture tensor's exact source values.
pub(super) fn det_fill(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((seed * 131 + i * 17) % 101) as f32 / 500.0) - 0.1)
        .collect()
}

// Build a minimal but valid `llama` GGUF (2 layers, dim 8) as bytes, with all
// tensors F32 and a small gpt2 tokenizer block. Deterministic weights.
pub(super) fn tiny_llama_gguf_bytes() -> Vec<u8> {
    tiny_llama_gguf_bytes_ex(false)
}

// Same fixture as `tiny_llama_gguf_bytes`, but when `with_qkv_bias` is true
// also emits deterministic F32 `blk.{i}.attn_{q,k,v}.bias` tensors (used by
// `write_llama_gguf_roundtrips_qkv_biases` to exercise the bias-emission
// path, which the plain fixture never touches).
pub(super) fn tiny_llama_gguf_bytes_ex(with_qkv_bias: bool) -> Vec<u8> {
    build_tiny_llama(8, 2, with_qkv_bias)
}

// Generalized fixture builder. `dim` also sets `ffn = 2·dim`; a `dim` that
// is a multiple of 256 makes every projection k-quantizable end-to-end
// (used by `write_llama_gguf_emits_kquant_at_dim_256`).
pub(super) fn build_tiny_llama(dim: usize, n_layers: usize, with_qkv_bias: bool) -> Vec<u8> {
    use crate::gguf::{MetaValue, GGML_F32};
    let (n_heads, ffn, vocab, ctx) = (2usize, 2 * dim, 32usize, 16usize);
    let head_dim = dim / n_heads;

    let gen = det_fill;

    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("llama".into()));
    b.meta("general.name", MetaValue::String("tiny".into()));
    b.meta("llama.embedding_length", MetaValue::U32(dim as u32));
    b.meta("llama.block_count", MetaValue::U32(n_layers as u32));
    b.meta("llama.attention.head_count", MetaValue::U32(n_heads as u32));
    b.meta(
        "llama.attention.head_count_kv",
        MetaValue::U32(n_heads as u32),
    );
    b.meta("llama.feed_forward_length", MetaValue::U32(ffn as u32));
    b.meta("llama.context_length", MetaValue::U32(ctx as u32));
    b.meta(
        "llama.attention.layer_norm_rms_epsilon",
        MetaValue::F32(1e-5),
    );
    // Minimal gpt2 tokenizer block (metadata only — round-trip target).
    let toks: Vec<MetaValue> = (0..vocab)
        .map(|i| MetaValue::String(format!("t{i}")))
        .collect();
    b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
    b.meta("tokenizer.ggml.tokens", MetaValue::Array(toks));

    let f32t = |b: &mut GgufBuilder, name: &str, dims: &[u64], seed: usize| {
        let n: usize = dims.iter().product::<u64>() as usize;
        b.tensor(name, dims, GGML_F32, f32s_to_le_bytes(&gen(seed, n)));
    };

    // token_embd: dims [dim, vocab], data row-major [vocab, dim].
    f32t(&mut b, "token_embd.weight", &[dim as u64, vocab as u64], 1);
    for i in 0..n_layers {
        let p = format!("blk.{i}");
        f32t(
            &mut b,
            &format!("{p}.attn_norm.weight"),
            &[dim as u64],
            10 + i,
        );
        // Projections: GGUF dims [n_in, n_out].
        f32t(
            &mut b,
            &format!("{p}.attn_q.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            20 + i,
        );
        if with_qkv_bias {
            f32t(
                &mut b,
                &format!("{p}.attn_q.bias"),
                &[(n_heads * head_dim) as u64],
                300 + i,
            );
        }
        f32t(
            &mut b,
            &format!("{p}.attn_k.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            30 + i,
        );
        if with_qkv_bias {
            f32t(
                &mut b,
                &format!("{p}.attn_k.bias"),
                &[(n_heads * head_dim) as u64],
                310 + i,
            );
        }
        f32t(
            &mut b,
            &format!("{p}.attn_v.weight"),
            &[dim as u64, (n_heads * head_dim) as u64],
            40 + i,
        );
        if with_qkv_bias {
            f32t(
                &mut b,
                &format!("{p}.attn_v.bias"),
                &[(n_heads * head_dim) as u64],
                320 + i,
            );
        }
        f32t(
            &mut b,
            &format!("{p}.attn_output.weight"),
            &[(n_heads * head_dim) as u64, dim as u64],
            50 + i,
        );
        f32t(
            &mut b,
            &format!("{p}.ffn_norm.weight"),
            &[dim as u64],
            60 + i,
        );
        f32t(
            &mut b,
            &format!("{p}.ffn_gate.weight"),
            &[dim as u64, ffn as u64],
            70 + i,
        );
        f32t(
            &mut b,
            &format!("{p}.ffn_up.weight"),
            &[dim as u64, ffn as u64],
            80 + i,
        );
        f32t(
            &mut b,
            &format!("{p}.ffn_down.weight"),
            &[ffn as u64, dim as u64],
            90 + i,
        );
    }
    f32t(&mut b, "output_norm.weight", &[dim as u64], 200);
    f32t(&mut b, "output.weight", &[dim as u64, vocab as u64], 201);
    b.into_bytes()
}
