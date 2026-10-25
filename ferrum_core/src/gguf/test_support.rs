//! Shared fixtures for the unit tests of this module tree.
use crate::error::Result;
use crate::gguf::format::{
    align_up, DEFAULT_ALIGNMENT, GGML_F32, GGML_Q4_0, GGML_Q8_0, GGUF_MAGIC, QK, VT_F32, VT_STRING,
    VT_U32,
};
use crate::gguf::header::check_extents;
use crate::gguf::{Gguf, MetaValue, TensorInfo};
use crate::quant::QKind;

pub(super) fn put_u32(o: &mut Vec<u8>, v: u32) {
    o.extend_from_slice(&v.to_le_bytes());
}

pub(super) fn put_u64(o: &mut Vec<u8>, v: u64) {
    o.extend_from_slice(&v.to_le_bytes());
}

pub(super) fn put_str(o: &mut Vec<u8>, s: &str) {
    put_u64(o, s.len() as u64);
    o.extend_from_slice(s.as_bytes());
}

pub(super) fn f32_to_f16(x: f32) -> u16 {
    // Round-to-nearest-even is unnecessary for the small exact values used
    // in tests; a truncating conversion suffices and stays exact for them.
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if x == 0.0 {
        return sign << 15;
    }
    (sign << 15) | ((exp as u16) << 10) | mant as u16
}

/// Build a GGUF with one metadata string, one F32, one Q8_0, and one Q4_0
/// tensor. Returns the bytes plus the exact f32 values each tensor encodes.
pub(super) fn synth_gguf() -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut header = Vec::new();
    put_u32(&mut header, GGUF_MAGIC);
    put_u32(&mut header, 3); // version
    put_u64(&mut header, 3); // tensor_count
    put_u64(&mut header, 1); // kv_count

    // metadata: general.architecture = "ferrum-test"
    put_str(&mut header, "general.architecture");
    put_u32(&mut header, VT_STRING);
    put_str(&mut header, "ferrum-test");

    // Tensor data we will encode.
    let f32_vals: Vec<f32> = (0..QK as i32).map(|i| (i - 16) as f32 * 0.25).collect();
    // Q8_0: exact multiples of d.
    let q8_d = 0.5f32;
    let q8_vals: Vec<f32> = (0..QK as i32).map(|i| q8_d * (i - 16) as f32).collect();
    // Q4_0: d * (nibble-8), nibble 0..15 → value in [-8,7]·d.
    let q4_d = 0.25f32;
    let q4_vals: Vec<f32> = (0..QK as i32)
        .map(|i| q4_d * ((i % 15) - 8) as f32)
        .collect();

    let mut data = Vec::new();
    // F32 tensor.
    let off_f32 = data.len() as u64;
    for &v in &f32_vals {
        data.extend_from_slice(&v.to_le_bytes());
    }
    // Q8_0 tensor: f16 d, then 32 i8.
    let off_q8 = data.len() as u64;
    data.extend_from_slice(&f32_to_f16(q8_d).to_le_bytes());
    for i in 0..QK as i32 {
        data.push(((i - 16) as i8) as u8);
    }
    // Q4_0 tensor: f16 d, then 16 packed bytes (low→j, high→j+16).
    let off_q4 = data.len() as u64;
    data.extend_from_slice(&f32_to_f16(q4_d).to_le_bytes());
    for j in 0..(QK / 2) {
        // Stored nibble = decoded_value + 8; here decoded_value = (idx%15) - 8,
        // so the nibble is simply idx % 15.
        let lo = (j as i32 % 15) as u8 & 0x0F;
        let hi = ((j + QK / 2) as i32 % 15) as u8 & 0x0F;
        data.push((hi << 4) | lo);
    }

    // Tensor directory.
    let mut dir = Vec::new();
    for (name, ty, off) in [
        ("t_f32", GGML_F32, off_f32),
        ("t_q8", GGML_Q8_0, off_q8),
        ("t_q4", GGML_Q4_0, off_q4),
    ] {
        put_str(&mut dir, name);
        put_u32(&mut dir, 1); // n_dims
        put_u64(&mut dir, QK as u64); // dims[0]
        put_u32(&mut dir, ty);
        put_u64(&mut dir, off);
    }

    let mut bytes = header;
    bytes.extend_from_slice(&dir);
    // Pad to default alignment (32) before the data section.
    let pad = align_up(bytes.len(), DEFAULT_ALIGNMENT as usize) - bytes.len();
    bytes.extend(std::iter::repeat_n(0u8, pad));
    bytes.extend_from_slice(&data);

    // Recompute q4 expected values to match the writer's nibble layout.
    let mut q4_expected = vec![0.0f32; QK];
    for j in 0..(QK / 2) {
        q4_expected[j] = q4_d * ((j as i32 % 15) - 8) as f32;
        q4_expected[j + QK / 2] = q4_d * (((j + QK / 2) as i32 % 15) - 8) as f32;
    }
    let _ = (&q8_vals, &q4_vals);
    (bytes, f32_vals, q8_vals, q4_expected)
}

pub(super) fn put_f32(o: &mut Vec<u8>, v: f32) {
    o.extend_from_slice(&v.to_bits().to_le_bytes());
}

pub(super) fn kv_u32(o: &mut Vec<u8>, key: &str, v: u32) {
    put_str(o, key);
    put_u32(o, VT_U32);
    put_u32(o, v);
}

pub(super) fn kv_f32(o: &mut Vec<u8>, key: &str, v: f32) {
    put_str(o, key);
    put_u32(o, VT_F32);
    put_f32(o, v);
}

pub(super) fn kv_str(o: &mut Vec<u8>, key: &str, v: &str) {
    put_str(o, key);
    put_u32(o, VT_STRING);
    put_str(o, v);
}

pub(super) struct TBuf {
    pub(super) infos: Vec<u8>,
    pub(super) data: Vec<u8>,
    pub(super) count: u32,
}

impl TBuf {
    pub(super) fn new() -> Self {
        Self {
            infos: Vec::new(),
            data: Vec::new(),
            count: 0,
        }
    }
    /// Append an F32 tensor with GGUF dims `ne` (fastest axis first) and
    /// row-major values.
    pub(super) fn add(&mut self, name: &str, ne: &[u64], vals: &[f32]) {
        let off = self.data.len() as u64;
        for &v in vals {
            put_f32(&mut self.data, v);
        }
        put_str(&mut self.infos, name);
        put_u32(&mut self.infos, ne.len() as u32);
        for &d in ne {
            put_u64(&mut self.infos, d);
        }
        put_u32(&mut self.infos, GGML_F32);
        put_u64(&mut self.infos, off);
        self.count += 1;
    }
}

/// A tiny but complete llama/qwen2-arch GGUF (2 layers, MHA), f32 weights.
/// `with_output` controls whether an explicit LM head is written (else the
/// loader ties it to the embedding); `with_bias` adds Qwen2-style q/k/v
/// attention biases. `n_meta` must equal the metadata KV count written.
pub(super) fn synth_llama_cfg(arch: &str, with_output: bool, with_bias: bool) -> Vec<u8> {
    let (vocab, dim, n_heads, n_layers, ffn) = (10usize, 8usize, 2usize, 2usize, 16usize);
    let head_dim = dim / n_heads;
    let qd = n_heads * head_dim; // = dim here
    let mut seed = 12345u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    };

    let mut t = TBuf::new();
    t.add(
        "token_embd.weight",
        &[dim as u64, vocab as u64],
        &rnd(vocab * dim),
    );
    for i in 0..n_layers {
        let p = format!("blk.{i}");
        t.add(
            &format!("{p}.attn_norm.weight"),
            &[dim as u64],
            &vec![1.0; dim],
        );
        t.add(
            &format!("{p}.attn_q.weight"),
            &[dim as u64, qd as u64],
            &rnd(dim * qd),
        );
        t.add(
            &format!("{p}.attn_k.weight"),
            &[dim as u64, qd as u64],
            &rnd(dim * qd),
        );
        t.add(
            &format!("{p}.attn_v.weight"),
            &[dim as u64, qd as u64],
            &rnd(dim * qd),
        );
        if with_bias {
            t.add(&format!("{p}.attn_q.bias"), &[qd as u64], &rnd(qd));
            t.add(&format!("{p}.attn_k.bias"), &[qd as u64], &rnd(qd));
            t.add(&format!("{p}.attn_v.bias"), &[qd as u64], &rnd(qd));
        }
        t.add(
            &format!("{p}.attn_output.weight"),
            &[qd as u64, dim as u64],
            &rnd(qd * dim),
        );
        t.add(
            &format!("{p}.ffn_norm.weight"),
            &[dim as u64],
            &vec![1.0; dim],
        );
        t.add(
            &format!("{p}.ffn_gate.weight"),
            &[dim as u64, ffn as u64],
            &rnd(dim * ffn),
        );
        t.add(
            &format!("{p}.ffn_up.weight"),
            &[dim as u64, ffn as u64],
            &rnd(dim * ffn),
        );
        t.add(
            &format!("{p}.ffn_down.weight"),
            &[ffn as u64, dim as u64],
            &rnd(ffn * dim),
        );
    }
    t.add("output_norm.weight", &[dim as u64], &vec![1.0; dim]);
    if with_output {
        t.add(
            "output.weight",
            &[dim as u64, vocab as u64],
            &rnd(dim * vocab),
        );
    }

    let mut meta = Vec::new();
    kv_str(&mut meta, "general.architecture", arch);
    kv_u32(&mut meta, &format!("{arch}.embedding_length"), dim as u32);
    kv_u32(&mut meta, &format!("{arch}.block_count"), n_layers as u32);
    kv_u32(
        &mut meta,
        &format!("{arch}.attention.head_count"),
        n_heads as u32,
    );
    kv_u32(
        &mut meta,
        &format!("{arch}.attention.head_count_kv"),
        n_heads as u32,
    );
    kv_u32(
        &mut meta,
        &format!("{arch}.feed_forward_length"),
        ffn as u32,
    );
    kv_u32(&mut meta, &format!("{arch}.context_length"), 32);
    kv_f32(
        &mut meta,
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
        1e-5,
    );
    kv_f32(&mut meta, &format!("{arch}.rope.freq_base"), 10000.0);
    let n_meta = 9u64;

    let mut bytes = Vec::new();
    put_u32(&mut bytes, GGUF_MAGIC);
    put_u32(&mut bytes, 3);
    put_u64(&mut bytes, t.count as u64);
    put_u64(&mut bytes, n_meta);
    bytes.extend_from_slice(&meta);
    bytes.extend_from_slice(&t.infos);
    let pad = align_up(bytes.len(), DEFAULT_ALIGNMENT as usize) - bytes.len();
    bytes.extend(std::iter::repeat_n(0u8, pad));
    bytes.extend_from_slice(&t.data);
    bytes
}

pub(super) fn synth_llama() -> Vec<u8> {
    synth_llama_cfg("llama", true, false)
}

pub(super) fn ti(name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        dims: dims.to_vec(),
        ggml_type,
        offset,
    }
}

pub(super) fn extents_err(ts: &[TensorInfo], avail: usize) -> String {
    check_extents(ts, avail).unwrap_err().to_string()
}

/// `n` equal F32 tensors of `floats` elements, laid out by the writer
/// (aligned, in order). Returns the bytes and the data-section start.
pub(super) fn n_tensor_file(n: usize, floats: usize) -> (Vec<u8>, usize) {
    let mut b = crate::gguf_write::GgufBuilder::new();
    for i in 0..n {
        let data = (0..floats)
            .flat_map(|j| ((i * floats + j) as f32).to_le_bytes())
            .collect();
        b.tensor(&format!("t{i}"), &[floats as u64], GGML_F32, data);
    }
    let bytes = b.into_bytes();
    let stride = (floats * 4).next_multiple_of(32);
    let start = bytes.len() - ((n - 1) * stride + floats * 4);
    (bytes, start)
}

/// Run both header paths (`parse` from memory, `open` streamed from disk).
pub(super) fn parse_and_open(bytes: &[u8]) -> (Result<Gguf>, Result<Gguf>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "ferrum_gguf_extents_{}_{}.gguf",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).unwrap();
    let opened = Gguf::open(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    (Gguf::parse(bytes.to_vec()), opened)
}

/// A llama GGUF from (dims, tensors) via the shared writer. `meta` values
/// are U32 unless listed in `meta64` (U64).
pub(super) fn llama_file(
    meta: &[(&str, u32)],
    meta64: &[(&str, u64)],
    ts: &[(&str, &[u64])],
) -> Gguf {
    let mut b = crate::gguf_write::GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("llama".into()));
    for &(k, v) in meta {
        b.meta(&format!("llama.{k}"), MetaValue::U32(v));
    }
    for &(k, v) in meta64 {
        b.meta(&format!("llama.{k}"), MetaValue::U64(v));
    }
    for &(name, dims) in ts {
        let n = dims.iter().product::<u64>() as usize;
        b.tensor(name, dims, GGML_F32, vec![0u8; 4 * n]);
    }
    Gguf::parse(b.into_bytes()).unwrap()
}

/// Run `load_llama` in both precisions; both must fail fast with `want`.
pub(super) fn llama_err(g: &Gguf, want: &str) {
    let t = std::time::Instant::now();
    for prec in [Some(QKind::Int8), None] {
        let e = g.load_llama_prec(prec).err().expect("must be rejected");
        assert!(e.to_string().contains(want), "{e}");
    }
    assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
}

/// Valid 1-layer llama shapes: dim 2, 1 head (head_dim 2), ffn 2, vocab 3.
pub(super) const L_META: [(&str, u32); 4] = [
    ("embedding_length", 2),
    ("block_count", 1),
    ("attention.head_count", 1),
    ("feed_forward_length", 2),
];

pub(super) fn l_tensors() -> Vec<(&'static str, Vec<u64>)> {
    [
        ("token_embd.weight", vec![2, 3]),
        ("blk.0.attn_norm.weight", vec![2]),
        ("blk.0.attn_q.weight", vec![2, 2]),
        ("blk.0.attn_k.weight", vec![2, 2]),
        ("blk.0.attn_v.weight", vec![2, 2]),
        ("blk.0.attn_output.weight", vec![2, 2]),
        ("blk.0.ffn_norm.weight", vec![2]),
        ("blk.0.ffn_gate.weight", vec![2, 2]),
        ("blk.0.ffn_up.weight", vec![2, 2]),
        ("blk.0.ffn_down.weight", vec![2, 2]),
        ("output_norm.weight", vec![2]),
    ]
    .into()
}

/// `l_tensors()` with `name`'s dims replaced (or `name` dropped if `None`),
/// plus `extra` metadata overriding `L_META`.
pub(super) fn l_file(name: &str, dims: Option<Vec<u64>>, extra: &[(&str, u32)]) -> Gguf {
    let mut meta: Vec<(&str, u32)> = L_META.to_vec();
    meta.retain(|(k, _)| !extra.iter().any(|(e, _)| e == k));
    meta.extend_from_slice(extra);
    let ts: Vec<(&str, Vec<u64>)> = l_tensors()
        .into_iter()
        .filter_map(|(n, d)| match (n == name, &dims) {
            (true, Some(nd)) => Some((n, nd.clone())),
            (true, None) => None,
            _ => Some((n, d)),
        })
        .collect();
    let ts: Vec<(&str, &[u64])> = ts.iter().map(|(n, d)| (*n, &d[..])).collect();
    llama_file(&meta, &[], &ts)
}
