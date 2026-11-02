//! Shared fixtures for the unit tests of this module tree.
use crate::activation::Activation;
use crate::error::Result;
use crate::gguf::{Gguf, MetaValue, GGML_F32};
use crate::gguf_write::{encode_tensor, GgufBuilder};
use crate::layer::{ActivationLayer, Embedding, LayerNorm, Linear, TransformerBlock};
use crate::loader::decode::from_gguf;
use crate::meta::{ModelMetadata, Normalizer, TaskType};
use crate::model::Sequential;
use crate::slm::{GenerativeSLM, TransformerConfig};
use crate::tensor::Tensor;
use crate::tokenizer::ByteBpeTokenizer;

pub(super) fn make_bundle() -> (Sequential, Normalizer, ModelMetadata) {
    let l1 = Linear::new(
        4,
        8,
        (0..32).map(|i| i as f32 * 0.01).collect(),
        vec![0.0; 8],
    )
    .unwrap();
    let l2 = Linear::new(
        8,
        3,
        (0..24).map(|i| i as f32 * -0.01).collect(),
        vec![0.1; 3],
    )
    .unwrap();
    let model = Sequential::new()
        .with(Box::new(l1))
        .with(Box::new(ActivationLayer::new(Activation::ReLU)))
        .with(Box::new(l2))
        .with(Box::new(ActivationLayer::new(Activation::Softmax)));
    let norm = Normalizer {
        means: vec![5.8, 3.1, 3.7, 1.2],
        stds: vec![0.8, 0.4, 1.7, 0.8],
    };
    let meta = ModelMetadata {
        dataset_name: "test".into(),
        task: TaskType::Classification,
        feature_names: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        feature_ranges: vec![[0.0, 10.0]; 4],
        class_names: vec!["X".into(), "Y".into(), "Z".into()],
        target_name: "".into(),
        target_range: [0.0, 2.0],
        input_dim: 4,
        output_dim: 3,
        tokenizer_state: String::new(),
    };
    (model, norm, meta)
}

/// A transformer bundle whose matrices are ≥ QUANT_MIN_LEN, so int4/int8
/// quantization actually engages (the small `make_embedding_bundle` stays
/// f32). `Embedding → TransformerBlock → LayerNorm → Linear(head) → Softmax`.
pub(super) fn make_embedding_bundle_big() -> (Sequential, Normalizer, ModelMetadata) {
    let vocab = 10usize;
    let ctx = 4usize;
    let dim = 64usize;
    let heads = 4usize;
    let hidden = 128usize;
    let f = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 * s).sin() * 0.2).collect() };
    let emb = Embedding::new(vocab, ctx, dim, f(vocab * dim, 0.01), f(ctx * dim, 0.02)).unwrap();
    let tb = TransformerBlock::new(
        ctx,
        heads,
        dim,
        vec![1.0; dim],
        vec![0.0; dim],
        f(dim * dim, 0.013),
        vec![0.0; dim],
        f(dim * dim, 0.017),
        vec![0.0; dim],
        f(dim * dim, 0.019),
        vec![0.0; dim],
        f(dim * dim, 0.023),
        vec![0.0; dim],
        vec![1.0; dim],
        vec![0.0; dim],
        f(dim * hidden, 0.007),
        vec![0.0; hidden],
        f(hidden * dim, 0.009),
        vec![0.0; dim],
    )
    .unwrap();
    let lnf = LayerNorm::new(dim, vec![1.0; dim], vec![0.0; dim]).unwrap();
    let head = Linear::new(dim, vocab, f(dim * vocab, 0.005), vec![0.0; vocab]).unwrap();
    let model = Sequential::new()
        .with(Box::new(emb))
        .with(Box::new(tb))
        .with(Box::new(lnf))
        .with(Box::new(head))
        .with(Box::new(ActivationLayer::new(Activation::Softmax)));
    let norm = Normalizer {
        means: vec![],
        stds: vec![],
    };
    let meta = ModelMetadata {
        dataset_name: "big".into(),
        task: TaskType::TransformerSLM,
        feature_names: (0..ctx).map(|i| format!("c_{i}")).collect(),
        feature_ranges: vec![[0.0, vocab as f32]; ctx],
        class_names: (0..vocab).map(|i| format!("{i:x}")).collect(),
        target_name: "next".into(),
        target_range: [0.0, vocab as f32],
        input_dim: ctx,
        output_dim: vocab,
        tokenizer_state: String::new(),
    };
    (model, norm, meta)
}

pub(super) fn make_embedding_bundle() -> (Sequential, Normalizer, ModelMetadata) {
    let vocab_size = 5;
    let max_seq_len = 4;
    let embedding_dim = 8;
    let emb = Embedding::new(
        vocab_size,
        max_seq_len,
        embedding_dim,
        (0..vocab_size * embedding_dim)
            .map(|i| i as f32 * 0.1)
            .collect(),
        (0..max_seq_len * embedding_dim)
            .map(|i| i as f32 * 0.01)
            .collect(),
    )
    .unwrap();
    let lm_head = Linear::new(
        embedding_dim,
        vocab_size,
        (0..embedding_dim * vocab_size)
            .map(|i| i as f32 * 0.01)
            .collect(),
        vec![0.0; vocab_size],
    )
    .unwrap();
    let model = Sequential::new()
        .with(Box::new(emb))
        .with(Box::new(ActivationLayer::new(Activation::Softmax)))
        .with(Box::new(lm_head));
    let norm = Normalizer {
        means: vec![],
        stds: vec![],
    };
    let vocab: Vec<String> = "abcde".chars().map(|c| c.to_string()).collect();
    let meta = ModelMetadata {
        dataset_name: "tiny_slm".into(),
        task: TaskType::TransformerSLM,
        feature_names: (0..max_seq_len).map(|i| format!("c_{i}")).collect(),
        feature_ranges: vec![[0.0, vocab_size as f32]; max_seq_len],
        class_names: vocab,
        target_name: "next_char".into(),
        target_range: [0.0, vocab_size as f32],
        input_dim: max_seq_len,
        output_dim: vocab_size,
        tokenizer_state: String::new(),
    };
    (model, norm, meta)
}

#[derive(Debug)]
pub(super) struct DummyLayer;

impl crate::layer::Layer for DummyLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
    fn name(&self) -> String {
        "DummyLayer".to_string()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(super) fn gguf_corpus() -> String {
    let words = [
        "the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran", "far", "away", "home",
    ];
    (0..600)
        .map(|i| {
            let end = if i % 4 == 3 { ". " } else { " " };
            format!(
                "{} {} {} {}{end}",
                words[i % 12],
                words[(i * 7 + 3) % 12],
                i % 50,
                words[(i * 5 + 1) % 12]
            )
        })
        .collect()
}

/// Distinct, deterministic parameter values (a swapped tensor shows up).
pub(super) fn vals(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 31 + seed * 977) as f32 * 0.137).sin() * 0.3)
        .collect()
}

pub(super) fn slm_meta(ctx: usize, vocab: usize, tokenizer_state: String) -> ModelMetadata {
    ModelMetadata {
        dataset_name: "gguf test".into(),
        task: TaskType::TransformerSLM,
        feature_names: (0..ctx).map(|i| format!("c_{i}")).collect(),
        feature_ranges: vec![[0.0, vocab as f32]; ctx],
        class_names: vec![],
        target_name: "next_char".into(),
        target_range: [0.0, vocab as f32],
        input_dim: ctx,
        output_dim: vocab,
        tokenizer_state,
    }
}

pub(super) type Layers = Vec<Box<dyn crate::layer::Layer>>;

pub(super) fn block(ctx: usize, d: usize, h: usize, s: usize) -> Box<dyn crate::layer::Layer> {
    Box::new(
        TransformerBlock::new(
            ctx,
            4,
            d,
            vals(d, s),
            vals(d, s + 1),
            vals(d * d, s + 2),
            vals(d, s + 3),
            vals(d * d, s + 4),
            vals(d, s + 5),
            vals(d * d, s + 6),
            vals(d, s + 7),
            vals(d * d, s + 8),
            vals(d, s + 9),
            vals(d, s + 10),
            vals(d, s + 11),
            vals(d * h, s + 12),
            vals(h, s + 13),
            vals(h * d, s + 14),
            vals(d, s + 15),
        )
        .unwrap(),
    )
}

pub(super) fn seq(layers: Layers) -> Sequential {
    layers.into_iter().fold(Sequential::new(), |m, l| m.with(l))
}

/// The layers of a GPT-2-codec model built directly (fast, every parameter
/// distinct): `Embedding, TransformerBlock×2, LayerNorm, Linear, Softmax`,
/// plus its BPE metadata.
pub(super) fn bpe_layers(embed: usize, hidden: usize, tie: bool) -> (Layers, ModelMetadata) {
    let tok = ByteBpeTokenizer::train(&gguf_corpus(), 300).unwrap();
    let (v, ctx, d, h) = (tok.vocab_size(), 8, embed, hidden);
    let token = vals(v * d, 1);
    let head = if tie {
        crate::gguf::transpose_2d(&token, v, d)
    } else {
        vals(d * v, 92)
    };
    let layers: Layers = vec![
        Box::new(Embedding::new(v, ctx, d, token, vals(ctx * d, 2)).unwrap()),
        block(ctx, d, h, 10),
        block(ctx, d, h, 30),
        Box::new(LayerNorm::new(d, vals(d, 90), vals(d, 91)).unwrap()),
        Box::new(Linear::without_bias(d, v, head).unwrap()),
        Box::new(ActivationLayer::new(Activation::Softmax)),
    ];
    (layers, slm_meta(ctx, v, tok.encode_state()))
}

pub(super) fn empty_norm() -> Normalizer {
    Normalizer {
        means: vec![],
        stds: vec![],
    }
}

pub(super) fn bpe_parts(
    embed: usize,
    hidden: usize,
    tie: bool,
) -> (Sequential, Normalizer, ModelMetadata) {
    let (layers, meta) = bpe_layers(embed, hidden, tie);
    (seq(layers), empty_norm(), meta)
}

pub(super) fn ids_input(n: usize, vocab: usize) -> Tensor {
    Tensor::matrix(
        1,
        n,
        (0..n).map(|i| ((i * 37 + 5) % vocab) as f32).collect(),
    )
    .unwrap()
}

pub(super) fn fwd(m: &Sequential, meta: &ModelMetadata) -> Vec<f32> {
    m.forward(&ids_input(meta.input_dim, meta.output_dim))
        .unwrap()
        .data
}

pub(super) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The reference tiny config (0.1M params), trained briefly.
pub(super) fn trained_slm(qat: bool) -> GenerativeSLM {
    let cfg = TransformerConfig {
        context_len: 64,
        embed_dim: 64,
        num_heads: 4,
        num_blocks: 2,
        hidden_dim: 128,
        vocab_size: 512,
        weight_tying: true,
        max_tokens: 8 * 16 * 64, // 8 optimizer steps: enough to move every weight
        qat,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(
        &gguf_corpus(),
        &cfg,
        &mut crate::rng::Rng::new(1),
        |_, _| {},
    )
    .unwrap()
}

pub(super) fn parse(bytes: &[u8]) -> Gguf {
    Gguf::parse(bytes.to_vec()).unwrap()
}

pub(super) fn arch(bytes: &[u8]) -> String {
    parse(bytes).architecture().unwrap().to_string()
}

pub(super) type Tensors = Vec<(String, Vec<u64>, Vec<f32>)>;

pub(super) type Kv = std::collections::BTreeMap<String, MetaValue>;

pub(super) type GgufEdit = Box<dyn Fn(&mut Kv, &mut Tensors)>;

pub(super) type LayerEdit = Box<dyn Fn(&mut Layers)>;

/// Re-emit an F32-only GGUF after editing its metadata and tensors.
pub(super) fn rebuild(bytes: &[u8], edit: impl FnOnce(&mut Kv, &mut Tensors)) -> Vec<u8> {
    let g = parse(bytes);
    let mut meta = g.metadata.clone();
    let mut ts: Tensors = g
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.dims.clone(), g.dequantize(t).unwrap()))
        .collect();
    edit(&mut meta, &mut ts);
    let mut b = GgufBuilder::new();
    for (k, v) in meta {
        b.meta(&k, v);
    }
    for (n, d, v) in ts {
        b.tensor(&n, &d, GGML_F32, encode_tensor(&v, GGML_F32).unwrap());
    }
    b.into_bytes()
}

pub(super) fn load_err(bytes: &[u8]) -> String {
    from_gguf(bytes)
        .err()
        .expect("expected an error")
        .to_string()
}

pub(super) const FINF_HINT: &str =
    "old Ferrum .bin (FINF) model — this format was removed; retrain to produce a .gguf";

pub(super) fn load_err_of<T>(r: Result<T>) -> String {
    r.err().expect("expected error").to_string()
}
