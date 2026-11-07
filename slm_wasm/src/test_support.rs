//! Tiny trained models shared by the unit tests (the bytes the browser would fetch).

use ferrum_core::model::Sequential;
use ferrum_core::{
    to_bytes, Embedding, GenerativeSLM, LayerNorm, Linear, ModelMetadata, Normalizer, Rng,
    TaskType, TransformerBlock, TransformerConfig,
};

const CORPUS: &str = "the cat sat on the mat. the dog sat on the log. ";

fn config(vocab_size: usize) -> TransformerConfig {
    TransformerConfig {
        context_len: 16,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 2,
        hidden_dim: 32,
        epochs: 3,
        vocab_size,
        ..TransformerConfig::default()
    }
}

fn train(vocab_size: usize) -> GenerativeSLM {
    let corpus = CORPUS.repeat(20);
    GenerativeSLM::train_transformer_config(
        &corpus,
        &config(vocab_size),
        &mut Rng::new(3),
        |_, _| {},
    )
    .unwrap()
}

use std::sync::OnceLock;

static BYTES_F32: OnceLock<Vec<u8>> = OnceLock::new();
static BYTES_INT8: OnceLock<Vec<u8>> = OnceLock::new();
static BYTES_CHAR: OnceLock<Vec<u8>> = OnceLock::new();

/// A tiny BPE model, f32 or int8, as the library and as bytes.
pub fn trained(quantized: bool) -> (GenerativeSLM, Vec<u8>) {
    let cell = if quantized { &BYTES_INT8 } else { &BYTES_F32 };
    let bytes = cell.get_or_init(|| {
        let m = train(280);
        if quantized {
            m.to_bytes_quantized().unwrap()
        } else {
            m.to_bytes().unwrap()
        }
    }).clone();
    (GenerativeSLM::from_bytes(&bytes).unwrap(), bytes)
}

/// A tiny character-level model (`vocab_size == 0`), f32.
pub fn trained_char() -> (GenerativeSLM, Vec<u8>) {
    let bytes = BYTES_CHAR.get_or_init(|| {
        train(0).to_bytes().unwrap()
    }).clone();
    (GenerativeSLM::from_bytes(&bytes).unwrap(), bytes)
}

/// The layers and metadata of a hand-built (untrained) text model: vocab 10, context 8, one block.
#[allow(clippy::type_complexity)]
pub fn hand_made() -> (
    Embedding,
    TransformerBlock,
    LayerNorm,
    Linear,
    Normalizer,
    ModelMetadata,
) {
    let vocab_size = 10;
    let context_len = 8;
    let embed_dim = 16;
    let num_heads = 2;
    let hidden_dim = 32;
    let mut rng = Rng::new(99);
    let scale = (1.0 / embed_dim as f32).sqrt();

    let emb = Embedding::new(
        vocab_size,
        context_len,
        embed_dim,
        (0..vocab_size * embed_dim)
            .map(|_| rng.next_normal() * scale)
            .collect(),
        (0..context_len * embed_dim)
            .map(|_| rng.next_normal() * scale)
            .collect(),
    )
    .unwrap();

    let tb = TransformerBlock::new(
        context_len,
        num_heads,
        embed_dim,
        vec![1.0; embed_dim],
        vec![0.0; embed_dim],
        (0..embed_dim * embed_dim)
            .map(|_| rng.next_normal() * scale)
            .collect(),
        vec![0.0; embed_dim],
        (0..embed_dim * embed_dim)
            .map(|_| rng.next_normal() * scale)
            .collect(),
        vec![0.0; embed_dim],
        (0..embed_dim * embed_dim)
            .map(|_| rng.next_normal() * scale)
            .collect(),
        vec![0.0; embed_dim],
        (0..embed_dim * embed_dim)
            .map(|_| rng.next_normal() * scale)
            .collect(),
        vec![0.0; embed_dim],
        vec![1.0; embed_dim],
        vec![0.0; embed_dim],
        (0..embed_dim * hidden_dim)
            .map(|_| rng.next_normal() * scale)
            .collect(),
        vec![0.0; hidden_dim],
        (0..hidden_dim * embed_dim)
            .map(|_| rng.next_normal() * scale)
            .collect(),
        vec![0.0; embed_dim],
    )
    .unwrap();

    let ln_final = LayerNorm::new(embed_dim, vec![1.0; embed_dim], vec![0.0; embed_dim]).unwrap();
    let lm_head = Linear::new(
        embed_dim,
        vocab_size,
        (0..embed_dim * vocab_size)
            .map(|_| rng.next_normal() * scale)
            .collect(),
        vec![0.0; vocab_size],
    )
    .unwrap();

    let vocab_strs: Vec<String> = (0..vocab_size)
        .map(|i| ((b'a' + i as u8) as char).to_string())
        .collect();
    let norm = Normalizer {
        means: vec![],
        stds: vec![],
    };
    let meta = ModelMetadata {
        dataset_name: "slm_test".into(),
        task: TaskType::TransformerSLM,
        feature_names: (0..context_len).map(|i| format!("c_{i}")).collect(),
        feature_ranges: vec![[0.0, vocab_size as f32]; context_len],
        class_names: vocab_strs,
        target_name: "next_char".into(),
        target_range: [0.0, vocab_size as f32],
        input_dim: context_len,
        output_dim: vocab_size,
        tokenizer_state: String::new(),
    };
    (emb, tb, ln_final, lm_head, norm, meta)
}

/// [`hand_made`] as GGUF bytes.
pub fn hand_made_bytes() -> Vec<u8> {
    let (emb, tb, ln, head, norm, meta) = hand_made();
    let model = Sequential::new()
        .with(Box::new(emb))
        .with(Box::new(tb))
        .with(Box::new(ln))
        .with(Box::new(head));
    to_bytes(&model, &norm, &meta).unwrap()
}
