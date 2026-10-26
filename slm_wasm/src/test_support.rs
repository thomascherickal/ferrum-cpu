//! Tiny trained models shared by the unit tests (the bytes the browser would fetch).

use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

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

/// A tiny BPE model, f32 or int8, as the library and as bytes.
pub fn trained(quantized: bool) -> (GenerativeSLM, Vec<u8>) {
    let m = train(280);
    let bytes = if quantized {
        m.to_bytes_quantized().unwrap()
    } else {
        m.to_bytes().unwrap()
    };
    (GenerativeSLM::from_bytes(&bytes).unwrap(), bytes)
}

/// A tiny character-level model (`vocab_size == 0`), f32.
pub fn trained_char() -> (GenerativeSLM, Vec<u8>) {
    let m = train(0);
    let bytes = m.to_bytes().unwrap();
    (GenerativeSLM::from_bytes(&bytes).unwrap(), bytes)
}
