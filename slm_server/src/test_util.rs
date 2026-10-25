//! Shared helpers for the unit tests.

use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};

pub(crate) fn model(vocab_size: usize) -> GenerativeSLM {
    let cfg = TransformerConfig {
        context_len: 8,
        embed_dim: 8,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 16,
        epochs: 1,
        vocab_size,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(
        &"abcabc abc\n".repeat(30),
        &cfg,
        &mut Rng::new(1),
        |_, _| {},
    )
    .unwrap()
}
