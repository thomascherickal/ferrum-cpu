//! Shared helpers for the unit tests.

use crate::imported::Imported;
use ferrum_core::{GenerativeSLM, Gguf, GgufTokenizer, Rng, TransformerConfig};

#[path = "../tests/common/llama.rs"]
pub(crate) mod llama;

/// The imported backend for GGUF `bytes`, at f32.
pub(crate) fn imported(bytes: &[u8]) -> Imported {
    let g = Gguf::parse(bytes.to_vec()).unwrap();
    Imported::new(
        g.load_llama_prec(None).unwrap(),
        GgufTokenizer::from_gguf(&g).unwrap(),
        &g,
    )
}

/// A file in the temp directory, removed on drop.
pub(crate) struct TempFile(std::path::PathBuf);

impl TempFile {
    pub(crate) fn new(name: &str, bytes: &[u8]) -> Self {
        let p = std::env::temp_dir().join(format!("ferrum-serve-{}-{name}", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        Self(p)
    }
    pub(crate) fn path(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

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
