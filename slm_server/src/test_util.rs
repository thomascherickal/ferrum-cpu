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

/// The fixtures above are what every other unit test stands on; these pin their invariants.
#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::MetaValue;

    #[test]
    fn imported_builds_the_tiny_llama_with_its_letter_tokenizer() {
        let m = imported(&llama::tiny_llama(&[]));
        // "abc": three letter tokens, the first one unscored.
        let s = m.score_rolling("abc").unwrap();
        assert_eq!(s.n_tokens, 2);
        assert!(s.logprob.is_finite() && s.logprob < 0.0);
    }

    #[test]
    fn tiny_llama_takes_extra_metadata_and_another_architecture() {
        let g = Gguf::parse(llama::tiny_llama(&[(
            "general.architecture",
            MetaValue::String("qwen2".into()),
        )]))
        .unwrap();
        assert_eq!(g.architecture(), Some("qwen2"));
        assert!(
            g.meta("qwen2.block_count").is_some(),
            "keys follow the architecture"
        );
        assert_eq!(llama::letter_vocab().len(), 32);
        assert_eq!(llama::letter_vocab()[llama::BOS as usize], "<f0>");
    }

    #[test]
    fn temp_file_holds_the_bytes_until_dropped() {
        let f = TempFile::new("selftest.bin", b"abc");
        let p = f.path().to_string();
        assert_eq!(std::fs::read(&p).unwrap(), b"abc");
        drop(f);
        assert!(!std::path::Path::new(&p).exists());
    }

    #[test]
    fn model_is_character_level_at_vocab_zero_and_bpe_otherwise() {
        let chars = model(0);
        assert!(chars.meta.tokenizer_state.is_empty());
        assert_eq!(chars.meta.input_dim, 8, "context 8");
        let bpe = model(270);
        assert!(!bpe.meta.tokenizer_state.is_empty());
    }
}
