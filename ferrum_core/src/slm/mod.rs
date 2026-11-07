//! Generic offline Edge Generative SLM Library Engine.

mod cached;
mod config;
mod corpus;
mod generate;
mod memory;
mod metrics_log;
mod persist;
mod sampling;
mod scoring;
mod stream;
#[cfg(test)]
mod test_support;
mod token_stream;
mod train;
mod train_disk;
mod train_tokens;
mod training_loop;

pub use self::config::{TransformerConfig, ValidationConfig, ValidationProgress};
pub use self::corpus::{char_to_hex, corpus_vocab, hex_to_char, tokenize_corpus};
pub use self::generate::{Generation, StopReason};
pub use self::sampling::SamplingParams;
pub(crate) use self::sampling::{argmax_first, sample_with_params};
pub use self::scoring::{Evaluation, Score};
pub use self::token_stream::GenStream;
pub use self::train_disk::{stream_text_to_token_file, STREAM_CORPUS_THRESHOLD};

use crate::error::{InferError, Result};
use crate::meta::{ModelMetadata, Normalizer, TaskType};
use crate::model::Sequential;
use crate::tokenizer::ByteBpeTokenizer;
use std::sync::{Arc, Mutex};

/// Unified Causal Small Language Model (SLM) for off-grid edge generative AI.
pub struct GenerativeSLM {
    pub model: Sequential,
    pub norm: Normalizer,
    pub meta: ModelMetadata,
    /// The BPE tokenizer parsed from `meta.tokenizer_state`, built once.
    tokenizer: TokenizerCache,
}

/// One-slot cache of the parsed tokenizer, keyed by the state string it was
/// built from, so a caller that edits the public `meta.tokenizer_state` can
/// never read a stale tokenizer. Errors are not cached.
#[derive(Default)]
struct TokenizerCache(Mutex<Option<(String, Arc<ByteBpeTokenizer>)>>);

impl GenerativeSLM {
    /// The BPE tokenizer for this model (`None` for character-level models),
    /// parsed from `meta.tokenizer_state` on first use and shared afterwards.
    fn bpe(&self) -> Result<Option<Arc<ByteBpeTokenizer>>> {
        let state = &self.meta.tokenizer_state;
        if state.is_empty() {
            return Ok(None);
        }
        let mut slot = self.tokenizer.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((built_from, tok)) = slot.as_ref() {
            if built_from == state {
                return Ok(Some(Arc::clone(tok)));
            }
        }
        let tok = Arc::new(ByteBpeTokenizer::from_state(state)?);
        *slot = Some((state.clone(), Arc::clone(&tok)));
        Ok(Some(tok))
    }

    /// Construct a new instance manually from sequential layers, normalizer, and metadata.
    pub fn new(model: Sequential, norm: Normalizer, meta: ModelMetadata) -> Self {
        vprintln!(
            "[slm::GenerativeSLM::new] Creating SLM with {} layers, input_dim={}, output_dim={}",
            model.len(),
            meta.input_dim,
            meta.output_dim
        );
        Self {
            model,
            norm,
            meta,
            tokenizer: TokenizerCache::default(),
        }
    }

    /// The conditioning token placed before the first scored or generated
    /// token: the tokenizer's EOS id when it has one (document boundary), else 0.
    pub fn bos_id(&self) -> usize {
        self.bpe()
            .ok()
            .flatten()
            .and_then(|t| t.eos_id())
            .unwrap_or(0)
    }

    /// Token ids for `text`. BPE models use the document-aware encoder;
    /// character-level models map each char (CR dropped) and error on a char
    /// outside the model vocabulary instead of silently using id 0.
    pub fn encode_text(&self, text: &str) -> Result<Vec<usize>> {
        if let Some(tok) = self.bpe()? {
            return Ok(tok.encode_docs(text));
        }
        self.require_token_id_model()?;
        let index: std::collections::HashMap<&str, usize> = self
            .meta
            .class_names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        text.chars()
            .filter(|&c| c != '\r')
            .map(|ch| {
                index.get(char_to_hex(ch).as_str()).copied().ok_or_else(|| {
                    InferError::DimMismatch(format!(
                        "character {ch:?} is not in the model vocabulary"
                    ))
                })
            })
            .collect()
    }

    /// Error unless this is a token-ID model (`TaskType::TransformerSLM`).
    /// Legacy one-hot MLP SLM files (flattened one-hot input) are no longer
    /// supported for generation or evaluation.
    fn require_token_id_model(&self) -> Result<()> {
        if self.meta.task == TaskType::TransformerSLM {
            Ok(())
        } else {
            Err(InferError::DimMismatch(format!(
                "not a token-ID SLM (task {:?}); legacy one-hot MLP models are not supported",
                self.meta.task
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::meta::TaskType;
    use crate::slm::test_support::{tiny_bpe_slm, tiny_char_slm};
    use crate::tokenizer::ByteBpeTokenizer;
    use std::sync::Arc;

    #[test]
    fn char_level_encode_drops_cr_and_bos_is_zero() {
        let slm = tiny_char_slm();
        assert_eq!(slm.bos_id(), 0);
        assert_eq!(
            slm.encode_text("ab\r\nc").unwrap(),
            slm.encode_text("ab\nc").unwrap()
        );
        assert!(slm.encode_text("").unwrap().is_empty());
        assert_eq!(slm.score_rolling("abc").unwrap().n_tokens, 3);
    }

    #[test]
    fn bpe_bos_is_eos_and_encode_text_matches_tokenizer() {
        let slm = tiny_bpe_slm();
        let tok = ByteBpeTokenizer::from_state(&slm.meta.tokenizer_state).unwrap();
        assert_eq!(slm.bos_id(), tok.eos_id().unwrap());
        assert_eq!(slm.encode_text("the cat").unwrap(), tok.encode("the cat"));
        assert!(slm.encode_text("").unwrap().is_empty());
    }

    #[test]
    fn bos_id_is_zero_when_tokenizer_state_is_unusable() {
        let mut slm = tiny_bpe_slm();
        slm.meta.tokenizer_state = "not,a,state".into();
        assert_eq!(slm.bos_id(), 0);
    }

    /// Regression (deferred minor): `bos_id`/`encode_text`/`generate_tokens`
    /// rebuilt the BPE tokenizer on every call. It is now built once, rebuilt
    /// only when `meta.tokenizer_state` changes, and errors are not cached.
    #[test]
    fn bpe_tokenizer_is_built_once_and_tracks_state_edits() {
        let mut slm = tiny_bpe_slm();
        let a = slm.bpe().unwrap().unwrap();
        let b = slm.bpe().unwrap().unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "second call must reuse the cached tokenizer"
        );
        // Bit-identical to a freshly parsed tokenizer.
        let fresh = ByteBpeTokenizer::from_state(&slm.meta.tokenizer_state).unwrap();
        assert_eq!(
            slm.encode_text("the cat sat").unwrap(),
            fresh.encode_docs("the cat sat")
        );
        assert_eq!(slm.bos_id(), fresh.eos_id().unwrap_or(0));
        // An edited state invalidates the cache; a broken one errors, and the
        // good entry is not poisoned by it.
        let good = slm.meta.tokenizer_state.clone();
        slm.meta.tokenizer_state = "not,a,state".into();
        assert!(slm.bpe().is_err());
        assert!(slm.encode_text("x").is_err());
        slm.meta.tokenizer_state = good;
        let c = slm.bpe().unwrap().unwrap();
        assert_eq!(
            slm.encode_text("the cat sat").unwrap(),
            c.encode_docs("the cat sat")
        );
    }

    #[test]
    fn char_level_models_have_no_bpe_tokenizer() {
        assert!(tiny_char_slm().bpe().unwrap().is_none());
    }

    #[test]
    fn bpe_cache_survives_a_poisoned_lock() {
        let slm = tiny_bpe_slm();
        let _ = slm.bpe().unwrap();
        std::thread::scope(|sc| {
            let r = sc.spawn(|| {
                let _g = slm.tokenizer.0.lock().unwrap();
                panic!("poison the cache lock");
            });
            assert!(r.join().is_err());
        });
        assert!(slm.tokenizer.0.is_poisoned());
        assert!(slm.bpe().unwrap().is_some());
    }

    #[test]
    fn encode_text_rejects_legacy_non_token_models() {
        let mut slm = tiny_char_slm();
        slm.meta.task = TaskType::Classification;
        let err = slm.encode_text("abc").unwrap_err().to_string();
        assert!(err.contains("not a token-ID SLM"), "{err}");
    }
}
