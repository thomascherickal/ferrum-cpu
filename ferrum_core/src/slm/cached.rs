//! KV-cached incremental forward driver for models with transformer blocks.
use crate::error::{InferError, Result};
use crate::layer::{Embedding, KvCache, TransformerBlock};
use crate::model::Sequential;
use crate::tensor::Tensor;

/// Incremental, KV-cached driver over an inference `Sequential` whose layers are
/// `Embedding → TransformerBlock × N → LayerNorm → Linear → Softmax`.
///
/// Feeding tokens one at a time with a per-block [`KvCache`] makes each new-token
/// step O(context) instead of the O(context²) of re-running a full forward over
/// the whole window every token (which `generate` did before). This mirrors the
/// WASM `TransformerSLMModel` path so native (CLI/GUI) generation runs on the
/// same fast inference engine as the browser build.
///
/// [`CachedTransformer::try_new`] returns `None` for models without
/// `TransformerBlock`s (e.g. the embedded-MLP family), so callers transparently
/// fall back to the full-forward loop.
pub(super) struct CachedTransformer<'a> {
    model: &'a Sequential,
    /// One cache per `TransformerBlock`, in layer order.
    caches: Vec<KvCache>,
    /// Absolute sequence position of the next token to be fed.
    pos: usize,
    /// Block context length = cache capacity = positional-table size.
    capacity: usize,
}

#[cfg(test)]
thread_local! {
    /// Test-only count of `CachedTransformer::feed` calls on this thread.
    pub(super) static FEEDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl<'a> CachedTransformer<'a> {
    /// Build a driver if `model` contains at least one `TransformerBlock`.
    pub(super) fn try_new(model: &'a Sequential) -> Option<Self> {
        let mut caches = Vec::new();
        let mut capacity = 0;
        for layer in model.layers() {
            if let Some(tb) = layer.as_any().downcast_ref::<TransformerBlock>() {
                capacity = tb.context_len();
                caches.push(KvCache::new(tb.context_len(), tb.embedding_dim()));
            }
        }
        if caches.is_empty() {
            None
        } else {
            Some(Self {
                model,
                caches,
                pos: 0,
                capacity,
            })
        }
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Drop all cached positions (start a fresh window at position 0).
    fn reset(&mut self) {
        for c in &mut self.caches {
            c.clear();
        }
        self.pos = 0;
    }

    /// Feed one token through the cached path, returning the model's final
    /// softmax row — the next-token probability distribution. Walks the layer
    /// list exactly like a full forward, but routes the embedding through
    /// `embed_one(token, pos)` and each block through `forward_with_cache`.
    pub(super) fn feed(&mut self, token: usize) -> Result<Vec<f32>> {
        #[cfg(test)]
        FEEDS.with(|f| f.set(f.get() + 1));
        let mut x: Option<Tensor> = None;
        let mut block_idx = 0usize;
        for layer in self.model.layers() {
            let any = layer.as_any();
            if let Some(emb) = any.downcast_ref::<Embedding>() {
                x = Some(emb.embed_one(token, self.pos)?);
            } else if let Some(tb) = any.downcast_ref::<TransformerBlock>() {
                let cur = x.ok_or_else(|| {
                    InferError::DimMismatch("Embedding must precede TransformerBlock".into())
                })?;
                x = Some(tb.forward_with_cache(&cur, &mut self.caches[block_idx])?);
                block_idx += 1;
            } else {
                let cur = x.ok_or_else(|| {
                    InferError::DimMismatch("model must start with an Embedding layer".into())
                })?;
                x = Some(layer.forward(&cur)?);
            }
        }
        self.pos += 1;
        x.map(|t| t.data)
            .ok_or_else(|| InferError::DimMismatch("empty model".into()))
    }

    /// Reset and feed `ids` (positions 0..ids.len()), returning the
    /// distribution after the last token. `ids.len()` must be ≤ `capacity`.
    pub(super) fn prime(&mut self, ids: &[usize]) -> Result<Vec<f32>> {
        self.reset();
        let mut dist = Vec::new();
        for &t in ids {
            dist = self.feed(t)?;
        }
        Ok(dist)
    }
}

#[cfg(test)]
mod tests {
    use super::CachedTransformer;
    use crate::model::Sequential;
    use crate::rng::Rng;
    use crate::slm::test_support::tiny_transformer_model;
    use crate::tensor::Tensor;
    use crate::train_transformer::TransformerNet;

    /// I1 core guarantee: priming the KV cache with a full `context_len` window
    /// yields the same next-token distribution as a single full forward over
    /// that window (its last row). This is what makes the cached generation a
    /// drop-in, faster replacement for the O(context²) per-token full forward.
    #[test]
    fn cached_prime_matches_full_forward_last_row() {
        let model = tiny_transformer_model();
        let context_len = 8;
        let ids = [1usize, 0, 3, 5, 2, 4, 1, 3]; // exactly one window

        let x = Tensor::matrix(1, context_len, ids.iter().map(|&t| t as f32).collect()).unwrap();
        let full = model.forward(&x).unwrap();
        let (rows, cols) = full.matrix_dims().unwrap();
        let full_last = &full.data[(rows - 1) * cols..];

        let mut cached = CachedTransformer::try_new(&model).expect("model has transformer blocks");
        assert_eq!(cached.capacity(), context_len);
        let dist = cached.prime(&ids).unwrap();

        assert_eq!(dist.len(), full_last.len());
        for (a, b) in dist.iter().zip(full_last) {
            assert!((a - b).abs() < 1e-5, "cached {a} vs full-forward {b}");
        }
        // Both are proper probability distributions (model ends in Softmax).
        assert!((dist.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }

    /// Feeding a partial prefix (window below capacity) matches a full forward
    /// over a window whose first `prefix_len` rows are those tokens: the cached
    /// path's distribution after the last fed token equals that window's row at
    /// the same position. Confirms the incremental O(context) feed is exact, not
    /// just the full-window prime.
    #[test]
    fn cached_partial_prefix_matches_full_forward_row() {
        let model = tiny_transformer_model();
        let context_len = 8;
        let prefix = [4usize, 1, 5, 2]; // 4 of 8 positions
        let p = prefix.len() - 1; // position of the last fed token

        // Full forward over a window holding `prefix` at positions 0..p (the
        // remaining slots are arbitrary; causal masking makes row p depend only
        // on positions 0..=p).
        let mut window = prefix.to_vec();
        window.resize(context_len, 0);
        let x = Tensor::matrix(1, context_len, window.iter().map(|&t| t as f32).collect()).unwrap();
        let full = model.forward(&x).unwrap();
        let (_rows, cols) = full.matrix_dims().unwrap();
        let row_p = &full.data[p * cols..(p + 1) * cols];

        let mut cached = CachedTransformer::try_new(&model).unwrap();
        let dist = cached.prime(&prefix).unwrap();
        for (a, b) in dist.iter().zip(row_p) {
            assert!((a - b).abs() < 1e-5, "cached prefix {a} vs full row {b}");
        }
    }

    /// The cached path is correct for BPE-sized vocabularies and large token
    /// IDs — not just the tiny char vocab. Drives a window of high-valued tokens
    /// through the cache and checks it tracks a full forward, confirming the BPE
    /// generation path feeds the cache valid positions/ids.
    #[test]
    fn cached_prime_matches_full_forward_bpe_vocab() {
        let mut rng = Rng::new(19);
        // vocab=300 (BPE base 256 + merges), context=6, embed=8, heads=2, 2 blocks
        let model = TransformerNet::new(300, 6, 8, 2, 16, 2, &mut rng)
            .unwrap()
            .to_inference()
            .unwrap();
        let context_len = 6;
        let ids = [257usize, 12, 299, 0, 130, 256]; // spans the byte base and merges

        let x = Tensor::matrix(1, context_len, ids.iter().map(|&t| t as f32).collect()).unwrap();
        let full = model.forward(&x).unwrap();
        let (rows, cols) = full.matrix_dims().unwrap();
        let full_last = &full.data[(rows - 1) * cols..];

        let mut cached = CachedTransformer::try_new(&model).unwrap();
        let dist = cached.prime(&ids).unwrap();
        assert_eq!(dist.len(), full_last.len());
        for (a, b) in dist.iter().zip(full_last) {
            assert!((a - b).abs() < 1e-5, "cached {a} vs full-forward {b}");
        }
    }

    /// Models without transformer blocks have no KV cache to drive, so the
    /// driver declines and the caller falls back to the full-forward loop.
    #[test]
    fn try_new_declines_non_transformer_models() {
        let linear = crate::layer::Linear::new(4, 6, vec![0.0; 24], vec![0.0; 6]).unwrap();
        let model = Sequential::new().with(Box::new(linear));
        assert!(CachedTransformer::try_new(&model).is_none());
    }
}
