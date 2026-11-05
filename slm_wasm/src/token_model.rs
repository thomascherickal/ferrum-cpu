//! The logic of the browser `TransformerSLMModel` as plain Rust returning `Result<_, String>`, so
//! it is tested natively; `lib.rs` only forwards each call and turns the `String` into a `JsValue`.

use ferrum_core::layer::{Embedding, KvCache, TransformerBlock};
use ferrum_core::{from_bytes, Sequential, TaskType, Tensor};

pub(crate) struct TokenModel {
    model: Sequential,
    meta_json: String,
    vocab_size: usize,
    context_len: usize,
    num_heads: usize,
    num_blocks: usize,
    /// One KV cache per TransformerBlock, in layer order.
    caches: Vec<KvCache>,
    /// Sequence position of the next token fed to the cached path.
    cache_pos: usize,
}

impl TokenModel {
    pub(crate) fn new(bytes: &[u8]) -> Result<Self, String> {
        let (model, _norm, meta) = from_bytes(bytes).map_err(|e| e.to_string())?;
        if meta.task != TaskType::TransformerSLM {
            return Err("Expected TransformerSLM task type".to_string());
        }
        // Count transformer blocks, extract num_heads, and build one KV cache
        // per block for the incremental generation path.
        let mut num_blocks = 0usize;
        let mut num_heads = 4usize; // default
        let mut caches = Vec::new();
        for layer in model.layers() {
            if let Some(tb) = layer.as_any().downcast_ref::<TransformerBlock>() {
                if num_blocks == 0 {
                    num_heads = tb.num_heads();
                }
                num_blocks += 1;
                caches.push(KvCache::new(tb.context_len(), tb.embedding_dim()));
            }
        }
        Ok(Self {
            vocab_size: meta.output_dim,
            context_len: meta.input_dim,
            meta_json: meta.to_json(),
            model,
            num_heads,
            num_blocks,
            caches,
            cache_pos: 0,
        })
    }

    pub(crate) fn metadata(&self) -> String {
        self.meta_json.clone()
    }

    pub(crate) fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub(crate) fn context_len(&self) -> usize {
        self.context_len
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.num_blocks
    }

    /// Next-token probabilities for exactly `context_len` token ids (as f32).
    pub(crate) fn predict_next(&self, context: &[f32]) -> Result<Vec<f32>, String> {
        if context.len() != self.context_len {
            return Err(format!(
                "Context must have exactly {} elements, got {}",
                self.context_len,
                context.len()
            ));
        }
        // `context` has exactly `context_len` entries, so this shape is valid by construction.
        let input = Tensor {
            shape: vec![1, self.context_len],
            data: context.to_vec(),
        };
        let out = self.model.forward(&input).map_err(|e| e.to_string())?;
        // The last row (the final `cols` values) is the next-token prediction.
        let cols = out.shape.last().copied().unwrap_or(1).max(1);
        Ok(out.data[out.data.len().saturating_sub(cols)..].to_vec())
    }

    /// Reset the KV caches, feed `context`, return the distribution after its last token.
    pub(crate) fn prime(&mut self, context: &[f32]) -> Result<Vec<f32>, String> {
        if context.is_empty() || context.len() > self.context_len {
            return Err(format!(
                "Context must have 1..={} tokens, got {}",
                self.context_len,
                context.len()
            ));
        }
        for c in &mut self.caches {
            c.clear();
        }
        self.cache_pos = 0;
        let mut probs = Vec::new();
        for &tok in context {
            // `as usize` saturates: a negative or NaN id would silently become token 0.
            let id = tok.round();
            if id.is_nan() || id < 0.0 {
                return Err(format!("Token id {tok} is not a non-negative number"));
            }
            probs = self.feed_token(id as usize)?;
        }
        Ok(probs)
    }

    /// One token through the KV-cached path; needs a prior `prime`.
    pub(crate) fn predict_next_cached(&mut self, token_id: usize) -> Result<Vec<f32>, String> {
        if self.cache_pos == 0 {
            return Err("Call prime(context) before predict_next_cached".to_string());
        }
        self.feed_token(token_id)
    }

    pub(crate) fn cached_len(&self) -> usize {
        self.cache_pos
    }

    fn feed_token(&mut self, token: usize) -> Result<Vec<f32>, String> {
        let js = |e: ferrum_core::InferError| e.to_string();
        let mut x: Option<Tensor> = None;
        let mut block_idx = 0usize;
        for layer in self.model.layers() {
            let any = layer.as_any();
            if let Some(emb) = any.downcast_ref::<Embedding>() {
                x = Some(emb.embed_one(token, self.cache_pos).map_err(js)?);
            } else if let Some(tb) = any.downcast_ref::<TransformerBlock>() {
                let cur =
                    x.ok_or_else(|| "model has no Embedding before TransformerBlock".to_string())?;
                x = Some(
                    tb.forward_with_cache(&cur, &mut self.caches[block_idx])
                        .map_err(js)?,
                );
                block_idx += 1;
            } else {
                let cur =
                    x.ok_or_else(|| "model must start with an Embedding layer".to_string())?;
                x = Some(layer.forward(&cur).map_err(js)?);
            }
        }
        self.cache_pos += 1;
        x.map(|t| t.data).ok_or_else(|| "empty model".to_string())
    }

    /// Flat `[num_heads x T x T]` attention weights of the last block's last forward pass.
    pub(crate) fn get_last_attention_weights(&self) -> Vec<f32> {
        self.model
            .layers()
            .iter()
            .rev()
            .find_map(|l| l.as_any().downcast_ref::<TransformerBlock>())
            .map(|tb| tb.last_attention.borrow().clone())
            .unwrap_or_default()
    }

    /// Temperature-scaled draw from `probs`; `random_value` in `[0, 1]`. Empty input gives 0.
    /// `r < cumsum` (not `<=`) so a zero-mass leading entry is never picked at `r == 0`.
    pub(crate) fn sample_from_probs(
        &self,
        probs: &[f32],
        temperature: f32,
        random_value: f32,
    ) -> usize {
        let scaled: Vec<f32> = probs
            .iter()
            .map(|&p| (p.ln() / temperature.max(0.01)).exp())
            .collect();
        let sum: f32 = scaled.iter().sum();
        let r = random_value.clamp(0.0, 1.0 - 1e-7);
        let mut cumsum = 0.0f32;
        for (i, &v) in scaled.iter().enumerate() {
            cumsum += v / sum;
            if r < cumsum {
                return i;
            }
        }
        scaled.len().saturating_sub(1)
    }

    /// Shannon entropy (0 = certain, ln(V) = uniform).
    pub(crate) fn entropy(&self, probs: &[f32]) -> f32 {
        probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// The top-k token indices, most probable first.
    pub(crate) fn top_k_indices(&self, probs: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.iter().take(k).map(|&(i, _)| i).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::TokenModel;
    use crate::test_support::{hand_made, hand_made_bytes, trained};
    use ferrum_core::model::Sequential;
    use ferrum_core::{from_bytes, to_bytes, Rng, SamplingParams, TaskType, Tensor};

    fn ids_f32(ids: &[usize]) -> Vec<f32> {
        ids.iter().map(|&i| i as f32).collect()
    }

    /// A model, its first 16 token ids as f32, and the library's forward pass on them.
    fn setup() -> (TokenModel, Vec<f32>, Vec<f32>) {
        let (lib, bytes) = trained(true);
        let mut ids = vec![lib.bos_id()];
        ids.extend(
            lib.encode_text("the cat sat on the mat. the dog sat on the log. the cat sat")
                .unwrap(),
        );
        let ctx = ids_f32(&ids[..16]);
        let out = lib
            .model
            .forward(&Tensor::matrix(1, 16, ctx.clone()).unwrap())
            .unwrap();
        let (rows, cols) = out.matrix_dims().unwrap();
        let want = out.data[(rows - 1) * cols..].to_vec();
        (TokenModel::new(&bytes).unwrap(), ctx, want)
    }

    #[test]
    fn predict_next_matches_library_forward() {
        let (m, ctx, want) = setup();
        assert_eq!(m.predict_next(&ctx).unwrap(), want);
        assert_eq!(want.len(), m.vocab_size());
    }

    #[test]
    fn prime_then_cached_equals_full_forward() {
        let (mut m, ctx, want) = setup();
        m.prime(&ctx[..8]).unwrap();
        let mut got = Vec::new();
        for &id in &ctx[8..] {
            got = m.predict_next_cached(id as usize).unwrap();
        }
        assert_eq!(m.cached_len(), 16);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // Priming again restarts from position 0.
        assert_eq!(m.prime(&ctx).unwrap(), got);
        assert_eq!(m.cached_len(), 16);
    }

    #[test]
    fn predict_next_rejects_out_of_vocab_id() {
        let (mut m, mut ctx, _) = setup();
        ctx[3] = m.vocab_size() as f32;
        assert!(m.predict_next(&ctx).is_err());
        assert!(m.prime(&ctx).is_err());
        let (_, ctx, _) = setup();
        m.prime(&ctx[..4]).unwrap();
        assert!(m.predict_next_cached(m.vocab_size()).is_err());
    }

    /// Regression (audit 2026-11-05, A1): a negative or NaN id from JS saturated to token 0.
    #[test]
    fn predict_next_and_prime_reject_negative_and_nan_ids() {
        let (mut m, ctx, _) = setup();
        for bad in [-1.0, f32::NAN] {
            let mut c = ctx.clone();
            c[3] = bad;
            assert!(m.predict_next(&c).is_err(), "predict_next took {bad}");
            assert!(m.prime(&c).is_err(), "prime took {bad}");
        }
    }

    #[test]
    fn predict_next_rejects_empty_context() {
        let (mut m, ctx, _) = setup();
        let e = m.predict_next(&[]).unwrap_err();
        assert_eq!(e, "Context must have exactly 16 elements, got 0");
        assert_eq!(
            m.prime(&[]).unwrap_err(),
            "Context must have 1..=16 tokens, got 0"
        );
        let long = [ctx.clone(), vec![0.0]].concat();
        assert_eq!(
            m.prime(&long).unwrap_err(),
            "Context must have 1..=16 tokens, got 17"
        );
        assert_eq!(
            m.predict_next_cached(0).unwrap_err(),
            "Call prime(context) before predict_next_cached"
        );
    }

    #[test]
    fn sample_from_probs_temperature_zero_is_argmax() {
        let (m, _, _) = setup();
        let probs = [0.1, 0.2, 0.6, 0.1];
        for r in [0.0, 0.3, 0.99, 1.0] {
            assert_eq!(m.sample_from_probs(&probs, 0.0, r), 2, "r={r}");
        }
    }

    #[test]
    fn sample_from_probs_bad_input_never_panics() {
        let (m, _, _) = setup();
        assert_eq!(m.sample_from_probs(&[], 1.0, 0.5), 0);
        assert!(m.sample_from_probs(&[f32::NAN, f32::NAN], 1.0, 0.5) < 2);
        assert!(m.sample_from_probs(&[0.0, 0.0, 0.0], 1.0, 0.5) < 3);
        assert!(m.sample_from_probs(&[0.5, 0.5], f32::NAN, f32::NAN) < 2);
        // A draw past the end of the cumulative sum lands on the last index.
        assert_eq!(m.sample_from_probs(&[0.25, 0.25], 1.0, 1.0), 1);
    }

    #[test]
    fn top_k_indices_k_larger_than_vocab() {
        let (m, _, _) = setup();
        assert_eq!(m.top_k_indices(&[0.1, 0.7, 0.2], 10), vec![1, 2, 0]);
        assert_eq!(m.top_k_indices(&[0.1, 0.7, 0.2], 0), Vec::<usize>::new());
        assert!(m.top_k_indices(&[], 3).is_empty());
    }

    #[test]
    fn entropy_of_one_hot_is_zero() {
        let (m, _, _) = setup();
        assert_eq!(m.entropy(&[0.0, 1.0, 0.0]), 0.0);
        assert!((m.entropy(&[0.25; 4]) - 4f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn attention_weights_shape_is_heads_x_len() {
        let (m, ctx, _) = setup();
        assert!(m.get_last_attention_weights().is_empty());
        m.predict_next(&ctx).unwrap();
        assert_eq!(
            m.get_last_attention_weights().len(),
            m.num_heads() * 16 * 16
        );
    }

    #[test]
    fn reports_the_trained_models_shape() {
        let (m, _, _) = setup();
        assert_eq!((m.context_len(), m.num_heads(), m.num_layers()), (16, 2, 2));
        assert_eq!(m.cached_len(), 0);
        assert!(m.metadata().contains("\"input_dim\""));
    }

    #[test]
    fn new_rejects_garbage_and_non_text_models() {
        assert!(TokenModel::new(b"CORRUPT").is_err());
        let (_, bytes) = trained(false);
        let (model, norm, mut meta) = from_bytes(&bytes).unwrap();
        meta.task = TaskType::Classification;
        let other = to_bytes(&model, &norm, &meta).unwrap();
        assert_eq!(
            TokenModel::new(&other).err().unwrap(),
            "Expected TransformerSLM task type"
        );
    }

    #[test]
    fn hand_made_model_loads_and_predicts() {
        let bytes = hand_made_bytes();
        let m = TokenModel::new(&bytes).unwrap();
        assert_eq!((m.vocab_size(), m.context_len(), m.num_heads()), (10, 8, 2));
        let probs = m
            .predict_next(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();
        assert_eq!(probs.len(), 10);
        assert!(probs.iter().all(|v| v.is_finite()));
        assert!(TokenModel::new(b"CORRUPT").is_err());
        assert!(TokenModel::new(&[]).is_err());
    }

    #[test]
    fn a_model_without_the_expected_layers_errors_in_the_cached_path() {
        let mut m = TokenModel::new(&hand_made_bytes()).unwrap();
        m.model = Sequential::new();
        assert_eq!(m.prime(&[1.0]).unwrap_err(), "empty model");

        let (_, tb, ln, ..) = hand_made();
        m.model = Sequential::new().with(Box::new(tb));
        assert_eq!(
            m.prime(&[1.0]).unwrap_err(),
            "model has no Embedding before TransformerBlock"
        );
        m.model = Sequential::new().with(Box::new(ln));
        assert_eq!(
            m.prime(&[1.0]).unwrap_err(),
            "model must start with an Embedding layer"
        );
    }

    #[test]
    fn cached_log_probabilities_sum_to_the_librarys_rolling_score() {
        let (lib, bytes) = trained(true);
        let mut w = TokenModel::new(&bytes).unwrap();
        let text = "the cat sat";
        let mut ids = vec![lib.bos_id()];
        ids.extend(lib.encode_text(text).unwrap());
        assert!(ids.len() <= 16);
        let mut probs = w.prime(&ids_f32(&ids[..1])).unwrap();
        let mut logprob = 0.0f64;
        for &id in &ids[1..] {
            logprob += (probs[id] as f64).ln();
            probs = w.predict_next_cached(id).unwrap();
        }
        let want = lib.score_rolling(text).unwrap();
        assert_eq!(want.n_tokens, ids.len() - 1);
        assert!((logprob - want.logprob).abs() < 1e-3);
        // Greedy next token from the browser path equals the library's greedy continuation.
        let next = w.top_k_indices(&probs, 1)[0];
        let sampling = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..SamplingParams::default()
        };
        let cont = lib
            .generate_tokens(text, 1, &[], &sampling, &mut Rng::new(0))
            .unwrap();
        let mut with_next = ids.clone();
        with_next.push(next);
        assert_eq!(
            lib.encode_text(&format!("{text}{}", cont.text)).unwrap(),
            with_next[1..].to_vec()
        );
    }
}
