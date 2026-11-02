//! WASM bindings for ferrum Transformer SLMs.
//!
//! `TransformerSLMModel` loads a GGUF transformer SLM and exposes next-token
//! prediction (full forward or KV-cached), sampling helpers, and the last
//! attention map for visualization — so a trained model runs in any browser.
//!
//! `Slm` is the text API: encode, decode, generate and score with the same tokenizer and sampler
//! as the CLI, for BPE and character-level models. Its logic lives in `text.rs` so it is tested
//! natively.

#![forbid(unsafe_code)]

mod text;
mod token_model;

#[cfg(test)]
mod test_support;

use text::{GenParams, TextSlm};
use token_model::TokenModel;
use wasm_bindgen::prelude::*;

/// A text model in the browser: `new Slm(bytes)`, then `encode`, `decode`, `generate`, `score`.
///
/// ```js
/// const bytes = new Uint8Array(await (await fetch('model.gguf')).arrayBuffer());
/// const slm = new Slm(bytes);
/// const info = JSON.parse(slm.info());   // { params, vocab, context, quant, tokenizer }
/// const ids = slm.encode('the cat');     // Uint32Array
/// const text = slm.decode(ids);          // 'the cat'
/// const out = slm.generate('the cat', 40, 0.8, 0, 1.0, 1.0, 42, ['\n']);
/// const logp = slm.score('the cat sat'); // rolling log-probability
///
/// // Streaming: pull the text piece by piece (run it in a Web Worker, see demo/worker.js).
/// slm.start('the cat', 40, 0.8, 0, 1.0, 1.0, 42, []);  // same arguments as generate
/// for (let c; (c = slm.next_chunk()) !== undefined; ) console.log(c);
/// slm.cancel();                          // abandon a generation early
/// ```
///
/// `generate` returns only the continuation (the prompt is not repeated) and equals
/// `train_transformer generate <model> <prompt> --max-tokens N --temp T --top-k K --top-p P
/// --repetition-penalty R --gen-seed SEED --stop S` minus the prompt. Errors are thrown as
/// JavaScript `Error`s.
/// Core errors are strings; the browser sees them as `Error`s.
fn js_err(e: String) -> JsError {
    JsError::new(&e)
}

#[wasm_bindgen]
pub struct Slm(TextSlm);

#[wasm_bindgen]
impl Slm {
    /// Load a Ferrum GGUF text model from bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Slm, JsError> {
        Ok(Slm(TextSlm::from_bytes(bytes).map_err(js_err)?))
    }

    /// Model facts as JSON: `{"params","vocab","context","quant","tokenizer"}`.
    pub fn info(&self) -> String {
        self.0.info()
    }

    /// Token ids for `text`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, JsError> {
        self.0.encode(text).map_err(js_err)
    }

    /// Text for token ids.
    pub fn decode(&self, ids: &[u32]) -> Result<String, JsError> {
        self.0.decode(ids).map_err(js_err)
    }

    /// Generate up to `max_tokens` tokens after `prompt`; stops early at any `stop` string (not
    /// included) or the end-of-text token. `temperature` 0 is greedy, `top_k` 0 and `top_p` 1
    /// disable those filters, `repetition_penalty` 1 disables the penalty. `seed` fixes the draw.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u32,
        stop: Vec<String>,
    ) -> Result<String, JsError> {
        let p = GenParams::new(
            max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            u64::from(seed),
            stop,
        )
        .map_err(js_err)?;
        self.0.generate(prompt, &p).map_err(js_err)
    }

    /// Begin a generation to pull with `next_chunk` (same arguments as `generate`). The chunks
    /// add up to what `generate` returns. Starting again replaces a generation in progress.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u32,
        stop: Vec<String>,
    ) -> Result<(), JsError> {
        let p = GenParams::new(
            max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            u64::from(seed),
            stop,
        )
        .map_err(js_err)?;
        self.0.start(prompt, &p).map_err(js_err)
    }

    /// The next piece of text, or `undefined` when the generation is over (or none was started).
    /// Each call does one or a few tokens of work, so a page can stay responsive between calls.
    pub fn next_chunk(&mut self) -> Result<Option<String>, JsError> {
        self.0.next_chunk().map_err(js_err)
    }

    /// Tokens generated so far by the latest `start` (still readable once it has ended).
    pub fn tokens_generated(&self) -> usize {
        self.0.tokens_generated()
    }

    /// Abandon the generation in progress.
    pub fn cancel(&mut self) {
        self.0.cancel();
    }

    /// Rolling log-probability of `text` (higher is likelier; every token is scored).
    pub fn score(&self, text: &str) -> Result<f64, JsError> {
        self.0.score(text).map_err(js_err)
    }
}

/// Edge Small Language Model — causal Transformer running in WASM.
///
/// JavaScript usage:
/// ```js
/// const resp = await fetch('model.gguf');
/// const bytes = new Uint8Array(await resp.arrayBuffer());
/// const slm = new TransformerSLMModel(bytes);
///
/// const meta = JSON.parse(slm.metadata());
/// const vocab = meta.class_names;           // idx → char mapping
/// const contextLen = meta.input_dim;
///
/// // Feed a context of integer token IDs
/// const context = new Float32Array([12, 5, 3, ...]);
/// const probs = slm.predict_next(context);  // Float32Array length vocab_size
///
/// // After inference, read attention weights for visualization
/// const attn = slm.get_last_attention_weights(); // Float32Array [heads × T × T]
/// const numHeads = slm.num_heads();
/// const contextLength = slm.context_len();
/// ```
#[wasm_bindgen]
pub struct TransformerSLMModel(TokenModel);

fn js_val(e: String) -> JsValue {
    JsValue::from_str(&e)
}

#[wasm_bindgen]
impl TransformerSLMModel {
    /// Construct from GGUF bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<TransformerSLMModel, JsValue> {
        Ok(Self(TokenModel::new(bytes).map_err(js_val)?))
    }

    /// Return model metadata JSON (vocab, context length, architecture details).
    pub fn metadata(&self) -> String {
        self.0.metadata()
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.0.num_heads()
    }

    /// Vocabulary size (number of distinct tokens).
    pub fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }

    /// Context length (number of characters the model sees at once).
    pub fn context_len(&self) -> usize {
        self.0.context_len()
    }

    /// Number of Transformer blocks.
    pub fn num_layers(&self) -> usize {
        self.0.num_layers()
    }

    /// Run inference on a context of token indices.
    ///
    /// `context` must have exactly `context_len` elements (Float32Array of usize-as-f32).
    /// Returns a Float32Array of length `vocab_size` with next-token probabilities.
    pub fn predict_next(&self, context: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.0.predict_next(context).map_err(js_val)
    }

    /// Reset the KV caches and prime them with a context of token IDs.
    ///
    /// After priming, call `predict_next_cached` to extend the sequence one
    /// token at a time at O(T) per token instead of re-running the full
    /// O(T²) context. Returns the next-token probabilities after the last
    /// context token (same as `predict_next` on the full context).
    pub fn prime(&mut self, context: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.0.prime(context).map_err(js_val)
    }

    /// Feed one token through the KV-cached path and return the next-token
    /// probabilities. Requires a prior `prime()`. Errors when the cache is
    /// full (`context_len` positions) — re-`prime` with a fresh window then.
    pub fn predict_next_cached(&mut self, token_id: usize) -> Result<Vec<f32>, JsValue> {
        self.0.predict_next_cached(token_id).map_err(js_val)
    }

    /// Number of positions currently held in the KV caches.
    pub fn cached_len(&self) -> usize {
        self.0.cached_len()
    }

    /// Return the self-attention weights from the LAST Transformer block's last forward pass.
    ///
    /// Returns a flat Float32Array of shape [num_heads × context_len × context_len].
    /// For `num_heads=4` and `context_len=32`, this is 4096 floats.
    ///
    /// In JavaScript, index as: `attn[h * T * T + i * T + j]` where h=head, i=query pos, j=key pos.
    pub fn get_last_attention_weights(&self) -> Vec<f32> {
        self.0.get_last_attention_weights()
    }

    /// Sample a next token index from probabilities using temperature scaling.
    /// `probs` = output of `predict_next`. `temperature` = 1.0 is neutral.
    /// Lower temperature → more deterministic. Higher → more random.
    pub fn sample_from_probs(&self, probs: &[f32], temperature: f32, random_value: f32) -> usize {
        self.0.sample_from_probs(probs, temperature, random_value)
    }

    /// Compute Shannon entropy of a probability distribution (0 = certain, ln(V) = uniform).
    pub fn entropy(&self, probs: &[f32]) -> f32 {
        self.0.entropy(probs)
    }

    /// Return the top-k token indices sorted by probability (descending).
    pub fn top_k_indices(&self, probs: &[f32], k: usize) -> Vec<usize> {
        self.0.top_k_indices(probs, k)
    }
}

#[cfg(test)]
mod tests {
    use ferrum_core::model::Sequential;
    use ferrum_core::{
        from_bytes, to_bytes, Embedding, LayerNorm, Linear, ModelMetadata, Normalizer, Rng,
        TaskType, Tensor, TransformerBlock,
    };

    fn slm_bytes() -> Vec<u8> {
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

        let ln_final =
            LayerNorm::new(embed_dim, vec![1.0; embed_dim], vec![0.0; embed_dim]).unwrap();
        let lm_head = Linear::new(
            embed_dim,
            vocab_size,
            (0..embed_dim * vocab_size)
                .map(|_| rng.next_normal() * scale)
                .collect(),
            vec![0.0; vocab_size],
        )
        .unwrap();

        let model = Sequential::new()
            .with(Box::new(emb))
            .with(Box::new(tb))
            .with(Box::new(ln_final))
            .with(Box::new(lm_head));

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
        to_bytes(&model, &norm, &meta).unwrap()
    }

    #[test]
    fn slm_loads_and_predicts() {
        let bytes = slm_bytes();
        let (model, _norm, meta) = from_bytes(&bytes).unwrap();
        assert_eq!(meta.task, TaskType::TransformerSLM);
        let context: Vec<f32> = (0..8).map(|i| (i % 10) as f32).collect();
        let x = Tensor::matrix(1, 8, context).unwrap();
        let out = model.forward(&x).unwrap();
        // Should produce [1, vocab_size] or [T, vocab] — check it's finite
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn slm_metadata_has_vocab() {
        let bytes = slm_bytes();
        let (_model, _norm, meta) = from_bytes(&bytes).unwrap();
        assert_eq!(meta.class_names.len(), 10);
        assert_eq!(meta.class_names[0], "a");
        assert_eq!(meta.input_dim, 8);
        assert_eq!(meta.output_dim, 10);
    }

    #[test]
    fn corrupt_bytes_error_gracefully() {
        assert!(from_bytes(b"CORRUPT").is_err());
        assert!(from_bytes(&[]).is_err());
    }

    use crate::test_support::trained;

    fn ids_f32(ids: &[usize]) -> Vec<f32> {
        ids.iter().map(|&i| i as f32).collect()
    }

    #[test]
    fn wasm_model_reports_the_trained_models_shape() {
        let (lib, bytes) = trained(true);
        let w = super::TransformerSLMModel::new(&bytes).unwrap();
        assert_eq!(w.vocab_size(), lib.meta.output_dim);
        assert_eq!(w.context_len(), 16);
        assert_eq!(w.num_heads(), 2);
        assert_eq!(w.num_layers(), 2);
        assert_eq!(w.cached_len(), 0);
        assert!(w.metadata().contains("\"input_dim\""), "{}", w.metadata());
    }

    #[test]
    fn predict_next_matches_the_library_model_and_the_cached_path() {
        for quantized in [false, true] {
            let (lib, bytes) = trained(quantized);
            let mut w = super::TransformerSLMModel::new(&bytes).unwrap();
            let mut ids = vec![lib.bos_id()];
            ids.extend(
                lib.encode_text(
                    "the cat sat on the mat. the dog sat on the log. the cat sat on the mat.",
                )
                .unwrap(),
            );
            assert!(ids.len() >= 16, "{}", ids.len());
            let ctx = ids_f32(&ids[..16]);

            // The library's own forward pass on the same window.
            let t = Tensor::matrix(1, 16, ctx.clone()).unwrap();
            let want = lib.model.forward(&t).unwrap();
            let (rows, cols) = want.matrix_dims().unwrap();
            let want = &want.data[(rows - 1) * cols..];

            let full = w.predict_next(&ctx).unwrap();
            assert_eq!(full.len(), lib.meta.output_dim);
            assert_eq!(full, want, "quantized={quantized}");
            assert!((full.iter().sum::<f32>() - 1.0).abs() < 1e-3);

            // Priming and extending the KV cache gives the same distribution (to rounding).
            w.prime(&ctx[..8]).unwrap();
            let mut cached = Vec::new();
            for &id in &ids[8..16] {
                cached = w.predict_next_cached(id).unwrap();
            }
            assert_eq!(w.cached_len(), 16);
            for (a, b) in cached.iter().zip(&full) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
            assert_eq!(w.prime(&ctx).unwrap(), cached);
        }
    }

    #[test]
    fn cached_log_probabilities_sum_to_the_librarys_rolling_score() {
        let (lib, bytes) = trained(true);
        let mut w = super::TransformerSLMModel::new(&bytes).unwrap();
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
        assert!(
            (logprob - want.logprob).abs() < 1e-3,
            "{logprob} vs {}",
            want.logprob
        );
        // Greedy next token from the browser path equals the library's greedy continuation.
        let next = w.top_k_indices(&probs, 1)[0];
        let cont = lib
            .generate_tokens(
                text,
                1,
                &[],
                &ferrum_core::SamplingParams {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    repetition_penalty: 1.0,
                },
                &mut Rng::new(0),
            )
            .unwrap();
        let mut with_next = ids.clone();
        with_next.push(next);
        assert_eq!(
            lib.encode_text(&format!("{text}{}", cont.text)).unwrap(),
            with_next[1..].to_vec()
        );
    }
}
