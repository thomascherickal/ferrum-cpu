//! Training entry points that build a GenerativeSLM from a corpus.
use crate::error::{InferError, Result};
use crate::meta::{ModelMetadata, Normalizer, TaskType};
use crate::rng::Rng;
use crate::slm::config::{TransformerConfig, ValidationConfig, ValidationProgress};
use crate::slm::corpus::{tokenize_for_lm_cfg, LmTokens};
use crate::slm::scoring::Evaluation;
use crate::slm::training_loop::{prepare_net, run_training};
use crate::slm::{GenerativeSLM, TokenizerCache};
use crate::train_transformer::{TokenSource, TransformerNet};

impl GenerativeSLM {
    /// Train a true decoder-only causal Transformer SLM on a raw text corpus.
    ///
    /// Unlike the flat one-hot MLP, this trains
    /// token + positional embeddings, `num_blocks` causal multi-head attention
    /// blocks, and an LM head end-to-end with Adam, using next-token loss at
    /// every position. The exported model serializes to Ferrum GGUF and runs in
    /// WASM via `TransformerSLMModel` (including KV-cached generation).
    ///
    /// `vocab_size` selects the tokenizer: `0` is character-level (the corpus's
    /// sorted character set), and any value `>= 256` trains a byte-level BPE
    /// tokenizer of that size whose merge list is stored inside the model.
    #[allow(clippy::too_many_arguments)]
    pub fn train_transformer(
        corpus: &str,
        context_len: usize,
        embed_dim: usize,
        num_heads: usize,
        num_blocks: usize,
        hidden_dim: usize,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        vocab_size: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        Self::train_transformer_with_callback(
            corpus,
            context_len,
            embed_dim,
            num_heads,
            num_blocks,
            hidden_dim,
            epochs,
            lr,
            batch_size,
            vocab_size,
            rng,
            |_, _| {},
        )
    }

    /// [`GenerativeSLM::train_transformer`] with an `(epoch, loss)` callback.
    #[allow(clippy::too_many_arguments)]
    pub fn train_transformer_with_callback<F>(
        corpus: &str,
        context_len: usize,
        embed_dim: usize,
        num_heads: usize,
        num_blocks: usize,
        hidden_dim: usize,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        vocab_size: usize,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        // `threads = 1` is the serial path, bit-for-bit identical to before.
        let cfg = legacy_config(
            context_len,
            embed_dim,
            num_heads,
            num_blocks,
            hidden_dim,
            epochs,
            lr,
            batch_size,
            vocab_size,
        );
        Self::train_transformer_inner(corpus, &cfg, 1, rng, &mut progress_callback)
    }

    /// Data-parallel [`GenerativeSLM::train_transformer_with_callback`]:
    /// `threads` worker threads split each minibatch and their gradients are
    /// reduced before each optimizer step (see
    /// [`crate::train_transformer::train_transformer_epoch_threaded`]).
    ///
    /// Pass `0` to auto-detect the machine's parallelism via
    /// [`crate::num_threads`]. `1` forces the serial path. Results are
    /// reproducible for a fixed `threads` value; the model is identical to the
    /// serial one when `threads` resolves to `1`.
    #[allow(clippy::too_many_arguments)]
    pub fn train_transformer_threaded_with_callback<F>(
        corpus: &str,
        context_len: usize,
        embed_dim: usize,
        num_heads: usize,
        num_blocks: usize,
        hidden_dim: usize,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        vocab_size: usize,
        threads: usize,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let threads = if threads == 0 {
            crate::parallel::num_threads()
        } else {
            threads
        };
        let cfg = legacy_config(
            context_len,
            embed_dim,
            num_heads,
            num_blocks,
            hidden_dim,
            epochs,
            lr,
            batch_size,
            vocab_size,
        );
        Self::train_transformer_inner(corpus, &cfg, threads, rng, &mut progress_callback)
    }

    /// Shared implementation behind the serial and threaded transformer trainers.
    fn train_transformer_inner(
        corpus: &str,
        cfg: &TransformerConfig,
        threads: usize,
        rng: &mut Rng,
        progress_callback: &mut dyn FnMut(usize, f32),
    ) -> Result<Self> {
        let tc = tokenize_for_lm_cfg(corpus, cfg)?;
        vprintln!("[slm::train_transformer] corpus={} tokens, vocab={}, bpe={}, ctx={}, dim={}, heads={}, blocks={}, hidden={}, threads={}",
            tc.tokens.len(), tc.vocab_size, !tc.tokenizer_state.is_empty(),
            cfg.context_len, cfg.embed_dim, cfg.num_heads, cfg.num_blocks, cfg.hidden_dim, threads);
        let mut net = prepare_net(cfg, tc.vocab_size, rng)?;
        run_training(
            &mut net,
            &mut tc.tokens.as_slice(),
            cfg,
            threads,
            rng,
            &mut |_, ep, loss| {
                progress_callback(ep, loss);
                Ok(true)
            },
        )?;
        Self::build_transformer_slm(&net, &tc)
    }

    /// Construct an inference [`GenerativeSLM`] from a trained transformer net
    /// and its tokenization (shared by the plain and validation-aware trainers).
    pub(super) fn build_transformer_slm(net: &TransformerNet, tc: &LmTokens) -> Result<Self> {
        let model_vocab = tc.vocab_size;
        let context_len = net.context_len();
        // QAT: ship exactly the Q8_0 values the training saw, so an int8 export
        // changes nothing (snapped on a weights-only copy, see `to_inference_snapped`).
        let model = net.to_inference_snapped()?;
        let meta = ModelMetadata {
            dataset_name: "GenerativeSLM Transformer".into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..context_len).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, model_vocab as f32]; context_len],
            class_names: tc.class_names.clone(),
            target_name: "next_char".into(),
            target_range: [0.0, model_vocab as f32],
            input_dim: context_len,
            output_dim: model_vocab,
            tokenizer_state: tc.tokenizer_state.clone(),
        };
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        Ok(Self {
            model,
            norm,
            meta,
            tokenizer: TokenizerCache::default(),
        })
    }

    /// [`GenerativeSLM::train_transformer_with_callback`] driven by a
    /// [`TransformerConfig`]. Training is int8 quantization-aware (QAT).
    pub fn train_transformer_config<F>(
        corpus: &str,
        cfg: &TransformerConfig,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        Self::train_transformer_inner(corpus, cfg, 1, rng, &mut progress_callback)
    }

    /// Data-parallel [`GenerativeSLM::train_transformer_config`]: trains with
    /// `threads` worker threads (`0` = auto-detect via [`crate::num_threads`],
    /// `1` = serial). Training stays int8 quantization-aware (QAT) and the
    /// result is reproducible for a fixed `threads` value.
    pub fn train_transformer_config_threaded<F>(
        corpus: &str,
        cfg: &TransformerConfig,
        threads: usize,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let threads = if threads == 0 {
            crate::parallel::num_threads()
        } else {
            threads
        };
        Self::train_transformer_inner(corpus, cfg, threads, rng, &mut progress_callback)
    }

    /// Train a transformer SLM with an internal validation split, early stopping
    /// on validation perplexity, and best-epoch checkpointing (T5).
    ///
    /// The corpus is split by character count: the final `val.val_fraction` is
    /// held out for validation and the rest is used for training (the
    /// tokenizer/vocabulary is fit on the training portion only, as it must be).
    /// After every epoch the in-progress model is scored on the held-out text
    /// with [`GenerativeSLM::evaluate`]; the weights with the **lowest validation
    /// cross-entropy** are retained and returned — not necessarily the final
    /// epoch's. When `val.patience > 0`, training stops early once validation
    /// fails to improve for that many consecutive epochs.
    ///
    /// `threads` follows the same convention as
    /// [`GenerativeSLM::train_transformer_config_threaded`] (`0` = auto, `1` =
    /// serial). The callback receives a [`ValidationProgress`] each epoch.
    pub fn train_transformer_config_validated<F>(
        corpus: &str,
        cfg: &TransformerConfig,
        threads: usize,
        val: &ValidationConfig,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(&ValidationProgress),
    {
        Self::train_validated_dyn(corpus, cfg, threads, val, rng, &mut progress_callback)
    }

    /// [`GenerativeSLM::train_transformer_config_validated`] without the callback's type
    /// parameter: one compiled copy of the logic.
    fn train_validated_dyn(
        corpus: &str,
        cfg: &TransformerConfig,
        threads: usize,
        val: &ValidationConfig,
        rng: &mut Rng,
        progress_callback: &mut dyn FnMut(&ValidationProgress),
    ) -> Result<Self> {
        if !(val.val_fraction > 0.0 && val.val_fraction < 1.0) {
            return Err(InferError::DimMismatch(
                "val_fraction must be in the open interval (0, 1)".into(),
            ));
        }
        let threads = if threads == 0 {
            crate::parallel::num_threads()
        } else {
            threads
        };

        // Split the corpus by character: head = train, tail = validation.
        let chars: Vec<char> = corpus.chars().collect();
        let total = chars.len();
        let val_chars = ((total as f32) * val.val_fraction).round() as usize;
        let val_chars = val_chars.clamp(1, total.saturating_sub(1));
        let split = total - val_chars;
        let train_text: String = chars[..split].iter().collect();
        let val_text: String = chars[split..].iter().collect();

        let tc = tokenize_for_lm_cfg(&train_text, cfg)?;
        vprintln!(
            "[slm::train_validated] train={} chars, val={} chars, vocab={}, patience={}",
            train_text.chars().count(),
            val_text.chars().count(),
            tc.vocab_size,
            val.patience
        );

        let mut source = tc.tokens.as_slice();
        Self::validated_loop(
            &tc,
            &mut source,
            cfg,
            threads,
            val,
            rng,
            &|c| c.evaluate(&val_text),
            progress_callback,
        )
    }

    /// The epoch loop shared by the validated trainers (text and token file): trains over
    /// `source`, scores each epoch's model with `eval` on the held-out data, keeps the best
    /// and stops early after `val.patience` epochs without improvement.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn validated_loop<S: TokenSource>(
        tc: &LmTokens,
        source: &mut S,
        cfg: &TransformerConfig,
        threads: usize,
        val: &ValidationConfig,
        rng: &mut Rng,
        eval: &dyn Fn(&Self) -> Result<Evaluation>,
        progress_callback: &mut dyn FnMut(&ValidationProgress),
    ) -> Result<Self> {
        let mut net = prepare_net(cfg, tc.vocab_size, rng)?;
        let mut best: Option<Self> = None;
        let mut best_ce = f32::INFINITY;
        let mut stale = 0usize;
        // A resumed model is the incumbent: later epochs must beat its score.
        if net.step_count() > 0 {
            let candidate = Self::build_transformer_slm(&net, tc)?;
            best_ce = eval(&candidate)?.cross_entropy;
            best = Some(candidate);
        }

        run_training(
            &mut net,
            source,
            cfg,
            threads,
            rng,
            &mut |net, ep, train_loss| {
                // Score the in-progress model on the held-out split.
                let candidate = Self::build_transformer_slm(net, tc)?;
                let ev = eval(&candidate)?;

                let is_best = ev.cross_entropy < best_ce;
                if is_best {
                    best_ce = ev.cross_entropy;
                    best = Some(candidate);
                    stale = 0;
                } else {
                    stale += 1;
                }

                progress_callback(&ValidationProgress {
                    epoch: ep,
                    train_loss,
                    val: ev,
                    is_best,
                });

                if val.patience > 0 && stale >= val.patience {
                    vprintln!("[slm::train_validated] early stop at epoch {ep} (no val improvement for {} epochs)", stale);
                    return Ok(false);
                }
                Ok(true)
            },
        )?;

        // `best` is set unless zero epochs ran: epoch 1 improves on +inf.
        best.ok_or_else(|| InferError::DimMismatch("training ran zero epochs".into()))
    }
}

/// [`TransformerConfig`] for the positional (legacy) training API: stride 1,
/// so those entry points keep their original per-epoch semantics.
#[allow(clippy::too_many_arguments)]
fn legacy_config(
    context_len: usize,
    embed_dim: usize,
    num_heads: usize,
    num_blocks: usize,
    hidden_dim: usize,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    vocab_size: usize,
) -> TransformerConfig {
    TransformerConfig {
        context_len,
        embed_dim,
        num_heads,
        num_blocks,
        hidden_dim,
        epochs,
        lr,
        batch_size,
        vocab_size,
        window_stride: 1,
        ..TransformerConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use crate::rng::Rng;
    use crate::slm::config::{TransformerConfig, ValidationConfig};
    use crate::slm::corpus::{tokenize_for_lm, tokenize_for_lm_cfg};
    use crate::slm::generate::StopReason;
    use crate::slm::sampling::SamplingParams;
    use crate::slm::test_support::{ignore_progress, ignore_validation, t5_cfg, tiny_bpe_slm};
    use crate::slm::GenerativeSLM;
    use crate::tokenizer::ByteBpeTokenizer;

    /// The returned model is the best-by-validation checkpoint, not the last
    /// epoch's. Re-scoring the returned model on the reconstructed validation
    /// split must reproduce the lowest cross-entropy seen during training.
    #[test]
    fn validated_training_returns_best_checkpoint() {
        let corpus: String = "abcd".repeat(80); // 320 chars, periodic
        let frac = 0.2f32;
        let vcfg = ValidationConfig {
            val_fraction: frac,
            patience: 0,
        };
        let mut evals: Vec<(usize, f32, bool)> = Vec::new();
        let mut rng = Rng::new(7);
        let slm = GenerativeSLM::train_transformer_config_validated(
            &corpus,
            &t5_cfg(12),
            1,
            &vcfg,
            &mut rng,
            |p| evals.push((p.epoch, p.val.cross_entropy, p.is_best)),
        )
        .unwrap();

        assert_eq!(evals.len(), 12, "patience=0 runs every epoch");
        assert!(
            evals.iter().any(|&(_, _, best)| best),
            "no epoch was ever best"
        );
        let min_ce = evals
            .iter()
            .map(|&(_, ce, _)| ce)
            .fold(f32::INFINITY, f32::min);

        // Reconstruct the validation split exactly as the trainer does.
        let chars: Vec<char> = corpus.chars().collect();
        let total = chars.len();
        let val_chars = ((total as f32) * frac).round() as usize;
        let val_chars = val_chars.clamp(1, total - 1);
        let val_text: String = chars[total - val_chars..].iter().collect();

        let returned_ce = slm.evaluate(&val_text).unwrap().cross_entropy;
        assert!(
            (returned_ce - min_ce).abs() < 1e-4,
            "returned model CE {returned_ce} is not the best {min_ce}"
        );
    }

    /// Early stopping halts training once validation stops improving for
    /// `patience` epochs (here the periodic corpus's val loss converges), so the
    /// run ends well before the generous epoch budget.
    #[test]
    fn validated_training_early_stops_on_plateau() {
        let corpus: String = "abcd".repeat(80);
        let vcfg = ValidationConfig {
            val_fraction: 0.2,
            patience: 3,
        };
        let mut count = 0usize;
        let mut rng = Rng::new(3);
        let _ = GenerativeSLM::train_transformer_config_validated(
            &corpus,
            &t5_cfg(3000),
            1,
            &vcfg,
            &mut rng,
            |_| count += 1,
        )
        .unwrap();
        assert!(
            count < 3000,
            "early stopping never triggered (ran {count} epochs)"
        );
        assert!(
            count >= 4,
            "must run at least patience+1 epochs (ran {count})"
        );
    }

    #[test]
    fn validated_training_rejects_bad_fraction() {
        let mut rng = Rng::new(1);
        // The last fraction is valid: it makes the run reach the callback.
        for f in [0.0f32, 1.0, -0.1, 1.5, 0.25] {
            let vcfg = ValidationConfig {
                val_fraction: f,
                patience: 0,
            };
            let r = GenerativeSLM::train_transformer_config_validated(
                &"abcd".repeat(20),
                &t5_cfg(2),
                1,
                &vcfg,
                &mut rng,
                ignore_validation,
            );
            assert_eq!(r.is_err(), f != 0.25, "val_fraction {f}");
        }
    }

    #[test]
    fn validated_training_refuses_an_absurd_batch() {
        // Regression shape of the non-validated test: ctx^2 attention scores cannot fit.
        let cfg = TransformerConfig {
            context_len: 100_000,
            ..t5_cfg(1)
        };
        let e = GenerativeSLM::train_transformer_config_validated(
            &"abcde fghij ".repeat(12_000),
            &cfg,
            1,
            &ValidationConfig::default(),
            &mut Rng::new(1),
            ignore_validation,
        )
        .err()
        .expect("must refuse");
        assert!(e.to_string().contains("training would need"), "{e}");
    }

    #[test]
    fn training_on_docs_registers_eos_and_bos_uses_it() {
        let corpus = "the cat sat.<|endoftext|>the dog ran.<|endoftext|>".repeat(30);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 8,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 16,
            epochs: 1,
            vocab_size: 280,
            ..TransformerConfig::default()
        };
        let slm =
            GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(2), |_, _| {})
                .unwrap();
        let tok = ByteBpeTokenizer::from_state(&slm.meta.tokenizer_state).unwrap();
        assert_eq!(slm.bos_id(), tok.eos_id().unwrap());
        assert_eq!(tok.vocab_size(), slm.meta.output_dim);
        // Separators in text are scored as the EOS token, not as literal bytes.
        let doc = slm.encode_text("the cat sat.<|endoftext|>").unwrap();
        assert_eq!(doc.last(), tok.eos_id().as_ref());
        assert_eq!(doc.len(), tok.encode("the cat sat.").len() + 1);
    }

    #[test]
    fn legacy_positional_trainer_also_uses_doc_aware_tokenization() {
        // tokenize_for_lm (positional train_transformer API) path.
        let corpus = "the cat sat.<|endoftext|>the dog ran.<|endoftext|>".repeat(30);
        let tc = tokenize_for_lm(&corpus, 8, 280).unwrap();
        let tok = ByteBpeTokenizer::from_state(&tc.tokenizer_state).unwrap();
        let eos = tok.eos_id().unwrap();
        assert_eq!(tc.tokens.iter().filter(|&&t| t as usize == eos).count(), 60);
        // Char-level corpora are untouched by document handling.
        let c = tokenize_for_lm("abcabcabcabc", 4, 0).unwrap();
        assert!(c.tokenizer_state.is_empty());
    }

    #[test]
    fn tokenizer_path_is_created_then_reused_and_short_corpus_errors() {
        let dir = std::env::temp_dir().join(format!("ferrum_tokpath_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("tok.gguf");
        let corpus = "the cat sat.<|endoftext|>the dog ran.<|endoftext|>".repeat(30);
        let cfg = TransformerConfig {
            context_len: 8,
            vocab_size: 280,
            tokenizer_path: path.to_str().unwrap().into(),
            ..TransformerConfig::default()
        };
        let first = tokenize_for_lm_cfg(&corpus, &cfg).unwrap();
        assert!(path.exists(), "tokenizer file created (with parent dirs)");
        // Second call reuses the saved tokenizer: identical result.
        let again = tokenize_for_lm_cfg(&corpus, &cfg).unwrap();
        assert_eq!(first.tokens, again.tokens);
        assert_eq!(first.tokenizer_state, again.tokenizer_state);
        // The saved file is a GGUF container, not the old text form.
        assert_eq!(&std::fs::read(&path).unwrap()[..4], b"GGUF");
        // An old text-form tokenizer file is a clear error, not silent retraining.
        let old = dir.join("old.gguf");
        std::fs::write(&old, "256,257;258,259").unwrap();
        let old_cfg = TransformerConfig {
            tokenizer_path: old.to_str().unwrap().into(),
            ..cfg.clone()
        };
        let err = tokenize_for_lm_cfg(&corpus, &old_cfg)
            .err()
            .expect("old text tokenizer must be rejected")
            .to_string();
        assert!(err.contains("old tokenizer file (pre-GGUF)"), "{err}");
        // The old file is left untouched.
        assert_eq!(std::fs::read(&old).unwrap(), b"256,257;258,259");
        // A bare file name has no parent component to create.
        let bare_cfg = TransformerConfig {
            tokenizer_path: dir.join("bare.gguf").to_str().unwrap().into(),
            ..cfg.clone()
        };
        assert!(tokenize_for_lm_cfg(&corpus, &bare_cfg).is_ok());
        // Corpus no longer than the context window is rejected.
        let tiny = TransformerConfig {
            context_len: 10_000,
            ..cfg.clone()
        };
        let err = tokenize_for_lm_cfg(&corpus, &tiny)
            .err()
            .expect("short corpus must be rejected")
            .to_string();
        assert!(err.contains("longer than the context window"), "{err}");
        // Char-level / no path falls through to tokenize_for_lm.
        let no_path = TransformerConfig {
            tokenizer_path: String::new(),
            ..cfg
        };
        assert!(!tokenize_for_lm_cfg(&corpus, &no_path)
            .unwrap()
            .tokenizer_state
            .is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn absurd_context_or_batch_is_refused_before_training_allocates() {
        // Regression: --context 100000 asked the allocator for 160 GB (ctx^2 attention
        // scores) and aborted; a huge --batch likewise.
        let corpus = "abcde fghij ".repeat(12_000); // 144k chars
        let huge_ctx = TransformerConfig {
            context_len: 100_000,
            embed_dim: 8,
            num_heads: 1,
            num_blocks: 1,
            hidden_dim: 8,
            epochs: 1,
            vocab_size: 0,
            ..TransformerConfig::default()
        };
        let e = GenerativeSLM::train_transformer_config(
            &corpus,
            &huge_ctx,
            &mut Rng::new(1),
            ignore_progress,
        )
        .err()
        .expect("must refuse");
        assert!(e.to_string().contains("training would need"), "{e}");
        let huge_batch = TransformerConfig {
            context_len: 512,
            embed_dim: 64,
            num_heads: 4,
            num_blocks: 4,
            hidden_dim: 256,
            batch_size: usize::MAX,
            epochs: 1,
            vocab_size: 0,
            ..TransformerConfig::default()
        };
        let e = GenerativeSLM::train_transformer_config(
            &corpus,
            &huge_batch,
            &mut Rng::new(1),
            ignore_progress,
        )
        .err()
        .expect("must refuse");
        assert!(e.to_string().contains("training would need"), "{e}");
    }

    #[test]
    fn absurd_char_budgets_do_not_overflow() {
        // Regression: `num_chars * 8 + 64` wrapped (release) or panicked (debug) near usize::MAX.
        let slm = tiny_bpe_slm();
        let p = SamplingParams::with_temperature(0.0);
        // A stop string greedy decoding is known to produce ends each run early.
        let free = slm
            .generate_tokens("the cat", 40, &[], &p, &mut Rng::new(1))
            .unwrap();
        let stop: String = free.text.chars().skip(10).take(2).collect();
        assert_eq!(stop.chars().count(), 2, "{free:?}");
        for budget in [usize::MAX, usize::MAX / 8, usize::MAX / 8 + 1] {
            let out = slm
                .generate_until("the cat", budget, &stop, &p, &mut Rng::new(1))
                .unwrap();
            assert!(out.starts_with("the cat") && out.len() < 100, "{out:?}");
        }
        let g = slm
            .generate_tokens("the cat", usize::MAX, &[stop], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Stop);
    }

    #[test]
    fn threads_zero_means_auto_and_matches_an_explicit_count() {
        let corpus = "the cat sat on the mat. ".repeat(30);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 16,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 32,
            epochs: 1,
            batch_size: 4,
            ..TransformerConfig::default()
        };
        let auto = GenerativeSLM::train_transformer_config_threaded(
            &corpus,
            &cfg,
            0,
            &mut Rng::new(5),
            |_, _| {},
        )
        .unwrap();
        let explicit = GenerativeSLM::train_transformer_config_threaded(
            &corpus,
            &cfg,
            crate::parallel::num_threads(),
            &mut Rng::new(5),
            |_, _| {},
        )
        .unwrap();
        assert_eq!(auto.to_bytes().unwrap(), explicit.to_bytes().unwrap());
    }

    #[test]
    fn positional_threaded_trainer_treats_zero_threads_as_auto() {
        let corpus = "the cat sat on the mat. ".repeat(30);
        let train = |threads: usize| {
            GenerativeSLM::train_transformer_threaded_with_callback(
                &corpus,
                8,
                16,
                2,
                1,
                32,
                1,
                0.01,
                4,
                0,
                threads,
                &mut Rng::new(5),
                |_, _| {},
            )
            .unwrap()
            .to_bytes()
            .unwrap()
        };
        assert_eq!(train(0), train(crate::parallel::num_threads()));
    }
}
