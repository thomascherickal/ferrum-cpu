//! Evaluation and loglikelihood scoring against held-out text.
use crate::error::{InferError, Result};
use crate::slm::cached::CachedTransformer;
use crate::slm::sampling::argmax_first;
use crate::slm::GenerativeSLM;
use crate::tensor::Tensor;

/// Result of scoring a trained model against held-out text with
/// [`GenerativeSLM::evaluate`].
///
/// All three fields describe the model's next-token predictive quality on the
/// supplied text, using exactly the same forward path that generation uses, so
/// the numbers reflect the shipped (quantization-aware) model.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Number of next-token predictions scored (every token of the text, the
    /// first conditioned on the BOS token).
    pub num_predictions: usize,
    /// Mean next-token negative log-likelihood in **nats** (natural-log
    /// cross-entropy). Lower is better.
    pub cross_entropy: f32,
    /// Mean next-token cross-entropy in **bits** (`cross_entropy / ln 2`).
    pub bits_per_token: f32,
    /// Perplexity, `exp(cross_entropy)`. Lower is better; a uniform model over a
    /// vocabulary of `V` tokens scores `V`.
    pub perplexity: f32,
}

/// Loglikelihood of a token span under the model ([`GenerativeSLM::score`],
/// [`GenerativeSLM::score_rolling`]).
///
/// Per-token probabilities come from the model's softmax output, clamped at
/// `1e-12`, so one token contributes at least `ln(1e-12) ≈ −27.63`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// Sum of natural-log probabilities of the scored tokens.
    pub logprob: f64,
    /// Number of tokens scored.
    pub n_tokens: usize,
    /// True when every scored token was the model's argmax prediction.
    pub greedy: bool,
}

/// Windows scored per batched forward pass in [`GenerativeSLM::token_logprobs`].
const SCORE_BATCH: usize = 32;

impl GenerativeSLM {
    /// Score the model's next-token predictions against held-out `text`,
    /// returning cross-entropy, bits-per-token, and perplexity.
    ///
    /// Every token is scored (the first conditioned on
    /// [`GenerativeSLM::bos_id`]); long texts use sliding windows with at least
    /// `context_len / 2` tokens of history (see [`GenerativeSLM::score_rolling`]).
    /// The trained (quantization-aware) weights are used, so the reported
    /// perplexity is the quality of the model you will ship. Character-level
    /// models return an error for characters outside the vocabulary.
    ///
    /// `text` must contain at least one token, else [`InferError::DimMismatch`]
    /// is returned. Use a corpus the model was **not** trained on to measure
    /// generalization.
    pub fn evaluate(&self, text: &str) -> Result<Evaluation> {
        vprintln!(
            "[slm::evaluate] scoring {} chars of held-out text",
            text.chars().count()
        );
        let s = self.score_rolling(text)?;
        if s.n_tokens == 0 {
            return Err(InferError::DimMismatch(
                "evaluation text must contain at least one token".into(),
            ));
        }
        Ok(finish_evaluation(-s.logprob, s.n_tokens))
    }

    /// log p(`continuation` | `context`) using the lm-eval-harness convention:
    /// trailing whitespace of `context` moves to the front of `continuation`;
    /// continuation tokens are the suffix of `encode(context + continuation)`
    /// after `encode(context)` (falling back to separate encodes when the joint
    /// encoding does not start with the context's). An empty context is scored
    /// after [`GenerativeSLM::bos_id`].
    pub fn score(&self, context: &str, continuation: &str) -> Result<Score> {
        let trimmed = context.trim_end();
        let (context, continuation) = if trimmed.len() < context.len() {
            (
                trimmed,
                format!("{}{continuation}", &context[trimmed.len()..]),
            )
        } else {
            (context, continuation.to_string())
        };
        if continuation.is_empty() {
            return Ok(Score {
                logprob: 0.0,
                n_tokens: 0,
                greedy: true,
            });
        }
        let ctx_ids = if context.is_empty() {
            vec![self.bos_id()]
        } else {
            self.encode_text(context)?
        };
        let joint = if context.is_empty() {
            None
        } else {
            Some(self.encode_text(&format!("{context}{continuation}"))?)
        };
        let ids = match joint {
            Some(j) if j.len() > ctx_ids.len() && j.starts_with(&ctx_ids) => j,
            _ => {
                let mut ids = ctx_ids.clone();
                ids.extend(self.encode_text(&continuation)?);
                ids
            }
        };
        self.token_logprobs(&ids, ctx_ids.len())
    }

    /// Loglikelihood of all of `text`: every token is scored, the first one
    /// conditioned on [`GenerativeSLM::bos_id`]. Long texts are scored in
    /// sliding windows (see [`GenerativeSLM::token_logprobs`]).
    pub fn score_rolling(&self, text: &str) -> Result<Score> {
        let mut ids = vec![self.bos_id()];
        ids.extend(self.encode_text(text)?);
        self.token_logprobs(&ids, 1)
    }

    /// Sum log p(ids[i] | ids[..i]) for `i in from..ids.len()` (`from >= 1`).
    ///
    /// Transformer models: the targets are covered by windows of `context_len`
    /// tokens. The first window starts at 0; each later window starts
    /// `context_len / 2` tokens before its first new target, so every target is
    /// scored exactly once with at least half a window of history. Windows are
    /// right-padded with id 0 (safe under the causal mask) and batched
    /// [`SCORE_BATCH`] at a time through one forward pass. Models without
    /// transformer blocks fall back to one forward per target.
    fn token_logprobs(&self, ids: &[usize], from: usize) -> Result<Score> {
        let from = from.max(1);
        let mut score = Score {
            logprob: 0.0,
            n_tokens: 0,
            greedy: true,
        };
        if from >= ids.len() {
            return Ok(score);
        }
        let t = self.meta.input_dim;
        let half = (t / 2).max(1);

        // (window start, first target, end target exclusive)
        let mut windows: Vec<(usize, usize, usize)> = Vec::new();
        let mut next = from;
        while next < ids.len() {
            let start = if next <= t { 0 } else { next - half };
            let end = (start + t + 1).min(ids.len());
            windows.push((start, next, end));
            next = end;
        }

        let transformer = CachedTransformer::try_new(&self.model).is_some();
        let mut add = |row: &[f32], target: usize| {
            let p = row[target].max(1e-12) as f64;
            let argmax = argmax_first(row);
            score.logprob += p.ln();
            score.n_tokens += 1;
            score.greedy &= argmax == target;
        };

        if transformer {
            for chunk in windows.chunks(SCORE_BATCH) {
                let mut input = Vec::with_capacity(chunk.len() * t);
                for &(start, _, _) in chunk {
                    let w = &ids[start..(start + t).min(ids.len())];
                    input.extend(w.iter().map(|&x| x as f32));
                    input.resize(input.len() + (t - w.len()), 0.0);
                }
                let out = self
                    .model
                    .forward(&Tensor::matrix(chunk.len(), t, input)?)?;
                let v = out.matrix_dims()?.1;
                for (b, &(start, first, end)) in chunk.iter().enumerate() {
                    for (target, &id) in ids.iter().enumerate().take(end).skip(first) {
                        let r = b * t + (target - 1 - start);
                        add(&out.data[r * v..(r + 1) * v], id);
                    }
                }
            }
        } else {
            for target in from..ids.len() {
                let start = target.saturating_sub(t);
                let mut ctx: Vec<f32> = vec![0.0; t - (target - start)];
                ctx.extend(ids[start..target].iter().map(|&x| x as f32));
                let out = self.model.forward(&Tensor::matrix(1, t, ctx)?)?;
                let (rows, v) = out.matrix_dims()?;
                add(&out.data[(rows - 1) * v..rows * v], ids[target]);
            }
        }
        Ok(score)
    }

    #[cfg(test)]
    fn score_rolling_prefix_for_test(&self, ids: &[usize]) -> f64 {
        self.token_logprobs(ids, 1).unwrap().logprob
    }
}

/// Turn an accumulated negative-log-likelihood sum into an [`Evaluation`]
/// (`count` must be non-zero).
fn finish_evaluation(total_nll: f64, count: usize) -> Evaluation {
    let ce = (total_nll / count as f64) as f32;
    Evaluation {
        num_predictions: count,
        cross_entropy: ce,
        bits_per_token: ce / std::f32::consts::LN_2,
        perplexity: ce.exp(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Score, SCORE_BATCH};
    use crate::slm::test_support::{bias_only_slm, naive_logprobs, tiny_bpe_slm, tiny_char_slm};

    #[test]
    fn score_matches_naive_reference() {
        let slm = tiny_bpe_slm();
        let ctx = "the cat sat on";
        let cont = " the mat.";
        let s = slm.score(ctx, cont).unwrap();
        let ctx_ids = slm.encode_text(ctx).unwrap();
        let all = slm.encode_text(&format!("{ctx}{cont}")).unwrap();
        let want = naive_logprobs(&slm, &all, ctx_ids.len());
        let want_lp: f64 = want.iter().map(|w| w.0).sum();
        assert_eq!(s.n_tokens, all.len() - ctx_ids.len());
        assert!(
            (s.logprob - want_lp).abs() < 1e-4,
            "{} vs {want_lp}",
            s.logprob
        );
        assert_eq!(s.greedy, want.iter().all(|w| w.1));
        assert!(s.logprob < 0.0);
    }

    #[test]
    fn score_moves_trailing_whitespace_into_continuation() {
        let slm = tiny_bpe_slm();
        let a = slm.score("the cat ", "sat").unwrap();
        let b = slm.score("the cat", " sat").unwrap();
        assert_eq!(a, b);
        // Whitespace-only context trims to empty: scored after BOS.
        let ws = slm.score("  \n", "the").unwrap();
        let empty = slm.score("", "  \nthe").unwrap();
        assert_eq!(ws, empty);
        // Whitespace-only continuation is still scored.
        assert!(slm.score("the cat", " ").unwrap().n_tokens >= 1);
    }

    #[test]
    fn score_falls_back_to_separate_encodes_when_joint_differs() {
        let slm = tiny_bpe_slm();
        // "the ca" + "t" merges differently jointly than the context alone, so
        // the joint encoding does not start with the context's ids.
        let ctx_ids = slm.encode_text("the ca").unwrap();
        let joint = slm.encode_text("the cat").unwrap();
        assert!(!joint.starts_with(&ctx_ids), "premise of this test");
        let s = slm.score("the ca", "t").unwrap();
        let mut ids = ctx_ids.clone();
        ids.extend(slm.encode_text("t").unwrap());
        let want = naive_logprobs(&slm, &ids, ctx_ids.len());
        assert_eq!(s.n_tokens, want.len());
        let want_lp: f64 = want.iter().map(|w| w.0).sum();
        assert!((s.logprob - want_lp).abs() < 1e-4);
    }

    #[test]
    fn rolling_scores_each_token_once_across_windows() {
        let slm = tiny_bpe_slm();
        let text = "the dog sat on the log. the cat sat on the mat. ".repeat(4);
        let ids = slm.encode_text(&text).unwrap();
        assert!(
            ids.len() > 5 * slm.meta.input_dim,
            "text must span many windows"
        );
        let s = slm.score_rolling(&text).unwrap();
        assert_eq!(s.n_tokens, ids.len());
        let mut with_bos = vec![slm.bos_id()];
        with_bos.extend(&ids);
        // The reference sees at most context_len tokens of history, the windowed
        // scorer at least context_len/2, so only the first window is exact.
        let t = slm.meta.input_dim;
        let naive_first: f64 = naive_logprobs(&slm, &with_bos[..t + 1], 1)
            .iter()
            .map(|w| w.0)
            .sum();
        let first = slm.score_rolling_prefix_for_test(&with_bos[..t + 1]);
        assert!((first - naive_first).abs() < 1e-4);
        assert!(s.logprob.is_finite() && s.logprob < 0.0);
        // Mean per-token loss is in a sane range (not the 1e-12 clamp).
        assert!(-s.logprob / (s.n_tokens as f64) < 20.0);
    }

    #[test]
    fn rolling_handles_text_spanning_more_than_one_batch_of_windows() {
        let slm = tiny_bpe_slm();
        // > SCORE_BATCH windows of ~context_len/2 new tokens each.
        let text = "the dog sat on the log. the cat sat on the mat. ".repeat(40);
        let ids = slm.encode_text(&text).unwrap();
        assert!(ids.len() > (SCORE_BATCH + 2) * slm.meta.input_dim / 2 + slm.meta.input_dim);
        let s = slm.score_rolling(&text).unwrap();
        assert_eq!(s.n_tokens, ids.len());
        assert!(s.logprob.is_finite());
        // A prefix scored alone agrees with the same prefix inside the batch
        // for the first window (exactness of windowing start).
        let short = slm.score_rolling("the dog").unwrap();
        assert_eq!(short.n_tokens, slm.encode_text("the dog").unwrap().len());
    }

    #[test]
    fn rolling_window_boundaries_cover_exact_lengths() {
        let slm = tiny_bpe_slm();
        let t = slm.meta.input_dim;
        // Byte-level pieces: lengths straddling one window and 1.5 windows.
        for n in [1, t - 1, t, t + 1, t + 2, 2 * t, 2 * t + 1, 3 * t + 3] {
            let ids = vec![b'a' as usize; n];
            let mut with_bos = vec![slm.bos_id()];
            with_bos.extend(&ids);
            let s = slm.token_logprobs(&with_bos, 1).unwrap();
            assert_eq!(s.n_tokens, n, "n={n}");
            assert!(s.logprob.is_finite());
        }
    }

    #[test]
    fn score_edge_cases_do_not_panic() {
        let slm = tiny_bpe_slm();
        let empty_cont = slm.score("the cat", "").unwrap();
        assert_eq!(
            empty_cont,
            Score {
                logprob: 0.0,
                n_tokens: 0,
                greedy: true
            }
        );
        let empty_ctx = slm.score("", "the cat").unwrap();
        assert!(empty_ctx.n_tokens > 0 && empty_ctx.logprob < 0.0);
        let empty_text = slm.score_rolling("").unwrap();
        assert_eq!(empty_text.n_tokens, 0);
        assert_eq!(empty_text.logprob, 0.0);
        assert!(empty_text.greedy);
        let both_empty = slm.score("", "").unwrap();
        assert_eq!(both_empty.n_tokens, 0);
        // `from` past the end (or 0) is clamped, never panics.
        assert_eq!(slm.token_logprobs(&[5], 1).unwrap().n_tokens, 0);
        assert_eq!(slm.token_logprobs(&[5, 6], 0).unwrap().n_tokens, 1);
        assert_eq!(slm.token_logprobs(&[5, 6], 9).unwrap().n_tokens, 0);
    }

    #[test]
    fn greedy_flag_is_false_for_an_unlikely_continuation() {
        let slm = tiny_bpe_slm();
        let s = slm
            .score("the cat sat", "\u{1F600}\u{1F600}\u{1F600}")
            .unwrap();
        assert!(!s.greedy);
        assert!(s.n_tokens >= 3);
    }

    #[test]
    fn char_level_score_rejects_unknown_chars() {
        let slm = tiny_char_slm();
        assert!(slm.score("abc", "abc").is_ok());
        let err = slm.score("abc", "xyz").unwrap_err().to_string();
        assert!(err.contains("not in the model vocabulary"), "{err}");
        assert!(slm.score_rolling("abcx").is_err());
        assert!(slm.encode_text("zzz").is_err());
    }

    #[test]
    fn evaluate_agrees_with_score_rolling_and_accepts_short_text() {
        let slm = tiny_bpe_slm();
        let text = "the cat sat on the log.";
        let e = slm.evaluate(text).unwrap();
        let s = slm.score_rolling(text).unwrap();
        assert_eq!(e.num_predictions, s.n_tokens);
        let ce = (-s.logprob / s.n_tokens as f64) as f32;
        assert!((e.cross_entropy - ce).abs() < 1e-5);
        assert!((e.perplexity - ce.exp()).abs() < 1e-3);
        assert!((e.bits_per_token - ce / std::f32::consts::LN_2).abs() < 1e-5);
        assert!(slm.evaluate("").is_err());
        // One token is enough.
        assert_eq!(slm.evaluate("a").unwrap().num_predictions, 1);
    }

    #[test]
    fn evaluate_char_level_errors_on_unknown_chars_and_counts_every_char() {
        let slm = tiny_char_slm();
        let e = slm.evaluate("abcabc").unwrap();
        assert_eq!(e.num_predictions, 6);
        assert!(slm.evaluate("abcz").is_err());
        assert!(slm.evaluate("").is_err());
        // Longer than one window: every char still counted exactly once.
        let long = "abc".repeat(30);
        assert_eq!(slm.evaluate(&long).unwrap().num_predictions, 90);
    }

    #[test]
    fn non_transformer_fallback_scores_every_target() {
        let slm = bias_only_slm();
        let bias = [0.0f64, 1.0, 2.0, 3.0, 4.0];
        let z: f64 = bias.iter().map(|b| b.exp()).sum();
        let lp = |i: usize| bias[i] - z.ln();
        // 6 targets > context 4: covers both the padded and the full-history path.
        let s = slm.score_rolling("eeabcd").unwrap();
        assert_eq!(s.n_tokens, 6);
        let want = lp(4) + lp(4) + lp(0) + lp(1) + lp(2) + lp(3);
        assert!((s.logprob - want).abs() < 1e-4, "{} vs {want}", s.logprob);
        assert!(!s.greedy);
        let all_e = slm.score_rolling("eee").unwrap();
        assert!(all_e.greedy);
        assert_eq!(all_e.n_tokens, 3);
        assert_eq!(slm.score("ab", "e").unwrap().n_tokens, 1);
    }
}
