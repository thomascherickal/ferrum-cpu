//! Rolling log-likelihood of a token sequence under an imported Llama/Qwen model
//! ([`LlamaModel::score_rolling`]); the counterpart of `GenerativeSLM::score_rolling`.
use crate::error::{InferError, Result};
use crate::llm::LlamaModel;

/// Log-likelihood of a token sequence ([`LlamaModel::score_rolling`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlamaScore {
    /// Sum of natural-log probabilities of the scored tokens.
    pub logprob: f64,
    /// Tokens scored: every token but the first.
    pub n_tokens: usize,
}

/// `(start, first target, end target exclusive)` windows covering targets `1..n`. A window feeds
/// `ids[start..min(start + window, n)]` and scores the predictions for `first..end`. The first
/// window starts at 0; each later one starts `window / 2` tokens before its first new target, so
/// every target is scored once with at least `window / 2` tokens of history.
fn plan_windows(n: usize, window: usize) -> Vec<(usize, usize, usize)> {
    let half = (window / 2).max(1);
    let mut out = Vec::new();
    let mut next = 1;
    while next < n {
        let start = if next <= window { 0 } else { next - half };
        let end = (start + window + 1).min(n);
        out.push((start, next, end));
        next = end;
    }
    out
}

/// `ln softmax(logits)[target]`, computed in f64 so large vocabularies stay accurate.
fn log_prob(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
    logits[target] as f64 - max - sum.ln()
}

impl LlamaModel {
    /// Score `ids` with sliding windows of `window` tokens (at most the model's context
    /// length). Every token except the first is scored; to score the first one too, put the
    /// model's BOS token in front, as the tokenizer's `prefix_bos` does for generation.
    ///
    /// Window semantics (identical to `GenerativeSLM::score_rolling`): the first window starts at
    /// token 0; every later window starts `window / 2` tokens before its first unscored token,
    /// so each token is scored exactly once, with at least `window / 2` tokens of history (the
    /// whole prefix while it fits). A text that fits one window is a single forward pass. Unlike
    /// the native scorer there is no probability floor: the log-softmax is exact.
    pub fn score_rolling(&self, ids: &[usize], window: usize) -> Result<LlamaScore> {
        if window < 2 || window > self.cfg.context_len {
            return Err(InferError::DimMismatch(format!(
                "window must be between 2 and the model's context length {} (got {window})",
                self.cfg.context_len
            )));
        }
        if ids.len() < 2 {
            return Err(InferError::DimMismatch(
                "scoring needs at least two tokens (one to condition on, one to predict)".into(),
            ));
        }
        let (mut logprob, mut n_tokens) = (0.0, 0);
        for (start, first, end) in plan_windows(ids.len(), window) {
            let input = &ids[start..(start + window).min(ids.len())];
            let logits = self.forward_tokens(input)?;
            let v = self.cfg.vocab_size;
            for (target, &id) in ids.iter().enumerate().take(end).skip(first) {
                let row = target - 1 - start;
                logprob += log_prob(&logits.data[row * v..(row + 1) * v], id);
                n_tokens += 1;
            }
        }
        Ok(LlamaScore { logprob, n_tokens })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_every_target_once_with_half_a_window_of_history() {
        for (n, w) in [(2, 2), (5, 8), (9, 8), (40, 8), (41, 7), (100, 16)] {
            let plan = plan_windows(n, w);
            let mut next = 1;
            for &(start, first, end) in &plan {
                assert_eq!(first, next, "n={n} w={w}: gap or overlap");
                assert!(end > first && end - start <= w + 1);
                assert!(start == 0 || first - start >= w / 2, "n={n} w={w}");
                next = end;
            }
            assert_eq!(next, n, "n={n} w={w}: all targets covered");
        }
        assert_eq!(
            plan_windows(5, 8),
            vec![(0, 1, 5)],
            "short text: one window"
        );
    }

    #[test]
    fn log_prob_is_a_normalised_log_softmax() {
        let logits = [1.0f32, 2.0, 3.0, -1.0];
        let total: f64 = (0..4).map(|t| log_prob(&logits, t).exp()).sum();
        assert!((total - 1.0).abs() < 1e-12, "{total}");
        assert!(log_prob(&[0.0, 0.0], 1) + std::f64::consts::LN_2 < 1e-12);
    }
}
