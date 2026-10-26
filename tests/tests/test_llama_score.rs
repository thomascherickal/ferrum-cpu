//! `LlamaModel::score_rolling` against a manual log-softmax over the model's own logits, on the
//! synthetic tiny llama (2 layers, dim 8, vocab 10, context 16).
mod synth;

use ferrum_core::{Gguf, LlamaModel};

fn model() -> LlamaModel {
    Gguf::parse(synth::synth_llama())
        .unwrap()
        .load_llama_prec(None)
        .unwrap()
}

/// ln p(ids[i] | ids[start..i]) for each `(i, start)`: one forward per target (the
/// obviously-correct reference).
fn manual(m: &LlamaModel, ids: &[usize], targets: &[(usize, usize)]) -> f64 {
    let v = m.cfg.vocab_size;
    targets
        .iter()
        .map(|&(i, start)| {
            let logits = m.forward_tokens(&ids[start..i]).unwrap();
            let row = &logits.data[(i - start - 1) * v..(i - start) * v];
            let max = row.iter().copied().fold(f32::MIN, f32::max) as f64;
            let z: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
            row[ids[i]] as f64 - max - z.ln()
        })
        .sum()
}

#[test]
fn one_window_equals_the_manual_sum_of_log_probs() {
    let m = model();
    let ids: Vec<usize> = (0..12).map(|i| (i * 7 + 3) % 10).collect();
    let s = m.score_rolling(&ids, m.cfg.context_len).unwrap();
    assert_eq!(s.n_tokens, 11);
    let targets: Vec<(usize, usize)> = (1..ids.len()).map(|i| (i, 0)).collect();
    let want = manual(&m, &ids, &targets);
    assert!((s.logprob - want).abs() < 1e-4, "{} vs {want}", s.logprob);
    // Finite, negative, and near the uniform baseline ln(10) per token for an untrained model.
    assert!(s.logprob.is_finite() && s.logprob < 0.0);
    let per = -s.logprob / s.n_tokens as f64;
    assert!(per > 0.5 && per < 6.0, "{per}");
}

#[test]
fn long_text_slides_windows_and_scores_each_token_once() {
    let m = model();
    let (n, w) = (50usize, 8usize);
    let ids: Vec<usize> = (0..n).map(|i| (i * i + 1) % 10).collect();
    let s = m.score_rolling(&ids, w).unwrap();
    assert_eq!(s.n_tokens, n - 1);
    // The documented windows: the first starts at 0 and feeds `w` tokens (targets 1..=w); a later
    // one starts w/2 tokens before its first new target and again feeds `w` tokens.
    let (mut targets, mut next) = (Vec::new(), 1);
    while next < n {
        let start = if next <= w { 0 } else { next - w / 2 };
        let end = (start + w + 1).min(n);
        targets.extend((next..end).map(|i| (i, start)));
        next = end;
    }
    assert_eq!(targets.len(), n - 1);
    let want = manual(&m, &ids, &targets);
    assert!((s.logprob - want).abs() < 1e-3, "{} vs {want}", s.logprob);
}

#[test]
fn bad_window_or_short_input_is_an_error() {
    let m = model();
    let ids = [1usize, 2, 3];
    assert!(m.score_rolling(&ids, 1).is_err());
    assert!(m.score_rolling(&ids, m.cfg.context_len + 1).is_err());
    assert!(m.score_rolling(&ids[..1], 8).is_err());
}
