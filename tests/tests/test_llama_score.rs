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

#[test]
fn score_from_scores_only_the_continuation_and_reports_greedy() {
    let m = model();
    let ids: Vec<usize> = (0..12).map(|i| (i * 7 + 3) % 10).collect();
    // From 1 it is score_rolling, bit for bit.
    let (s, _) = m.score_from(&ids, 1, 8).unwrap();
    assert_eq!(s, m.score_rolling(&ids, 8).unwrap());
    // From 7 (a window of 8): targets 7..12, the first window starting at 0, then at 5.
    let (s, greedy) = m.score_from(&ids, 7, 8).unwrap();
    assert_eq!(s.n_tokens, 5);
    let want = manual(&m, &ids, &[(7, 0), (8, 0), (9, 5), (10, 5), (11, 5)]);
    assert!((s.logprob - want).abs() < 1e-4, "{} vs {want}", s.logprob);
    assert!(!greedy, "an arbitrary sequence is not the argmax path");
    // The model's own greedy continuation is.
    let v = m.cfg.vocab_size;
    let mut path = vec![3usize, 1];
    for _ in 0..4 {
        let l = m.forward_tokens(&path).unwrap();
        let row = &l.data[(path.len() - 1) * v..path.len() * v];
        let best = (0..v).fold(0, |b, i| if row[i] > row[b] { i } else { b });
        path.push(best);
    }
    assert!(m.score_from(&path, 2, m.cfg.context_len).unwrap().1);
    // Nothing to score; a bad window is still refused.
    let (s, greedy) = m.score_from(&ids, ids.len(), 8).unwrap();
    assert_eq!((s.logprob, s.n_tokens, greedy), (0.0, 0, true));
    assert!(m.score_from(&ids, 1, 1).is_err());
}
