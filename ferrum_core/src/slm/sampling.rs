//! Decoding controls and the temperature / top-k / top-p / repetition-penalty sampler.
use crate::rng::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Decoding controls for generation (I2). Beyond temperature, these add the
/// standard knobs that tame the repetition loops a temperature-only sampler
/// falls into at low temperature:
///
/// - `top_k`: keep only the `k` highest-probability tokens (`0` disables).
/// - `top_p`: nucleus sampling — keep the smallest set of tokens whose
///   cumulative probability reaches `top_p` (`1.0` disables).
/// - `repetition_penalty`: down-weight tokens already present in the recent
///   context (`1.0` disables; typical values 1.05–1.3).
///
/// [`SamplingParams::with_temperature`] reproduces the previous temperature-only
/// behaviour exactly (all other knobs disabled), so `generate(.., temp, ..)` is
/// unchanged.
///
/// `cancel` is a cooperative stop for the decode loops: set the flag (from any
/// thread) and every [`crate::GenerativeSLM`] decode loop, and
/// [`crate::llm::LlamaModel::generate_cancellable`], stops before its next token
/// and returns what it has. It is a plain flag, not a clock, so it works on
/// `wasm32-unknown-unknown`. `Clone` shares the flag, so a stream made from these
/// params still sees it. `PartialEq` ignores it (it is a control, not a setting).
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Softmax temperature; `<= 0` selects greedy argmax decoding.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Cooperative cancel flag, checked once per token (`None`: never cancelled).
    pub cancel: Option<Arc<AtomicBool>>,
}

impl PartialEq for SamplingParams {
    fn eq(&self, o: &Self) -> bool {
        // Destructured, so a new field is a compile error here until it is compared or ignored.
        let Self {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            cancel: _,
        } = self;
        (temperature, top_k, top_p, repetition_penalty)
            == (&o.temperature, &o.top_k, &o.top_p, &o.repetition_penalty)
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            cancel: None,
        }
    }
}

impl SamplingParams {
    /// Has the `cancel` flag been set?
    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Relaxed))
    }

    /// Temperature-only sampling (top-k / top-p / repetition penalty disabled) —
    /// identical to the pre-I2 sampler.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Self::default()
        }
    }
}

/// Core logits sampler (I2): repetition penalty → temperature softmax → top-k →
/// top-p (nucleus) → renormalise → sample.
///
/// `recent` lists token IDs in the current context; the repetition penalty
/// down-weights their logits (dividing positive logits / multiplying negative
/// ones by `repetition_penalty`, the standard CTRL scheme). With
/// `SamplingParams::with_temperature` (top_k=0, top_p=1, penalty=1, recent
/// ignored) this draws exactly one `rng.next_f32()` from the temperature softmax
/// — bit-identical to the pre-I2 sampler.
/// Index of the largest value, the **first** on ties (what llama.cpp's greedy
/// sampler and `torch.argmax` return); `0` for an empty slice.
pub(crate) fn argmax_first(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in xs.iter().enumerate() {
        if x.total_cmp(&xs[best]).is_gt() {
            best = i;
        }
    }
    best
}

/// Inverse-CDF draw: the first index whose running mass reaches `r`. Rounding can leave the
/// total a hair under `r`; fall back to the last index that has any mass, and never pick a
/// token that top-k / top-p zeroed out (not even for `r == 0`).
fn pick_by_mass(probs: &[f32], r: f32) -> usize {
    let mut cumsum = 0.0f32;
    let mut last = 0;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last = i;
            cumsum += p;
            if r <= cumsum {
                return i;
            }
        }
    }
    last
}

pub(crate) fn sample_with_params(
    logits: &[f32],
    params: &SamplingParams,
    recent: &[usize],
    rng: &mut Rng,
) -> usize {
    let n = logits.len();

    // 1. Repetition penalty on the raw logits of already-seen tokens.
    let mut work: Vec<f32> = logits.to_vec();
    if (params.repetition_penalty - 1.0).abs() > f32::EPSILON && params.repetition_penalty > 0.0 {
        for &tok in recent {
            if let Some(l) = work.get_mut(tok) {
                *l = if *l > 0.0 {
                    *l / params.repetition_penalty
                } else {
                    *l * params.repetition_penalty
                };
            }
        }
    }

    // Temperature <= 0 is exact argmax (greedy decoding); no RNG draw.
    if params.temperature <= 0.0 {
        return argmax_first(&work);
    }
    let t = params.temperature.max(0.01);

    // 2. Temperature softmax (numerically stable).
    let max = work.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = work.iter().map(|&l| ((l - max) / t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    // `sum <= threshold || NaN` — a degenerate/non-finite softmax mass triggers
    // the random fallback (equivalent to the previous `!(sum > 1e-10)`).
    if sum.is_nan() || sum <= 1e-10 {
        let fallback = rng.next_u64() as usize % n;
        vprintln!(
            "[slm::sample] ⚠️  Near-zero softmax sum, random fallback idx={}",
            fallback
        );
        return fallback;
    }
    for p in &mut probs {
        *p /= sum;
    }

    // 3. top-k: zero out everything below the k-th largest probability.
    if params.top_k > 0 && params.top_k < n {
        let mut sorted = probs.clone();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let threshold = sorted[params.top_k - 1];
        for p in &mut probs {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    // 4. top-p (nucleus): keep the smallest high-prob set reaching `top_p`.
    if params.top_p < 1.0 {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let mut cum = 0.0f32;
        let mut keep = vec![false; n];
        for &i in &order {
            keep[i] = true; // always keep at least the top token
            cum += probs[i];
            if cum >= params.top_p {
                break;
            }
        }
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            }
        }
    }

    // 5. Renormalise the surviving mass and sample.
    // The top token always survives and carries at least 1/n of a unit mass, so
    // `sum2` is never zero or NaN here.
    let sum2: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum2;
    }
    pick_by_mass(&probs, rng.next_f32())
}

#[cfg(test)]
mod tests {
    use super::{argmax_first, pick_by_mass, sample_with_params, SamplingParams};
    use crate::rng::Rng;
    use crate::slm::test_support::greedy;

    #[test]
    fn sampling_ignores_out_of_vocabulary_recent_tokens() {
        let logits = [0.5f32, -1.2, 2.3];
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.5,
            ..SamplingParams::default()
        };
        let a = sample_with_params(&logits, &params, &[99], &mut Rng::new(1));
        assert_eq!(
            a,
            sample_with_params(&logits, &params, &[], &mut Rng::new(1))
        );
    }

    #[test]
    fn equality_ignores_the_cancel_flag_and_clones_share_it() {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let p = SamplingParams {
            cancel: Some(flag.clone()),
            ..SamplingParams::default()
        };
        assert_eq!(p, SamplingParams::default());
        assert_ne!(p, SamplingParams::with_temperature(0.5));
        let q = p.clone();
        assert!(!q.is_cancelled() && !SamplingParams::default().is_cancelled());
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(q.is_cancelled(), "a clone sees a flag set later");
    }

    #[test]
    fn sampling_top_p_one_equals_disabled() {
        let logits = [0.5f32, -1.2, 2.3, 0.0, 1.1];
        let off = SamplingParams::default();
        let one = SamplingParams {
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let (mut a, mut b) = (Rng::new(5), Rng::new(5));
        for _ in 0..100 {
            assert_eq!(
                sample_with_params(&logits, &off, &[], &mut a),
                sample_with_params(&logits, &one, &[], &mut b)
            );
        }
    }

    #[test]
    fn temperature_only_params_match_legacy_sampler() {
        // The pre-I2 sampler: temperature softmax + one next_f32 draw. The new
        // sampler with temperature-only params must reproduce it bit-for-bit.
        let logits = [0.5f32, -1.2, 2.3, 0.0, 1.1, -0.4];
        let temp = 0.8f32;
        let legacy = |logits: &[f32], rng: &mut Rng| -> usize {
            let t = temp.max(0.01);
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut p: Vec<f32> = logits.iter().map(|&l| ((l - max) / t).exp()).collect();
            let sum: f32 = p.iter().sum();
            for v in &mut p {
                *v /= sum;
            }
            let r = rng.next_f32();
            let mut c = 0.0;
            p.iter()
                .position(|&pi| {
                    c += pi;
                    r <= c
                })
                .unwrap_or(p.len() - 1)
        };
        let params = SamplingParams::with_temperature(temp);
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..200 {
            assert_eq!(
                legacy(&logits, &mut r1),
                sample_with_params(&logits, &params, &[], &mut r2)
            );
        }
    }

    #[test]
    fn top_k_restricts_support_to_k_tokens() {
        // One clearly-dominant pair; top_k=2 must never sample outside them.
        let logits = [5.0f32, 4.0, -2.0, -3.0, -5.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..SamplingParams::default()
        };
        let mut rng = Rng::new(7);
        for _ in 0..500 {
            let idx = sample_with_params(&logits, &params, &[], &mut rng);
            assert!(
                idx == 0 || idx == 1,
                "top_k=2 sampled out-of-support idx {idx}"
            );
        }
    }

    #[test]
    fn top_p_nucleus_keeps_minimal_high_prob_set() {
        // Token 0 alone exceeds p=0.5, so nucleus sampling must always pick it.
        let logits = [10.0f32, 1.0, 0.5, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            repetition_penalty: 1.0,
            ..SamplingParams::default()
        };
        let mut rng = Rng::new(3);
        for _ in 0..500 {
            assert_eq!(sample_with_params(&logits, &params, &[], &mut rng), 0);
        }
    }

    #[test]
    fn repetition_penalty_suppresses_recent_tokens() {
        // Token 0 dominates; penalising it (present in `recent`) shifts mass to 1.
        let logits = [4.0f32, 3.0, -2.0];
        let recent = [0usize];
        let penalised = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 100.0,
            ..SamplingParams::default()
        };
        let mut rng = Rng::new(11);
        let mut ones = 0;
        for _ in 0..1000 {
            if sample_with_params(&logits, &penalised, &recent, &mut rng) == 1 {
                ones += 1;
            }
        }
        // Without the penalty token 0 wins ~73% of the time; with a heavy penalty
        // token 1 should dominate.
        assert!(
            ones > 800,
            "repetition penalty did not redirect mass: {ones}/1000 → token 1"
        );
    }

    #[test]
    fn zero_temperature_is_argmax() {
        let logits = [0.1f32, 2.0, 1.9, -1.0];
        for t in [0.0, -1.0] {
            let p = SamplingParams {
                temperature: t,
                ..SamplingParams::default()
            };
            for s in 0..20 {
                assert_eq!(sample_with_params(&logits, &p, &[], &mut Rng::new(s)), 1);
            }
        }
        // Greedy still honours the repetition penalty (applied to raw logits).
        let p = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..SamplingParams::default()
        };
        assert_eq!(sample_with_params(&logits, &p, &[1], &mut Rng::new(0)), 2);
        // Greedy draws nothing from the RNG.
        let mut r = Rng::new(3);
        let before = r.next_u64();
        let mut r2 = Rng::new(3);
        sample_with_params(&logits, &greedy(), &[], &mut r2);
        assert_eq!(r2.next_u64(), before);
    }

    #[test]
    fn argmax_first_takes_the_first_of_equal_maxima() {
        assert_eq!(argmax_first(&[]), 0);
        assert_eq!(argmax_first(&[1.0]), 0);
        assert_eq!(argmax_first(&[0.1, 2.0, 2.0, 1.0]), 1);
        assert_eq!(argmax_first(&[3.0, 3.0, 3.0]), 0);
        assert_eq!(argmax_first(&[-1.0, -0.5, -0.5]), 1);
        assert_eq!(argmax_first(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
        // Greedy decoding uses it, regardless of the RNG.
        let p = SamplingParams::with_temperature(0.0);
        assert_eq!(
            sample_with_params(&[1.0, 5.0, 5.0], &p, &[], &mut Rng::new(1)),
            1
        );
    }

    #[test]
    fn pick_by_mass_never_returns_a_zeroed_token() {
        // Probabilities summing below r (rounding): the old fallback was the last index.
        assert_eq!(pick_by_mass(&[0.5, 0.4, 0.0, 0.0], 0.95), 1);
        assert_eq!(
            pick_by_mass(&[0.0, 0.0, 0.0], 0.5),
            0,
            "no mass at all: index 0"
        );
        assert_eq!(pick_by_mass(&[0.2, 0.3, 0.5], 0.0), 0);
        assert_eq!(pick_by_mass(&[0.2, 0.3, 0.5], 0.2), 0);
        assert_eq!(pick_by_mass(&[0.2, 0.3, 0.5], 0.21), 1);
        assert_eq!(pick_by_mass(&[0.2, 0.3, 0.5], 1.0), 2);
        // r = 0 must not pick a token with no mass.
        assert_eq!(pick_by_mass(&[0.0, 1.0], 0.0), 1);
    }
}
