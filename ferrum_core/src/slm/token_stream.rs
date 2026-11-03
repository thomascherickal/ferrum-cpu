//! Resumable token-level generation: the loop behind [`GenerativeSLM::generate_tokens_with`],
//! one step at a time, so a caller that cannot block (a browser page) can pull text as it is
//! produced and stop whenever it likes.

use crate::error::{InferError, Result};
use crate::rng::Rng;
use crate::slm::cached::{CacheParts, CachedTransformer};
use crate::slm::corpus::hex_to_char;
use crate::slm::generate::{drain_utf8, earliest_stop, held_back, Generation, StopReason};
use crate::slm::sampling::{sample_with_params, SamplingParams};
use crate::slm::GenerativeSLM;
use crate::tokenizer::ByteBpeTokenizer;
use std::sync::Arc;

/// The state of one generation in progress, made by [`GenerativeSLM::token_stream`] and advanced
/// by [`GenerativeSLM::stream_next`]. It owns everything but the model and the RNG, so it can be
/// kept between calls. Drive it with the model that made it.
pub struct GenStream {
    cache: Option<CacheParts>,
    bpe: Option<Arc<ByteBpeTokenizer>>,
    eos: Option<usize>,
    params: SamplingParams,
    stops: Vec<String>,
    max_stop: usize,
    max_tokens: usize,
    cap: usize,
    keep: usize,
    window: Vec<usize>,
    dist: Vec<f32>,
    text: String,
    pending: Vec<u8>,
    n_tokens: usize,
    emitted: usize,
    reason: StopReason,
    stopped: bool,
    done: bool,
    cancelled: bool,
}

impl GenStream {
    /// Tokens generated so far (the one that completed a stop string included, EOS not).
    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }

    /// True once [`GenerativeSLM::stream_next`] has returned `None`.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// The result so far; complete once the stream is done.
    pub fn into_generation(self) -> Generation {
        Generation {
            text: self.text,
            n_tokens: self.n_tokens,
            stop_reason: self.reason,
            cancelled: self.cancelled,
        }
    }
}

impl GenerativeSLM {
    /// Start a generation with the semantics of [`GenerativeSLM::generate_tokens`] (same prompt
    /// handling, stop strings and EOS) without running it: the prompt is encoded and primed, then
    /// [`GenerativeSLM::stream_next`] produces the text.
    pub fn token_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        stops: &[String],
        params: &SamplingParams,
    ) -> Result<GenStream> {
        self.require_token_id_model()?;
        let bpe = self.bpe()?;
        let eos = bpe.as_ref().and_then(|t| t.eos_id());
        let mut cached = CachedTransformer::try_new(&self.model).ok_or_else(|| {
            InferError::DimMismatch("generate_tokens needs a transformer model".into())
        })?;
        let cap = cached.capacity().max(1);
        let keep = (cap / 2).max(1);

        let mut window = self.encode_text(prompt)?;
        if window.is_empty() {
            window.push(self.bos_id());
        }
        if window.len() > cap {
            window.drain(..window.len() - cap);
        }
        let dist = cached.prime(&window)?;
        Ok(GenStream {
            cache: Some(cached.into_parts()),
            bpe,
            eos,
            params: params.clone(),
            stops: stops.to_vec(),
            max_stop: stops.iter().map(|s| s.len()).max().unwrap_or(0),
            max_tokens,
            cap,
            keep,
            window,
            dist,
            text: String::new(),
            pending: Vec::new(),
            n_tokens: 0,
            emitted: 0,
            reason: StopReason::MaxTokens,
            stopped: false,
            done: false,
            cancelled: false,
        })
    }

    /// Advance `stream` by one token and return the text that became safe to show (possibly
    /// empty: a partial UTF-8 character or a possible stop-string prefix is held back), or `None`
    /// when the generation is over. The fragments concatenate to exactly the final text, and stop
    /// text is never returned. `rng` must be the same generator on every call. Once the params'
    /// `cancel` flag is set it returns `None` before drawing another token, and the generation is
    /// the text returned so far ([`Generation::cancelled`]). A cancel wins over the final flush:
    /// if the flag is set before the call that would return the held-back tail, the tail is
    /// dropped and the call returns `None`.
    pub fn stream_next(&self, stream: &mut GenStream, rng: &mut Rng) -> Result<Option<String>> {
        let st = stream;
        if st.done {
            return Ok(None);
        }
        if st.params.is_cancelled() {
            // Keep exactly what was returned: a held-back stop prefix or half character is dropped.
            st.text.truncate(st.emitted);
            st.pending.clear();
            (st.cancelled, st.done) = (true, true);
            return Ok(None);
        }
        if !st.stopped && st.n_tokens < st.max_tokens {
            let logits: Vec<f32> = st.dist.iter().map(|&p| p.max(1e-12).ln()).collect();
            let next = sample_with_params(&logits, &st.params, &st.window, rng);
            if Some(next) == st.eos {
                st.reason = StopReason::Eos;
                st.stopped = true;
            } else {
                st.n_tokens += 1;
                let before = st.text.len();
                match &st.bpe {
                    Some(tok) => {
                        st.pending
                            .extend_from_slice(tok.token_bytes(next).unwrap_or(&[]));
                        drain_utf8(&mut st.pending, &mut st.text);
                    }
                    None => st.text.push(hex_to_char(&self.meta.class_names[next])),
                }
                if let Some(cut) = earliest_stop(&st.text, before, st.max_stop, &st.stops) {
                    st.text.truncate(cut);
                    st.reason = StopReason::Stop;
                    st.stopped = true;
                } else {
                    let safe = st.text.len() - held_back(&st.text[st.emitted..], &st.stops);
                    let fragment = st.text[st.emitted..safe].to_string();
                    st.emitted = safe;
                    st.window.push(next);
                    // ponytail: the cache is put back before `advanced?` below, so it is always here.
                    let parts = st.cache.take().expect("cache is restored after every step");
                    let mut cached = CachedTransformer::resume(&self.model, parts);
                    let advanced = if st.window.len() > st.cap {
                        st.window.drain(..st.window.len() - st.keep);
                        cached.prime(&st.window)
                    } else {
                        cached.feed(next)
                    };
                    st.cache = Some(cached.into_parts());
                    st.dist = advanced?;
                    return Ok(Some(fragment));
                }
            }
        }
        if st.reason != StopReason::Stop && !st.pending.is_empty() {
            // The flushed tail (a dangling lead byte becomes U+FFFD) can also complete a stop.
            let before = st.text.len();
            st.text.push_str(&String::from_utf8_lossy(&st.pending));
            if let Some(cut) = earliest_stop(&st.text, before, st.max_stop, &st.stops) {
                st.text.truncate(cut);
                st.reason = StopReason::Stop;
            }
        }
        st.pending.clear();
        st.done = true;
        // Nothing is pending any more: the rest is final (and stop-free).
        Ok(Some(
            st.text.get(st.emitted..).unwrap_or_default().to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::rng::Rng;
    use crate::slm::generate::StopReason;
    use crate::slm::sampling::SamplingParams;
    use crate::slm::test_support::{
        bias_only_slm, cancellable, state_after_draws, tiny_bpe_slm, tiny_char_slm, tiny_eos_slm,
        tiny_utf8_slm,
    };
    use crate::slm::GenerativeSLM;
    use std::sync::atomic::Ordering;

    /// Pull a whole stream: the fragments and the final generation.
    fn pull(
        m: &GenerativeSLM,
        prompt: &str,
        max: usize,
        stops: &[String],
        p: &SamplingParams,
        seed: u64,
    ) -> (Vec<String>, crate::slm::Generation) {
        let mut rng = Rng::new(seed);
        let mut s = m.token_stream(prompt, max, stops, p).unwrap();
        let mut frags = Vec::new();
        while let Some(f) = m.stream_next(&mut s, &mut rng).unwrap() {
            frags.push(f);
        }
        assert!(s.is_done());
        assert_eq!(m.stream_next(&mut s, &mut rng).unwrap(), None, "stays done");
        (frags, s.into_generation())
    }

    #[test]
    fn pulled_fragments_equal_generate_tokens_for_every_model_and_budget() {
        let sampled = SamplingParams {
            temperature: 1.0,
            top_k: 6,
            top_p: 0.95,
            repetition_penalty: 1.1,
            ..SamplingParams::default()
        };
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<String>>();
        // (model, prompt, two stop-string sets besides none)
        let cases = [
            (tiny_bpe_slm(), "the c", s(&["a"]), s(&["e ", "t"])),
            (tiny_char_slm(), "ab", s(&["c"]), s(&["b ", "a"])),
            (tiny_eos_slm(), "the c", s(&["a"]), s(&["t."])),
            (
                tiny_utf8_slm(),
                "\u{e9}",
                s(&["\u{fffd}"]),
                s(&["\u{e9}\u{e9}"]),
            ),
        ];
        for (mi, (m, prompt, s1, s2)) in cases.iter().enumerate() {
            for max in [0, 1, 7, 70] {
                // 70 tokens outruns the context: the cache is re-primed on the way.
                for stop in [vec![], s1.clone(), s2.clone()] {
                    for (pi, p) in [SamplingParams::default(), sampled.clone()]
                        .iter()
                        .enumerate()
                    {
                        let want = m
                            .generate_tokens(prompt, max, &stop, p, &mut Rng::new(9))
                            .unwrap();
                        let (frags, got) = pull(m, prompt, max, &stop, p, 9);
                        assert_eq!(got, want, "model {mi} max {max} stops {stop:?} params {pi}");
                        assert_eq!(frags.concat(), want.text);
                        // Streamed text never contains a stop string.
                        assert!(stop.iter().all(|x| !frags.concat().contains(x.as_str())));
                    }
                }
            }
        }
    }

    #[test]
    fn dropping_a_stream_early_leaves_a_prefix_and_the_model_reusable() {
        let m = tiny_bpe_slm();
        let p = SamplingParams::default();
        let full = m
            .generate_tokens("the c", 30, &[], &p, &mut Rng::new(2))
            .unwrap();
        let mut rng = Rng::new(2);
        let mut s = m.token_stream("the c", 30, &[], &p).unwrap();
        let mut seen = String::new();
        for _ in 0..5 {
            seen.push_str(&m.stream_next(&mut s, &mut rng).unwrap().unwrap());
        }
        assert_eq!(s.n_tokens(), 5);
        assert!(!s.is_done());
        assert!(full.text.starts_with(&seen), "{seen:?} vs {:?}", full.text);
        drop(s);
        let again = m
            .generate_tokens("the c", 30, &[], &p, &mut Rng::new(2))
            .unwrap();
        assert_eq!(again, full);
    }

    #[test]
    fn a_model_without_transformer_blocks_cannot_stream() {
        let m = bias_only_slm();
        assert!(m
            .token_stream("a", 5, &[], &SamplingParams::default())
            .is_err());
    }

    /// The token-level loop with a flag that `on_text` sets after `k` callbacks.
    fn tokens_cancelled_after(
        m: &GenerativeSLM,
        prompt: &str,
        k: usize,
        stops: &[String],
        seed: u64,
    ) -> (crate::slm::Generation, String, u64) {
        let (p, flag) = cancellable(SamplingParams::default());
        let mut rng = Rng::new(seed);
        let (mut calls, mut streamed) = (0, String::new());
        let g = m
            .generate_tokens_with(prompt, 50, stops, &p, &mut rng, &mut |f| {
                streamed.push_str(f);
                calls += 1;
                if calls >= k {
                    flag.store(true, Ordering::Relaxed);
                }
            })
            .unwrap();
        (g, streamed, rng.state())
    }

    #[test]
    fn cancel_before_first_token_returns_seed_only() {
        for m in [tiny_bpe_slm(), tiny_char_slm()] {
            let (p, flag) = cancellable(SamplingParams::default());
            flag.store(true, Ordering::Relaxed);
            let mut rng = Rng::new(4);
            let mut calls = 0;
            let g = m
                .generate_tokens_with("ab", 20, &[], &p, &mut rng, &mut |_| calls += 1)
                .unwrap();
            assert_eq!((g.text.as_str(), g.n_tokens, g.cancelled), ("", 0, true));
            assert_eq!(g.stop_reason, StopReason::MaxTokens);
            assert_eq!(calls, 0, "nothing is streamed");
            assert_eq!(rng.state(), Rng::new(4).state(), "no RNG draw");
        }
    }

    #[test]
    fn cancel_after_k_tokens_returns_k_tokens() {
        for m in [tiny_bpe_slm(), tiny_char_slm()] {
            for k in [1, 3, 7] {
                let (g, streamed, state) = tokens_cancelled_after(&m, "ab", k, &[], 6);
                let want = m
                    .generate_tokens("ab", k, &[], &SamplingParams::default(), &mut Rng::new(6))
                    .unwrap();
                assert_eq!((g.n_tokens, g.cancelled), (k, true), "k {k}");
                assert_eq!(g.text, want.text, "k {k}");
                assert_eq!(streamed, g.text);
                assert_eq!(state, state_after_draws(6, k), "exactly k draws");
            }
        }
    }

    #[test]
    fn cancel_mid_multibyte_keeps_valid_utf8() {
        // Byte-level model: every token is one byte of a two-byte character.
        let m = tiny_utf8_slm();
        let mut dropped = false;
        for k in 1..=6 {
            let (g, streamed, _) = tokens_cancelled_after(&m, "\u{e9}", k, &[], 3);
            assert!(!g.text.contains('\u{fffd}'), "k {k}: {:?}", g.text);
            assert_eq!(streamed, g.text, "k {k}");
            assert!(g.cancelled);
            dropped |= g.text.len() < g.n_tokens;
        }
        assert!(
            dropped,
            "some cancel landed between the two bytes of a character"
        );
    }

    #[test]
    fn no_cancel_flag_is_bit_identical_to_today() {
        let sampled = SamplingParams {
            temperature: 0.9,
            top_k: 5,
            top_p: 0.9,
            repetition_penalty: 1.2,
            ..SamplingParams::default()
        };
        for m in [tiny_bpe_slm(), tiny_char_slm()] {
            for p in [SamplingParams::default(), sampled.clone()] {
                let (with_flag, _flag) = cancellable(p.clone());
                let (mut a, mut b) = (Rng::new(8), Rng::new(8));
                let want = m.generate_tokens("ab", 70, &[], &p, &mut a).unwrap();
                let got = m
                    .generate_tokens("ab", 70, &[], &with_flag, &mut b)
                    .unwrap();
                assert_eq!(got, want);
                assert!(!got.cancelled);
                assert_eq!(b.state(), a.state(), "same RNG draws");
            }
        }
    }

    #[test]
    fn a_cancel_drops_held_back_stop_text_and_ends_the_stream() {
        let m = tiny_bpe_slm();
        let stops = vec!["t\u{1}".to_string()]; // every `t` is held back, the stop never lands
        let (p, flag) = cancellable(SamplingParams::default());
        let mut rng = Rng::new(2);
        let mut s = m.token_stream("the c", 60, &stops, &p).unwrap();
        let mut frags = String::new();
        while s.text.len() == s.emitted {
            frags.push_str(&m.stream_next(&mut s, &mut rng).unwrap().unwrap());
        }
        flag.store(true, Ordering::Relaxed);
        let n = s.n_tokens();
        assert_eq!(m.stream_next(&mut s, &mut rng).unwrap(), None);
        assert!(s.is_done());
        assert_eq!(m.stream_next(&mut s, &mut rng).unwrap(), None, "stays done");
        assert_eq!(s.n_tokens(), n, "no token after the cancel");
        let g = s.into_generation();
        assert!(g.cancelled);
        assert_eq!(g.text, frags, "the fragments are the whole text");
        assert!(!g.text.ends_with('t'));
    }

    #[test]
    fn eos_ends_the_stream_with_its_reason() {
        let m = tiny_eos_slm();
        let (_, g) = pull(&m, "ab", 50, &[], &SamplingParams::default(), 1);
        let want = m
            .generate_tokens("ab", 50, &[], &SamplingParams::default(), &mut Rng::new(1))
            .unwrap();
        assert_eq!(g, want);
        assert_eq!(g.stop_reason, StopReason::Eos);
    }
}
