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
        })
    }

    /// Advance `stream` by one token and return the text that became safe to show (possibly
    /// empty: a partial UTF-8 character or a possible stop-string prefix is held back), or `None`
    /// when the generation is over. The fragments concatenate to exactly the final text, and stop
    /// text is never returned. `rng` must be the same generator on every call.
    pub fn stream_next(&self, stream: &mut GenStream, rng: &mut Rng) -> Result<Option<String>> {
        let st = stream;
        if st.done {
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
                    let parts = st.cache.take().ok_or_else(|| {
                        InferError::DimMismatch("generation state lost its cache".into())
                    })?;
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
    use crate::slm::test_support::{tiny_bpe_slm, tiny_char_slm, tiny_eos_slm, tiny_utf8_slm};
    use crate::slm::GenerativeSLM;

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
