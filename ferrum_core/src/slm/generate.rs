//! Public generation entry points: text, streaming, stop strings and token-level generation.
use crate::error::Result;
use crate::rng::Rng;
use crate::slm::sampling::SamplingParams;
use crate::slm::GenerativeSLM;

/// Why [`GenerativeSLM::generate_tokens`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// `max_tokens` tokens were generated.
    MaxTokens,
    /// A stop string appeared (it is excluded from the text).
    Stop,
    /// The model emitted the EOS token.
    Eos,
}

/// Output of [`GenerativeSLM::generate_tokens`].
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    /// The continuation only (prompt excluded, stop string excluded).
    pub text: String,
    /// Tokens generated (including the one that completed a stop string; EOS
    /// itself is not counted).
    pub n_tokens: usize,
    pub stop_reason: StopReason,
    /// The `cancel` flag of [`SamplingParams`] ended the generation: `text` is
    /// what was produced (and streamed) before it, and `stop_reason` is
    /// [`StopReason::MaxTokens`].
    pub cancelled: bool,
}

impl GenerativeSLM {
    /// Autoregressively generate next-character sequence completions from a seed text.
    ///
    /// For BPE models (`meta.tokenizer_state` non-empty) generation runs over
    /// subword tokens but `num_chars` still counts **characters**: the output is
    /// `seed` followed by exactly `num_chars` newly generated characters
    /// (fewer only if generation is cut short). Character-level models keep the
    /// original per-character behaviour.
    pub fn generate(
        &self,
        seed: &str,
        num_chars: usize,
        temp: f32,
        rng: &mut Rng,
    ) -> Result<String> {
        // The full-string API is the streaming API with a no-op sink.
        self.generate_stream(seed, num_chars, temp, rng, |_| {})
    }

    /// Like [`GenerativeSLM::generate`] but with full decoding control (I2):
    /// temperature plus top-k, top-p (nucleus), and repetition penalty. See
    /// [`SamplingParams`].
    pub fn generate_with(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<String> {
        self.generate_stream_with(seed, num_chars, params, rng, |_| {})
    }

    /// Streaming counterpart of [`GenerativeSLM::generate`]: identical sampling
    /// and identical return value, but `on_text` is invoked with each newly
    /// generated **fragment** as soon as it is produced, so callers can print a
    /// completion token-by-token (a REPL, a server's SSE stream, a TUI) instead
    /// of waiting for the whole result.
    ///
    /// Guarantees:
    /// - Concatenating every fragment passed to `on_text` yields exactly the
    ///   generated continuation — i.e. the return value with the `seed` prefix
    ///   removed (equivalently, [`GenerativeSLM::generate_continuation`]'s
    ///   output).
    /// - The seed itself is never emitted.
    /// - For a fixed RNG the output and the fragment sequence are deterministic,
    ///   and the returned `String` equals what [`GenerativeSLM::generate`] would
    ///   return for the same arguments.
    ///
    /// Character-level models emit one character per step. BPE models emit
    /// decoded text as whole tokens land, holding back a partial trailing
    /// multi-byte character until it completes (so no `U+FFFD` placeholder is
    /// ever streamed and then revised).
    pub fn generate_stream<F>(
        &self,
        seed: &str,
        num_chars: usize,
        temp: f32,
        rng: &mut Rng,
        on_text: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        self.generate_stream_with(
            seed,
            num_chars,
            &SamplingParams::with_temperature(temp),
            rng,
            on_text,
        )
    }

    /// Streaming generation with full decoding control (I2) — the implementation
    /// behind [`GenerativeSLM::generate_stream`] (which passes a
    /// temperature-only [`SamplingParams`]). See [`SamplingParams`] for the
    /// top-k / top-p / repetition-penalty knobs.
    pub fn generate_stream_with<F>(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        rng: &mut Rng,
        mut on_text: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        self.generate_stream_core(seed, num_chars, params, None, rng, &mut on_text)
    }

    /// [`GenerativeSLM::generate_with`] that **stops early** at a natural
    /// boundary (I3): generation halts as soon as the `stop` string appears in
    /// the generated continuation (and the result ends right after it), or after
    /// `max_chars` characters — whichever comes first. Useful with a sentence
    /// terminator or a document separator (see the tokenizer's special tokens,
    /// K3). An empty `stop` disables early stopping.
    pub fn generate_until(
        &self,
        seed: &str,
        max_chars: usize,
        stop: &str,
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<String> {
        self.generate_stream_until(seed, max_chars, stop, params, rng, |_| {})
    }

    /// Streaming counterpart of [`GenerativeSLM::generate_until`] (I3).
    pub fn generate_stream_until<F>(
        &self,
        seed: &str,
        max_chars: usize,
        stop: &str,
        params: &SamplingParams,
        rng: &mut Rng,
        mut on_text: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let stop = if stop.is_empty() { None } else { Some(stop) };
        self.generate_stream_core(seed, max_chars, params, stop, rng, &mut on_text)
    }

    /// Generate a completion from `seed` and return **only the newly generated
    /// text**, excluding the seed prefix.
    ///
    /// [`GenerativeSLM::generate`] returns `seed` followed by the continuation
    /// (the natural shape for printing a running document); this is the
    /// ergonomic counterpart for callers that only want the model's own output
    /// — chat replies, autocomplete suggestions, code completion. The character
    /// budget (`num_chars`) and sampling semantics are identical to
    /// [`GenerativeSLM::generate`]; only the returned slice differs.
    pub fn generate_continuation(
        &self,
        seed: &str,
        num_chars: usize,
        temp: f32,
        rng: &mut Rng,
    ) -> Result<String> {
        let full = self.generate(seed, num_chars, temp, rng)?;
        // `generate` always returns the seed verbatim as a prefix, so skipping
        // the seed's character count yields exactly the continuation.
        let seed_chars = seed.chars().count();
        Ok(full.chars().skip(seed_chars).collect())
    }

    /// [`GenerativeSLM::generate_continuation`] with full decoding control (I2):
    /// returns only the newly generated text (seed prefix removed).
    pub fn generate_continuation_with(
        &self,
        seed: &str,
        num_chars: usize,
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<String> {
        let full = self.generate_with(seed, num_chars, params, rng)?;
        let seed_chars = seed.chars().count();
        Ok(full.chars().skip(seed_chars).collect())
    }

    /// Generate up to `max_tokens` tokens after `prompt`, returning only the
    /// continuation. Stops early at the first of `stops` to appear (excluded
    /// from the text) or when the model emits EOS. Any prompt length works: a
    /// prompt longer than the context keeps its most recent tokens; an empty
    /// prompt starts from [`GenerativeSLM::bos_id`].
    ///
    /// When the KV cache fills, it is re-primed from the most recent
    /// `context_len / 2` tokens, so the cost is one re-prime per
    /// `context_len / 2` generated tokens rather than one per token (learned
    /// absolute positions rule out sliding the cache by one).
    pub fn generate_tokens(
        &self,
        prompt: &str,
        max_tokens: usize,
        stops: &[String],
        params: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<Generation> {
        self.generate_tokens_with(prompt, max_tokens, stops, params, rng, &mut |_| {})
    }

    /// [`GenerativeSLM::generate_tokens`] that also hands the text to `on_text`
    /// as it is produced. The fragments concatenate to exactly the returned
    /// `text`: only the tail that could still grow into a stop string is held
    /// back until it is known not to, and stop text is never passed on.
    pub fn generate_tokens_with(
        &self,
        prompt: &str,
        max_tokens: usize,
        stops: &[String],
        params: &SamplingParams,
        rng: &mut Rng,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation> {
        let mut stream = self.token_stream(prompt, max_tokens, stops, params)?;
        while let Some(fragment) = self.stream_next(&mut stream, rng)? {
            on_text(&fragment);
        }
        Ok(stream.into_generation())
    }
}

/// Length in bytes of the longest suffix of `text` that is a proper prefix of
/// some stop string, i.e. the tail that might still turn into a stop.
pub(super) fn held_back(text: &str, stops: &[String]) -> usize {
    stops
        .iter()
        .flat_map(|s| {
            (1..s.len())
                .rev()
                .filter(|&k| s.is_char_boundary(k))
                .map(move |k| &s[..k])
        })
        .filter(|p| text.ends_with(p))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

/// Byte offset of the earliest stop string that ends inside `text[before..]`,
/// scanning only the tail (`max_stop` bytes before `before`) that could hold a
/// newly completed match. Empty stop strings never match.
pub(super) fn earliest_stop(
    text: &str,
    before: usize,
    max_stop: usize,
    stops: &[String],
) -> Option<usize> {
    let from = (0..=before.saturating_sub(max_stop))
        .rev()
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(0);
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text[from..].find(s.as_str()).map(|i| from + i))
        .min()
}

/// Move the longest valid UTF-8 prefix of `pending` onto `out`, replacing
/// invalid sequences with U+FFFD exactly like `String::from_utf8_lossy`, and
/// leave an incomplete trailing sequence in `pending` for the next token.
pub(super) fn drain_utf8(pending: &mut Vec<u8>, out: &mut String) {
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                out.push_str(s);
                pending.clear();
                return;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                out.push_str(&String::from_utf8_lossy(&pending[..valid]));
                match e.error_len() {
                    Some(bad) => {
                        out.push('\u{FFFD}');
                        pending.drain(..valid + bad);
                    }
                    None => {
                        pending.drain(..valid);
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{drain_utf8, held_back, Generation, StopReason};
    use crate::meta::TaskType;
    use crate::rng::Rng;
    use crate::slm::sampling::SamplingParams;
    use crate::slm::test_support::{
        bias_only_slm, greedy, tiny_bpe_slm, tiny_char_slm, tiny_eos_slm, tiny_utf8_slm,
    };
    use crate::slm::GenerativeSLM;

    #[test]
    fn generation_with_params_is_deterministic_and_preserves_seed() {
        // End-to-end: the *_with API runs the cached transformer path with the
        // extra knobs and stays deterministic / seed-preserving.
        let mut rng = Rng::new(7);
        let slm = GenerativeSLM::train_transformer(
            "abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc",
            4,
            8,
            2,
            1,
            16,
            30,
            0.01,
            8,
            0,
            &mut rng,
        )
        .unwrap();
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 3,
            top_p: 0.9,
            repetition_penalty: 1.2,
            ..SamplingParams::default()
        };
        let a = slm
            .generate_with("abca", 10, &params, &mut Rng::new(99))
            .unwrap();
        let b = slm
            .generate_with("abca", 10, &params, &mut Rng::new(99))
            .unwrap();
        assert_eq!(a, b, "generation must be deterministic for a fixed RNG");
        assert!(a.starts_with("abca"));
        assert_eq!(a.chars().count(), 4 + 10);
    }

    #[test]
    fn generate_until_stops_at_boundary_char_level() {
        let corpus: String = "abc".repeat(40);
        let mut rng = Rng::new(7);
        let slm =
            GenerativeSLM::train_transformer(&corpus, 4, 8, 2, 1, 16, 40, 0.01, 8, 0, &mut rng)
                .unwrap();
        let params = SamplingParams::with_temperature(0.1); // near-greedy

        let out = slm
            .generate_until("abca", 30, "c", &params, &mut Rng::new(2))
            .unwrap();
        assert!(out.starts_with("abca"));
        assert!(out.ends_with('c'), "should stop right after a 'c': {out:?}");
        assert!(out.chars().count() < 4 + 30, "did not stop early: {out:?}");

        // Deterministic for a fixed RNG.
        let a = slm
            .generate_until("abca", 30, "c", &params, &mut Rng::new(5))
            .unwrap();
        let b = slm
            .generate_until("abca", 30, "c", &params, &mut Rng::new(5))
            .unwrap();
        assert_eq!(a, b);

        // Empty stop disables early stopping → exactly the full budget.
        let full = slm
            .generate_until("abca", 12, "", &params, &mut Rng::new(5))
            .unwrap();
        assert_eq!(full.chars().count(), 4 + 12);
    }

    #[test]
    fn generate_stream_until_fragments_match_continuation() {
        let corpus: String = "abc".repeat(40);
        let mut rng = Rng::new(7);
        let slm =
            GenerativeSLM::train_transformer(&corpus, 4, 8, 2, 1, 16, 40, 0.01, 8, 0, &mut rng)
                .unwrap();
        let params = SamplingParams::with_temperature(0.1);

        let mut streamed = String::new();
        let full = slm
            .generate_stream_until("abca", 30, "c", &params, &mut Rng::new(9), |f| {
                streamed.push_str(f)
            })
            .unwrap();
        let cont: String = full.chars().skip("abca".chars().count()).collect();
        assert_eq!(
            streamed, cont,
            "streamed fragments must equal the stopped continuation"
        );
        assert!(full.ends_with('c'));
    }

    #[test]
    fn generate_until_stops_at_boundary_bpe() {
        let corpus = "the quick brown fox jumps over the lazy dog. the quick brown fox \
            jumps over the lazy dog. the quick brown fox jumps over the lazy dog. ";
        let mut rng = Rng::new(11);
        let slm =
            GenerativeSLM::train_transformer(corpus, 8, 16, 2, 1, 32, 30, 0.01, 8, 300, &mut rng)
                .unwrap();
        let params = SamplingParams::with_temperature(0.5);
        let seed = "the quick";

        let out = slm
            .generate_until(seed, 40, " ", &params, &mut Rng::new(3))
            .unwrap();
        assert!(out.starts_with(seed));
        let n = out.chars().count();
        // If it stopped before the budget, it must have stopped right after a space.
        if n < seed.chars().count() + 40 {
            assert!(out.ends_with(' '), "stopped but not at the boundary");
        }
        // Deterministic.
        let a = slm
            .generate_until(seed, 40, " ", &params, &mut Rng::new(8))
            .unwrap();
        let b = slm
            .generate_until(seed, 40, " ", &params, &mut Rng::new(8))
            .unwrap();
        assert_eq!(a, b);
    }

    /// Regression (deferred minor): an unknown seed char in char-level
    /// generation silently became token 0; it now errors naming the char.
    #[test]
    fn char_level_generate_rejects_an_unknown_seed_char() {
        let slm = tiny_char_slm();
        let p = greedy();
        for chars in [0usize, 5] {
            let err = slm
                .generate_with("abcXabc", chars, &p, &mut Rng::new(1))
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("'X'") && err.contains("not in the model vocabulary"),
                "{err}"
            );
        }
        // A valid seed still generates; the error path leaves no state behind.
        let ok = slm
            .generate_with("abcabcab", 5, &p, &mut Rng::new(1))
            .unwrap();
        assert!(ok.starts_with("abcabcab"));
    }

    #[test]
    fn generate_tokens_respects_budget_and_is_deterministic() {
        let slm = tiny_bpe_slm();
        let p = greedy();
        let a = slm
            .generate_tokens("the cat", 12, &[], &p, &mut Rng::new(1))
            .unwrap();
        let b = slm
            .generate_tokens("the cat", 12, &[], &p, &mut Rng::new(99))
            .unwrap();
        assert_eq!(a, b, "temperature 0 is argmax: the seed must not matter");
        assert_eq!(a.n_tokens, 12);
        assert_eq!(a.stop_reason, StopReason::MaxTokens);
        // Sampling at temperature 1 is deterministic for a fixed seed.
        let sp = SamplingParams::default();
        let s1 = slm
            .generate_tokens("the cat", 12, &[], &sp, &mut Rng::new(4))
            .unwrap();
        let s2 = slm
            .generate_tokens("the cat", 12, &[], &sp, &mut Rng::new(4))
            .unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn held_back_is_the_longest_proper_stop_prefix_at_the_tail() {
        let st = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(held_back("abc", &[]), 0);
        assert_eq!(held_back("abc", &st(&["xyz", ""])), 0);
        assert_eq!(held_back("ab#", &st(&["##end"])), 1);
        assert_eq!(held_back("a##e", &st(&["##end"])), 3);
        assert_eq!(
            held_back("ab##end", &st(&["##end"])),
            0,
            "a full match is cut, not held"
        );
        assert_eq!(held_back("a\u{e9}", &st(&["\u{e9}\u{e9}", "\u{e9}x"])), 2);
        assert_eq!(held_back("ab", &st(&["b!", "ab!"])), 2, "longest wins");
    }

    #[test]
    fn streamed_text_equals_the_returned_text_and_never_leaks_a_stop() {
        for (slm, prompt) in [(tiny_bpe_slm(), "the cat"), (tiny_char_slm(), "ab")] {
            let seeded = SamplingParams::default();
            for (params, seed) in [(greedy(), 1), (seeded, 4)] {
                let free = slm
                    .generate_tokens(prompt, 40, &[], &params, &mut Rng::new(seed))
                    .unwrap();
                // No stops, a stop that fires mid-way, one that straddles tokens,
                // one that never completes but whose first byte often appears.
                let mid: String = free.text.chars().skip(5).take(3).collect();
                let first: String = free.text.chars().take(1).chain("#nope".chars()).collect();
                for stops in [
                    vec![],
                    vec![mid.clone()],
                    vec![mid.clone(), "zzzz".to_string()],
                    vec![first],
                ] {
                    let want = slm
                        .generate_tokens(prompt, 40, &stops, &params, &mut Rng::new(seed))
                        .unwrap();
                    let mut frags: Vec<String> = Vec::new();
                    let got = slm
                        .generate_tokens_with(
                            prompt,
                            40,
                            &stops,
                            &params,
                            &mut Rng::new(seed),
                            &mut |t| frags.push(t.to_string()),
                        )
                        .unwrap();
                    assert_eq!(got, want, "{stops:?}");
                    assert_eq!(frags.concat(), want.text, "{stops:?}");
                    for f in &frags {
                        assert!(stops.iter().all(|s| !f.contains(s.as_str())), "{f:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn streaming_holds_back_only_a_possible_stop_prefix() {
        let slm = tiny_char_slm(); // one char per token
        let p = greedy();
        let free = slm
            .generate_tokens("ab", 12, &[], &p, &mut Rng::new(1))
            .unwrap();
        let stop: String = free.text.chars().skip(4).take(2).collect();
        let mut seen = String::new();
        let mut lag = Vec::new(); // chars produced so far minus chars shown, per call
        let mut n = 0;
        let g = slm
            .generate_tokens_with(
                "ab",
                12,
                std::slice::from_ref(&stop),
                &p,
                &mut Rng::new(1),
                &mut |t| {
                    n += 1;
                    seen.push_str(t);
                    lag.push(n - seen.chars().count() as i64);
                },
            )
            .unwrap();
        assert_eq!(seen, g.text);
        // Per token, at most the first char of the 2-char stop is withheld (the
        // last call is the final flush, which is not a token).
        lag.pop();
        assert!(lag.iter().all(|&l| l <= 1), "{lag:?}");
    }

    #[test]
    fn generate_tokens_zero_budget_returns_empty() {
        let slm = tiny_bpe_slm();
        let g = slm
            .generate_tokens("the cat", 0, &[], &greedy(), &mut Rng::new(1))
            .unwrap();
        assert_eq!(
            g,
            Generation {
                text: String::new(),
                n_tokens: 0,
                stop_reason: StopReason::MaxTokens,
                cancelled: false,
            }
        );
    }

    #[test]
    fn generate_tokens_stops_at_first_stop_and_excludes_it() {
        let slm = tiny_bpe_slm();
        let p = greedy();
        let free = slm
            .generate_tokens("the cat", 40, &[], &p, &mut Rng::new(1))
            .unwrap();
        // Pick a stop string that greedy decoding is known to produce.
        let stop = free.text.chars().skip(3).take(2).collect::<String>();
        let g = slm
            .generate_tokens(
                "the cat",
                40,
                &["#never#".into(), String::new(), stop.clone()],
                &p,
                &mut Rng::new(1),
            )
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Stop);
        assert!(
            !g.text.contains(&stop),
            "stop string must be excluded: {:?}",
            g.text
        );
        assert!(free.text.starts_with(&g.text));
        // Only empty stops: they never fire.
        let none = slm
            .generate_tokens("the cat", 10, &[String::new()], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(none.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn generate_tokens_stop_spanning_tokens_and_earliest_wins() {
        let slm = tiny_char_slm(); // one token per char
        let p = greedy();
        let free = slm
            .generate_tokens("ab", 12, &[], &p, &mut Rng::new(1))
            .unwrap();
        // One token per char, so a 2-char stop always spans two tokens.
        let early: String = free.text.chars().skip(1).take(2).collect();
        let late: String = free.text.chars().skip(6).take(2).collect();
        let at = free.text.find(&early).unwrap();
        let g = slm
            .generate_tokens("ab", 12, std::slice::from_ref(&early), &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Stop);
        assert_eq!(g.text, &free.text[..at]);
        assert_eq!(
            g.n_tokens,
            at + 2,
            "the token completing the stop is counted"
        );
        // Several stops: the earliest position in the text wins, whatever the list order.
        let cut = free
            .text
            .find(&early)
            .unwrap()
            .min(free.text.find(&late).unwrap());
        let g2 = slm
            .generate_tokens(
                "ab",
                12,
                &[late.clone(), early.clone()],
                &p,
                &mut Rng::new(1),
            )
            .unwrap();
        assert_eq!(g2.text, &free.text[..cut]);
        let g3 = slm
            .generate_tokens("ab", 12, &[early, late], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g3, g2);
    }

    #[test]
    fn generate_tokens_char_level_paths() {
        let slm = tiny_char_slm();
        let p = greedy();
        // Empty prompt starts from BOS.
        let g = slm
            .generate_tokens("", 5, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.n_tokens, 5);
        assert_eq!(g.text.chars().count(), 5);
        // Unknown prompt chars are an error, not id 0.
        assert!(slm
            .generate_tokens("xyz", 5, &[], &p, &mut Rng::new(1))
            .is_err());
    }

    #[test]
    fn generate_tokens_stops_at_eos() {
        let slm = tiny_eos_slm();
        let g = slm
            .generate_tokens("the cat", 60, &[], &greedy(), &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Eos, "{g:?}");
        assert!(g.n_tokens < 60);
        assert!(!g.text.contains("endoftext"));
    }

    #[test]
    fn generate_tokens_accepts_prompt_longer_than_context() {
        let slm = tiny_bpe_slm();
        let long = "the dog sat on the log. ".repeat(10);
        let p = greedy();
        let g = slm
            .generate_tokens(&long, 30, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.n_tokens, 30);
        let empty = slm
            .generate_tokens("", 5, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(empty.n_tokens, 5);
        // Generation beyond several cache windows re-primes without error.
        let far = slm
            .generate_tokens("the", 100, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(far.n_tokens, 100);
    }

    #[test]
    fn generate_tokens_flushes_incomplete_utf8_at_budget() {
        let slm = tiny_utf8_slm();
        let p = greedy();
        let full = slm
            .generate_tokens("\u{e9}", 4, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(full.text, "\u{e9}\u{e9}", "premise: model learned the char");
        // An odd budget ends mid-character: the dangling lead byte is flushed
        // lossily instead of being dropped.
        let odd = slm
            .generate_tokens("\u{e9}", 3, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(odd.text, "\u{e9}\u{fffd}");
        // A multi-byte stop string straddles two byte tokens.
        let stopped = slm
            .generate_tokens("\u{e9}", 8, &["\u{e9}".into()], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(stopped.stop_reason, StopReason::Stop);
        assert_eq!(stopped.text, "");
        let late = slm
            .generate_tokens("\u{e9}", 8, &["\u{e9}\u{e9}".into()], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(late.stop_reason, StopReason::Stop);
        assert_eq!(late.text, "");
    }

    /// Regression (deferred minor): a stop string completed only by the bytes
    /// flushed at the end of the budget (a dangling lead byte becomes U+FFFD)
    /// was never checked.
    #[test]
    fn generate_tokens_checks_stops_against_the_flushed_tail() {
        let slm = tiny_utf8_slm();
        let p = greedy();
        let stops = ["\u{fffd}".to_string()];
        let g = slm
            .generate_tokens("\u{e9}", 3, &stops, &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Stop);
        assert_eq!(g.text, "\u{e9}");
        assert_eq!(g.n_tokens, 3);
        // Same budget without the stop still flushes the lossy tail.
        let plain = slm
            .generate_tokens("\u{e9}", 3, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(plain.text, "\u{e9}\u{fffd}");
        assert_eq!(plain.stop_reason, StopReason::MaxTokens);
        // A non-matching stop leaves the flushed text alone.
        let other = slm
            .generate_tokens("\u{e9}", 3, &["zz".into()], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(other.text, "\u{e9}\u{fffd}");
    }

    #[test]
    fn generate_tokens_rejects_non_transformer_and_legacy_models() {
        let slm = bias_only_slm();
        let err = slm
            .generate_tokens("ab", 3, &[], &greedy(), &mut Rng::new(1))
            .unwrap_err()
            .to_string();
        assert!(err.contains("needs a transformer model"), "{err}");
        let mut legacy = tiny_char_slm();
        legacy.meta.task = TaskType::Classification;
        assert!(legacy
            .generate_tokens("ab", 3, &[], &greedy(), &mut Rng::new(1))
            .is_err());
        let mut bad = tiny_bpe_slm();
        bad.meta.tokenizer_state = "1,2,3".into();
        assert!(bad
            .generate_tokens("ab", 3, &[], &greedy(), &mut Rng::new(1))
            .is_err());
    }

    #[test]
    fn drain_utf8_handles_valid_invalid_and_incomplete() {
        let drain = |bytes: &[u8]| {
            let mut pending = bytes.to_vec();
            let mut out = String::new();
            drain_utf8(&mut pending, &mut out);
            (out, pending)
        };
        assert_eq!(drain(b""), (String::new(), vec![]));
        assert_eq!(drain(b"abc"), ("abc".into(), vec![]));
        assert_eq!(
            drain("h\u{e9}llo".as_bytes()),
            ("h\u{e9}llo".into(), vec![])
        );
        // Incomplete trailing sequence waits for the next token.
        assert_eq!(drain(&[0xC3]), (String::new(), vec![0xC3]));
        assert_eq!(drain(b"a\xE2\x82"), ("a".into(), vec![0xE2, 0x82]));
        // Invalid bytes become U+FFFD like from_utf8_lossy.
        assert_eq!(drain(&[0xFF, b'a']), ("\u{fffd}a".into(), vec![]));
        assert_eq!(
            drain(&[b'a', 0xFF, 0xFF, b'b']),
            ("a\u{fffd}\u{fffd}b".into(), vec![])
        );
        // Invalid followed by an incomplete tail: replace, keep the tail.
        assert_eq!(drain(&[0xFF, 0xC3]), ("\u{fffd}".into(), vec![0xC3]));
        // A lead byte followed by a non-continuation is invalid, the next char is kept.
        assert_eq!(drain(&[0xC3, b'a']), ("\u{fffd}a".into(), vec![]));
        // Appending: existing output is preserved.
        let mut pending = "\u{e9}".as_bytes()[1..].to_vec();
        let mut out = String::from("x");
        drain_utf8(&mut pending, &mut out);
        assert_eq!(out, "x\u{fffd}");
        // Equivalent to lossy decoding for complete input.
        let bytes = [b'a', 0xF0, 0x9F, 0x98, 0x80, 0xC0, b'z'];
        assert_eq!(drain(&bytes).0, String::from_utf8_lossy(&bytes));
    }

    #[test]
    fn generation_stops_at_eos_and_streams_the_same_text() {
        let slm = tiny_eos_slm();
        let p = greedy();
        let mut streamed = String::new();
        let out = slm
            .generate_stream_with("the cat", 200, &p, &mut Rng::new(1), |f| {
                streamed.push_str(f)
            })
            .unwrap();
        let g = slm
            .generate_tokens("the cat", 200, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.stop_reason, StopReason::Eos, "{g:?}");
        // The continuation ends at the document end (EOS), not at the budget.
        assert_eq!(out, format!("the cat{}", g.text));
        assert!(out.chars().count() < "the cat".len() + 200);
        assert_eq!(streamed, g.text);
        // With a stop string that never matches, EOS still ends generation.
        let until = slm
            .generate_until("the cat", 200, "#none#", &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(until, out);
    }

    #[test]
    fn generate_until_cuts_after_a_stop_spanning_tokens() {
        let slm = tiny_bpe_slm();
        let p = greedy();
        let free = slm
            .generate_tokens("the cat", 40, &[], &p, &mut Rng::new(1))
            .unwrap()
            .text;
        // A 4-char stop taken from the middle of the free run spans tokens.
        let stop: String = free.chars().skip(3).take(4).collect();
        let want_end = free.find(&stop).unwrap() + stop.len();
        let mut streamed = String::new();
        let out = slm
            .generate_stream_until("the cat", 40, &stop, &p, &mut Rng::new(1), |f| {
                streamed.push_str(f)
            })
            .unwrap();
        assert_eq!(out, format!("the cat{}", &free[..want_end]));
        assert_eq!(streamed, &free[..want_end]);
    }

    #[test]
    fn generate_until_honours_the_char_budget_over_a_later_stop() {
        let slm = tiny_bpe_slm();
        let p = greedy();
        let free = slm
            .generate_tokens("the cat", 40, &[], &p, &mut Rng::new(1))
            .unwrap()
            .text;
        // The stop sits past the 3-char budget: the budget wins.
        let stop: String = free.chars().skip(6).take(3).collect();
        let out = slm
            .generate_until("the cat", 3, &stop, &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(
            out,
            format!("the cat{}", free.chars().take(3).collect::<String>())
        );
    }

    /// Every generation entry point survives every odd sampling knob and seed shape.
    #[test]
    fn generation_matrix_never_panics_and_honours_budgets() {
        let nan = f32::NAN;
        let inf = f32::INFINITY;
        let temps = [0.0, 1e-9, 1.0, 50.0, inf, nan, -1.0];
        let top_ks = [0usize, 1, 3, 10_000, usize::MAX];
        let top_ps = [0.0f32, 0.5, 1.0, 2.0, -1.0, nan];
        let reps = [0.0f32, -1.0, 1.0, 1.5, nan, inf];
        for (name, slm, long_seed) in [
            ("bpe", tiny_bpe_slm(), "the dog sat on the log. ".repeat(10)),
            ("char", tiny_char_slm(), "abc abc\n".repeat(10)),
        ] {
            let seeds = ["", "a", "ab", long_seed.as_str()];
            let mut rng = Rng::new(9);
            let mut case = 0usize;
            for &t in &temps {
                for &k in &top_ks {
                    for &p in &top_ps {
                        for &r in &reps {
                            case += 1;
                            // Thin the 7*5*6*6 grid deterministically but keep every value hit.
                            if !(case.is_multiple_of(7) || k == 3 && p == 0.5 && r == 1.5) {
                                continue;
                            }
                            let params = SamplingParams {
                                temperature: t,
                                top_k: k,
                                top_p: p,
                                repetition_penalty: r,
                                ..SamplingParams::default()
                            };
                            for seed in seeds {
                                for n in [0usize, 1, 5] {
                                    let ctx = format!("{name} {params:?} seed={seed:?} n={n}");
                                    let out = slm
                                        .generate_with(seed, n, &params, &mut rng)
                                        .unwrap_or_else(|e| panic!("{ctx}: {e}"));
                                    assert!(out.starts_with(seed), "{ctx}: {out:?}");
                                    assert!(
                                        out.chars().count() <= seed.chars().count() + n,
                                        "{ctx}: {out:?}"
                                    );
                                    let g = slm
                                        .generate_tokens(
                                            seed,
                                            n,
                                            &["a".into(), "".into()],
                                            &params,
                                            &mut rng,
                                        )
                                        .unwrap_or_else(|e| panic!("{ctx}: {e}"));
                                    assert!(g.n_tokens <= n, "{ctx}: {g:?}");
                                    if n == 0 {
                                        assert_eq!(g.stop_reason, StopReason::MaxTokens, "{ctx}");
                                        assert!(g.text.is_empty(), "{ctx}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn char_level_empty_seed_generates_like_the_token_path() {
        let slm = tiny_char_slm();
        let p = SamplingParams::with_temperature(0.0);
        let streamed = std::cell::RefCell::new(String::new());
        let out = slm
            .generate_stream_with("", 12, &p, &mut Rng::new(1), |f| {
                streamed.borrow_mut().push_str(f)
            })
            .unwrap();
        assert_eq!(out.chars().count(), 12, "{out:?}");
        assert_eq!(*streamed.borrow(), out);
        // Same greedy text as the token-budgeted path, which also starts from token 0.
        let g = slm
            .generate_tokens("", 12, &[], &p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(g.text, out);
        // A zero budget returns the (empty) seed untouched.
        assert_eq!(slm.generate_with("", 0, &p, &mut Rng::new(1)).unwrap(), "");
    }

    /// A stop string may start in one token and end in a later one; whatever the alignment, the
    /// output is cut exactly at its first occurrence (both generation paths), and with several
    /// stops at whichever occurs first.
    #[test]
    fn stop_strings_cut_at_the_first_occurrence_at_every_alignment() {
        for slm in [tiny_bpe_slm(), tiny_char_slm()] {
            let p = SamplingParams::with_temperature(0.0);
            let seed = if slm.meta.tokenizer_state.is_empty() {
                "abc"
            } else {
                "the cat"
            };
            let free = slm
                .generate_tokens(seed, 48, &[], &p, &mut Rng::new(1))
                .unwrap()
                .text;
            let chars: Vec<char> = free.chars().collect();
            assert!(chars.len() >= 20, "{free:?}");
            for len in [1usize, 2, 3, 5] {
                for k in 0..chars.len() - len {
                    let stop: String = chars[k..k + len].iter().collect();
                    let want: String = free[..free.find(&stop).unwrap()].to_string();
                    let g = slm
                        .generate_tokens(
                            seed,
                            48,
                            std::slice::from_ref(&stop),
                            &p,
                            &mut Rng::new(1),
                        )
                        .unwrap();
                    assert_eq!(g.text, want, "stop {stop:?} at {k}");
                    assert_eq!(g.stop_reason, StopReason::Stop);
                    let legacy = slm
                        .generate_until(seed, 48, &stop, &p, &mut Rng::new(1))
                        .unwrap();
                    assert!(
                        legacy.starts_with(&format!("{seed}{want}")),
                        "legacy {legacy:?} vs {seed}{want}"
                    );
                    // With several stops, generation halts when the first one completes and the
                    // text is cut at the earliest start among those present then (HF / lm-eval).
                    let other: String = chars[(k + len).min(chars.len() - 1)..]
                        .iter()
                        .take(2)
                        .collect();
                    for stops in [
                        vec![stop.clone(), other.clone()],
                        vec![other.clone(), stop.clone()],
                    ] {
                        let both = slm
                            .generate_tokens(seed, 48, &stops, &p, &mut Rng::new(1))
                            .unwrap();
                        let halt = (1..=chars.len())
                            .map(|n| chars[..n].iter().collect::<String>())
                            .find(|pre| stops.iter().any(|s| pre.contains(s.as_str())))
                            .unwrap();
                        let cut = stops
                            .iter()
                            .filter_map(|s| halt.find(s.as_str()))
                            .min()
                            .unwrap();
                        assert_eq!(both.text, halt[..cut], "{stops:?}");
                    }
                }
            }
        }
    }
}
