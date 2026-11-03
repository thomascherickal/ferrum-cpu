//! The imported-model backend: a Llama/Qwen [`LlamaModel`] with the GGUF's own tokenizer,
//! answering the same three primitives as a Ferrum model with the same JSON shapes.
//!
//! Conventions (Manual §9.7): scoring and generation both prepend the BOS token exactly when
//! the file says so (`prefix_bos`: `add_bos_token` and a BOS id), like `run-gguf` and
//! `eval-gguf`; generation is a raw completion (no chat template) whose prompt is cut from the
//! left to leave room for `max_tokens`; texts longer than the scoring window (the context,
//! at most 512 tokens like `eval-gguf`) are scored in overlapping windows.

use crate::backend::{GenRequest, DEFAULT_MAX_TOKENS};
use ferrum_core::{GgufTokenizer, InferError, LlamaModel, LlamaScore, Result, Rng, SamplingParams};

/// Largest scoring window, in tokens: what `eval-gguf` uses, and it bounds the
/// `window x vocab` f32 logits one forward pass holds.
pub(crate) const MAX_SCORE_WINDOW: usize = 512;

/// The scoring window of a model with context `ctx_len`.
fn score_window(ctx_len: usize) -> usize {
    ctx_len.min(MAX_SCORE_WINDOW)
}

/// Cut `ids` (the first `n_bos` of which are the BOS prefix) from the left so that the prompt
/// and `max_new` generated tokens fit the `ctx`-token window: the BOS and the most recent
/// tokens are kept, and nothing is ever refused for length. A `max_new` of a whole window or
/// more cannot be honoured (this model has no sliding window); the prompt then keeps the whole
/// window less one slot, as the Ferrum path keeps a full window, and the generation ends where
/// the window does.
fn truncate_prompt(mut ids: Vec<usize>, n_bos: usize, ctx: usize, max_new: usize) -> Vec<usize> {
    let last = ctx.saturating_sub(1).max(1);
    let budget = if max_new >= ctx {
        last
    } else {
        (ctx - max_new).clamp(1, last)
    };
    let keep = budget.saturating_sub(n_bos).max(1);
    if ids.len() > n_bos + keep {
        ids.drain(n_bos..ids.len() - keep);
    }
    ids
}

pub(crate) struct Imported {
    pub(crate) model: LlamaModel,
    pub(crate) tok: GgufTokenizer,
}

/// Output of [`Imported::generate`]: the continuation (stop string excluded) and the tokens
/// generated (the one that completed a stop string included, EOS excluded), as on the Ferrum
/// path.
#[derive(Debug, PartialEq)]
pub(crate) struct Generated {
    pub(crate) text: String,
    pub(crate) n_tokens: usize,
}

/// Byte offset of the earliest stop string in `text[from..]` (`from` floored to a character
/// boundary).
fn earliest_stop(text: &str, from: usize, stops: &[String]) -> Option<usize> {
    let from = (0..=from.min(text.len()))
        .rev()
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(0);
    stops
        .iter()
        .filter_map(|s| text[from..].find(s.as_str()).map(|i| from + i))
        .min()
}

impl Imported {
    /// The BOS token when the file says to prepend one (`add_bos_token` and a BOS id): the
    /// same rule as generation and `eval-gguf`.
    fn scoring_prefix(&self) -> Vec<usize> {
        self.tok.prefix_bos().into_iter().collect()
    }

    fn score_ids(&self, ids: &[usize], from: usize) -> Result<(LlamaScore, bool)> {
        self.model
            .score_from(ids, from, score_window(self.model.cfg.context_len))
    }

    /// Loglikelihood of `continuation` after `context`, split like `GenerativeSLM::score`:
    /// trailing whitespace of the context moves onto the continuation, and the joint
    /// encoding is used when it extends the context's tokens.
    pub(crate) fn score(&self, context: &str, continuation: &str) -> Result<(LlamaScore, bool)> {
        let trimmed = context.trim_end();
        let (context, continuation) = (
            trimmed,
            format!("{}{continuation}", &context[trimmed.len()..]),
        );
        if continuation.is_empty() {
            return Ok((
                LlamaScore {
                    logprob: 0.0,
                    n_tokens: 0,
                },
                true,
            ));
        }
        let mut ctx_ids = self.scoring_prefix();
        ctx_ids.extend(self.tok.encode(context));
        if ctx_ids.is_empty() {
            return Err(InferError::DimMismatch(
                "context is empty and the model prepends no BOS token".into(),
            ));
        }
        let mut joint = self.scoring_prefix();
        joint.extend(self.tok.encode(&format!("{context}{continuation}")));
        let ids = if joint.len() > ctx_ids.len() && joint.starts_with(&ctx_ids) {
            joint
        } else {
            let mut ids = ctx_ids.clone();
            ids.extend(self.tok.encode(&continuation));
            ids
        };
        self.score_ids(&ids, ctx_ids.len())
    }

    /// Loglikelihood of all of `text`. With a BOS every token is scored; without one (the file
    /// does not prepend any) the first token has nothing to condition on and is not scored.
    /// A context longer than the window needs no cutting: the windows slide so every scored
    /// token keeps at least half a window of the most recent history.
    pub(crate) fn score_rolling(&self, text: &str) -> Result<LlamaScore> {
        let mut ids = self.scoring_prefix();
        ids.extend(self.tok.encode(text));
        self.score_ids(&ids, 1).map(|(s, _)| s)
    }

    /// The text of `out`, generated after the prompt token `anchor`: decoded together with the
    /// anchor so a leading space survives SentencePiece's dummy-prefix stripping.
    fn continuation_text(&self, anchor: usize, out: &[usize]) -> String {
        let mut ids = Vec::with_capacity(out.len() + 1);
        ids.push(anchor);
        ids.extend_from_slice(out);
        let full = self.tok.decode(&ids);
        match full.strip_prefix(self.tok.decode(&[anchor]).as_str()) {
            Some(rest) => rest.to_string(),
            None => self.tok.decode(out),
        }
    }

    /// Raw completion of `g.prompt`: up to `max_tokens` tokens, stopping at EOS or at the
    /// first stop string (excluded from the text). A prompt too long for the context window
    /// together with `max_tokens` is cut from the left ([`truncate_prompt`]), and generation
    /// ends when the window is full.
    pub(crate) fn generate(&self, g: &GenRequest) -> Result<Generated> {
        let mut prompt: Vec<usize> = self.tok.prefix_bos().into_iter().collect();
        let n_bos = prompt.len();
        prompt.extend(self.tok.encode(&g.prompt));
        let max_new = g.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let prompt = truncate_prompt(prompt, n_bos, self.model.cfg.context_len, max_new);
        let params = SamplingParams {
            temperature: g.temperature,
            top_k: g.top_k,
            top_p: g.top_p,
            repetition_penalty: g.repetition_penalty.unwrap_or(1.0),
            ..SamplingParams::default()
        };
        let eos: Vec<usize> = self.tok.eos().into_iter().collect();
        let anchor = prompt.last().copied().unwrap_or(0);
        // Rescan this many bytes before the old end: a stop may straddle it, and lossy UTF-8
        // decoding may rewrite a few trailing bytes once a character completes.
        let back = g.stop.iter().map(String::len).max().unwrap_or(0) + 8;
        let (mut text, mut scanned, mut hit) = (String::new(), 0usize, None);
        // ponytail: every token re-decodes the whole continuation (O(n^2) in max_tokens,
        // at most 4096); keep a running decode if long generations get slow.
        let mut out = self.model.generate_until(
            &prompt,
            max_new,
            &params,
            &eos,
            &mut Rng::new(g.seed),
            &mut |out| {
                if g.stop.is_empty() {
                    return false;
                }
                text = self.continuation_text(anchor, out);
                hit = earliest_stop(&text, scanned.saturating_sub(back), &g.stop);
                scanned = text.len();
                hit.is_some()
            },
        )?;
        if out.last().is_some_and(|t| eos.contains(t)) {
            out.pop();
        }
        let text = match hit {
            Some(i) => text[..i].to_string(),
            None => self.continuation_text(anchor, &out),
        };
        Ok(Generated {
            text,
            n_tokens: out.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{imported, llama};
    use ferrum_core::MetaValue;

    fn req(prompt: &str, max: usize, stop: &[&str]) -> GenRequest {
        GenRequest {
            prompt: prompt.into(),
            max_tokens: Some(max),
            stop: stop.iter().map(|s| s.to_string()).collect(),
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            repetition_penalty: None,
        }
    }

    #[test]
    fn earliest_stop_scans_from_a_char_boundary() {
        let stops = vec!["c".to_string(), "ab".to_string()];
        assert_eq!(earliest_stop("xxabc", 0, &stops), Some(2));
        assert_eq!(earliest_stop("xxabc", 3, &stops), Some(4));
        assert_eq!(earliest_stop("xx", 9, &stops), None);
        // `from` inside a multi-byte character is floored to its start.
        assert_eq!(earliest_stop("éc", 1, &stops), Some(2));
    }

    #[test]
    fn rolling_prepends_bos_when_the_gguf_defines_one() {
        let m = imported(&llama::tiny_llama_bos_eos());
        let ids = m.tok.encode("abc");
        let s = m.score_rolling("abc").unwrap();
        let mut with_bos = vec![llama::BOS as usize];
        with_bos.extend(&ids);
        assert_eq!(s, m.model.score_rolling(&with_bos, 16).unwrap());
        assert_eq!(s.n_tokens, ids.len());
        // Without a BOS the first token is not scored.
        let m = imported(&llama::tiny_llama(&[]));
        let s = m.score_rolling("abc").unwrap();
        assert_eq!(s, m.model.score_rolling(&ids, 16).unwrap());
        assert_eq!(s.n_tokens, ids.len() - 1);
        assert_eq!(m.score_rolling("").unwrap().n_tokens, 0);
    }

    #[test]
    fn scoring_bos_follows_add_bos_token() {
        let with = |add: bool| {
            imported(&llama::tiny_llama(&[
                ("tokenizer.ggml.bos_token_id", MetaValue::U32(llama::BOS)),
                ("tokenizer.ggml.add_bos_token", MetaValue::Bool(add)),
            ]))
        };
        let (on, off) = (with(true), with(false));
        let ids = on.tok.encode("ab");
        let (a, b) = (
            on.score_rolling("ab").unwrap(),
            off.score_rolling("ab").unwrap(),
        );
        // add_bos_token=true: BOS conditions the first token, which is scored.
        let mut bos_ids = vec![llama::BOS as usize];
        bos_ids.extend(&ids);
        assert_eq!(a, on.model.score_rolling(&bos_ids, 16).unwrap());
        assert_eq!(a.n_tokens, 2);
        // add_bos_token=false: no BOS despite the id, so the first token is not scored.
        assert_eq!(b, off.model.score_rolling(&ids, 16).unwrap());
        assert_eq!(b.n_tokens, 1);
        assert_ne!(a.logprob, b.logprob);
        // The same rule for a continuation: an empty context needs the BOS.
        assert_eq!(on.score("", "ab").unwrap().0.n_tokens, 2);
        let e = off.score("", "ab").unwrap_err().to_string();
        assert!(
            e.contains("context is empty and the model prepends no BOS token"),
            "{e}"
        );
    }

    #[test]
    fn the_scoring_window_is_capped_at_512() {
        assert_eq!(score_window(16), 16);
        assert_eq!(score_window(512), 512);
        assert_eq!(score_window(8192), 512);
    }

    #[test]
    fn prompts_are_cut_from_the_left_to_leave_room_for_max_tokens() {
        let ids: Vec<usize> = (100..120).collect();
        // Fits: untouched.
        assert_eq!(truncate_prompt(ids[..4].to_vec(), 0, 16, 8), ids[..4]);
        // Cut to ctx - max_new most recent tokens.
        assert_eq!(truncate_prompt(ids.clone(), 0, 16, 6), ids[10..]);
        // The BOS stays in front and counts against the budget.
        let mut want = vec![100];
        want.extend(&ids[11..]);
        assert_eq!(truncate_prompt(ids.clone(), 1, 16, 6), want);
        // max_tokens that fills the window alone cannot be honoured: the window less one slot.
        assert_eq!(truncate_prompt(ids.clone(), 0, 16, 99), ids[5..]);
        let mut want = vec![100];
        want.extend(&ids[6..]);
        assert_eq!(truncate_prompt(ids.clone(), 1, 16, 16), want);
        assert_eq!(truncate_prompt(ids.clone(), 0, 16, 15), ids[19..]);
        assert_eq!(truncate_prompt(ids.clone(), 1, 16, 15), vec![100, 119]);
        assert_eq!(truncate_prompt(ids.clone(), 0, 16, 0), ids[5..]);
        assert!(truncate_prompt(vec![], 0, 16, 4).is_empty());
    }

    #[test]
    fn a_long_prompt_generates_as_its_truncated_tail_and_never_fails() {
        let m = imported(&llama::tiny_llama_bos_eos());
        let long = "abcdefghijklmnopqrstuvwxyz".repeat(2);
        let ids = m.tok.encode(&long);
        assert!(ids.len() > 16);
        let g = m.generate(&req(&long, 4, &[])).unwrap();
        // 16 - 4 = 12 slots: BOS + the 11 most recent tokens.
        let mut prompt = vec![llama::BOS as usize];
        prompt.extend(&ids[ids.len() - 11..]);
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let mut want = m
            .model
            .generate(
                &prompt,
                4,
                &params,
                &[llama::EOS as usize],
                &mut Rng::new(0),
            )
            .unwrap();
        if want.last() == Some(&(llama::EOS as usize)) {
            want.pop();
        }
        assert_eq!((g.n_tokens, g.text), (want.len(), m.tok.decode(&want)));
        // max_tokens beyond the window still answers.
        assert!(m.generate(&req(&long, 99, &[])).is_ok());
    }

    #[test]
    fn score_conditions_on_the_context_and_moves_trailing_space() {
        let m = imported(&llama::tiny_llama_bos_eos());
        let bos = llama::BOS as usize;
        let (s, greedy) = m.score("ab ", "cd").unwrap();
        let mut ids = vec![bos];
        ids.extend(m.tok.encode("ab cd"));
        let ctx = 1 + m.tok.encode("ab").len();
        assert_eq!((s, greedy), m.model.score_from(&ids, ctx, 16).unwrap());
        assert_eq!(s.n_tokens, ids.len() - ctx);
        // An empty context conditions on BOS alone; an empty continuation scores nothing.
        let (s, _) = m.score("", "ab").unwrap();
        assert_eq!(s.n_tokens, m.tok.encode("ab").len());
        assert_eq!(m.score("ab", "").unwrap().0.n_tokens, 0);
        // Without a BOS an empty context is refused.
        let m = imported(&llama::tiny_llama(&[]));
        let e = m.score("", "ab").unwrap_err().to_string();
        assert!(
            e.contains("context is empty and the model prepends no BOS token"),
            "{e}"
        );
        assert_eq!(m.score("a", "b").unwrap().0.n_tokens, 1);
    }

    #[test]
    fn score_falls_back_when_the_joint_encoding_does_not_extend_the_context() {
        // BPE merges nothing here, but an unknown character drops out of the joint encoding:
        // encode("a") = [a], encode("a€") = [a] (no byte tokens) -> not longer, so the
        // continuation is encoded on its own.
        let m = imported(&llama::tiny_llama_bos_eos());
        let (s, _) = m.score("a", "b").unwrap();
        assert_eq!(s.n_tokens, 1);
        let (s, _) = m.score("a", "€").unwrap();
        assert_eq!(s.n_tokens, m.tok.encode("€").len());
    }

    #[test]
    fn greedy_generation_equals_the_library_and_strips_eos() {
        let m = imported(&llama::tiny_llama_bos_eos());
        let mut prompt = vec![llama::BOS as usize];
        prompt.extend(m.tok.encode("ab"));
        let params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let mut want = m
            .model
            .generate(
                &prompt,
                6,
                &params,
                &[llama::EOS as usize],
                &mut Rng::new(0),
            )
            .unwrap();
        let g = m.generate(&req("ab", 6, &[])).unwrap();
        if want.last() == Some(&(llama::EOS as usize)) {
            want.pop();
        }
        assert_eq!(g.n_tokens, want.len());
        assert_eq!(g.text, m.tok.decode(&want));
    }

    #[test]
    fn stop_strings_and_max_tokens_end_the_generation() {
        let m = imported(&llama::tiny_llama(&[]));
        let full = m.generate(&req("ab", 10, &[])).unwrap();
        assert_eq!(full.n_tokens, 10, "{full:?}");
        assert_eq!(m.generate(&req("ab", 3, &[])).unwrap().n_tokens, 3);
        // Stop at the first occurrence of the third generated character.
        let ch = full.text.chars().nth(2).unwrap().to_string();
        let cut = full.text.find(&ch).unwrap();
        let g = m.generate(&req("ab", 10, &[&ch])).unwrap();
        assert_eq!(g.text, full.text[..cut]);
        assert!(g.n_tokens <= 3 && g.n_tokens >= 1, "{g:?}");
        // A stop that never appears changes nothing.
        assert_eq!(m.generate(&req("ab", 10, &["zzzz"])).unwrap(), full);
        // The context window (16) caps the generation; a long prompt is cut, not refused.
        assert_eq!(m.generate(&req("ab", 99, &[])).unwrap().n_tokens, 14);
        assert!(m.generate(&req(&"a".repeat(16), 1, &[])).is_ok());
        assert!(
            m.generate(&req("", 1, &[])).is_err(),
            "an empty prompt without BOS"
        );
    }

    #[test]
    fn continuation_text_keeps_a_leading_space() {
        let m = imported(&llama::tiny_llama(&[]));
        let space = m.tok.token_id("\u{120}").unwrap();
        let a = m.tok.token_id("a").unwrap();
        assert_eq!(m.continuation_text(a, &[space, a]), " a");
        // SentencePiece drops the dummy-prefix space of a decode: decoded alone, a
        // continuation that starts a new word would lose its space; after the anchor it keeps it.
        let m = imported(&llama::tiny_llama_spm());
        let (wa, b) = (
            m.tok.token_id("\u{2581}a").unwrap(),
            m.tok.token_id("b").unwrap(),
        );
        let a = m.tok.token_id("a").unwrap();
        assert_eq!(m.tok.decode(&[wa, b]), "ab");
        assert_eq!(m.continuation_text(a, &[wa, b]), " ab");
        // An anchor whose decode is not a prefix (a lone byte of a UTF-8 sequence) falls back.
        let m = imported(&llama::tiny_llama(&[
            (
                "tokenizer.ggml.tokens",
                MetaValue::Array(
                    ["<0xC3>", "<0xA9>", "a"]
                        .iter()
                        .map(|t| MetaValue::String(t.to_string()))
                        .chain((3..32).map(|i| MetaValue::String(format!("<f{i}>"))))
                        .collect(),
                ),
            ),
            ("tokenizer.ggml.model", MetaValue::String("llama".into())),
        ]));
        assert_eq!(m.continuation_text(0, &[1, 2]), m.tok.decode(&[1, 2]));
    }
}
