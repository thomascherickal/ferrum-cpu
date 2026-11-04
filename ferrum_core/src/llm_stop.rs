//! Text-level stop strings for imported models: the one matcher behind `run-gguf`, the GUI's
//! GGUF tab and the server's `/v1/generate` on a Llama/Qwen GGUF, so the three cannot drift.
//!
//! Semantics match the Ferrum models' `stop`: the first (earliest) occurrence in the generated
//! text wins, the stop string itself is not part of the output, and a stop that spans token
//! boundaries still matches. Pair it with [`crate::LlamaModel::generate_until`]: call
//! [`StopMatcher::check`] from the hook, then [`StopMatcher::cut`] for the text.
#![warn(missing_docs)]

use crate::gguf_tokenizer::GgufTokenizer;

/// Byte offset of the earliest stop string in `text[from..]` (`from` floored to a character
/// boundary).
pub fn earliest_stop(text: &str, from: usize, stops: &[String]) -> Option<usize> {
    let from = (0..=from.min(text.len()))
        .rev()
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(0);
    stops
        .iter()
        .filter_map(|s| text[from..].find(s.as_str()).map(|i| from + i))
        .min()
}

/// The text of `out`, generated after the prompt token `anchor`: decoded together with the
/// anchor so a leading space survives SentencePiece's dummy-prefix stripping.
pub fn continuation_text(tok: &GgufTokenizer, anchor: usize, out: &[usize]) -> String {
    let mut ids = Vec::with_capacity(out.len() + 1);
    ids.push(anchor);
    ids.extend_from_slice(out);
    let full = tok.decode(&ids);
    match full.strip_prefix(tok.decode(&[anchor]).as_str()) {
        Some(rest) => rest.to_string(),
        None => tok.decode(out),
    }
}

/// Watches the generated tokens for the first stop string. Empty stops never match (and cost
/// nothing: no decoding).
pub struct StopMatcher<'a> {
    tok: &'a GgufTokenizer,
    anchor: usize,
    stops: Vec<String>,
    // Rescan this many bytes before the old end: a stop may straddle it, and lossy UTF-8
    // decoding may rewrite a few trailing bytes once a character completes.
    back: usize,
    text: String,
    scanned: usize,
    hit: Option<usize>,
}

impl<'a> StopMatcher<'a> {
    /// `anchor` is the last prompt token (see [`continuation_text`]). Empty stop strings are
    /// ignored.
    pub fn new(tok: &'a GgufTokenizer, anchor: usize, stops: &[String]) -> Self {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let back = stops.iter().map(String::len).max().unwrap_or(0) + 8;
        Self {
            tok,
            anchor,
            stops,
            back,
            text: String::new(),
            scanned: 0,
            hit: None,
        }
    }

    /// True once the tokens generated so far (all of them, every call) complete a stop string.
    // ponytail: every call re-decodes the whole continuation (O(n^2) in the token budget);
    // keep a running decode if long generations get slow.
    pub fn check(&mut self, out: &[usize]) -> bool {
        if self.stops.is_empty() {
            return false;
        }
        self.text = continuation_text(self.tok, self.anchor, out);
        self.hit = earliest_stop(
            &self.text,
            self.scanned.saturating_sub(self.back),
            &self.stops,
        );
        self.scanned = self.text.len();
        self.hit.is_some()
    }

    /// The continuation cut before the matched stop string; `None` when none matched.
    pub fn cut(&self) -> Option<&str> {
        self.hit.map(|i| &self.text[..i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;
    use crate::gguf_write::GgufBuilder;
    use crate::MetaValue;

    /// Byte-level BPE vocabulary: `a`..`z` are ids 0..25, then `Ã` (26) and `©` (27), whose
    /// two bytes together decode to `é`.
    fn tok() -> GgufTokenizer {
        let mut v: Vec<MetaValue> = ('a'..='z')
            .map(|c| MetaValue::String(c.to_string()))
            .collect();
        v.push(MetaValue::String("\u{c3}".into()));
        v.push(MetaValue::String("\u{a9}".into()));
        let mut b = GgufBuilder::new();
        b.meta("tokenizer.ggml.model", MetaValue::String("gpt2".into()));
        b.meta("tokenizer.ggml.tokens", MetaValue::Array(v));
        GgufTokenizer::from_gguf(&Gguf::parse(b.into_bytes()).unwrap()).unwrap()
    }

    fn ids(s: &str) -> Vec<usize> {
        s.bytes().map(|b| (b - b'a') as usize).collect()
    }

    fn stops(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| s.to_string()).collect()
    }

    /// Feed `out` one token at a time, like the generate hook; the step that matched.
    fn feed(m: &mut StopMatcher, out: &[usize]) -> Option<usize> {
        (1..=out.len()).find(|&n| m.check(&out[..n]))
    }

    #[test]
    fn earliest_stop_scans_from_a_char_boundary() {
        let s = stops(&["c", "ab"]);
        assert_eq!(earliest_stop("xxabc", 0, &s), Some(2));
        assert_eq!(earliest_stop("xxabc", 3, &s), Some(4));
        assert_eq!(earliest_stop("xx", 9, &s), None);
        // `from` inside a multi-byte character is floored to its start.
        assert_eq!(earliest_stop("\u{e9}c", 1, &s), Some(2));
    }

    #[test]
    fn a_stop_inside_one_token_cuts_before_it() {
        let t = tok();
        let s = stops(&["cd"]);
        let mut m = StopMatcher::new(&t, 0, &s);
        assert_eq!(feed(&mut m, &ids("abcde")), Some(4));
        assert_eq!(m.cut(), Some("ab"));
    }

    #[test]
    fn a_stop_spanning_two_tokens_matches_and_the_end_matches() {
        let t = tok();
        let s = stops(&["bc"]);
        let mut m = StopMatcher::new(&t, 0, &s);
        assert_eq!(feed(&mut m, &ids("abc")), Some(3), "stop at the very end");
        assert_eq!(m.cut(), Some("a"));
        let s = stops(&["a"]);
        let mut m = StopMatcher::new(&t, 0, &s);
        assert_eq!(feed(&mut m, &ids("abc")), Some(1));
        assert_eq!(m.cut(), Some(""), "a stop at the start leaves nothing");
    }

    #[test]
    fn no_match_means_no_cut_and_a_held_back_prefix_is_released() {
        let t = tok();
        let s = stops(&["abc"]);
        let mut m = StopMatcher::new(&t, 0, &s);
        // "ab" is a prefix of the stop, but "abd" completes nothing.
        assert_eq!(feed(&mut m, &ids("abdef")), None);
        assert_eq!(m.cut(), None);
    }

    #[test]
    fn the_earliest_stop_wins_whatever_its_position_in_the_list() {
        let t = tok();
        let s = stops(&["de", "bc"]);
        let mut m = StopMatcher::new(&t, 0, &s);
        assert_eq!(feed(&mut m, &ids("abcde")), Some(3));
        assert_eq!(m.cut(), Some("a"));
    }

    #[test]
    fn no_stops_or_only_empty_ones_never_match() {
        let t = tok();
        for list in [stops(&[]), stops(&[""])] {
            let mut m = StopMatcher::new(&t, 0, &list);
            assert_eq!(feed(&mut m, &ids("abc")), None);
            assert_eq!(m.cut(), None);
        }
    }

    #[test]
    fn a_stop_after_a_multi_byte_character_matches() {
        let t = tok();
        let s = stops(&["\u{e9}b"]);
        let mut m = StopMatcher::new(&t, 0, &s);
        // a, Ã, ©, b: the character is incomplete after Ã and whole after ©.
        assert_eq!(feed(&mut m, &[0, 26, 27, 1]), Some(4));
        assert_eq!(m.cut(), Some("a"));
        let s = stops(&["\u{e9}"]);
        let mut m = StopMatcher::new(&t, 0, &s);
        assert_eq!(feed(&mut m, &[0, 26, 27, 1]), Some(3));
        assert_eq!(m.cut(), Some("a"));
    }

    #[test]
    fn continuation_text_decodes_after_the_anchor() {
        let t = tok();
        assert_eq!(continuation_text(&t, 0, &ids("bc")), "bc");
    }
}
