//! Live text for imported-GGUF generation: turns the growing token list into decoded
//! fragments. The Generate tab's `gen-fragment` mechanism, for the GGUF tab.

use ferrum_core::GgufTokenizer;

/// Decodes the tokens produced so far and hands only the new text to `sink`.
pub struct FragmentStream<'a> {
    tok: &'a GgufTokenizer,
    sent: String,
    sink: Box<dyn FnMut(&str) + 'a>,
}

impl<'a> FragmentStream<'a> {
    pub fn new(tok: &'a GgufTokenizer, sink: impl FnMut(&str) + 'a) -> Self {
        FragmentStream {
            tok,
            sent: String::new(),
            sink: Box::new(sink),
        }
    }

    /// Call with every token generated so far; sends whatever text is new.
    /// A trailing U+FFFD (a half-decoded character) is held back until the next token.
    /// ponytail: re-decodes the whole output each token (O(n^2)); decode a window if
    /// very long runs make it show.
    pub fn push(&mut self, out: &[usize]) {
        let text = self.tok.decode(out);
        let ready = text.strip_suffix('\u{fffd}').unwrap_or(&text);
        if let Some(new) = ready
            .strip_prefix(self.sent.as_str())
            .filter(|n| !n.is_empty())
        {
            (self.sink)(new);
            self.sent = ready.to_string();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::Tiny;
    use ferrum_core::Gguf;

    fn tok() -> GgufTokenizer {
        GgufTokenizer::from_gguf(&Gguf::parse(Tiny::default().build()).unwrap()).unwrap()
    }

    #[test]
    fn fragments_concatenate_to_the_full_decode() {
        let t = tok();
        let ids: Vec<usize> = (0..t.vocab_size().min(6)).collect();
        let mut got = String::new();
        {
            let mut s = FragmentStream::new(&t, |f| got.push_str(f));
            for n in 1..=ids.len() {
                s.push(&ids[..n]);
            }
        }
        assert_eq!(got, t.decode(&ids));
    }

    #[test]
    fn pushing_the_same_tokens_twice_sends_nothing_new() {
        let t = tok();
        let mut n = 0;
        {
            let mut s = FragmentStream::new(&t, |_| n += 1);
            s.push(&[1, 2]);
            s.push(&[1, 2]);
        }
        assert!(n <= 1);
    }
}
