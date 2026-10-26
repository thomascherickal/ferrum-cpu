//! Where training windows come from: a slice of ids in memory, or a [`TokenFile`] streamed from
//! disk. The epoch code asks only for `len()` and for one window at a time, so both sources
//! draw the same RNG sequence and cut the same windows (bit-identical training).
use crate::error::Result;
use crate::token_file::TokenFile;

/// A stored token id that widens to `usize` at the window boundary, so a corpus
/// can be held as `u32` (half the RAM) while the model code keeps using `usize`.
pub(crate) trait TokenId: Copy + Send + Sync {
    fn widen(self) -> usize;
}

impl TokenId for usize {
    fn widen(self) -> usize {
        self
    }
}

impl TokenId for u32 {
    fn widen(self) -> usize {
        self as usize
    }
}

/// A token stream the trainer cuts `(input, target)` windows from.
pub(crate) trait TokenSource {
    /// Number of tokens.
    fn len(&self) -> usize;

    /// Append to `input` the `t` ids at `start..start + t` and to `targets` the `t` ids at
    /// `start + 1..start + t + 1` (`start + t < len()`).
    fn cut(
        &mut self,
        start: usize,
        t: usize,
        input: &mut Vec<usize>,
        targets: &mut Vec<usize>,
    ) -> Result<()>;
}

impl<T: TokenId> TokenSource for &[T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn cut(
        &mut self,
        start: usize,
        t: usize,
        input: &mut Vec<usize>,
        targets: &mut Vec<usize>,
    ) -> Result<()> {
        input.extend(self[start..start + t].iter().map(|&x| x.widen()));
        targets.extend(self[start + 1..start + t + 1].iter().map(|&x| x.widen()));
        Ok(())
    }
}

impl TokenSource for TokenFile {
    fn len(&self) -> usize {
        TokenFile::len(self)
    }

    /// One seek and one read of `t + 1` ids; memory is that window, not the corpus.
    fn cut(
        &mut self,
        start: usize,
        t: usize,
        input: &mut Vec<usize>,
        targets: &mut Vec<usize>,
    ) -> Result<()> {
        let mut w = Vec::with_capacity(t + 1);
        self.read_window(start, t + 1, &mut w)?;
        input.extend(w[..t].iter().map(|&x| x as usize));
        targets.extend(w[1..].iter().map(|&x| x as usize));
        Ok(())
    }
}

/// The first `len` tokens of another source: the training part of a train/validation split.
pub(crate) struct Prefix<'a, S: TokenSource> {
    pub(crate) inner: &'a mut S,
    pub(crate) len: usize,
}

impl<S: TokenSource> TokenSource for Prefix<'_, S> {
    fn len(&self) -> usize {
        self.len
    }

    fn cut(
        &mut self,
        start: usize,
        t: usize,
        input: &mut Vec<usize>,
        targets: &mut Vec<usize>,
    ) -> Result<()> {
        debug_assert!(start + t < self.len);
        self.inner.cut(start, t, input, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_file::{write_token_file, TokenFileMeta};

    fn file(ids: &[u32], vocab: usize) -> TokenFile {
        let d = std::env::temp_dir().join(format!("ferrum_tsrc_{vocab}_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let path = d.join("t.tok.gguf").to_str().unwrap().to_string();
        let meta = TokenFileMeta {
            vocab_size: vocab,
            eos_id: None,
            tokenizer_sha256: "x".into(),
        };
        write_token_file(&path, &meta, |s| s.push(ids)).unwrap();
        TokenFile::open(&path).unwrap()
    }

    #[test]
    fn a_file_cuts_the_same_windows_as_the_slice_and_a_prefix_hides_the_rest() {
        let ids: Vec<u32> = (0..500).map(|i| (i * 7 % 301) as u32).collect();
        let mut f = file(&ids, 301);
        let mut s: &[u32] = &ids;
        assert_eq!(TokenSource::len(&f), TokenSource::len(&s));
        for (start, t) in [(0, 16), (100, 1), (483, 16), (499 - 8, 8)] {
            let (mut a, mut b, mut c, mut d) = (vec![], vec![], vec![], vec![]);
            f.cut(start, t, &mut a, &mut b).unwrap();
            s.cut(start, t, &mut c, &mut d).unwrap();
            assert_eq!((a, b), (c, d), "window {start}+{t}");
        }
        let p = Prefix {
            inner: &mut f,
            len: 400,
        };
        assert_eq!(p.len(), 400);
        // A window past the end of the file is an error, not a panic.
        let (mut a, mut b) = (vec![], vec![]);
        assert!(f.cut(495, 16, &mut a, &mut b).is_err());
    }
}
