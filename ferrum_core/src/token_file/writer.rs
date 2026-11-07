//! One-pass atomic writer of a `ferrum-tokens` file.
use super::{element_type, TokenFileMeta, TokenFileStats, TENSOR, TOKEN_FILE_ARCH, VERSION};
use crate::error::{InferError, Result};
use crate::fs_atomic::atomic_write_with;
use crate::gguf::{MetaValue, GGML_I32};
use crate::gguf_write::GgufBuilder;
use std::io::{BufWriter, Seek, SeekFrom, Write};

/// The GGUF header (metadata, directory, alignment padding) for `n` tokens in `docs` documents.
/// Every count is a fixed-width integer, so its length does not depend on `n` or `docs`:
/// the writer emits a placeholder first and rewrites it in place once the counts are known.
pub(super) fn header_bytes(meta: &TokenFileMeta, n: u64, docs: u64) -> Vec<u8> {
    let (ty, _) = element_type(meta.vocab_size);
    let mut b = GgufBuilder::new();
    b.meta(
        "general.architecture",
        MetaValue::String(TOKEN_FILE_ARCH.into()),
    )
    .meta("ferrum.tokens.version", MetaValue::U32(VERSION))
    .meta(
        "ferrum.tokens.vocab_size",
        MetaValue::U32(meta.vocab_size as u32),
    )
    .meta("ferrum.tokens.token_count", MetaValue::U64(n))
    .meta("ferrum.tokens.document_count", MetaValue::U64(docs))
    .meta(
        "ferrum.tokens.tokenizer_sha256",
        MetaValue::String(meta.tokenizer_sha256.clone()),
    );
    if let Some(eos) = meta.eos_id {
        b.meta("ferrum.tokens.eos_id", MetaValue::U32(eos));
    }
    // A tensor that claims `n` ids but carries no data yields exactly the header bytes.
    b.tensor(TENSOR, &[n], ty, Vec::new());
    b.into_bytes()
}

/// Receives the ids of a token file being written (see [`write_token_file`]).
pub struct TokenSink<'a> {
    w: BufWriter<&'a mut std::fs::File>,
    meta: &'a TokenFileMeta,
    wide: bool,
    tokens: u64,
    eos_seen: u64,
}

impl TokenSink<'_> {
    /// Ids pushed so far.
    pub fn len(&self) -> u64 {
        self.tokens
    }

    /// True when no id was pushed yet.
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// Append `ids`; one outside the vocabulary is an error (nothing is kept).
    pub fn push(&mut self, ids: &[u32]) -> Result<()> {
        for &id in ids {
            if id as usize >= self.meta.vocab_size {
                return Err(InferError::Format(format!(
                    "token {} is id {id}, outside the vocabulary of {}",
                    self.tokens, self.meta.vocab_size
                )));
            }
            self.eos_seen += u64::from(self.meta.eos_id == Some(id));
            if self.wide {
                self.w.write_all(&id.to_le_bytes())?;
            } else {
                self.w.write_all(&(id as i16).to_le_bytes())?;
            }
            self.tokens += 1;
        }
        Ok(())
    }
}

/// Write a `ferrum-tokens` GGUF to `path` in one pass with O(1) memory: `produce` pushes ids into
/// the [`TokenSink`] (a placeholder header, then the ids through a 1 MiB buffer, then the header
/// rewritten with the real counts). The file appears at `path` only when complete (temp file +
/// rename); an error from `produce`, or an id `>= meta.vocab_size`, aborts the write and is
/// returned, leaving any existing file untouched.
///
/// Ceiling: a `u32` vocabulary and `u64` token count; 2 bytes per token while the vocabulary is
/// at most 32768, else 4.
pub fn write_token_file(
    path: &str,
    meta: &TokenFileMeta,
    produce: impl FnOnce(&mut TokenSink) -> Result<()>,
) -> Result<TokenFileStats> {
    let bad = |m: String| InferError::Format(m);
    if meta.vocab_size == 0 || meta.vocab_size > i32::MAX as usize {
        return Err(bad(format!("vocab_size {} out of range", meta.vocab_size)));
    }
    if meta.eos_id.is_some_and(|e| e as usize >= meta.vocab_size) {
        return Err(bad("eos_id is outside the vocabulary".into()));
    }
    let (ty, width) = element_type(meta.vocab_size);
    let mut failure: Option<InferError> = None;
    let mut stats = TokenFileStats::default();
    atomic_write_with(path, |f| {
        let placeholder = header_bytes(meta, 0, 0);
        f.write_all(&placeholder)?;
        let mut sink = TokenSink {
            w: BufWriter::with_capacity(1 << 20, &mut *f),
            meta,
            wide: ty == GGML_I32,
            tokens: 0,
            eos_seen: 0,
        };
        if let Err(e) = produce(&mut sink) {
            failure = Some(e);
            return Err(std::io::Error::other("aborted"));
        }
        sink.w.flush()?;
        let (tokens, eos_seen) = (sink.tokens, sink.eos_seen);
        drop(sink);
        stats.tokens = tokens;
        stats.documents = if tokens == 0 { 0 } else { eos_seen + 1 };
        let header = header_bytes(meta, stats.tokens, stats.documents);
        debug_assert_eq!(header.len(), placeholder.len());
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&header)?;
        stats.bytes = header.len() as u64 + tokens * width as u64;
        Ok(())
    })
    .map_err(|e| failure.take().unwrap_or(e))?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(vocab: usize, eos: Option<u32>) -> TokenFileMeta {
        TokenFileMeta {
            vocab_size: vocab,
            eos_id: eos,
            tokenizer_sha256: "cd".repeat(32),
        }
    }

    fn path(name: &str) -> String {
        let d =
            std::env::temp_dir().join(format!("ferrum_tokwriter_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.join("w.tok.gguf").to_str().unwrap().to_string()
    }

    fn contains(hay: &[u8], needle: &str) -> bool {
        hay.windows(needle.len()).any(|w| w == needle.as_bytes())
    }

    #[test]
    fn header_names_the_kind_and_writes_eos_only_when_present() {
        let with = header_bytes(&meta(10, Some(3)), 5, 2);
        let without = header_bytes(&meta(10, None), 5, 2);
        for h in [&with, &without] {
            assert!(h.starts_with(b"GGUF"));
            assert!(contains(h, TOKEN_FILE_ARCH) && contains(h, TENSOR));
            assert!(contains(h, &"cd".repeat(32)));
        }
        assert!(contains(&with, "ferrum.tokens.eos_id"));
        assert!(!contains(&without, "ferrum.tokens.eos_id"));
    }

    /// The ids after the header are little-endian i16 for a small vocabulary and u32 above
    /// 32768, and `stats` reports the exact file size and the document count.
    #[test]
    fn ids_are_stored_little_endian_in_the_vocabulary_width() {
        for (vocab, wide) in [(300usize, false), (40_000, true)] {
            let p = path(&format!("le{vocab}"));
            let ids = [1u32, 299, 7, 0];
            let stats = write_token_file(&p, &meta(vocab, Some(7)), |s| s.push(&ids)).unwrap();
            let bytes = std::fs::read(&p).unwrap();
            assert_eq!(stats.bytes, bytes.len() as u64);
            assert_eq!(
                (stats.tokens, stats.documents),
                (4, 2),
                "one EOS splits two documents"
            );
            let want: Vec<u8> = if wide {
                ids.iter().flat_map(|i| i.to_le_bytes()).collect()
            } else {
                ids.iter().flat_map(|&i| (i as i16).to_le_bytes()).collect()
            };
            assert_eq!(bytes[bytes.len() - want.len()..], want[..], "vocab {vocab}");
            assert_eq!(
                bytes[..bytes.len() - want.len()],
                header_bytes(&meta(vocab, Some(7)), 4, 2)[..],
                "the placeholder header was rewritten with the real counts"
            );
        }
    }

    fn never_run(_: &mut TokenSink) -> Result<()> { Ok(()) }

    #[test]
    fn bad_meta_is_rejected_before_any_file_appears() {
        let p_ok = path("meta_ok");
        write_token_file(&p_ok, &meta(100, None), never_run).unwrap();
        let _ = std::fs::remove_file(&p_ok);
        let p = path("meta");
        for m in [
            meta(0, None),
            meta(i32::MAX as usize + 1, None),
            meta(5, Some(5)),
        ] {
            assert!(matches!(write_token_file(&p, &m, never_run), Err(InferError::Format(_))));
        }
        assert!(!std::path::Path::new(&p).exists());
    }

    #[test]
    fn an_empty_stream_has_no_documents() {
        let p = path("empty");
        let stats = write_token_file(&p, &meta(10, Some(1)), |_| Ok(())).unwrap();
        assert_eq!((stats.tokens, stats.documents), (0, 0));
    }
}
