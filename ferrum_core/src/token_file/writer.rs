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
