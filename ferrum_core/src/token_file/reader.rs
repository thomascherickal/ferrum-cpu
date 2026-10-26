//! Streaming reader of a `ferrum-tokens` file.
use super::{element_type, TENSOR, TOKEN_FILE_ARCH, VERSION};
use crate::error::{InferError, Result};
use crate::gguf::{Gguf, MetaValue};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

fn bad(m: String) -> InferError {
    InferError::Format(m)
}

/// Whether `path` starts like a GGUF file (magic and a version this reader accepts): a cheap
/// sniff so `train` can tell a token file from a text corpus by content, not by extension.
/// A text file cannot match: bytes 4..8 would have to be the control characters `02 00 00 00`.
pub fn is_gguf_file(path: &str) -> bool {
    let mut head = [0u8; 8];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut head))
        .is_ok()
        && head.starts_with(b"GGUF")
        && matches!(
            u32::from_le_bytes([head[4], head[5], head[6], head[7]]),
            2 | 3
        )
}

/// An open token file. Holds the file handle and a scratch buffer of one window, never the
/// tokens: reading `len` ids costs `len * width` bytes of buffer, whatever the file's size.
#[derive(Debug)]
pub struct TokenFile {
    file: File,
    /// Absolute offset of the first id.
    pub(super) data_start: u64,
    /// Bytes per id (2 or 4).
    width: usize,
    tokens: usize,
    vocab_size: usize,
    documents: u64,
    eos_id: Option<u32>,
    tokenizer_sha256: String,
    scratch: Vec<u8>,
}

impl TokenFile {
    /// Open and validate `path`: GGUF header, `ferrum-tokens` architecture, version, metadata,
    /// one 1-D integer `tokens` tensor whose length matches `token_count` and lies inside the
    /// file (a truncated file fails here). The ids themselves are range-checked as they are read.
    pub fn open(path: &str) -> Result<Self> {
        let g = Gguf::open(path)?;
        if g.architecture() != Some(TOKEN_FILE_ARCH) {
            return Err(bad(format!(
                "{path} is not a {TOKEN_FILE_ARCH} file (general.architecture = {:?})",
                g.architecture()
            )));
        }
        let num = |key: &str| {
            g.meta(&format!("ferrum.tokens.{key}"))
                .and_then(MetaValue::as_usize)
                .ok_or_else(|| bad(format!("{path}: key 'ferrum.tokens.{key}' missing")))
        };
        if num("version")? != VERSION as usize {
            return Err(bad(format!("{path}: unsupported token file version")));
        }
        let vocab_size = num("vocab_size")?;
        if vocab_size == 0 || vocab_size > i32::MAX as usize {
            return Err(bad(format!("{path}: vocab_size {vocab_size} out of range")));
        }
        let tokens = num("token_count")?;
        let documents = num("document_count")? as u64;
        let eos_id = match g.meta("ferrum.tokens.eos_id") {
            None => None,
            Some(v) => Some(
                v.as_usize()
                    .filter(|&e| e < vocab_size)
                    .ok_or_else(|| bad(format!("{path}: eos_id is outside the vocabulary")))?
                    as u32,
            ),
        };
        let tokenizer_sha256 = g
            .meta("ferrum.tokens.tokenizer_sha256")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| {
                bad(format!(
                    "{path}: key 'ferrum.tokens.tokenizer_sha256' missing"
                ))
            })?
            .to_string();
        let (ty, width) = element_type(vocab_size);
        let t = match g.tensors.as_slice() {
            [t] if t.name == TENSOR => t,
            _ => {
                return Err(bad(format!(
                    "{path}: expected exactly one '{TENSOR}' tensor"
                )))
            }
        };
        if t.ggml_type != ty || t.dims != [tokens as u64] {
            return Err(bad(format!(
                "{path}: tensor '{TENSOR}' must be 1-D [{tokens}] of type {ty} for a vocabulary of {vocab_size}"
            )));
        }
        let data_start = (g.data_offset() as u64)
            .checked_add(t.offset)
            .ok_or_else(|| bad(format!("{path}: tensor offset overflows")))?;
        let end = tokens as u64 * width as u64 + data_start;
        if end > g.file_len() as u64 {
            return Err(bad(format!(
                "{path}: token data runs past the end of the file"
            )));
        }
        Ok(Self {
            file: File::open(path)?,
            data_start,
            width,
            tokens,
            vocab_size,
            documents,
            eos_id,
            tokenizer_sha256,
            scratch: Vec::new(),
        })
    }

    /// Number of ids in the file.
    pub fn len(&self) -> usize {
        self.tokens
    }

    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn documents(&self) -> u64 {
        self.documents
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// sha256 of the tokenizer file the ids were made with.
    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    /// Bytes of scratch the reader currently holds (one window's worth at most so far).
    pub fn buffered_bytes(&self) -> usize {
        self.scratch.capacity()
    }

    /// Replace `out` with ids `start..start + len`. One seek and one read of `len * width`
    /// bytes; an id outside the vocabulary (a corrupt file) is an error.
    pub fn read_window(&mut self, start: usize, len: usize, out: &mut Vec<u32>) -> Result<()> {
        if start.checked_add(len).is_none_or(|e| e > self.tokens) {
            return Err(InferError::DimMismatch(format!(
                "token window {start}+{len} is past the {} tokens in the file",
                self.tokens
            )));
        }
        self.scratch.resize(len * self.width, 0);
        self.file.seek(SeekFrom::Start(
            self.data_start + (start * self.width) as u64,
        ))?;
        self.file.read_exact(&mut self.scratch)?;
        out.clear();
        if self.width == 2 {
            out.extend(
                self.scratch
                    .chunks_exact(2)
                    .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]]))),
            );
        } else {
            out.extend(
                self.scratch
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
        }
        if let Some(pos) = out.iter().position(|&id| id as usize >= self.vocab_size) {
            return Err(bad(format!(
                "token {} is id {}, outside the vocabulary of {}",
                start + pos,
                out[pos],
                self.vocab_size
            )));
        }
        Ok(())
    }
}
