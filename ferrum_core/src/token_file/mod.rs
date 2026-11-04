//! The pre-tokenized corpus file (`general.architecture = "ferrum-tokens"`): a GGUF v3
//! container holding one `tokens` tensor of token ids (I16 when the vocabulary fits, else I32),
//! so a corpus larger than RAM can be trained on by streaming windows from disk.
//!
//! Metadata (`ferrum.tokens.*`): `version`, `vocab_size`, `token_count`, `document_count`,
//! `eos_id` (absent when the tokenizer has none) and `tokenizer_sha256` (the sha256 of the
//! tokenizer file the ids came from; the caller computes and compares it, the engine has no
//! hash). Documents are separated by the EOS id exactly as `ByteBpeTokenizer::encode_docs`
//! writes them, so `document_count` is one more than the number of EOS ids (0 when empty).
//!
//! Output is [`write_token_file`] (one pass, atomic, ids pushed into a [`TokenSink`]); the reader
//! is [`TokenFile`] (`Read` + `Seek` on the tensor data, no whole-file read, no mmap).
#![warn(missing_docs)]

mod reader;
mod writer;

pub use reader::{is_gguf_file, TokenFile};
pub use writer::{write_token_file, TokenSink};

use crate::gguf::{GGML_I16, GGML_I32};

/// `general.architecture` of a token file.
pub const TOKEN_FILE_ARCH: &str = "ferrum-tokens";
/// Layout version under `ferrum.tokens.version`.
pub(crate) const VERSION: u32 = 1;
/// Name of the single tensor.
pub(crate) const TENSOR: &str = "tokens";

/// What a token file records about its ids (everything but the ids themselves).
#[derive(Clone, Debug, PartialEq)]
pub struct TokenFileMeta {
    /// Ids are in `0..vocab_size`.
    pub vocab_size: usize,
    /// The id that separates documents, when the tokenizer has one.
    pub eos_id: Option<u32>,
    /// Lowercase hex sha256 of the tokenizer file (any non-empty string the caller can compare).
    pub tokenizer_sha256: String,
}

/// What a finished token file holds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenFileStats {
    /// Ids written.
    pub tokens: u64,
    /// Documents written (each ends with the EOS id when the tokenizer has one).
    pub documents: u64,
    /// Size of the file in bytes.
    pub bytes: u64,
}

/// The tensor type and bytes per id for a vocabulary: I16 holds ids up to 32767.
pub(crate) fn element_type(vocab_size: usize) -> (u32, usize) {
    if vocab_size <= i16::MAX as usize + 1 {
        (GGML_I16, 2)
    } else {
        (GGML_I32, 4)
    }
}

#[cfg(test)]
mod tests;
