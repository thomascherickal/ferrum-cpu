//! Corpus tokenization for language-model training (char-level and BPE).
use crate::error::{InferError, Result};
use crate::slm::config::TransformerConfig;
use crate::tokenizer::ByteBpeTokenizer;
use std::collections::HashSet;

/// Translates a character into its exact lowercase hex-string representation.
pub fn char_to_hex(ch: char) -> String {
    format!("{:x}", ch as u32)
}

/// Decodes an exact hex-string representation back into its character representation.
pub fn hex_to_char(hex: &str) -> char {
    let code = u32::from_str_radix(hex, 16).unwrap_or(32); // Fallback to space on parse error
    std::char::from_u32(code).unwrap_or(' ')
}

/// The sorted character vocabulary of a corpus (always includes ' ' and '\n',
/// never '\r').
pub fn corpus_vocab(corpus: &str) -> Vec<char> {
    let mut vocab: HashSet<char> = corpus.chars().filter(|&c| c != '\r').collect();
    vocab.insert(' ');
    vocab.insert('\n');
    let mut vocab_vec: Vec<char> = vocab.into_iter().collect();
    vocab_vec.sort();
    vocab_vec
}

/// Everything the token-ID language-model paths need to size a network and
/// describe it in metadata, produced by [`tokenize_for_lm`].
pub(super) struct LmTokens {
    /// Token-ID stream over the whole corpus.
    pub(super) tokens: Vec<usize>,
    /// Number of distinct token IDs to embed and predict (the model's
    /// embedding rows and LM-head width).
    pub(super) vocab_size: usize,
    /// Per-class hex code-point names for character-level models; empty for BPE
    /// (where decoding goes through the stored tokenizer instead).
    pub(super) class_names: Vec<String>,
    /// Serialized BPE merge list; empty for character-level models.
    pub(super) tokenizer_state: String,
}

/// [`tokenize_for_lm`], reusing (or creating) `cfg.tokenizer_path` for BPE.
pub(super) fn tokenize_for_lm_cfg(corpus: &str, cfg: &TransformerConfig) -> Result<LmTokens> {
    let path = std::path::Path::new(&cfg.tokenizer_path);
    if cfg.vocab_size == 0 || cfg.tokenizer_path.is_empty() {
        return tokenize_for_lm(corpus, cfg.context_len, cfg.vocab_size);
    }
    let tok = if path.exists() {
        vprintln!("[slm::tokenize] reusing tokenizer {}", path.display());
        ByteBpeTokenizer::from_gguf_bytes(&std::fs::read(path)?)?
    } else {
        let tok = ByteBpeTokenizer::train_with_docs(corpus, cfg.vocab_size)?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, tok.to_gguf_bytes())?;
        tok
    };
    lm_tokens_from_bpe(&tok, corpus, cfg.context_len)
}

/// Tokenize `corpus` for the transformer / embedded language-model paths.
///
/// `vocab_size == 0` selects character-level tokenization (the sorted corpus
/// vocabulary, identical to [`tokenize_corpus`]). `vocab_size >= 256` trains a
/// byte-level [`ByteBpeTokenizer`] of that target size and encodes the corpus
/// with it, so the same trainer, QAT, and GGUF serialization work unchanged on
/// subword tokens. Values in `1..256` are rejected (the 256-byte base
/// vocabulary is irreducible).
pub(super) fn tokenize_for_lm(
    corpus: &str,
    context_len: usize,
    vocab_size: usize,
) -> Result<LmTokens> {
    if vocab_size == 0 {
        let (vocab_vec, tokens) = tokenize_corpus(corpus, context_len)?;
        let class_names = vocab_vec.iter().map(|&ch| char_to_hex(ch)).collect();
        return Ok(LmTokens {
            vocab_size: vocab_vec.len(),
            tokens,
            class_names,
            tokenizer_state: String::new(),
        });
    }
    let tok = ByteBpeTokenizer::train_with_docs(corpus, vocab_size)?;
    lm_tokens_from_bpe(&tok, corpus, context_len)
}

fn lm_tokens_from_bpe(
    tok: &ByteBpeTokenizer,
    corpus: &str,
    context_len: usize,
) -> Result<LmTokens> {
    let tokens = tok.encode_docs(corpus);
    if tokens.len() < context_len + 1 {
        return Err(InferError::DimMismatch(
            "BPE-tokenized corpus must be longer than the context window".into(),
        ));
    }
    vprintln!(
        "[slm::tokenize_for_lm] BPE: vocab={}, {} tokens",
        tok.vocab_size(),
        tokens.len()
    );
    Ok(LmTokens {
        vocab_size: tok.vocab_size(),
        tokens,
        class_names: Vec::new(),
        tokenizer_state: tok.encode_state(),
    })
}

/// Tokenize a corpus into (sorted vocabulary, token-ID stream) for
/// transformer training. The vocabulary always includes ' ' and '\n'.
pub fn tokenize_corpus(corpus: &str, context_len: usize) -> Result<(Vec<char>, Vec<usize>)> {
    let chars: Vec<char> = corpus.chars().filter(|&c| c != '\r').collect();
    if chars.len() < context_len + 1 {
        return Err(InferError::DimMismatch(
            "Corpus must be longer than the context window".into(),
        ));
    }
    let vocab_vec = corpus_vocab(corpus);
    let tokens: Vec<usize> = chars
        .iter()
        .map(|c| vocab_vec.binary_search(c).unwrap_or(0))
        .collect();
    Ok((vocab_vec, tokens))
}
