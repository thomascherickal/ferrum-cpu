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
    /// Token-ID stream over the whole corpus, held as `u32` (half the RAM of
    /// `usize`) and widened where a training window is cut.
    pub(super) tokens: Vec<u32>,
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
        crate::fs_atomic::atomic_write(&cfg.tokenizer_path, &tok.to_gguf_bytes())?;
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
        let (vocab_vec, tokens) = tokenize_chars(corpus, context_len)?;
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
    let tokens = narrow_ids(tok.encode_docs(corpus))?;
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

/// Token ids as `u32` with no spare capacity. An id beyond `u32` cannot occur
/// (the vocabulary is far smaller) but is an error rather than a wrap.
fn narrow_ids(ids: Vec<usize>) -> Result<Vec<u32>> {
    let mut out = ids
        .into_iter()
        .map(|i| u32::try_from(i).map_err(|_| InferError::Format(format!("token id {i} > u32"))))
        .collect::<Result<Vec<u32>>>()?;
    out.shrink_to_fit();
    Ok(out)
}

/// Tokenize a corpus into (sorted vocabulary, token-ID stream) for
/// transformer training. The vocabulary always includes ' ' and '\n'.
pub fn tokenize_corpus(corpus: &str, context_len: usize) -> Result<(Vec<char>, Vec<usize>)> {
    let (vocab, ids) = tokenize_chars(corpus, context_len)?;
    Ok((vocab, ids.into_iter().map(|i| i as usize).collect()))
}

/// [`tokenize_corpus`] with the ids held as `u32` (a char vocabulary is at most
/// the Unicode scalar count, far below `u32::MAX`).
fn tokenize_chars(corpus: &str, context_len: usize) -> Result<(Vec<char>, Vec<u32>)> {
    let chars: Vec<char> = corpus.chars().filter(|&c| c != '\r').collect();
    if chars.len() < context_len + 1 {
        return Err(InferError::DimMismatch(
            "Corpus must be longer than the context window".into(),
        ));
    }
    let vocab_vec = corpus_vocab(corpus);
    let tokens: Vec<u32> = chars
        .iter()
        .map(|c| vocab_vec.binary_search(c).unwrap_or(0) as u32)
        .collect();
    Ok((vocab_vec, tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slm::config::TransformerConfig;
    use crate::slm::corpus::tokenize_for_lm_cfg;

    #[test]
    fn corpus_ids_are_held_as_u32_at_half_the_memory_and_match_the_public_api() {
        let corpus = "the cat sat on the mat. ".repeat(40);
        // Character level: same ids as the public `tokenize_corpus`.
        let c = super::tokenize_for_lm(&corpus, 8, 0).unwrap();
        let (_, wide) = super::tokenize_corpus(&corpus, 8).unwrap();
        assert!(c
            .tokens
            .iter()
            .map(|&t| t as usize)
            .eq(wide.iter().copied()));
        // BPE: exactly the tokenizer's ids, in `len * 4` bytes with no spare capacity.
        let b = super::tokenize_for_lm(&corpus, 8, 300).unwrap();
        let tok = crate::tokenizer::ByteBpeTokenizer::from_state(&b.tokenizer_state).unwrap();
        assert!(b
            .tokens
            .iter()
            .map(|&t| t as usize)
            .eq(tok.encode_docs(&corpus)));
        for t in [&c.tokens, &b.tokens] {
            assert_eq!(t.capacity(), t.len());
            assert_eq!(
                std::mem::size_of_val(&t[..]) * 2,
                t.len() * std::mem::size_of::<usize>()
            );
        }
    }

    #[test]
    fn a_bare_tokenizer_filename_is_created_in_the_working_directory_and_reused() {
        let path = format!("ferrum_corpus_tok_{}.gguf", std::process::id());
        let cfg = TransformerConfig {
            context_len: 8,
            vocab_size: 300,
            tokenizer_path: path.clone(),
            ..TransformerConfig::default()
        };
        let corpus = "the cat sat on the mat. ".repeat(30);
        let first = tokenize_for_lm_cfg(&corpus, &cfg);
        let created = std::path::Path::new(&path).exists();
        let again = tokenize_for_lm_cfg(&corpus, &cfg);
        let _ = std::fs::remove_file(&path);
        assert!(created);
        assert_eq!(first.unwrap().tokens, again.unwrap().tokens);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn narrow_ids_rejects_overflow() {
        assert!(narrow_ids(vec![usize::MAX]).is_err());
    }
}
