//! Opening a `ferrum-tokens` file as a training corpus, with the tokenizer it was made with.

use crate::hash::sha256_hex;
use ferrum_core::{ByteBpeTokenizer, TokenFile};

/// Open `path` as a token file and load `tokenizer`, checking that the file's recorded tokenizer
/// sha256 and vocabulary match it. Training needs the tokenizer anyway (it is stored in the
/// model), so a missing or different one is an error here, before any training time.
/// `char_level` is the "vocab 0" request, which a BPE token file cannot honour.
pub fn open_token_file(
    path: &str,
    tokenizer: Option<&str>,
    char_level: bool,
) -> Result<(TokenFile, ByteBpeTokenizer), String> {
    let file = TokenFile::open(path).map_err(|e| e.to_string())?;
    let tok_path = tokenizer.ok_or(
        "training from a token file needs --tokenizer <tok.gguf>, the file `tokenize` used",
    )?;
    if char_level {
        return Err(
            "a token file is BPE-tokenized; --vocab 0 (character-level) cannot be used".into(),
        );
    }
    let bytes =
        std::fs::read(tok_path).map_err(|e| format!("cannot read tokenizer {tok_path}: {e}"))?;
    let sha = sha256_hex(&bytes);
    if sha != file.tokenizer_sha256() {
        return Err(format!(
            "{tok_path} is not the tokenizer {path} was made with (sha256 {sha}, the token file \
             recorded {}); re-run `tokenize` with it",
            file.tokenizer_sha256()
        ));
    }
    let tok = ByteBpeTokenizer::from_gguf_bytes(&bytes).map_err(|e| e.to_string())?;
    if tok.vocab_size() != file.vocab_size() {
        return Err(format!(
            "{tok_path} has a vocabulary of {} but {path} was made with {}",
            tok.vocab_size(),
            file.vocab_size()
        ));
    }
    Ok((file, tok))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::{tokenize, TokenizeOptions};
    use ferrum_core::{StopCondition, DOC_SEP};

    #[test]
    fn opens_with_the_right_tokenizer_and_refuses_the_wrong_one() {
        let d = std::env::temp_dir().join(format!("ferrum_tokcorpus_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let text = (0..8)
            .map(|i| format!("story {i} of the small brown fox"))
            .collect::<Vec<_>>()
            .join(DOC_SEP);
        std::fs::write(d.join("c.txt"), &text).unwrap();
        let tok = ByteBpeTokenizer::train(&text, 300).unwrap();
        let other = ByteBpeTokenizer::train("a completely different corpus of words", 280).unwrap();
        let s = |n: &str| d.join(n).to_string_lossy().into_owned();
        std::fs::write(s("tok.gguf"), tok.to_gguf_bytes()).unwrap();
        std::fs::write(s("other.gguf"), other.to_gguf_bytes()).unwrap();
        tokenize(
            &TokenizeOptions {
                input: s("c.txt"),
                tokenizer: s("tok.gguf"),
                out: s("c.tok.gguf"),
                doc_sep: DOC_SEP.into(),
                stop: StopCondition::none(),
            },
            &mut |_| {},
        )
        .unwrap();
        let (file, t) = open_token_file(&s("c.tok.gguf"), Some(&s("tok.gguf")), false).unwrap();
        assert_eq!(t.vocab_size(), file.vocab_size());
        let err = |tok: Option<&str>, chars: bool| {
            open_token_file(&s("c.tok.gguf"), tok, chars).err().unwrap()
        };
        assert!(err(None, false).contains("needs --tokenizer"));
        assert!(err(Some(&s("tok.gguf")), true).contains("--vocab 0"));
        assert!(err(Some(&s("other.gguf")), false).contains("is not the tokenizer"));
        assert!(err(Some(&s("nope.gguf")), false).starts_with("cannot read tokenizer "));
        assert!(open_token_file(&s("c.txt"), Some(&s("tok.gguf")), false).is_err());
        let _ = std::fs::remove_dir_all(d);
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_tokcorpus_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn token_corpus_rejects_non_token_gguf() {
        let d = tmp("nontok");
        let tok = ByteBpeTokenizer::train("a small corpus of words to learn from", 280).unwrap();
        let p = d.join("tok.gguf");
        std::fs::write(&p, tok.to_gguf_bytes()).unwrap();
        // A tokenizer file is a GGUF, but not a ferrum-tokens one.
        let e = open_token_file(p.to_str().unwrap(), p.to_str(), false)
            .err()
            .unwrap();
        assert!(e.contains("ferrum-tokens"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_token_file_whose_vocabulary_differs_from_its_tokenizer_is_refused() {
        let d = tmp("vocab");
        let tok = ByteBpeTokenizer::train("a small corpus of words to learn from", 280).unwrap();
        let bytes = tok.to_gguf_bytes();
        let p = d.join("tok.gguf");
        std::fs::write(&p, &bytes).unwrap();
        // The right tokenizer sha256, but a recorded vocabulary one larger than the tokenizer's.
        let meta = ferrum_core::TokenFileMeta {
            vocab_size: tok.vocab_size() + 1,
            eos_id: None,
            tokenizer_sha256: sha256_hex(&bytes),
        };
        let tokens = d.join("c.tok.gguf");
        ferrum_core::write_token_file(tokens.to_str().unwrap(), &meta, |s| s.push(&[1, 2, 3]))
            .unwrap();
        let e = open_token_file(tokens.to_str().unwrap(), p.to_str(), false)
            .err()
            .unwrap();
        assert!(e.contains("has a vocabulary of"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
