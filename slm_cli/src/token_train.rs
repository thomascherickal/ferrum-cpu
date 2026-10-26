//! `train <tokens.gguf>`: open a pre-tokenized file and the tokenizer it was made with.

use crate::args::Args;
use crate::util::sha256_hex;
use ferrum_core::{ByteBpeTokenizer, TokenFile};

/// Open `path` as a token file and load `--tokenizer`, checking that the file's recorded
/// tokenizer sha256 and vocabulary match it. Training needs the tokenizer anyway (it is stored
/// in the model), so a missing or different one is an error here, before any training time.
pub(crate) fn open(
    args: &Args,
    path: &str,
) -> Result<(TokenFile, ByteBpeTokenizer), Box<dyn std::error::Error>> {
    let file = TokenFile::open(path)?;
    let tok_path = args.value("tokenizer").ok_or(
        "training from a token file needs --tokenizer <tok.gguf>, the file `tokenize` used",
    )?;
    if args.get("vocab", 1usize) == 0 {
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
        )
        .into());
    }
    let tok = ByteBpeTokenizer::from_gguf_bytes(&bytes)?;
    if tok.vocab_size() != file.vocab_size() {
        return Err(format!(
            "{tok_path} has a vocabulary of {} but {path} was made with {}",
            tok.vocab_size(),
            file.vocab_size()
        )
        .into());
    }
    Ok((file, tok))
}

/// Seed text for `--sample`: the first few tokens of the corpus, decoded.
pub(crate) fn sample_seed(
    file: &mut TokenFile,
    tok: &ByteBpeTokenizer,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut ids = Vec::new();
    file.read_window(0, file.len().min(8), &mut ids)?;
    Ok(tok.decode(&ids.iter().map(|&i| i as usize).collect::<Vec<_>>()))
}
