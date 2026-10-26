//! `train <tokens.gguf>`: open a pre-tokenized file and the tokenizer it was made with.

use crate::args::Args;
use ferrum_core::{ByteBpeTokenizer, TokenFile};
use slm_cli::token_corpus::open_token_file;

/// [`open_token_file`] with the `--tokenizer` and `--vocab` flags.
pub(crate) fn open(
    args: &Args,
    path: &str,
) -> Result<(TokenFile, ByteBpeTokenizer), Box<dyn std::error::Error>> {
    Ok(open_token_file(
        path,
        args.value("tokenizer"),
        args.get("vocab", 1usize) == 0,
    )?)
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
