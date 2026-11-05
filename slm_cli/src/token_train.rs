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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{args, Tmp};
    use slm_cli::tokenize::{tokenize, TokenizeOptions};

    const TEXT: &str = "the small brown fox jumps over the lazy dog and the small cat";

    /// A tokenizer file and the token file made with it, in `t`.
    fn token_file(t: &Tmp, text: &str) -> (String, String) {
        let tok = ByteBpeTokenizer::train(TEXT, 280).unwrap();
        let tok_path = t.write("tok.gguf", tok.to_gguf_bytes());
        let corpus = t.write("c.txt", text);
        let out = t.path("c.tok.gguf");
        tokenize(
            &TokenizeOptions {
                input: corpus,
                tokenizer: tok_path.clone(),
                out: out.clone(),
                doc_sep: ferrum_core::DOC_SEP.into(),
                stop: ferrum_core::StopCondition::none(),
            },
            &mut |_| {},
        )
        .unwrap();
        (tok_path, out)
    }

    #[test]
    fn open_reads_tokenizer_and_vocab_flags() {
        let t = Tmp::new("tokentrain_open");
        let (tok, file) = token_file(&t, TEXT);
        let (f, bpe) = open(
            &args("train", &[&file, "m.gguf", "--tokenizer", &tok]),
            &file,
        )
        .unwrap();
        assert_eq!(bpe.vocab_size(), f.vocab_size());
        let e = open(&args("train", &[&file, "m.gguf"]), &file)
            .err()
            .unwrap();
        assert!(e.to_string().contains("needs --tokenizer"), "{e}");
        let e = open(
            &args(
                "train",
                &[&file, "m.gguf", "--tokenizer", &tok, "--vocab", "0"],
            ),
            &file,
        )
        .err()
        .unwrap();
        assert!(e.to_string().contains("--vocab 0"), "{e}");
    }

    #[test]
    fn sample_seed_decodes_the_first_tokens_of_the_corpus() {
        let t = Tmp::new("tokentrain_seed");
        let (tok, file) = token_file(&t, TEXT);
        let a = args("train", &[&file, "m.gguf", "--tokenizer", &tok]);
        let (mut f, bpe) = open(&a, &file).unwrap();
        assert!(f.len() > 8);
        let seed = sample_seed(&mut f, &bpe).unwrap();
        assert!(!seed.is_empty() && TEXT.starts_with(&seed), "{seed:?}");
        assert!(seed.len() < TEXT.len(), "only the first 8 tokens");
        // A corpus shorter than 8 tokens is decoded whole.
        let short = Tmp::new("tokentrain_short");
        let (tok, file) = token_file(&short, "the fox");
        let (mut f, bpe) = open(
            &args("train", &[&file, "m.gguf", "--tokenizer", &tok]),
            &file,
        )
        .unwrap();
        assert!(f.len() < 8);
        assert!(sample_seed(&mut f, &bpe).unwrap().starts_with("the fox"));
    }
}
