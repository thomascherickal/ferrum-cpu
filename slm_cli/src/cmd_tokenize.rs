//! `train_transformer tokenize`: parse the flags, run [`slm_cli::tokenize`], print the summary.

use crate::args::Args;
use slm_cli::tokenize::{tokenize, TokenizeOptions};

pub(crate) fn cmd_tokenize(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: train_transformer tokenize <corpus.txt|dir> --tokenizer <tok.gguf> --out <train.tok.gguf> [--doc-sep S]";
    let input = args.positional.first().ok_or(usage)?;
    let tok_path = args.value("tokenizer").ok_or(usage)?;
    let out = args.value("out").ok_or(usage)?;
    let t0 = std::time::Instant::now();
    let r = tokenize(
        &TokenizeOptions {
            input: input.to_string(),
            tokenizer: tok_path.to_string(),
            out: out.to_string(),
            doc_sep: args
                .value("doc-sep")
                .unwrap_or(ferrum_core::DOC_SEP)
                .to_string(),
            stop: ferrum_core::StopCondition::none(),
        },
        &mut |_| {},
    )?;
    println!(
        "tokenized {input} → {out}: {} tokens, {} documents, vocab {} ({} bytes in, {} bytes out, {:.1}s)",
        r.tokens,
        r.documents,
        r.vocab_size,
        r.bytes_in,
        r.bytes_out,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::TOKENIZE_FLAGS;

    fn parse(v: &[&str]) -> Args {
        let raw: Vec<String> = v.iter().map(|x| x.to_string()).collect();
        Args::parse_for("tokenize", &raw, TOKENIZE_FLAGS).unwrap()
    }

    #[test]
    fn cmd_tokenize_needs_input_tokenizer_and_out_then_writes_the_token_file() {
        for a in [
            parse(&["--tokenizer", "t", "--out", "o"]),
            parse(&["c.txt", "--out", "o"]),
            parse(&["c.txt", "--tokenizer", "t"]),
        ] {
            let e = cmd_tokenize(&a).unwrap_err().to_string();
            assert!(e.contains("usage: train_transformer tokenize"), "{e}");
        }
        let d = std::env::temp_dir().join(format!("ferrum_cmd_tokenize_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let text = "the quick brown fox<|endoftext|>jumps over the lazy dog";
        std::fs::write(d.join("c.txt"), text).unwrap();
        let tok = ferrum_core::ByteBpeTokenizer::train(text, 280).unwrap();
        std::fs::write(d.join("tok.gguf"), tok.to_gguf_bytes()).unwrap();
        let s = |n: &str| d.join(n).to_str().unwrap().to_string();
        cmd_tokenize(&parse(&[
            &s("c.txt"),
            "--tokenizer",
            &s("tok.gguf"),
            "--out",
            &s("c.tok.gguf"),
        ]))
        .unwrap();
        assert!(!ferrum_core::TokenFile::open(&s("c.tok.gguf"))
            .unwrap()
            .is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }
}
