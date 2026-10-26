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
