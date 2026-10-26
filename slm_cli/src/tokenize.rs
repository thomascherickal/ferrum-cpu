//! `tokenize`: a corpus of any size -> a `ferrum-tokens` GGUF that `train` streams from disk.

use crate::args::Args;
use crate::prepare::{expand_inputs, for_each_raw_doc};
use crate::util::sha256_hex;
use crate::validate::{check_distinct_paths, check_writable};
use ferrum_core::{write_token_file, ByteBpeTokenizer, InferError, TokenFileMeta};
use sha2::Sha256;
use std::path::{Path, PathBuf};

/// Encode `files` (in order) with `tok` and write the ids to `out`. Each file is streamed
/// document by document (memory is the largest document plus a 1 MiB chunk), and the ids are
/// exactly those of `tok.encode_docs` over the files joined with the separator: an EOS id
/// between every two documents, including across file boundaries. Returns
/// `(tokens, documents, bytes read, bytes written)`.
pub(crate) fn tokenize_files(
    files: &[PathBuf],
    tok: &ByteBpeTokenizer,
    tokenizer_sha256: String,
    sep: &str,
    out: &str,
) -> Result<(u64, u64, u64, u64), Box<dyn std::error::Error>> {
    let eos = tok.eos_id().ok_or(
        "the tokenizer has no <|endoftext|> id; retrain it (a current `train` run always adds one)",
    )?;
    let meta = TokenFileMeta {
        vocab_size: tok.vocab_size(),
        eos_id: Some(eos as u32),
        tokenizer_sha256,
    };
    let mut bytes_in = 0u64;
    let stats = write_token_file(out, &meta, |sink| {
        let mut first = true;
        let mut ids: Vec<u32> = Vec::new();
        for path in files {
            let (mut hasher, mut total) = (Sha256::default(), 0u64);
            for_each_raw_doc(path, sep, &mut hasher, &mut total, |raw, offset| {
                let text = std::str::from_utf8(raw).map_err(|e| {
                    format!(
                        "{}: invalid UTF-8 at byte {} (`prepare --lossy` repairs it)",
                        path.display(),
                        offset + e.valid_up_to() as u64
                    )
                })?;
                ids.clear();
                if !first {
                    ids.push(eos as u32);
                }
                first = false;
                for id in tok.encode_docs(text) {
                    ids.push(u32::try_from(id).map_err(|_| format!("token id {id} > u32"))?);
                }
                sink.push(&ids).map_err(|e| e.to_string())
            })
            .map_err(InferError::Format)?;
            bytes_in += total;
        }
        if sink.is_empty() {
            return Err(InferError::Format(
                "the corpus is empty: no tokens to write".into(),
            ));
        }
        Ok(())
    })?;
    Ok((stats.tokens, stats.documents, bytes_in, stats.bytes))
}

pub(crate) fn cmd_tokenize(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: train_transformer tokenize <corpus.txt|dir> --tokenizer <tok.gguf> --out <train.tok.gguf> [--doc-sep S]";
    let input = args.positional.first().ok_or(usage)?;
    let tok_path = args.value("tokenizer").ok_or(usage)?;
    let out = args.value("out").ok_or(usage)?;
    let sep = args.value("doc-sep").unwrap_or(ferrum_core::DOC_SEP);
    check_distinct_paths(&[
        ("the corpus", input),
        ("--tokenizer", tok_path),
        ("--out", out),
    ])?;
    let tok_bytes = std::fs::read(tok_path).map_err(|e| {
        format!(
            "cannot read tokenizer {tok_path}: {e} (create one with `train <corpus.txt> <model.gguf> --tokenizer {tok_path}`)"
        )
    })?;
    let tok = ByteBpeTokenizer::from_gguf_bytes(&tok_bytes)?;
    check_writable(out)?;
    let files = expand_inputs(std::slice::from_ref(input), Path::new(out))?;
    if files.is_empty() {
        return Err(format!("no files found under {input}").into());
    }
    let t0 = std::time::Instant::now();
    let (tokens, docs, bytes_in, bytes_out) =
        tokenize_files(&files, &tok, sha256_hex(&tok_bytes), sep, out)?;
    println!(
        "tokenized {input} → {out}: {tokens} tokens, {docs} documents, vocab {} ({bytes_in} bytes in, {bytes_out} bytes out, {:.1}s)",
        tok.vocab_size(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
