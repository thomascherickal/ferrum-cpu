//! `tokenize`: a corpus of any size -> a `ferrum-tokens` GGUF that `train` streams from disk.
//!
//! [`tokenize`] is the whole command without argument parsing or printing; the `tokenize`
//! subcommand and the GUI both call it, so they write the same ids.

use crate::hash::sha256_hex;
use crate::outputs::{check_distinct_paths, check_writable};
use crate::prepare::{expand_inputs, for_each_raw_doc};
use ferrum_core::{write_token_file, ByteBpeTokenizer, InferError, StopCondition, TokenFileMeta};
use sha2::Sha256;
use std::path::{Path, PathBuf};

pub struct TokenizeOptions {
    /// A file or a folder of text files.
    pub input: String,
    /// The tokenizer file (`ferrum-tokenizer` GGUF) the ids come from.
    pub tokenizer: String,
    /// The `ferrum-tokens` file to write.
    pub out: String,
    pub doc_sep: String,
    /// Polled between documents; a stop ends the run with no file written.
    pub stop: StopCondition,
}

/// Reported at the start of every input file and about every [`PROGRESS_EVERY`] documents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizeProgress {
    pub file: usize,
    pub files: usize,
    pub path: String,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub tokens: u64,
}

pub const PROGRESS_EVERY: u64 = 2048;

/// The message of a run ended by `TokenizeOptions::stop`.
pub const CANCELLED: &str = "tokenize cancelled; no file was written";

#[derive(Clone, Debug)]
pub struct TokenizeReport {
    pub tokens: u64,
    pub documents: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub vocab_size: usize,
    pub files: usize,
}

/// Encode `files` (in order) with `tok` and write the ids to `out`. Each file is streamed
/// document by document (memory is the largest document plus a 1 MiB chunk), and the ids are
/// exactly those of `tok.encode_docs` over the files joined with the separator: an EOS id
/// between every two documents, including across file boundaries. Returns
/// `(tokens, documents, bytes read, bytes written)`.
pub fn tokenize_files(
    files: &[PathBuf],
    tok: &ByteBpeTokenizer,
    tokenizer_sha256: String,
    sep: &str,
    out: &str,
    stop: &StopCondition,
    progress: &mut dyn FnMut(TokenizeProgress),
) -> Result<(u64, u64, u64, u64), String> {
    let eos = tok.eos_id().ok_or(
        "the tokenizer has no <|endoftext|> id; retrain it (a current `train` run always adds one)",
    )?;
    let meta = TokenFileMeta {
        vocab_size: tok.vocab_size(),
        eos_id: Some(eos as u32),
        tokenizer_sha256,
    };
    let sizes: Vec<u64> = files
        .iter()
        .map(|p| std::fs::metadata(p).map_or(0, |m| m.len()))
        .collect();
    let bytes_total: u64 = sizes.iter().sum();
    let mut bytes_in = 0u64;
    let stats = write_token_file(out, &meta, |sink| {
        let mut first = true;
        let mut ids: Vec<u32> = Vec::new();
        let (mut tokens, mut docs_seen) = (0u64, 0u64);
        for (i, path) in files.iter().enumerate() {
            let (mut hasher, mut total) = (Sha256::default(), 0u64);
            let report = |bytes_done: u64, tokens: u64| TokenizeProgress {
                file: i + 1,
                files: files.len(),
                path: path.to_string_lossy().into_owned(),
                bytes_done,
                bytes_total,
                tokens,
            };
            progress(report(bytes_in, tokens));
            for_each_raw_doc(path, sep, &mut hasher, &mut total, |raw, offset| {
                // ponytail: polled between documents, so one huge document is read to its end first.
                if stop.should_stop().is_some() {
                    return Err(CANCELLED.to_string());
                }
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
                tokens += ids.len() as u64;
                docs_seen += 1;
                if docs_seen % PROGRESS_EVERY == 0 {
                    progress(report(bytes_in + offset, tokens));
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
    })
    .map_err(|e| {
        let text = e.to_string();
        // The library wraps the callback's message; a stop keeps its own plain text.
        if text.ends_with(CANCELLED) {
            CANCELLED.to_string()
        } else {
            text
        }
    })?;
    Ok((stats.tokens, stats.documents, bytes_in, stats.bytes))
}

/// The `tokenize` command: check the paths, read the tokenizer, stream the corpus into `out`.
pub fn tokenize(
    o: &TokenizeOptions,
    progress: &mut dyn FnMut(TokenizeProgress),
) -> Result<TokenizeReport, String> {
    check_distinct_paths(&[
        ("the corpus", &o.input),
        ("--tokenizer", &o.tokenizer),
        ("--out", &o.out),
    ])?;
    let tok_bytes = std::fs::read(&o.tokenizer).map_err(|e| {
        format!(
            "cannot read tokenizer {}: {e} (create one with `train <corpus.txt> <model.gguf> --tokenizer {}`)",
            o.tokenizer, o.tokenizer
        )
    })?;
    let tok = ByteBpeTokenizer::from_gguf_bytes(&tok_bytes).map_err(|e| e.to_string())?;
    check_writable(&o.out)?;
    let files = expand_inputs(std::slice::from_ref(&o.input), Path::new(&o.out))
        .map_err(|e| e.to_string())?;
    if files.is_empty() {
        return Err(format!("no files found under {}", o.input));
    }
    let (tokens, documents, bytes_in, bytes_out) = tokenize_files(
        &files,
        &tok,
        sha256_hex(&tok_bytes),
        &o.doc_sep,
        &o.out,
        &o.stop,
        progress,
    )?;
    Ok(TokenizeReport {
        tokens,
        documents,
        bytes_in,
        bytes_out,
        vocab_size: tok.vocab_size(),
        files: files.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{TokenFile, DOC_SEP};

    fn tmp(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("ferrum_tokenize_lib_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn docs(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| format!("Document {i}: the quick brown fox jumps over the lazy dog."))
            .collect()
    }

    /// A tokenizer trained on `text`, written to `dir/tok.gguf`.
    fn tokenizer(dir: &Path, text: &str) -> (PathBuf, ByteBpeTokenizer) {
        let tok = ByteBpeTokenizer::train(text, 300).unwrap();
        let p = dir.join("tok.gguf");
        std::fs::write(&p, tok.to_gguf_bytes()).unwrap();
        (p, tok)
    }

    fn options(dir: &Path, input: &Path) -> TokenizeOptions {
        TokenizeOptions {
            input: input.to_string_lossy().into(),
            tokenizer: dir.join("tok.gguf").to_string_lossy().into(),
            out: dir.join("c.tok.gguf").to_string_lossy().into(),
            doc_sep: DOC_SEP.into(),
            stop: StopCondition::none(),
        }
    }

    #[test]
    fn the_ids_are_those_of_encode_docs_over_the_joined_documents() {
        let d = tmp("ids");
        let docs = docs(6);
        let joined = docs.join(DOC_SEP);
        let (_, tok) = tokenizer(&d, &joined);
        // Two files: the document boundary between them is an EOS like any other.
        std::fs::write(d.join("a.txt"), docs[..3].join(DOC_SEP)).unwrap();
        std::fs::write(d.join("b.txt"), docs[3..].join(DOC_SEP)).unwrap();
        let corpus = d.join("corpus");
        std::fs::create_dir_all(&corpus).unwrap();
        std::fs::rename(d.join("a.txt"), corpus.join("a.txt")).unwrap();
        std::fs::rename(d.join("b.txt"), corpus.join("b.txt")).unwrap();
        let mut seen = Vec::new();
        let r = tokenize(&options(&d, &corpus), &mut |p| seen.push(p)).unwrap();
        let want: Vec<usize> = tok.encode_docs(&joined);
        assert_eq!(
            (r.tokens as usize, r.files, r.vocab_size),
            (want.len(), 2, tok.vocab_size())
        );
        assert_eq!(r.documents, 6);
        let mut file = TokenFile::open(&options(&d, &corpus).out).unwrap();
        let mut got = Vec::new();
        file.read_window(0, file.len(), &mut got).unwrap();
        assert_eq!(got.iter().map(|&i| i as usize).collect::<Vec<_>>(), want);
        // One progress report at the start of each file, the second after the first file's bytes.
        assert_eq!(seen.len(), 2);
        assert_eq!((seen[0].file, seen[0].bytes_done), (1, 0));
        assert!(seen[1].bytes_done > 0 && seen[1].bytes_done < seen[1].bytes_total);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn errors_name_the_problem_and_a_stop_writes_nothing() {
        let d = tmp("errors");
        let docs = docs(4);
        let text = d.join("c.txt");
        std::fs::write(&text, docs.join(DOC_SEP)).unwrap();
        let (tokp, _) = tokenizer(&d, &docs.join(DOC_SEP));
        let o = options(&d, &text);

        let mut missing = options(&d, &text);
        missing.tokenizer = d.join("nope.gguf").to_string_lossy().into();
        assert!(tokenize(&missing, &mut |_| {})
            .unwrap_err()
            .starts_with("cannot read tokenizer "));
        let mut same = options(&d, &text);
        same.out = same.input.clone();
        assert!(tokenize(&same, &mut |_| {})
            .unwrap_err()
            .contains("are the same file"));
        let empty = d.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(tokenize(&options(&d, &empty), &mut |_| {})
            .unwrap_err()
            .starts_with("no files found under"));

        // Invalid UTF-8: the file and the byte.
        let bad = d.join("bad.txt");
        std::fs::write(&bad, b"fine text\xff").unwrap();
        let e = tokenize(&options(&d, &bad), &mut |_| {}).unwrap_err();
        assert!(
            e.contains("bad.txt") && e.contains("invalid UTF-8 at byte 9"),
            "{e}"
        );
        assert!(!Path::new(&o.out).exists(), "a failed run writes nothing");

        // A stop requested at the first document.
        o.stop
            .cancel_flag()
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(tokenize(&o, &mut |_| {}).unwrap_err(), CANCELLED);
        assert!(!Path::new(&o.out).exists() && tokp.exists());
        let _ = std::fs::remove_dir_all(d);
    }
}
