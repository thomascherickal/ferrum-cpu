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
    let mut bytes_in = 0u64;
    let stats = write_token_file(out, &meta, |sink| {
        bytes_in = encode_files(files, tok, eos, sep, stop, progress, &mut |ids| {
            sink.push(ids)
        })?;
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

/// The ids of `files` (see [`tokenize_files`]), handed to `push` one document at a time; the
/// seam lets a test stand in for the token file. Returns the bytes read.
fn encode_files(
    files: &[PathBuf],
    tok: &ByteBpeTokenizer,
    eos: usize,
    sep: &str,
    stop: &StopCondition,
    progress: &mut dyn FnMut(TokenizeProgress),
    push: &mut dyn FnMut(&[u32]) -> ferrum_core::Result<()>,
) -> ferrum_core::Result<u64> {
    let sizes: Vec<u64> = files
        .iter()
        .map(|p| std::fs::metadata(p).map_or(0, |m| m.len()))
        .collect();
    let bytes_total: u64 = sizes.iter().sum();
    let mut bytes_in = 0u64;
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
            // ponytail: ids are below the vocabulary size, which `write_token_file` caps at
            // i32::MAX, so the cast cannot truncate.
            ids.extend(tok.encode_docs(text).into_iter().map(|id| id as u32));
            tokens += ids.len() as u64;
            docs_seen += 1;
            if docs_seen % PROGRESS_EVERY == 0 {
                progress(report(bytes_in + offset, tokens));
            }
            push(&ids).map_err(|e| e.to_string())
        })
        .map_err(InferError::Format)?;
        bytes_in += total;
    }
    Ok(bytes_in)
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
    fn a_failing_write_ends_the_run_with_its_error() {
        let d = tmp("failwrite");
        let docs = docs(2);
        let text = d.join("c.txt");
        std::fs::write(&text, docs.join(DOC_SEP)).unwrap();
        let (_, tok) = tokenizer(&d, &docs.join(DOC_SEP));
        let eos = tok.eos_id().unwrap();
        let stop = StopCondition::none();
        let mut full = |_: &[u32]| Err(InferError::Io("disk full".into()));
        let r = encode_files(&[text], &tok, eos, DOC_SEP, &stop, &mut |_| {}, &mut full);
        let err = r.unwrap_err().to_string();
        assert_eq!(err, "model format error: i/o error: disk full");
        let _ = std::fs::remove_dir_all(d);
    }

    fn dummy_progress(_: TokenizeProgress) {}

    #[test]
    fn errors_name_the_problem_and_a_stop_writes_nothing() {
        dummy_progress(TokenizeProgress {
            file: 0,
            files: 0,
            path: String::new(),
            bytes_done: 0,
            bytes_total: 0,
            tokens: 0,
        });
        let d = tmp("errors");
        let docs = docs(4);
        let text = d.join("c.txt");
        std::fs::write(&text, docs.join(DOC_SEP)).unwrap();
        let (tokp, _) = tokenizer(&d, &docs.join(DOC_SEP));
        let o = options(&d, &text);

        let mut missing = options(&d, &text);
        missing.tokenizer = d.join("nope.gguf").to_string_lossy().into();
        assert!(tokenize(&missing, &mut dummy_progress)
            .unwrap_err()
            .starts_with("cannot read tokenizer "));
        let mut same = options(&d, &text);
        same.out = same.input.clone();
        assert!(tokenize(&same, &mut dummy_progress)
            .unwrap_err()
            .contains("are the same file"));
        let empty = d.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(tokenize(&options(&d, &empty), &mut dummy_progress)
            .unwrap_err()
            .starts_with("no files found under"));
        let mut unwrite = options(&d, &text);
        unwrite.out = "/nonexistent_dir_12345/out.tokens".into();
        assert!(tokenize(&unwrite, &mut dummy_progress).is_err());

        // Invalid UTF-8: the file and the byte.
        let bad = d.join("bad.txt");
        std::fs::write(&bad, b"fine text\xff").unwrap();
        let e = tokenize(&options(&d, &bad), &mut dummy_progress).unwrap_err();
        assert!(
            e.contains("bad.txt") && e.contains("invalid UTF-8 at byte 9"),
            "{e}"
        );
        assert!(!Path::new(&o.out).exists(), "a failed run writes nothing");

        let empty = d.join("empty.txt");
        std::fs::write(&empty, "").unwrap();
        let e = tokenize(&options(&d, &empty), &mut dummy_progress).unwrap_err();
        assert!(e.contains("corpus is empty"), "{e}");

        // A stop requested at the first document.
        o.stop
            .cancel_flag()
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(tokenize(&o, &mut dummy_progress).unwrap_err(), CANCELLED);
        assert!(!Path::new(&o.out).exists() && tokp.exists());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_tokenizer_without_eos_an_unreadable_tokenizer_and_a_missing_input_are_refused() {
        let d = tmp("noeos");
        let text = d.join("c.txt");
        std::fs::write(&text, "some text").unwrap();
        // A byte-level tokenizer has no <|endoftext|> id.
        std::fs::write(
            d.join("tok.gguf"),
            ByteBpeTokenizer::byte_level().to_gguf_bytes(),
        )
        .unwrap();
        let e = tokenize(&options(&d, &text), &mut dummy_progress).unwrap_err();
        assert!(e.contains("no <|endoftext|> id"), "{e}");
        // A file that is not a tokenizer.
        std::fs::write(d.join("tok.gguf"), b"not a gguf at all").unwrap();
        assert!(tokenize(&options(&d, &text), &mut dummy_progress).is_err());
        // A tokenizer that is fine, with an input that does not exist.
        tokenizer(&d, "some text to learn from");
        let e = tokenize(&options(&d, &d.join("missing.txt")), &mut dummy_progress).unwrap_err();
        assert!(e.contains("missing.txt"), "{e}");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_progress_report_comes_every_progress_every_documents() {
        let d = tmp("every");
        let n = PROGRESS_EVERY as usize + 1;
        let text = (0..n)
            .map(|i| format!("doc {i}"))
            .collect::<Vec<_>>()
            .join(DOC_SEP);
        let text_path = d.join("c.txt");
        std::fs::write(&text_path, &text).unwrap();
        tokenizer(&d, &text);
        let mut seen = Vec::new();
        let r = tokenize(&options(&d, &text_path), &mut |p| seen.push(p)).unwrap();
        assert_eq!(r.documents, n as u64);
        // The start of the file and the report at document PROGRESS_EVERY.
        assert_eq!(seen.len(), 2);
        assert!(seen[1].tokens > 0 && seen[1].bytes_done > 0);
        let _ = std::fs::remove_dir_all(d);
    }
}
