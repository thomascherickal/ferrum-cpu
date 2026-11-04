use super::*;
use ferrum_core::{ByteBpeTokenizer, TokenFile};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_gui_tok_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn docs(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("Story {i}: a small brown fox jumps over the lazy dog, again and again."))
        .collect()
}

/// A corpus folder (two files, one document boundary between them) and its tokenizer file.
fn setup(dir: &Path) -> (PathBuf, PathBuf, ByteBpeTokenizer, String) {
    let docs = docs(40);
    let joined = docs.join(DOC_SEP);
    let corpus = dir.join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::write(corpus.join("a.txt"), docs[..25].join(DOC_SEP)).unwrap();
    std::fs::write(corpus.join("b.txt"), docs[25..].join(DOC_SEP)).unwrap();
    let tok = ByteBpeTokenizer::train(&joined, 300).unwrap();
    let tokp = dir.join("tok.gguf");
    std::fs::write(&tokp, tok.to_gguf_bytes()).unwrap();
    (corpus, tokp, tok, joined)
}

fn params(corpus: &Path, tok: &Path, out: &Path) -> TokenizeParams {
    TokenizeParams {
        input: corpus.to_string_lossy().into(),
        tokenizer_path: tok.to_string_lossy().into(),
        out_path: out.to_string_lossy().into(),
        doc_sep: DOC_SEP.into(),
    }
}

fn ids(path: &Path) -> Vec<usize> {
    let mut f = TokenFile::open(&path.to_string_lossy()).unwrap();
    let mut got = Vec::new();
    f.read_window(0, f.len(), &mut got).unwrap();
    got.into_iter().map(|i| i as usize).collect()
}

#[test]
fn the_gui_writes_the_ids_the_tokenizer_and_the_cli_library_give() {
    let d = tmp("ids");
    let (corpus, tokp, tok, joined) = setup(&d);
    let out = d.join("c.tok.gguf");
    let mut seen = Vec::new();
    let r = do_tokenize(
        &params(&corpus, &tokp, &out),
        StopCondition::none(),
        &mut |p| seen.push(p),
    )
    .unwrap();
    // The library's own encoding of the same documents.
    let want = tok.encode_docs(&joined);
    assert_eq!(ids(&out), want);
    assert_eq!(
        (r.tokens as usize, r.documents, r.files, r.vocab_size),
        (want.len(), 40, 2, tok.vocab_size())
    );
    // The CLI's function gives the same file bytes and report.
    let cli_out = d.join("cli.tok.gguf");
    let cli = slm_cli::tokenize::tokenize(
        &slm_cli::tokenize::TokenizeOptions {
            input: corpus.to_string_lossy().into(),
            tokenizer: tokp.to_string_lossy().into(),
            out: cli_out.to_string_lossy().into(),
            doc_sep: DOC_SEP.into(),
            stop: StopCondition::none(),
        },
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(
        std::fs::read(&out).unwrap(),
        std::fs::read(&cli_out).unwrap()
    );
    assert_eq!(
        (r.tokens, r.documents, r.bytes_in, r.bytes_out),
        (cli.tokens, cli.documents, cli.bytes_in, cli.bytes_out)
    );
    assert!(
        r.summary.starts_with("tokenized ")
            && r.summary
                .contains(&format!("{} tokens, 40 documents", r.tokens)),
        "{}",
        r.summary
    );
    // Progress: the start of each file, then the second file's offset.
    assert_eq!(seen.len(), 2);
    assert_eq!((seen[0].file, seen[0].files, seen[0].bytes_done), (1, 2, 0));
    assert!(seen[1].bytes_done > 0 && seen[1].tokens > 0);
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn errors_use_the_guis_words_and_a_stop_writes_no_file() {
    let d = tmp("errors");
    let (corpus, tokp, _, _) = setup(&d);
    let out = d.join("c.tok.gguf");
    let run = |p: &TokenizeParams| do_tokenize(p, StopCondition::none(), &mut |_| {}).unwrap_err();

    let blank = TokenizeParams {
        input: "  ".into(),
        ..params(&corpus, &tokp, &out)
    };
    assert!(run(&blank).contains("add the text file or folder"));
    let missing = params(&d.join("nope"), &tokp, &out);
    assert!(run(&missing).contains("does not exist"));
    let no_tok = params(&corpus, &d.join("none.gguf"), &out);
    assert!(run(&no_tok).contains("does not exist"));
    assert!(run(&params(&corpus, &tokp, Path::new("/proc/x.gguf"))).contains("not allowed"));
    // A tokenizer file that is not one: the library's message, without the "model format error" prefix noise.
    let junk = d.join("junk.gguf");
    std::fs::write(&junk, b"not a gguf").unwrap();
    let e = run(&params(&corpus, &junk, &out));
    assert!(!e.starts_with("model format error"), "{e}");
    // Invalid UTF-8 names the file and the byte, and points at the Prepare tab.
    let bad = d.join("bad.txt");
    std::fs::write(&bad, b"fine text\xff").unwrap();
    let e = run(&params(&bad, &tokp, &out));
    assert!(
        e.contains("bad.txt") && e.contains("invalid UTF-8 at byte 9") && e.contains("Prepare tab"),
        "{e}"
    );
    assert!(!out.exists());
    // Output and input the same file.
    assert!(run(&params(&bad, &tokp, &bad)).contains("same file"));

    // A stop: nothing is written, and the message says so.
    let stop = StopCondition::none();
    stop.cancel_flag().store(true, Ordering::Relaxed);
    let e = do_tokenize(&params(&corpus, &tokp, &out), stop, &mut |_| {}).unwrap_err();
    assert_eq!(e, slm_cli::tokenize::CANCELLED);
    assert!(!out.exists());
    let _ = std::fs::remove_dir_all(d);
}
