//! `train_transformer tokenize` through the real binary: the ids in the token file are exactly
//! the ids `ByteBpeTokenizer::encode_docs` gives the same text, the way `train` tokenizes it.
use ferrum_core::{ByteBpeTokenizer, TokenFile, DOC_SEP};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::Command;

struct Dir(PathBuf);

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("ferrum_tokcli_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
    fn s(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_string()
    }
}

/// (success, stdout + stderr)
fn tt(args: &[&str]) -> (bool, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .unwrap();
    (
        o.status.success(),
        String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr),
    )
}

/// 40 short documents joined by the separator, with a multi-byte character and a CRLF.
fn corpus(n: usize) -> String {
    let words = [
        "the",
        "cat",
        "sat",
        "on",
        "a",
        "mat",
        "é",
        "dog\r\nran",
        "and",
        "then",
    ];
    let mut s = 7u64;
    (0..n)
        .map(|_| {
            (0..12)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    words[(s >> 33) as usize % words.len()]
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join(DOC_SEP)
}

/// Write a vocab-300 tokenizer trained on `text`; returns it with its file bytes.
fn tokenizer(d: &Dir, text: &str) -> (ByteBpeTokenizer, Vec<u8>) {
    let tok = ByteBpeTokenizer::train_with_docs(text, 300).unwrap();
    let bytes = tok.to_gguf_bytes();
    std::fs::write(d.s("tok.gguf"), &bytes).unwrap();
    (tok, bytes)
}

fn all_ids(path: &str) -> (TokenFile, Vec<usize>) {
    let mut f = TokenFile::open(path).unwrap();
    let (mut out, mut ids) = (Vec::new(), Vec::new());
    for start in (0..f.len()).step_by(1000) {
        f.read_window(start, 1000.min(f.len() - start), &mut out)
            .unwrap();
        ids.extend(out.iter().map(|&i| i as usize));
    }
    (f, ids)
}

#[test]
fn a_token_file_holds_exactly_the_ids_training_would_see() {
    let d = Dir::new("same");
    let text = corpus(40);
    std::fs::write(d.s("c.txt"), &text).unwrap();
    let (tok, tok_bytes) = tokenizer(&d, &text);
    let (ok, out) = tt(&[
        "tokenize",
        &d.s("c.txt"),
        "--tokenizer",
        &d.s("tok.gguf"),
        "--out",
        &d.s("t.tok.gguf"),
    ]);
    assert!(ok, "{out}");
    assert!(
        out.contains("tokenized") && out.contains("40 documents"),
        "{out}"
    );
    let want = tok.encode_docs(&text);
    let (f, got) = all_ids(&d.s("t.tok.gguf"));
    assert_eq!(got, want);
    assert_eq!(
        (f.len(), f.documents(), f.vocab_size()),
        (want.len(), 40, tok.vocab_size())
    );
    assert_eq!(f.eos_id(), tok.eos_id().map(|e| e as u32));
    let sha: String = Sha256::digest(&tok_bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(f.tokenizer_sha256(), sha);
}

#[test]
fn a_document_larger_than_the_read_chunk_still_matches() {
    let d = Dir::new("big");
    // About 1.3 MB in documents of ~20 KB each: separators fall on both sides of the 1 MiB chunk edge.
    let text = (0..64)
        .map(|i| corpus(900 + i))
        .collect::<Vec<_>>()
        .join(DOC_SEP);
    assert!(text.len() > (1 << 20) + 1000, "{}", text.len());
    std::fs::write(d.s("c.txt"), &text).unwrap();
    let (tok, _) = tokenizer(&d, &text[..200_000]);
    let (ok, out) = tt(&[
        "tokenize",
        &d.s("c.txt"),
        "--tokenizer",
        &d.s("tok.gguf"),
        "--out",
        &d.s("t.tok.gguf"),
    ]);
    assert!(ok, "{out}");
    assert_eq!(all_ids(&d.s("t.tok.gguf")).1, tok.encode_docs(&text));
}

#[test]
fn a_directory_is_its_files_in_order_joined_by_the_separator() {
    let d = Dir::new("dir");
    let (a, b) = (corpus(5), corpus(7));
    std::fs::create_dir_all(d.0.join("in/sub")).unwrap();
    std::fs::write(d.0.join("in/a.txt"), &a).unwrap();
    std::fs::write(d.0.join("in/sub/b.txt"), &b).unwrap();
    std::fs::write(d.0.join("in/.hidden"), "ignored").unwrap();
    let (tok, _) = tokenizer(&d, &format!("{a}{b}"));
    // The output sits inside the input directory: a second run must not read it back.
    let args = [
        "tokenize",
        &d.s("in"),
        "--tokenizer",
        &d.s("tok.gguf"),
        "--out",
        &d.s("in/out.tok.gguf"),
    ];
    for _ in 0..2 {
        let (ok, out) = tt(&args);
        assert!(ok, "{out}");
        assert_eq!(
            all_ids(&d.s("in/out.tok.gguf")).1,
            tok.encode_docs(&format!("{a}{DOC_SEP}{b}"))
        );
    }
}

#[test]
fn a_custom_separator_splits_documents_and_inserts_eos() {
    let d = Dir::new("sep");
    let text = "one two\n====\nthree four\n====\nfive";
    std::fs::write(d.s("c.txt"), text).unwrap();
    let (tok, _) = tokenizer(&d, text);
    let (ok, out) = tt(&[
        "tokenize",
        &d.s("c.txt"),
        "--tokenizer",
        &d.s("tok.gguf"),
        "--out",
        &d.s("t.tok.gguf"),
        "--doc-sep",
        "\n====\n",
    ]);
    assert!(ok, "{out}");
    let (f, got) = all_ids(&d.s("t.tok.gguf"));
    let want = tok.encode_docs(&text.replace("\n====\n", DOC_SEP));
    assert_eq!(got, want);
    assert_eq!(f.documents(), 3);
}

#[test]
fn bad_input_is_reported_and_leaves_no_output() {
    let d = Dir::new("err");
    let text = corpus(5);
    std::fs::write(d.s("c.txt"), &text).unwrap();
    let out_path = d.s("t.tok.gguf");
    let tok_path = d.s("tok.gguf");
    let (ok, m) = tt(&[
        "tokenize",
        &d.s("c.txt"),
        "--tokenizer",
        &tok_path,
        "--out",
        &out_path,
    ]);
    assert!(
        !ok && m.contains("cannot read tokenizer") && m.contains("--tokenizer"),
        "{m}"
    );
    tokenizer(&d, &text);
    let (ok, m) = tt(&[
        "tokenize",
        &d.s("c.txt"),
        "--tokenizer",
        &d.s("c.txt"),
        "--out",
        &out_path,
    ]);
    assert!(!ok && m.contains("same file"), "{m}");
    std::fs::write(d.s("junk.gguf"), b"not gguf").unwrap();
    let (ok, m) = tt(&[
        "tokenize",
        &d.s("c.txt"),
        "--tokenizer",
        &d.s("junk.gguf"),
        "--out",
        &out_path,
    ]);
    assert!(!ok && m.starts_with("error:"), "{m}");
    std::fs::write(d.s("bad.txt"), [b'a', 0xFF, b'b']).unwrap();
    let (ok, m) = tt(&[
        "tokenize",
        &d.s("bad.txt"),
        "--tokenizer",
        &tok_path,
        "--out",
        &out_path,
    ]);
    assert!(!ok && m.contains("invalid UTF-8 at byte 1"), "{m}");
    std::fs::write(d.s("empty.txt"), "").unwrap();
    let (ok, m) = tt(&[
        "tokenize",
        &d.s("empty.txt"),
        "--tokenizer",
        &tok_path,
        "--out",
        &out_path,
    ]);
    assert!(!ok && m.contains("empty"), "{m}");
    let (ok, m) = tt(&[
        "tokenize",
        &d.s("nope.txt"),
        "--tokenizer",
        &tok_path,
        "--out",
        &out_path,
    ]);
    assert!(!ok && m.contains("does not exist"), "{m}");
    assert!(!std::path::Path::new(&out_path).exists());
    assert!(!std::path::Path::new(&format!("{out_path}.tmp")).exists());
    // Missing flags and unknown flags.
    let (ok, m) = tt(&["tokenize", &d.s("c.txt")]);
    assert!(
        !ok && m.contains("usage: train_transformer tokenize"),
        "{m}"
    );
    let (ok, m) = tt(&["tokenize", &d.s("c.txt"), "--bogus", "1"]);
    assert!(
        !ok && m.contains("unknown flag --bogus for 'tokenize'"),
        "{m}"
    );
    let (ok, m) = tt(&["tokenize", "--help"]);
    assert!(
        ok && m.contains("--tokenizer <S>")
            && m.contains("--doc-sep <S>")
            && m.contains("--out <S>"),
        "{m}"
    );
}
