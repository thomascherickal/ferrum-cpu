use super::writer::header_bytes;
use super::*;
use crate::error::{InferError, Result};
use crate::gguf::{MetaValue, GGML_I32};
use crate::gguf_write::GgufBuilder;

fn scratch(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("ferrum_tokfile_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.join("t.tok.gguf").to_str().unwrap().to_string()
}

fn meta(vocab: usize, eos: Option<u32>) -> TokenFileMeta {
    TokenFileMeta {
        vocab_size: vocab,
        eos_id: eos,
        tokenizer_sha256: "ab".repeat(32),
    }
}

/// A deterministic id stream over `0..vocab`.
fn ids(n: usize, vocab: usize) -> Vec<u32> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as usize % vocab) as u32
        })
        .collect()
}

fn write(path: &str, v: &[u32], m: &TokenFileMeta) -> Result<TokenFileStats> {
    write_token_file(path, m, |sink| {
        // Pushed in uneven chunks, as a document stream would.
        v.chunks(7).try_for_each(|c| sink.push(c))
    })
}

#[test]
fn round_trips_in_both_widths_and_random_access_matches_the_source() {
    for (vocab, width) in [(300usize, 2usize), (32768, 2), (32769, 4), (50_000, 4)] {
        let path = scratch(&format!("rt{vocab}"));
        let v = ids(5000, vocab);
        let m = meta(vocab, Some(7));
        let st = write(&path, &v, &m).unwrap();
        assert_eq!(st.tokens, 5000);
        assert_eq!(st.bytes, std::fs::metadata(&path).unwrap().len());
        assert!(!std::path::Path::new(&format!("{path}.tmp")).exists());
        let mut f = TokenFile::open(&path).unwrap();
        assert_eq!(
            (f.len(), f.vocab_size(), f.eos_id()),
            (5000, vocab, Some(7))
        );
        assert_eq!(f.tokenizer_sha256(), m.tokenizer_sha256);
        assert_eq!(f.documents(), st.documents);
        assert_eq!(st.bytes, f.data_start + 5000 * width as u64);
        // Sliding, random and edge windows all equal the source slice.
        let mut out = Vec::new();
        let mut s = 99u64;
        let mut starts: Vec<(usize, usize)> =
            vec![(0, 5000), (0, 1), (4999, 1), (4983, 17), (123, 0)];
        for _ in 0..300 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let start = (s >> 33) as usize % 4990;
            starts.push((start, 1 + (s >> 20) as usize % 10));
        }
        for (start, len) in starts {
            f.read_window(start, len, &mut out).unwrap();
            assert_eq!(
                out,
                &v[start..start + len],
                "vocab {vocab} window {start}+{len}"
            );
        }
        assert!(f.read_window(4990, 11, &mut out).is_err(), "past the end");
        assert!(f.read_window(usize::MAX, 2, &mut out).is_err(), "overflow");
        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn counts_documents_by_eos() {
    let path = scratch("docs");
    // a EOS b b EOS (trailing EOS = a final empty document, like `encode_docs`).
    let st = write(&path, &[1, 9, 2, 3, 9], &meta(10, Some(9))).unwrap();
    assert_eq!(st.documents, 3);
    assert_eq!(TokenFile::open(&path).unwrap().documents(), 3);
    // No EOS in the tokenizer: one document. Empty: none.
    assert_eq!(
        write(&path, &[1, 2, 3], &meta(10, None)).unwrap().documents,
        1
    );
    let st = write(&path, &[], &meta(10, Some(9))).unwrap();
    assert_eq!((st.tokens, st.documents), (0, 0));
    let f = TokenFile::open(&path).unwrap();
    assert!(f.is_empty() && f.eos_id() == Some(9));
}

#[test]
fn header_length_does_not_depend_on_the_counts() {
    let m = meta(300, Some(1));
    assert_eq!(
        header_bytes(&m, 0, 0).len(),
        header_bytes(&m, u64::MAX, u64::MAX).len()
    );
}

#[test]
fn truncated_file_is_rejected_at_open() {
    let path = scratch("trunc");
    write(&path, &ids(1000, 300), &meta(300, None)).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    for cut in [1, 2, 500, bytes.len() - 100] {
        std::fs::write(&path, &bytes[..bytes.len() - cut.min(bytes.len() - 1)]).unwrap();
        assert!(TokenFile::open(&path).is_err(), "cut {cut}");
    }
    std::fs::write(&path, b"GGUF").unwrap();
    assert!(TokenFile::open(&path).is_err());
}

#[test]
fn other_gguf_kinds_and_bad_metadata_are_rejected() {
    let path = scratch("arch");
    let put = |b: GgufBuilder| std::fs::write(&path, b.into_bytes()).unwrap();
    let mut b = GgufBuilder::new();
    b.meta("general.architecture", MetaValue::String("gpt2".into()));
    put(b);
    let e = TokenFile::open(&path).unwrap_err().to_string();
    assert!(e.contains("not a ferrum-tokens file"), "{e}");

    // token_count says 5, the tensor holds 6.
    let ok = |count: u64, dims: u64, vocab: u32| {
        let mut b = GgufBuilder::new();
        b.meta(
            "general.architecture",
            MetaValue::String(TOKEN_FILE_ARCH.into()),
        )
        .meta("ferrum.tokens.version", MetaValue::U32(1))
        .meta("ferrum.tokens.vocab_size", MetaValue::U32(vocab))
        .meta("ferrum.tokens.token_count", MetaValue::U64(count))
        .meta("ferrum.tokens.document_count", MetaValue::U64(1))
        .meta(
            "ferrum.tokens.tokenizer_sha256",
            MetaValue::String("x".into()),
        );
        b.tensor(
            "tokens",
            &[dims],
            crate::gguf::GGML_I16,
            vec![0; dims as usize * 2],
        );
        b
    };
    put(ok(6, 6, 300));
    TokenFile::open(&path).unwrap();
    put(ok(5, 6, 300));
    assert!(TokenFile::open(&path)
        .unwrap_err()
        .to_string()
        .contains("must be 1-D [5]"));
    put(ok(6, 6, 0));
    assert!(TokenFile::open(&path)
        .unwrap_err()
        .to_string()
        .contains("vocab_size"));
    // I16 data for a vocabulary that needs I32.
    put(ok(6, 6, 40_000));
    assert!(TokenFile::open(&path).is_err());
    // Missing metadata key.
    let mut b = GgufBuilder::new();
    b.meta(
        "general.architecture",
        MetaValue::String(TOKEN_FILE_ARCH.into()),
    );
    put(b);
    assert!(TokenFile::open(&path)
        .unwrap_err()
        .to_string()
        .contains("missing"));
    // An I32 tensor of the right type is also fine when the vocabulary is large.
    let mut b = GgufBuilder::new();
    b.meta(
        "general.architecture",
        MetaValue::String(TOKEN_FILE_ARCH.into()),
    )
    .meta("ferrum.tokens.version", MetaValue::U32(1))
    .meta("ferrum.tokens.vocab_size", MetaValue::U32(40_000))
    .meta("ferrum.tokens.token_count", MetaValue::U64(2))
    .meta("ferrum.tokens.document_count", MetaValue::U64(1))
    .meta(
        "ferrum.tokens.tokenizer_sha256",
        MetaValue::String("x".into()),
    );
    b.tensor("tokens", &[2], GGML_I32, vec![0; 8]);
    put(b);
    TokenFile::open(&path).unwrap();
}

#[test]
fn ids_outside_the_vocabulary_are_rejected_on_write_and_on_read() {
    let path = scratch("range");
    let e = write(&path, &[1, 2, 300, 4], &meta(300, None))
        .unwrap_err()
        .to_string();
    assert!(e.contains("token 2 is id 300"), "{e}");
    assert!(!std::path::Path::new(&path).exists());
    assert!(!std::path::Path::new(&format!("{path}.tmp")).exists());
    // An error from the id source aborts the write too and is passed through.
    let e = write_token_file(&path, &meta(300, None), |sink| {
        sink.push(&[1])?;
        Err(InferError::Io("disk gone".into()))
    })
    .unwrap_err();
    assert_eq!(e, InferError::Io("disk gone".into()));
    assert!(!std::path::Path::new(&path).exists());
    // Bad meta.
    assert!(write(&path, &[], &meta(0, None)).is_err());
    assert!(write(&path, &[], &meta(10, Some(10))).is_err());

    // A corrupt file: the last id overwritten with 0xFFFF.
    write(&path, &[1, 2, 3, 4], &meta(300, None)).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let n = bytes.len();
    bytes[n - 2..].copy_from_slice(&[0xFF, 0xFF]);
    std::fs::write(&path, bytes).unwrap();
    let mut f = TokenFile::open(&path).unwrap();
    let mut out = Vec::new();
    f.read_window(0, 3, &mut out).unwrap();
    let e = f.read_window(0, 4, &mut out).unwrap_err().to_string();
    assert!(e.contains("token 3 is id 65535"), "{e}");
}

#[test]
fn the_reader_streams_it_holds_one_window_not_the_tokens() {
    let path = scratch("mem");
    let n = 200_000;
    write(&path, &ids(n, 300), &meta(300, None)).unwrap();
    let mut f = TokenFile::open(&path).unwrap();
    assert!(std::mem::size_of::<TokenFile>() < 256);
    assert_eq!(f.buffered_bytes(), 0, "nothing is cached at open");
    let mut out = Vec::new();
    for start in [0, 777, 199_000, n - 17] {
        f.read_window(start, 17, &mut out).unwrap();
    }
    assert_eq!(
        f.buffered_bytes(),
        17 * 2,
        "one window of I16 ids, not {n} tokens"
    );
    assert!(out.capacity() < 64);
}

/// A token-file GGUF with `version`, optionally without the tokenizer hash, carrying `tensors`
/// (each `[2]` I16 ids) instead of the single `tokens` one.
fn raw_token_file(version: u32, with_sha: bool, tensors: &[&str]) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.meta(
        "general.architecture",
        MetaValue::String(TOKEN_FILE_ARCH.into()),
    )
    .meta("ferrum.tokens.version", MetaValue::U32(version))
    .meta("ferrum.tokens.vocab_size", MetaValue::U32(300))
    .meta("ferrum.tokens.token_count", MetaValue::U64(2))
    .meta("ferrum.tokens.document_count", MetaValue::U64(1));
    if with_sha {
        b.meta(
            "ferrum.tokens.tokenizer_sha256",
            MetaValue::String("x".into()),
        );
    }
    for name in tensors {
        b.tensor(name, &[2], crate::gguf::GGML_I16, vec![0; 4]);
    }
    b.into_bytes()
}

#[test]
fn token_file_reader_rejects_each_malformed_header() {
    let path = scratch("hdr");
    let open_err = |bytes: Vec<u8>| {
        std::fs::write(&path, bytes).unwrap();
        TokenFile::open(&path).unwrap_err().to_string()
    };
    std::fs::write(&path, raw_token_file(1, true, &["tokens"])).unwrap();
    TokenFile::open(&path).unwrap();
    let e = open_err(raw_token_file(2, true, &["tokens"]));
    assert!(e.contains("unsupported token file version"), "{e}");
    let e = open_err(raw_token_file(1, false, &["tokens"]));
    assert!(e.contains("tokenizer_sha256"), "{e}");
    for tensors in [&[][..], &["tokens", "extra"], &["ids"]] {
        let e = open_err(raw_token_file(1, true, tensors));
        assert!(e.contains("expected exactly one 'tokens' tensor"), "{e}");
    }
}

#[test]
fn token_file_reader_wrong_dtype_and_truncated_tensor_fail_at_open() {
    let path = scratch("dtype");
    // I32 ids for a vocabulary that needs I16.
    let mut b = GgufBuilder::new();
    b.meta(
        "general.architecture",
        MetaValue::String(TOKEN_FILE_ARCH.into()),
    )
    .meta("ferrum.tokens.version", MetaValue::U32(1))
    .meta("ferrum.tokens.vocab_size", MetaValue::U32(300))
    .meta("ferrum.tokens.token_count", MetaValue::U64(2))
    .meta("ferrum.tokens.document_count", MetaValue::U64(1))
    .meta(
        "ferrum.tokens.tokenizer_sha256",
        MetaValue::String("x".into()),
    );
    b.tensor("tokens", &[2], GGML_I32, vec![0; 8]);
    std::fs::write(&path, b.into_bytes()).unwrap();
    assert!(TokenFile::open(&path)
        .unwrap_err()
        .to_string()
        .contains("must be 1-D [2] of type"));
    // The directory is intact but the ids were cut off: the GGUF extent check catches it, so the
    // reader never needs a check of its own.
    let mut bytes = raw_token_file(1, true, &["tokens"]);
    bytes.truncate(bytes.len() - 2);
    std::fs::write(&path, bytes).unwrap();
    let e = TokenFile::open(&path).unwrap_err().to_string();
    assert!(e.contains("runs past EOF"), "{e}");
}

#[test]
fn is_gguf_file_sniffs_magic_and_version() {
    let path = scratch("sniff");
    let p = |bytes: &[u8]| {
        std::fs::write(&path, bytes).unwrap();
        is_gguf_file(&path)
    };
    assert!(p(&raw_token_file(1, true, &["tokens"])));
    assert!(p(b"GGUF\x02\x00\x00\x00"));
    assert!(!p(b"GGUF\x01\x00\x00\x00"), "unsupported version");
    assert!(!p(b"GGUF\x03\x00"), "shorter than the sniffed head");
    assert!(!p(b"plain text corpus, not a model"));
    assert!(!is_gguf_file(&format!("{path}.missing")));
}

#[test]
fn the_sink_counts_what_it_was_given() {
    let path = scratch("sink");
    write_token_file(&path, &meta(10, None), |sink| {
        assert!(sink.is_empty());
        sink.push(&[1, 2, 3])?;
        assert_eq!((sink.len(), sink.is_empty()), (3, false));
        Ok(())
    })
    .unwrap();
}
