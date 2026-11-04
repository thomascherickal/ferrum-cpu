//! What kind of GGUF file a path holds, from its header: the detection behind
//! `train_transformer info`, shared with the GUI's Models tab so both describe
//! checkpoints, token files and tokenizer files the same way.

use ferrum_core::{ByteBpeTokenizer, Gguf, TokenFile, TransformerNet};

/// The kinds `info` tells apart.
#[derive(Debug, PartialEq)]
pub enum FileKind {
    /// A Ferrum model (`gpt2`, `ferrum`, no architecture) or a file that is not a
    /// readable GGUF: the model loader describes it or reports the precise error.
    Model,
    /// An imported model of this architecture (llama, qwen2, or one Ferrum cannot run).
    Imported(String),
    /// A file Ferrum writes that is not a model.
    Other(OtherFile),
}

/// A non-model Ferrum file: its `general.architecture`, a one-line description,
/// and the fields `info` prints, in order.
#[derive(Debug, PartialEq)]
pub struct OtherFile {
    pub architecture: &'static str,
    pub version: u32,
    pub kind: String,
    pub fields: Vec<(&'static str, String)>,
}

/// Detect the kind of the file at `path`. Only the non-model kinds are read further
/// (a checkpoint's header, a token file's header, a tokenizer file in full); an error
/// there is returned as text.
pub fn detect(path: &str) -> Result<FileKind, String> {
    let Ok(g) = Gguf::open(path) else {
        return Ok(FileKind::Model);
    };
    let other = |architecture, kind: String, fields| {
        Ok(FileKind::Other(OtherFile {
            architecture,
            version: g.version,
            kind,
            fields,
        }))
    };
    let read = || std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"));
    match g.architecture() {
        Some("gpt2" | "ferrum") | None => Ok(FileKind::Model),
        Some("ferrum-checkpoint") => {
            let c = TransformerNet::checkpoint_info(read()?).map_err(|e| e.to_string())?;
            other(
                "ferrum-checkpoint",
                "training checkpoint; continue it with train --resume".into(),
                vec![
                    ("Vocab", c.vocab.to_string()),
                    ("Context", c.context.to_string()),
                    ("Embed", c.embed.to_string()),
                    ("Heads", c.heads.to_string()),
                    ("Hidden", c.hidden.to_string()),
                    ("Blocks", c.blocks.to_string()),
                    ("Step", c.step.to_string()),
                ],
            )
        }
        Some("ferrum-tokens") => {
            let f = TokenFile::open(path).map_err(|e| e.to_string())?;
            other(
                "ferrum-tokens",
                format!("pre-tokenized corpus; train on it with train {path} ..."),
                vec![
                    ("Tokens", f.len().to_string()),
                    ("Documents", f.documents().to_string()),
                    ("Vocab", f.vocab_size().to_string()),
                    (
                        "EOS id",
                        f.eos_id().map_or("none".into(), |e| e.to_string()),
                    ),
                    ("Tokenizer", format!("sha256 {}", f.tokenizer_sha256())),
                ],
            )
        }
        Some("ferrum-tokenizer") => {
            let t = ByteBpeTokenizer::from_gguf_bytes(&read()?).map_err(|e| e.to_string())?;
            let merges = g
                .meta("tokenizer.ggml.merges")
                .and_then(|m| m.array_len())
                .unwrap_or(0);
            other(
                "ferrum-tokenizer",
                "BPE tokenizer file; reuse it with train --tokenizer".into(),
                vec![
                    ("Vocab", t.vocab_size().to_string()),
                    ("Merges", merges.to_string()),
                ],
            )
        }
        Some("ferrum-llama-checkpoint") => other(
            "ferrum-llama-checkpoint",
            "fine-tune checkpoint; apply it with run-gguf --resume".into(),
            vec![],
        ),
        Some(arch) => Ok(FileKind::Imported(arch.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{GgufBuilder, MetaValue};

    fn file(tag: &str, bytes: &[u8]) -> String {
        let p = std::env::temp_dir().join(format!("ferrum_kind_{tag}_{}", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn with_arch(arch: Option<&str>) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.meta("general.architecture", MetaValue::String(a.into()));
        }
        b.into_bytes()
    }

    #[test]
    fn models_imports_and_unreadable_files_are_told_apart_by_the_header() {
        for (tag, bytes, want) in [
            ("gpt2", with_arch(Some("gpt2")), FileKind::Model),
            ("ferrum", with_arch(Some("ferrum")), FileKind::Model),
            ("none", with_arch(None), FileKind::Model),
            ("junk", b"not a gguf".to_vec(), FileKind::Model),
            (
                "llama",
                with_arch(Some("llama")),
                FileKind::Imported("llama".into()),
            ),
            (
                "bert",
                with_arch(Some("bert")),
                FileKind::Imported("bert".into()),
            ),
        ] {
            let p = file(tag, &bytes);
            assert_eq!(detect(&p).unwrap(), want, "{tag}");
            let _ = std::fs::remove_file(p);
        }
        assert_eq!(detect("/no/such/file").unwrap(), FileKind::Model);
    }

    #[test]
    fn a_tokenizer_file_reports_its_vocab_and_merges() {
        let tok = ByteBpeTokenizer::byte_level();
        let p = file("tok", &tok.to_gguf_bytes());
        let FileKind::Other(o) = detect(&p).unwrap() else {
            panic!("not described")
        };
        assert_eq!(o.architecture, "ferrum-tokenizer");
        assert_eq!(o.version, 3);
        assert!(o.kind.contains("--tokenizer"), "{}", o.kind);
        assert_eq!(
            o.fields,
            [
                ("Vocab", tok.vocab_size().to_string()),
                ("Merges", "0".to_string())
            ]
        );
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn a_fine_tune_checkpoint_is_named_and_a_damaged_checkpoint_is_an_error() {
        let p = file("flck", &with_arch(Some("ferrum-llama-checkpoint")));
        let FileKind::Other(o) = detect(&p).unwrap() else {
            panic!("not described")
        };
        assert!(o.kind.contains("run-gguf --resume") && o.fields.is_empty());
        let _ = std::fs::remove_file(p);
        // A header that claims to be a checkpoint but lacks its keys, and the same for
        // a tokenizer file and a token file: the error is returned, not a panic.
        for arch in ["ferrum-checkpoint", "ferrum-tokenizer", "ferrum-tokens"] {
            let p = file(arch, &with_arch(Some(arch)));
            assert!(detect(&p).is_err(), "{arch}");
            let _ = std::fs::remove_file(p);
        }
    }
}
