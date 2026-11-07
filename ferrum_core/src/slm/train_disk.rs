//! Disk-streamed BPE training for text corpora of any size.
//!
//! Instead of loading the entire text file into a `String` in RAM, [`stream_text_to_token_file`]
//! and [`GenerativeSLM::train_transformer_from_path`] stream the corpus from disk in two bounded-
//! memory passes:
//! 1. Learn (or reuse from `cfg.tokenizer_path`) a [`ByteBpeTokenizer`] via
//!    [`ByteBpeTokenizer::train_from_file`].
//! 2. Stream-encode the text file through a [`BufReader`] into a `ferrum-tokens` [`TokenFile`] on
//!    disk, then train over its windows via [`GenerativeSLM::train_transformer_tokens`].

use crate::error::{InferError, Result};
use crate::rng::Rng;
use crate::slm::config::{TransformerConfig, ValidationConfig, ValidationProgress};
use crate::slm::GenerativeSLM;
use crate::token_file::{write_token_file, TokenFile, TokenFileMeta};
use crate::tokenizer::{ByteBpeTokenizer, DOC_SEP};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const BASE_VOCAB: usize = 256;

/// Byte-size threshold above which CLI and GUI training automatically stream a BPE text corpus
/// from disk through a temporary [`TokenFile`] instead of reading the whole file into RAM (8 MiB).
pub const STREAM_CORPUS_THRESHOLD: u64 = 8 * 1024 * 1024;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(1);

/// RAII guard that removes a temporary [`TokenFile`] when dropped.
pub(crate) struct TempTokenFile {
    path: PathBuf,
}

impl TempTokenFile {
    fn new(corpus_path: &Path) -> Self {
        let seq = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let stem = corpus_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("corpus");
        let name = format!(
            "ferrum_stream_{}_{}_{}.tok.gguf",
            stem,
            std::process::id(),
            seq
        );
        Self {
            path: std::env::temp_dir().join(name),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempTokenFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Stream-encode `reader` with `tok` (converting every [`DOC_SEP`] to `tok.eos_id()` when
/// registered) and pass encoded chunks to `push`. Bounded memory (`~64 KiB` buffer).
fn encode_reader_docs<R: BufRead>(
    mut reader: R,
    tok: &ByteBpeTokenizer,
    mut push: impl FnMut(&[u32]) -> Result<()>,
) -> Result<()> {
    let eos = tok.eos_id().map(|e| e as u32);
    let mut buf = String::new();
    let mut line = String::new();
    let mut ids = Vec::new();
    let mut emit_segment = |segment: &str| -> Result<()> {
        for (i, part) in segment.split(DOC_SEP).enumerate() {
            ids.clear();
            if i > 0 {
                if let Some(e) = eos {
                    ids.push(e);
                }
            }
            ids.extend(tok.encode(part).into_iter().map(|id| id as u32));
            if !ids.is_empty() {
                push(&ids)?;
            }
        }
        Ok(())
    };
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            if !buf.is_empty() {
                emit_segment(&buf)?;
            }
            break;
        }
        buf.push_str(&line);
        if buf.len() >= (1 << 16) && !line.trim_end_matches(['\r', '\n']).is_empty() {
            let split = if buf.ends_with('\n') {
                buf.len() - 1
            } else {
                buf.len()
            };
            emit_segment(&buf[..split])?;
            buf.drain(..split);
        }
    }
    Ok(())
}

/// Stream a text corpus at `corpus_path` from disk to build (or reuse from `cfg.tokenizer_path`)
/// a [`ByteBpeTokenizer`] and write a `ferrum-tokens` GGUF at `token_file_path` without holding
/// the full text or token array in RAM.
pub fn stream_text_to_token_file(
    corpus_path: &Path,
    token_file_path: &Path,
    cfg: &TransformerConfig,
) -> Result<(TokenFile, ByteBpeTokenizer)> {
    if cfg.vocab_size <= BASE_VOCAB {
        return Err(InferError::DimMismatch(format!(
            "disk-streamed training requires a BPE vocab_size >= {} (got {})",
            BASE_VOCAB + 1,
            cfg.vocab_size
        )));
    }
    let tok_path = Path::new(&cfg.tokenizer_path);
    let tok = if !cfg.tokenizer_path.is_empty() && tok_path.exists() {
        vprintln!("[slm::train_disk] reusing tokenizer {}", tok_path.display());
        ByteBpeTokenizer::from_gguf_bytes(&std::fs::read(tok_path)?)?
    } else {
        let tok = ByteBpeTokenizer::train_from_file(corpus_path, cfg.vocab_size)?;
        if !cfg.tokenizer_path.is_empty() {
            if let Some(parent) = tok_path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            crate::fs_atomic::atomic_write(&cfg.tokenizer_path, &tok.to_gguf_bytes())?;
        }
        tok
    };
    let meta = TokenFileMeta {
        vocab_size: tok.vocab_size(),
        eos_id: tok.eos_id().map(|e| e as u32),
        tokenizer_sha256: "streamed".into(),
    };
    let out_str = token_file_path.to_string_lossy();
    let file = std::fs::File::open(corpus_path)?;
    let reader = BufReader::with_capacity(1 << 20, file);
    write_token_file(&out_str, &meta, |sink| {
        encode_reader_docs(reader, &tok, |ids| sink.push(ids))
    })?;
    let tokens = TokenFile::open(&out_str)?;
    if tokens.len() < cfg.context_len + 1 {
        return Err(InferError::DimMismatch(
            "BPE-tokenized corpus must be longer than the context window".into(),
        ));
    }
    Ok((tokens, tok))
}

impl GenerativeSLM {
    /// Train a causal BPE Transformer by streaming `corpus_path` directly from disk into a
    /// temporary [`TokenFile`] (`threads`: `0` = auto, `1` = serial). Produces bit-identical
    /// weights to [`GenerativeSLM::train_transformer_config_threaded`] on the same text and seed
    /// while keeping memory bounded regardless of corpus size.
    pub fn train_transformer_from_path<F>(
        corpus_path: &Path,
        cfg: &TransformerConfig,
        threads: usize,
        rng: &mut Rng,
        progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let tmp = TempTokenFile::new(corpus_path);
        let (mut tokens, tok) = stream_text_to_token_file(corpus_path, tmp.path(), cfg)?;
        Self::train_transformer_tokens(&mut tokens, &tok, cfg, threads, rng, progress_callback)
    }

    /// Validated variant of [`GenerativeSLM::train_transformer_from_path`], holding out the tail
    /// `val.val_fraction` of the streamed [`TokenFile`] for validation and early stopping.
    pub fn train_transformer_from_path_validated<F>(
        corpus_path: &Path,
        cfg: &TransformerConfig,
        threads: usize,
        val: &ValidationConfig,
        rng: &mut Rng,
        progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(&ValidationProgress),
    {
        let tmp = TempTokenFile::new(corpus_path);
        let (mut tokens, tok) = stream_text_to_token_file(corpus_path, tmp.path(), cfg)?;
        Self::train_transformer_tokens_validated(
            &mut tokens,
            &tok,
            cfg,
            threads,
            val,
            rng,
            progress_callback,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{stream_text_to_token_file, TempTokenFile, STREAM_CORPUS_THRESHOLD};
    use crate::rng::Rng;
    use crate::slm::config::{TransformerConfig, ValidationConfig};
    use crate::slm::test_support::{ignore_progress, ignore_validation, t5_cfg};
    use crate::slm::GenerativeSLM;
    use crate::tokenizer::DOC_SEP;
    use std::path::PathBuf;

    struct DiskFixture {
        dir: PathBuf,
        corpus_path: PathBuf,
        corpus: String,
    }

    impl DiskFixture {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("ferrum_train_disk_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let corpus = (0..35)
                .map(|i| format!("the cat sat on mat {i}.\n\nthe dog ran to log {}.", i % 5))
                .collect::<Vec<_>>()
                .join(DOC_SEP);
            let corpus_path = dir.join("corpus.txt");
            std::fs::write(&corpus_path, &corpus).unwrap();
            Self {
                dir,
                corpus_path,
                corpus,
            }
        }
    }

    impl Drop for DiskFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn cfg() -> TransformerConfig {
        TransformerConfig {
            context_len: 8,
            batch_size: 4,
            vocab_size: 290,
            ..t5_cfg(2)
        }
    }

    #[test]
    fn stream_threshold_is_eight_mib() {
        assert_eq!(STREAM_CORPUS_THRESHOLD, 8 * 1024 * 1024);
    }

    #[test]
    fn train_from_path_is_bit_identical_to_in_memory_training() {
        let f = DiskFixture::new("bit_id");
        let c = cfg();
        let mem = GenerativeSLM::train_transformer_config_threaded(
            &f.corpus,
            &c,
            1,
            &mut Rng::new(42),
            ignore_progress,
        )
        .unwrap();
        let disk = GenerativeSLM::train_transformer_from_path(
            &f.corpus_path,
            &c,
            1,
            &mut Rng::new(42),
            ignore_progress,
        )
        .unwrap();
        assert_eq!(mem.to_bytes().unwrap(), disk.to_bytes().unwrap());
    }

    #[test]
    fn train_from_path_writes_and_reuses_tokenizer_path() {
        let f = DiskFixture::new("tok_reuse");
        let tok_path = f.dir.join("sub").join("tok.gguf");
        let c = TransformerConfig {
            tokenizer_path: tok_path.to_str().unwrap().to_string(),
            ..cfg()
        };
        let first = GenerativeSLM::train_transformer_from_path(
            &f.corpus_path,
            &c,
            1,
            &mut Rng::new(7),
            ignore_progress,
        )
        .unwrap();
        assert!(tok_path.exists(), "tokenizer should be saved on first run");
        let second = GenerativeSLM::train_transformer_from_path(
            &f.corpus_path,
            &c,
            1,
            &mut Rng::new(7),
            ignore_progress,
        )
        .unwrap();
        assert_eq!(first.to_bytes().unwrap(), second.to_bytes().unwrap());
    }

    #[test]
    fn train_from_path_validated_runs_and_cleans_up_temp_file() {
        let f = DiskFixture::new("val");
        let val = ValidationConfig {
            val_fraction: 0.2,
            patience: 1,
        };
        let mut epochs = 0usize;
        let slm = GenerativeSLM::train_transformer_from_path_validated(
            &f.corpus_path,
            &cfg(),
            1,
            &val,
            &mut Rng::new(11),
            |_| epochs += 1,
        )
        .unwrap();
        assert!(epochs >= 1);
        assert!(!slm.meta.tokenizer_state.is_empty());

        let tmp_path = {
            let tmp = TempTokenFile::new(&f.corpus_path);
            std::fs::write(tmp.path(), b"x").unwrap();
            assert!(tmp.path().exists());
            tmp.path().to_path_buf()
        };
        assert!(!tmp_path.exists(), "TempTokenFile must unlink on drop");
    }

    #[test]
    fn stream_text_to_token_file_handles_large_chunks_and_errors() {
        let f = DiskFixture::new("large_and_err");
        // Write > 64 KiB corpus so buffer-flush boundary in encode_reader_docs is exercised.
        let big_corpus = format!(
            "{}\n{}",
            "the cat sat on the mat and jumped over the moon.\n".repeat(1600),
            "final line without trailing newline"
        );
        let big_path = f.dir.join("big.txt");
        std::fs::write(&big_path, &big_corpus).unwrap();
        let tok_file = f.dir.join("big.tok.gguf");
        let (tf, tok) = stream_text_to_token_file(&big_path, &tok_file, &cfg()).unwrap();
        assert_eq!(tf.len(), tok.encode_docs(&big_corpus).len());

        // Error: vocab_size <= 256
        let bad_vocab = TransformerConfig {
            vocab_size: 0,
            ..cfg()
        };
        assert!(stream_text_to_token_file(&f.corpus_path, &tok_file, &bad_vocab).is_err());

        // Error: corpus too short for context window
        let short_path = f.dir.join("short.txt");
        std::fs::write(&short_path, "hi").unwrap();
        let long_ctx = TransformerConfig {
            context_len: 64,
            ..cfg()
        };
        assert!(stream_text_to_token_file(&short_path, &tok_file, &long_ctx).is_err());

        // Error: nonexistent file
        assert!(GenerativeSLM::train_transformer_from_path_validated(
            &f.dir.join("missing.txt"),
            &cfg(),
            1,
            &ValidationConfig {
                val_fraction: 0.2,
                patience: 0
            },
            &mut Rng::new(1),
            ignore_validation,
        )
        .is_err());
    }
}
