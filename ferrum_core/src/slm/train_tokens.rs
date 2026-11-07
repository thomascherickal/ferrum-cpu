//! Training from a pre-tokenized [`TokenFile`]: windows are streamed from disk, so the corpus
//! may be larger than RAM. The result is bit-identical to training on the same ids in memory
//! (same RNG draws, same window order), serial or threaded.
use crate::error::{InferError, Result};
use crate::rng::Rng;
use crate::slm::config::{TransformerConfig, ValidationConfig, ValidationProgress};
use crate::slm::corpus::LmTokens;
use crate::slm::training_loop::{prepare_net, run_training};
use crate::slm::GenerativeSLM;
use crate::token_file::TokenFile;
use crate::tokenizer::ByteBpeTokenizer;
use crate::train_transformer::Prefix;

/// What the model needs to know about the corpus (everything except the ids, which stay on
/// disk). Refuses a token file made with another tokenizer or one too short for a window.
fn describe(tokens: &TokenFile, tok: &ByteBpeTokenizer, context_len: usize) -> Result<LmTokens> {
    if tokens.vocab_size() != tok.vocab_size() {
        return Err(InferError::DimMismatch(format!(
            "the token file has a vocabulary of {} but the tokenizer has {}: it is not the \
             tokenizer the tokens were made with",
            tokens.vocab_size(),
            tok.vocab_size()
        )));
    }
    if tokens.len() < context_len + 1 {
        return Err(InferError::DimMismatch(
            "the token file must hold more tokens than the context window".into(),
        ));
    }
    Ok(LmTokens {
        tokens: Vec::new(),
        vocab_size: tok.vocab_size(),
        class_names: Vec::new(),
        tokenizer_state: tok.encode_state(),
    })
}

fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        crate::parallel::num_threads()
    } else {
        threads
    }
}

impl GenerativeSLM {
    /// [`GenerativeSLM::train_transformer_config_threaded`] over a [`TokenFile`] made with
    /// `tok` (`threads`: `0` = auto, `1` = serial). Only the windows of the current minibatch
    /// are in memory. The caller compares the tokenizer file's sha256 with
    /// [`TokenFile::tokenizer_sha256`]; here the vocabulary sizes must agree.
    pub fn train_transformer_tokens<F>(
        tokens: &mut TokenFile,
        tok: &ByteBpeTokenizer,
        cfg: &TransformerConfig,
        threads: usize,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, f32),
    {
        let tc = describe(tokens, tok, cfg.context_len)?;
        let mut net = prepare_net(cfg, tc.vocab_size, rng)?;
        run_training(&mut net, tokens, cfg, resolve_threads(threads), rng, &mut |_, ep, loss| { progress_callback(ep, loss); Ok(true) })?;
        Self::build_transformer_slm(&net, &tc)
    }

    /// [`GenerativeSLM::train_transformer_config_validated`] over a [`TokenFile`]. The tail
    /// `val.val_fraction` of the **tokens** (rounded, at least one) is held out and the head
    /// is trained on; the in-memory path splits the text by characters and refits the
    /// tokenizer on the head, which a fixed token file cannot do. The held-out tail is read
    /// into memory (8 bytes per token), so keep the fraction small for a huge corpus.
    pub fn train_transformer_tokens_validated<F>(
        tokens: &mut TokenFile,
        tok: &ByteBpeTokenizer,
        cfg: &TransformerConfig,
        threads: usize,
        val: &ValidationConfig,
        rng: &mut Rng,
        mut progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(&ValidationProgress),
    {
        Self::train_tokens_validated_dyn(
            tokens,
            tok,
            cfg,
            threads,
            val,
            rng,
            &mut progress_callback,
        )
    }

    /// [`GenerativeSLM::train_transformer_tokens_validated`] without the callback's type
    /// parameter: one compiled copy of the logic.
    fn train_tokens_validated_dyn(
        tokens: &mut TokenFile,
        tok: &ByteBpeTokenizer,
        cfg: &TransformerConfig,
        threads: usize,
        val: &ValidationConfig,
        rng: &mut Rng,
        progress_callback: &mut dyn FnMut(&ValidationProgress),
    ) -> Result<Self> {
        if !(val.val_fraction > 0.0 && val.val_fraction < 1.0) {
            return Err(InferError::DimMismatch(
                "val_fraction must be in the open interval (0, 1)".into(),
            ));
        }
        let tc = describe(tokens, tok, cfg.context_len)?;
        let total = tokens.len();
        let val_n = (((total as f32) * val.val_fraction).round() as usize).clamp(1, total - 1);
        let split = total - val_n;
        if split < cfg.context_len + 1 {
            return Err(InferError::DimMismatch(format!(
                "after holding out {val_n} validation tokens only {split} remain, fewer than the \
                 context window needs; lower --val"
            )));
        }
        let mut val_ids: Vec<usize> = Vec::with_capacity(val_n);
        let mut chunk = Vec::new();
        for start in (split..total).step_by(1 << 16) {
            tokens.read_window(start, (1 << 16).min(total - start), &mut chunk)?;
            val_ids.extend(chunk.iter().map(|&i| i as usize));
        }
        vprintln!(
            "[slm::train_tokens_validated] train={split} tokens, val={val_n} tokens, vocab={}, patience={}",
            tc.vocab_size,
            val.patience
        );
        let mut head = Prefix {
            inner: tokens,
            len: split,
        };
        Self::validated_loop(
            &tc,
            &mut head,
            cfg,
            resolve_threads(threads),
            val,
            rng,
            &|c| c.evaluate_ids(&val_ids),
            progress_callback,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::rng::Rng;
    use crate::slm::config::{TransformerConfig, ValidationConfig};
    use crate::slm::test_support::{ignore_progress, ignore_validation, t5_cfg};
    use crate::slm::GenerativeSLM;
    use crate::token_file::{write_token_file, TokenFile, TokenFileMeta};
    use crate::tokenizer::{ByteBpeTokenizer, DOC_SEP};

    /// A tokenizer, a token file of a small document corpus made with it, and the directory.
    struct Fixture {
        dir: std::path::PathBuf,
        tok: ByteBpeTokenizer,
        ids: Vec<u32>,
    }

    impl Fixture {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("ferrum_train_tokens_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let text = (0..30)
                .map(|i| format!("the cat sat on mat {i}. the dog ran to the log {}.", i % 5))
                .collect::<Vec<_>>()
                .join(DOC_SEP);
            let tok = ByteBpeTokenizer::train_with_docs(&text, 290).unwrap();
            let ids = tok.encode_docs(&text).iter().map(|&i| i as u32).collect();
            let f = Fixture { dir, tok, ids };
            f.write("t.tok.gguf", &f.ids);
            f
        }

        fn path(&self, name: &str) -> String {
            self.dir.join(name).to_str().unwrap().to_string()
        }

        fn write(&self, name: &str, ids: &[u32]) {
            let meta = TokenFileMeta {
                vocab_size: self.tok.vocab_size(),
                eos_id: self.tok.eos_id().map(|e| e as u32),
                tokenizer_sha256: "test".into(),
            };
            write_token_file(&self.path(name), &meta, |s| s.push(ids)).unwrap();
        }

        fn open(&self, name: &str) -> TokenFile {
            TokenFile::open(&self.path(name)).unwrap()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn cfg() -> TransformerConfig {
        TransformerConfig {
            context_len: 8,
            batch_size: 4,
            ..t5_cfg(2)
        }
    }

    #[test]
    fn train_tokens_auto_threads_match_an_explicit_count() {
        let f = Fixture::new("auto");
        let train = |threads: usize| {
            let mut file = f.open("t.tok.gguf");
            let mut losses = Vec::new();
            let slm = GenerativeSLM::train_transformer_tokens(
                &mut file,
                &f.tok,
                &cfg(),
                threads,
                &mut Rng::new(3),
                |ep, loss| losses.push((ep, loss.to_bits())),
            )
            .unwrap();
            (slm.to_bytes().unwrap(), losses)
        };
        let (auto, auto_losses) = train(0);
        let (explicit, explicit_losses) = train(crate::parallel::num_threads());
        assert_eq!(auto, explicit);
        assert_eq!(auto_losses, explicit_losses);
        assert_eq!(auto_losses.len(), 2);
    }

    #[test]
    fn train_tokens_validated_reports_each_epoch_and_returns_the_best() {
        let f = Fixture::new("val");
        let mut file = f.open("t.tok.gguf");
        let mut epochs = 0usize;
        let slm = GenerativeSLM::train_transformer_tokens_validated(
            &mut file,
            &f.tok,
            &cfg(),
            1,
            &ValidationConfig {
                val_fraction: 0.2,
                patience: 0,
            },
            &mut Rng::new(3),
            |p| epochs = p.epoch,
        )
        .unwrap();
        assert_eq!(epochs, 2);
        assert!(slm
            .evaluate_ids(&f.ids[..20].iter().map(|&i| i as usize).collect::<Vec<_>>())
            .is_ok());
    }

    #[test]
    fn train_tokens_rejects_vocab_mismatch() {
        let f = Fixture::new("mismatch");
        let other = ByteBpeTokenizer::train_with_docs("some other text. ".repeat(20).as_str(), 270)
            .unwrap();
        assert_ne!(other.vocab_size(), f.tok.vocab_size());
        let mut file = f.open("t.tok.gguf");
        let e = GenerativeSLM::train_transformer_tokens(
            &mut file,
            &other,
            &cfg(),
            1,
            &mut Rng::new(1),
            ignore_progress,
        )
        .err()
        .expect("must refuse");
        assert!(e.to_string().contains("not the tokenizer"), "{e}");
    }

    #[test]
    fn train_tokens_rejects_a_file_shorter_than_the_context() {
        let f = Fixture::new("short");
        f.write("short.tok.gguf", &f.ids[..8]);
        let mut file = f.open("short.tok.gguf");
        let e = GenerativeSLM::train_transformer_tokens(
            &mut file,
            &f.tok,
            &cfg(),
            1,
            &mut Rng::new(1),
            ignore_progress,
        )
        .err()
        .expect("must refuse");
        assert!(
            e.to_string().contains("more tokens than the context"),
            "{e}"
        );
    }

    #[test]
    fn train_tokens_validated_rejects_bad_splits_and_an_absurd_context() {
        let f = Fixture::new("badsplit");
        let run = |cfg: &TransformerConfig, fraction: f32| {
            let mut file = f.open("t.tok.gguf");
            GenerativeSLM::train_transformer_tokens_validated(
                &mut file,
                &f.tok,
                cfg,
                1,
                &ValidationConfig {
                    val_fraction: fraction,
                    patience: 0,
                },
                &mut Rng::new(1),
                ignore_validation,
            )
            .err()
            .expect("must fail")
            .to_string()
        };
        assert!(run(&cfg(), 0.0).contains("open interval"));
        assert!(run(&cfg(), 1.0).contains("open interval"));
        assert!(run(&cfg(), 0.999).contains("lower --val"));
        let big: Vec<u32> = f.ids.iter().copied().cycle().take(150_000).collect();
        f.write("big.tok.gguf", &big);
        let huge = TransformerConfig {
            context_len: 100_000,
            ..cfg()
        };
        let mut file = f.open("big.tok.gguf");
        let e = GenerativeSLM::train_transformer_tokens_validated(
            &mut file,
            &f.tok,
            &huge,
            1,
            &ValidationConfig {
                val_fraction: 0.1,
                patience: 0,
            },
            &mut Rng::new(1),
            ignore_validation,
        )
        .err()
        .expect("must refuse")
        .to_string();
        assert!(e.contains("training would need"), "{e}");
    }
}
