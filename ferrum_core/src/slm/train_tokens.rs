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
        run_training(
            &mut net,
            tokens,
            cfg,
            resolve_threads(threads),
            rng,
            |_, ep, loss| {
                progress_callback(ep, loss);
                Ok(true)
            },
        )?;
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
        progress_callback: F,
    ) -> Result<Self>
    where
        F: FnMut(&ValidationProgress),
    {
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
