//! The Train tab: train a transformer SLM in-process with streamed progress.

use crate::jobs::{Job, JobGuard};
use crate::paths::checked_path;
use crate::AppState;
use ferrum_core::{validate_for_training, GenerativeSLM, Rng, TransformerConfig, ValidationConfig};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Runtime, State};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainParams {
    /// Always "transformer" (the only training method).
    pub method: String,
    pub corpus_path: String,
    pub model_path: String,
    pub context_len: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_blocks: usize,
    pub hidden_dim: usize,
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub vocab_size: usize,
    pub seed: u64,
    pub threads: usize,
    pub verbose: bool,
    /// Transformer pipeline options (see [`TransformerConfig`]); absent fields
    /// take the config defaults.
    #[serde(default)]
    pub window_stride: usize,
    #[serde(default)]
    pub max_tokens: u64,
    #[serde(default)]
    pub weight_decay: f32,
    #[serde(default)]
    pub dropout: f32,
    #[serde(default)]
    pub grad_clip: f32,
    #[serde(default)]
    pub cosine_lr: bool,
    #[serde(default)]
    pub warmup_steps: u64,
    #[serde(default)]
    pub weight_tying: bool,
    #[serde(default = "default_true")]
    pub qat: bool,
    #[serde(default)]
    pub tokenizer_path: String,
    #[serde(default)]
    pub checkpoint_path: String,
    #[serde(default)]
    pub checkpoint_every: u64,
    #[serde(default)]
    pub resume: bool,
    /// Held-out validation fraction (`0` = off); best epoch is kept.
    #[serde(default)]
    pub val_fraction: f32,
    #[serde(default)]
    pub patience: usize,
    /// "I know, try anyway": skip the engine's up-front refusal of a run that would not fit in RAM.
    #[serde(default)]
    pub force: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TrainResult {
    pub model_path: String,
    pub method: String,
    pub seconds: f64,
    pub final_loss: f32,
    pub input_dim: usize,
    pub output_dim: usize,
    pub tokenizer: String,
    pub layers: usize,
    pub bytes: u64,
    /// A cancel ended the run early; the model so far was saved.
    pub stopped: bool,
    /// Optimizer step at which the run stopped (`None` unless `stopped`).
    pub stopped_at_step: Option<u64>,
}

/// Train a transformer SLM in-process, streaming
/// `train-progress` events and (if `verbose`) `engine-log` lines, then save it.
/// `cancel_job("train")` ends the run after the current optimizer step; the
/// model trained so far is still saved (and checkpointed, if a checkpoint path
/// is set), exactly as `train --max-seconds` does.
#[tauri::command]
pub async fn train_slm<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    params: TrainParams,
) -> Result<TrainResult, String> {
    let job = state.jobs.start(Job::Train)?;
    tauri::async_runtime::spawn_blocking(move || train_inner(app, params, job))
        .await
        .map_err(|e| format!("task error: {e}"))?
}

/// Open the Train tab's corpus as a token file with its tokenizer (the CLI's checks: the tokenizer
/// is the one the file was made with, and the vocabulary is BPE).
fn open_token_corpus(
    p: &TrainParams,
) -> Result<(ferrum_core::TokenFile, ferrum_core::ByteBpeTokenizer), String> {
    let tok = p.tokenizer_path.trim();
    slm_cli::token_corpus::open_token_file(
        &p.corpus_path,
        Some(tok).filter(|t| !t.is_empty()),
        p.vocab_size == 0,
    )
    .map_err(|e| {
        e.replace(
            "training from a token file needs --tokenizer <tok.gguf>, the file `tokenize` used",
            "training from a token file needs the Tokenizer file field: the file Tokenize corpus used",
        )
        .replace("--vocab 0 (character-level)", "Vocab 0 (character-level)")
        .replace("re-run `tokenize` with it", "run Tokenize corpus again with it")
    })
}

fn train_inner<R: Runtime>(
    app: AppHandle<R>,
    p: TrainParams,
    job: JobGuard,
) -> Result<TrainResult, String> {
    // ---- shared validation with clear messages ----
    checked_path("output model path", &p.model_path, false)?;
    checked_path("corpus path", &p.corpus_path, true)?;
    for (label, v) in [
        ("tokenizer path", &p.tokenizer_path),
        ("checkpoint path", &p.checkpoint_path),
    ] {
        if !v.trim().is_empty() {
            checked_path(label, v, false)?;
        }
    }
    if p.context_len == 0 {
        return Err("context length must be ≥ 1".into());
    }
    if p.epochs == 0 && p.max_tokens == 0 {
        return Err("epochs must be ≥ 1".into());
    }
    if p.batch_size == 0 {
        return Err("batch size must be ≥ 1".into());
    }
    if p.hidden_dim == 0 {
        return Err("hidden width must be ≥ 1".into());
    }
    if !(p.lr.is_finite() && p.lr > 0.0) {
        return Err("learning rate must be a positive number".into());
    }
    // A GGUF corpus is a token file (`tokenize`), detected by content like the CLI does: it is
    // streamed from disk instead of read here.
    let mut token_input = if ferrum_core::is_gguf_file(&p.corpus_path) {
        Some(open_token_corpus(&p)?)
    } else {
        None
    };
    let corpus = if token_input.is_some() {
        String::new()
    } else {
        let corpus = std::fs::read_to_string(&p.corpus_path)
            .map_err(|e| format!("cannot read corpus {}: {e}", p.corpus_path))?;
        if corpus.trim().is_empty() {
            return Err(format!("corpus {} is empty", p.corpus_path));
        }
        validate_for_training(&corpus, p.context_len).map_err(|e| e.to_string())?;
        corpus
    };

    ferrum_core::set_verbose(p.verbose);

    // Capture the most recent loss for the result without changing the callback
    // signature.
    let last_loss = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let last_cb = last_loss.clone();
    let app_cb = app.clone();
    // A token budget has no known epoch count: `total` is null then.
    let total = (p.max_tokens == 0).then_some(p.epochs);
    let progress = move |epoch: usize, loss: f32| {
        last_cb.store(loss.to_bits(), std::sync::atomic::Ordering::Relaxed);
        let _ = app_cb.emit(
            "train-progress",
            serde_json::json!({ "epoch": epoch, "total": total, "loss": loss }),
        );
    };

    let mut rng = Rng::new(p.seed);
    let t0 = std::time::Instant::now();

    let slm = match p.method.as_str() {
        "transformer" => {
            if p.num_heads == 0 || !p.embed_dim.is_multiple_of(p.num_heads) {
                ferrum_core::set_verbose(false);
                return Err(format!(
                    "embedding dim ({}) must be divisible by heads ({})",
                    p.embed_dim, p.num_heads
                ));
            }
            if p.num_blocks == 0 {
                ferrum_core::set_verbose(false);
                return Err("transformer blocks must be ≥ 1".into());
            }
            if p.vocab_size != 0 && p.vocab_size < 256 {
                ferrum_core::set_verbose(false);
                return Err("vocab must be 0 (character-level) or ≥ 256 (byte-level BPE)".into());
            }
            let cfg = TransformerConfig {
                context_len: p.context_len,
                embed_dim: p.embed_dim,
                num_heads: p.num_heads,
                num_blocks: p.num_blocks,
                hidden_dim: p.hidden_dim,
                epochs: p.epochs,
                lr: p.lr,
                batch_size: p.batch_size,
                vocab_size: p.vocab_size,
                weight_decay: p.weight_decay,
                dropout: p.dropout,
                window_stride: p.window_stride,
                max_tokens: p.max_tokens,
                qat: p.qat,
                grad_clip: p.grad_clip,
                cosine_lr: p.cosine_lr,
                warmup_steps: p.warmup_steps,
                weight_tying: p.weight_tying,
                tokenizer_path: p.tokenizer_path.trim().to_string(),
                checkpoint_path: p.checkpoint_path.trim().to_string(),
                checkpoint_every: p.checkpoint_every,
                resume: p.resume,
                log_every_steps: 0,
                stop: job.stop().clone(),
                skip_memory_check: p.force,
                ..TransformerConfig::default()
            };
            let val = ValidationConfig {
                val_fraction: p.val_fraction,
                patience: p.patience,
            };
            match (token_input.as_mut(), p.val_fraction > 0.0) {
                (Some((file, tok)), true) => GenerativeSLM::train_transformer_tokens_validated(
                    file,
                    tok,
                    &cfg,
                    p.threads,
                    &val,
                    &mut rng,
                    |v| progress(v.epoch, v.train_loss),
                ),
                (Some((file, tok)), false) => GenerativeSLM::train_transformer_tokens(
                    file, tok, &cfg, p.threads, &mut rng, progress,
                ),
                (None, true) => GenerativeSLM::train_transformer_config_validated(
                    &corpus,
                    &cfg,
                    p.threads,
                    &val,
                    &mut rng,
                    |v| progress(v.epoch, v.train_loss),
                ),
                (None, false) => GenerativeSLM::train_transformer_config_threaded(
                    &corpus, &cfg, p.threads, &mut rng, progress,
                ),
            }
        }
        other => {
            ferrum_core::set_verbose(false);
            return Err(format!("unknown training method: {other}"));
        }
    }
    .map_err(|e| {
        ferrum_core::set_verbose(false);
        format!("training failed: {e}")
    })?;

    let seconds = t0.elapsed().as_secs_f64();
    slm.save(&p.model_path)
        .map_err(|e| format!("cannot save model {}: {e}", p.model_path))?;
    ferrum_core::set_verbose(false);

    let bytes = std::fs::metadata(&p.model_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let stopped = job.stop().stopped();
    let tokenizer = if slm.meta.tokenizer_state.is_empty() {
        "character-level".to_string()
    } else {
        "byte-level BPE".to_string()
    };
    let result = TrainResult {
        model_path: p.model_path.clone(),
        method: p.method.clone(),
        seconds,
        final_loss: f32::from_bits(last_loss.load(std::sync::atomic::Ordering::Relaxed)),
        input_dim: slm.meta.input_dim,
        output_dim: slm.meta.output_dim,
        tokenizer,
        layers: slm.model.len(),
        bytes,
        stopped: stopped.is_some(),
        stopped_at_step: stopped.map(|(_, step)| step),
    };
    let _ = app.emit("train-done", result.clone());
    Ok(result)
}
