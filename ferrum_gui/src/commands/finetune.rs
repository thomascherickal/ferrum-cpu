//! The Fine-tune tab: AdamW-fine-tune an imported GGUF into a `.flck` checkpoint.

use super::guard::{available_memory_bytes, gguf_resident_bytes, ram_guard};
use super::stop::stop_condition;
use crate::infra::error::CmdError;
use crate::infra::jobs::{Job, JobGuard};
use crate::infra::paths::checked_path;
use crate::AppState;
use ferrum_core::{
    Adam, FinetuneRun, Gguf, GgufTokenizer, LlamaTrainer, LrSchedule, Rng, SamplingParams,
    StopCause,
};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Runtime, State};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinetuneParams {
    pub model_path: String,
    pub corpus_path: String,
    pub out_path: String,
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub seq_len: usize,
    pub warmup: u64,
    pub clip: f32,
    pub weight_decay: f32,
    pub dropout: f32,
    pub qat: bool,
    pub seed: u64,
    pub threads: usize,
    pub resume: Option<String>,
    pub force: bool,
    pub verbose: bool,
    /// Stop after the optimizer step that crosses this many seconds (the CLI's `--max-seconds`).
    #[serde(default)]
    pub max_seconds: Option<f64>,
    /// Generate a sample from the tuned model afterwards (the CLI's `--sample`).
    #[serde(default)]
    pub sample: bool,
    /// Sample temperature (`--temp`), used with `sample`.
    #[serde(default = "default_temp")]
    pub temp: f32,
    /// Sample length in tokens (`--max`), used with `sample`.
    #[serde(default = "default_sample_tokens")]
    pub sample_tokens: usize,
}

fn default_temp() -> f32 {
    0.8
}

fn default_sample_tokens() -> usize {
    48
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FinetuneResult {
    pub out_path: String,
    pub epochs: usize,
    pub first_loss: f32,
    pub final_loss: f32,
    pub seconds: f64,
    pub tokens: usize,
    pub steps: u64,
    pub bytes: u64,
    /// A cancel ended the run early; the checkpoint so far was saved and resumes.
    pub stopped: bool,
    /// The `--sample` text (`None` unless asked for, and not after a Stop).
    pub sample: Option<String>,
}

/// AdamW-fine-tune an imported llama/qwen2 GGUF on a text corpus and write a
/// `.flck` checkpoint that `run_gguf { resume }` can overlay on the base model.
/// Per-epoch progress streams as `finetune-progress` events. `cancel_job("finetune")`
/// ends the run after the current optimizer step and still writes the checkpoint,
/// so it can be resumed.
#[tauri::command]
pub async fn finetune_gguf<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    params: FinetuneParams,
) -> Result<FinetuneResult, CmdError> {
    let job = state.jobs.start(Job::Finetune)?;
    crate::infra::error::blocking(move || finetune_inner(app, params, job)).await
}

/// Write `bytes` to `path` through `<path>.tmp` and a rename, so a failure never leaves a
/// truncated file at `path` (and never a stray `.tmp`).
fn write_atomic(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, bytes)
        .and_then(|()| std::fs::rename(&tmp, path))
        .inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })
}

fn finetune_inner<R: Runtime>(
    app: AppHandle<R>,
    p: FinetuneParams,
    job: JobGuard,
) -> Result<FinetuneResult, String> {
    checked_path("output checkpoint path (.flck)", &p.out_path, false)?;
    checked_path("model path", &p.model_path, true)?;
    checked_path("corpus path", &p.corpus_path, true)?;
    if let Some(r) = p.resume.as_deref().filter(|s| !s.trim().is_empty()) {
        checked_path("resume checkpoint path", r, true)?;
    }
    if p.epochs == 0 {
        return Err("epochs must be ≥ 1".into());
    }
    if !(p.lr.is_finite() && p.lr > 0.0) {
        return Err("learning rate must be a positive number".into());
    }
    // Before hours of training, not after (the CLI's checks and messages).
    let stop = stop_condition(job.stop(), p.max_seconds)?;
    if p.sample && !(p.temp.is_finite() && p.temp >= 0.0) {
        return Err(format!(
            "--temp must be a finite number >= 0 (0 = greedy), got {}",
            p.temp
        ));
    }
    ferrum_core::set_verbose(p.verbose);
    fn clean<T, E: std::fmt::Display>(r: Result<T, E>, prefix: &str) -> Result<T, String> {
        if r.is_err() {
            ferrum_core::set_verbose(false);
        }
        crate::infra::fmt_err(r, prefix)
    }
    let cleanup = || ferrum_core::set_verbose(false);

    // 1. Open + f32-load the base model.
    let g = clean(
        Gguf::open(&p.model_path),
        &format!("cannot open {}", p.model_path),
    )?;
    let est = gguf_resident_bytes(&g, None); // f32
    let train_est = est.saturating_mul(4); // weights + grad + Adam m/v
    let hint = "fine-tune a smaller model or enable 'train anyway'";
    if let Err(e) = ram_guard(
        "training memory",
        "",
        train_est,
        available_memory_bytes(),
        p.force,
        hint,
    ) {
        cleanup();
        return Err(e);
    }
    let tok = GgufTokenizer::from_gguf(&g).map_err(|_| {
        cleanup();
        "this GGUF has no tokenizer; text fine-tuning needs one".to_string()
    })?;
    let model = clean(g.load_llama_prec(None), "cannot load model")?;

    // 2. Read + tokenize the corpus.
    let text = clean(
        std::fs::read_to_string(&p.corpus_path),
        &format!("cannot read corpus {}", p.corpus_path),
    )?;
    if text.trim().is_empty() {
        cleanup();
        return Err(format!("corpus {} is empty", p.corpus_path));
    }
    let tokens = corpus_tokens(&tok, &text);
    // The corpus identity the checkpoint records and a resume checks (the CLI's rule).
    let corpus_id = clean(
        slm_cli::hash::file_id(&p.corpus_path),
        &format!("cannot read corpus {}", p.corpus_path),
    )?;

    let seq = p.seq_len.min(model.cfg.context_len).max(2);
    let batch = p.batch_size.max(1);
    if tokens.len() < seq {
        cleanup();
        return Err(format!(
            "corpus tokenized to {} tokens but sequence length is {seq}; supply more text",
            tokens.len()
        ));
    }

    // 3. Build + configure the trainer.
    // ponytail: `load_llama_prec(None)` above yields f32 weights, the one thing `new` refuses.
    let mut tr = LlamaTrainer::new(model).expect("f32 weights");
    tr.set_optimizer(Adam::new(p.lr));
    tr.set_weight_decay(p.weight_decay.max(0.0));
    tr.set_dropout(p.dropout);
    tr.set_grad_clip(if p.clip > 0.0 { Some(p.clip) } else { None });
    tr.set_qat(p.qat);

    let num_windows = tokens.len() - seq + 1;
    let steps_per_epoch = num_windows.div_ceil(batch) as u64;

    // 4. Resume optimizer state if requested.
    let mut rng = if let Some(ckpt) = p.resume.as_ref().filter(|s| !s.trim().is_empty()) {
        let bytes = std::fs::read(ckpt).map_err(|e| {
            cleanup();
            format!("cannot read checkpoint {ckpt}: {e}")
        })?;
        tr.load_checkpoint_from_vec(bytes).map_err(|e| {
            cleanup();
            format!("cannot resume from {ckpt}: {e}")
        })?
    } else {
        Rng::new(p.seed)
    };
    // Refuses a resume on another corpus, batch or sequence length; notes changed lr settings
    // in the log. The returned step is where this run's first epoch began.
    let run = FinetuneRun {
        corpus_id: Some(corpus_id),
        batch,
        seq,
        lr: p.lr,
        warmup: p.warmup,
        clip: p.clip,
        weight_decay: p.weight_decay.max(0.0),
        dropout: p.dropout,
    };
    let base = tr.begin_run(&run, tokens.len()).map_err(|e| {
        cleanup();
        format!("cannot resume: {e}")
    })?;

    // The schedule spans the *cumulative* step timeline (after any resume), so a
    // resumed run does not start past total_steps — which would pin the LR at 0.
    if p.warmup > 0 {
        let total = base + steps_per_epoch * p.epochs as u64;
        tr.set_lr_schedule(Some(LrSchedule::warmup_cosine(
            p.lr,
            p.warmup,
            total.max(p.warmup + 1),
        )));
    }

    let threads = if p.threads == 0 {
        ferrum_core::num_threads()
    } else {
        p.threads
    };

    // 5. Train, streaming per-epoch progress.
    let t0 = std::time::Instant::now();
    let mut first_loss = f32::NAN;
    let mut final_loss = f32::NAN;
    let total_steps = base + steps_per_epoch * p.epochs as u64;
    let mut epochs_run = 0;
    let mut stopped = false;
    let mut cancelled = false;
    for e in 0..p.epochs {
        // A cancel ends the epoch after a step (never the last step of the run).
        let (loss, halted) =
            match tr.finetune_epoch_until(&tokens, seq, batch, &mut rng, threads, &mut |step| {
                let cause = if step < total_steps {
                    stop.should_stop()
                } else {
                    None
                };
                cancelled |= cause == Some(StopCause::Cancelled);
                cause.is_some()
            }) {
                Ok(r) => r,
                Err(err) => {
                    cleanup();
                    return Err(format!("fine-tuning failed: {err}"));
                }
            };
        epochs_run = e + 1;
        stopped = halted;
        if e == 0 {
            first_loss = loss;
        }
        final_loss = loss;
        let _ = app.emit(
            "finetune-progress",
            serde_json::json!({
                "epoch": e + 1, "total": p.epochs, "loss": loss, "ppl": loss.exp(),
            }),
        );
        if halted {
            break;
        }
    }
    let seconds = t0.elapsed().as_secs_f64();

    // Optional sample from the tuned model (not after a Stop), like the CLI.
    let sample = if p.sample && !cancelled {
        let mut prompt: Vec<usize> = tok.bos().into_iter().collect();
        if prompt.is_empty() {
            prompt.push(tokens[0]);
        }
        let stops: Vec<usize> = tok.eos().into_iter().collect();
        let mut grng = Rng::new(super::time_seed());
        let params = SamplingParams::with_temperature(p.temp);
        match tr
            .model
            .generate(&prompt, p.sample_tokens, &params, &stops, &mut grng)
        {
            Ok(out) => Some(tok.decode(&out)),
            Err(e) => {
                cleanup();
                return Err(format!("sample generation failed: {e}"));
            }
        }
    } else {
        None
    };

    // 6. Save the checkpoint.
    let bytes = tr.save_checkpoint(&rng);
    write_atomic(&p.out_path, &bytes).map_err(|e| {
        cleanup();
        format!("cannot write {}: {e}", p.out_path)
    })?;
    cleanup();

    let result = FinetuneResult {
        out_path: p.out_path.clone(),
        epochs: epochs_run,
        first_loss,
        final_loss,
        seconds,
        tokens: tokens.len(),
        steps: tr.step_count(),
        bytes: bytes.len() as u64,
        stopped,
        sample,
    };
    let _ = app.emit("finetune-done", result.clone());
    Ok(result)
}

/// The corpus token stream, identical to the CLI's: BOS only when the GGUF asks for it.
fn corpus_tokens(tok: &GgufTokenizer, text: &str) -> Vec<usize> {
    let mut tokens: Vec<usize> = tok.prefix_bos().into_iter().collect();
    tokens.extend(tok.encode(text));
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression (M14): the old code used `tok.bos()`, which prepended BOS even when the
    /// GGUF says `add_bos_token = false`; the CLI uses `prefix_bos()`.
    #[test]
    fn corpus_tokens_follow_add_bos_token_like_the_cli() {
        let tk = |flag: Option<bool>| {
            let t = crate::test_support::Tiny {
                add_bos: flag,
                ..Default::default()
            };
            let path = std::env::temp_dir().join(format!(
                "ferrum_gui_bos_{}_{:?}.gguf",
                std::process::id(),
                flag
            ));
            std::fs::write(&path, t.build()).unwrap();
            let g = Gguf::open(path.to_str().unwrap()).unwrap();
            let _ = std::fs::remove_file(&path);
            GgufTokenizer::from_gguf(&g).unwrap()
        };
        let off = tk(Some(false));
        assert_eq!(off.bos(), Some(0));
        assert_eq!(corpus_tokens(&off, "hi"), off.encode("hi"));
        let on = tk(Some(true));
        assert_eq!(corpus_tokens(&on, "hi")[0], 0);
        assert_eq!(corpus_tokens(&on, "hi").len(), on.encode("hi").len() + 1);
    }

    #[test]
    fn write_atomic_replaces_and_never_leaves_a_temp_file() {
        let dir = std::env::temp_dir().join(format!("ferrum_gui_atomic_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("c.flck").to_string_lossy().into_owned();
        write_atomic(&f, b"one").unwrap();
        write_atomic(&f, b"two!").unwrap();
        assert_eq!(std::fs::read(&f).unwrap(), b"two!");
        assert!(!std::path::Path::new(&format!("{f}.tmp")).exists());
        // A directory squatting on the temp name makes the write fail; the old file survives.
        std::fs::create_dir(format!("{f}.tmp")).unwrap();
        assert!(write_atomic(&f, b"three").is_err());
        assert_eq!(std::fs::read(&f).unwrap(), b"two!");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
