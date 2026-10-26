//! Tauri commands: the bridge between the webview and `ferrum_core`.
//!
//! Conventions:
//! - Every command returns `Result<T, String>` so failures surface as a clear,
//!   human-readable message in the GUI (requirement: clear error messages).
//! - Heavy/blocking work runs inside `spawn_blocking` so the UI thread stays
//!   responsive and events can stream while the command runs.

use crate::jobs::{Job, JobGuard};
use crate::paths::checked_path;
use crate::AppState;
use ferrum_core::{
    clean_corpus, corpus_stats, validate_for_training, Adam, CleanOptions, GenerativeSLM, Gguf,
    GgufTokenizer, LlamaTrainer, LrSchedule, QKind, Rng, SamplingParams, TransformerConfig,
    ValidationConfig,
};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Runtime, State};

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE)
        | 1
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Datasets — download + clean (requirement #1)
// ─────────────────────────────────────────────────────────────────────────────

/// HTTP client with bounded connect/read timeouts so a hung or slow host can
/// never block a download task indefinitely (G2).
pub(crate) fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout_read(std::time::Duration::from_secs(30))
        .timeout_write(std::time::Duration::from_secs(30))
        .build()
}

/// Synchronous core of [`download_text`]: validate the URL, fetch with timeouts,
/// and return at most `cap` bytes decoded as UTF-8. Separated from the Tauri
/// command so it is unit-testable without an `AppHandle`.
fn fetch_text(url: &str, cap: usize) -> Result<String, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("URL must start with http:// or https://".to_string());
    }
    let cap = cap.max(1);
    let resp = http_agent()
        .get(url)
        .call()
        .map_err(|e| format!("download failed: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .take(cap as u64)
        .read_to_end(&mut buf)
        .map_err(|e| format!("read failed: {e}"))?;
    if buf.is_empty() {
        return Err("downloaded resource was empty".to_string());
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Download a text resource (e.g. a Project Gutenberg `.txt`) and return it as a
/// UTF-8 string (lossily decoded), capped at `max_bytes` (default 8 MiB). Uses an
/// HTTP client with connect/read timeouts so a slow host cannot hang the task.
#[tauri::command]
pub async fn download_text(url: String, max_bytes: Option<usize>) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let cap = max_bytes.unwrap_or(8 * 1024 * 1024);
        fetch_text(&url, cap)
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanOpts {
    pub strip_gutenberg: bool,
    pub lowercase: bool,
    pub collapse_whitespace: bool,
    pub normalize_punctuation: bool,
    pub strip_control_chars: bool,
    pub max_chars: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanResult {
    pub cleaned: String,
    pub preview: String,
    pub chars: usize,
    pub bytes: usize,
    pub lines: usize,
    pub words: usize,
    pub unique_chars: usize,
}

/// Clean raw text into a training-ready corpus and report statistics. Pure;
/// does not touch disk.
#[tauri::command]
pub fn clean_text(raw: String, opts: CleanOpts) -> Result<CleanResult, String> {
    let options = CleanOptions {
        strip_gutenberg: opts.strip_gutenberg,
        lowercase: opts.lowercase,
        collapse_whitespace: opts.collapse_whitespace,
        normalize_punctuation: opts.normalize_punctuation,
        strip_control_chars: opts.strip_control_chars,
        max_chars: opts.max_chars.filter(|&n| n > 0),
    };
    let cleaned = clean_corpus(&raw, &options);
    if cleaned.trim().is_empty() {
        return Err("nothing left after cleaning — check the source and options".into());
    }
    let st = corpus_stats(&cleaned);
    let preview: String = cleaned.chars().take(4000).collect();
    Ok(CleanResult {
        preview,
        chars: st.chars,
        bytes: st.bytes,
        lines: st.lines,
        words: st.words,
        unique_chars: st.unique_chars,
        cleaned,
    })
}

/// Write a corpus to disk (creating parent directories), returning bytes written.
#[tauri::command]
pub fn save_corpus(text: String, path: String) -> Result<usize, String> {
    checked_path("output file path", &path, false)?;
    if let Some(parent) = Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {parent:?}: {e}"))?;
        }
    }
    std::fs::write(&path, text.as_bytes()).map_err(|e| format!("cannot write {path}: {e}"))?;
    Ok(text.len())
}

/// Read a UTF-8 text file (lossy), capped at `max_bytes` (default 16 MiB).
#[tauri::command]
pub fn read_text_file(path: String, max_bytes: Option<usize>) -> Result<String, String> {
    // Devices and FIFOs are refused up front; the cap still bounds huge regular files.
    checked_path("file path", &path, true)?;
    let cap = max_bytes.unwrap_or(16 * 1024 * 1024) as u64;
    let file = std::fs::File::open(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut data = Vec::new();
    file.take(cap)
        .read_to_end(&mut data)
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    Ok(String::from_utf8_lossy(&data).into_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Training (requirement #6) — transformer
// ─────────────────────────────────────────────────────────────────────────────

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
    let corpus = std::fs::read_to_string(&p.corpus_path)
        .map_err(|e| format!("cannot read corpus {}: {e}", p.corpus_path))?;
    if corpus.trim().is_empty() {
        return Err(format!("corpus {} is empty", p.corpus_path));
    }
    validate_for_training(&corpus, p.context_len).map_err(|e| e.to_string())?;

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
                ..TransformerConfig::default()
            };
            if p.val_fraction > 0.0 {
                let val = ValidationConfig {
                    val_fraction: p.val_fraction,
                    patience: p.patience,
                };
                GenerativeSLM::train_transformer_config_validated(
                    &corpus,
                    &cfg,
                    p.threads,
                    &val,
                    &mut rng,
                    |v| progress(v.epoch, v.train_loss),
                )
            } else {
                GenerativeSLM::train_transformer_config_threaded(
                    &corpus, &cfg, p.threads, &mut rng, progress,
                )
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

// ─────────────────────────────────────────────────────────────────────────────
// 3. Generation with streaming (requirement #6)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenParams {
    pub model_path: String,
    pub seed_text: String,
    pub num_chars: usize,
    /// Softmax temperature; `0` is exact argmax (greedy), like `--temp 0`.
    pub temp: f32,
    pub gen_seed: Option<u64>,
    pub stream: bool,
    pub verbose: bool,
    /// Keep only the `k` highest-probability tokens (`0` disables).
    #[serde(default)]
    pub top_k: usize,
    /// Nucleus sampling threshold (`1.0` disables).
    #[serde(default = "one_f32")]
    pub top_p: f32,
    /// Down-weight already-seen tokens (`1.0` disables).
    #[serde(default = "one_f32")]
    pub rep_penalty: f32,
    /// Token budget (`0` = none: `num_chars` counts characters, as before).
    #[serde(default)]
    pub max_tokens: usize,
    /// Stop strings, one per entry; generation ends before the first one that
    /// appears. `\n`, `\t` and `\\` are unescaped (a line cannot hold a real newline).
    #[serde(default)]
    pub stop: Vec<String>,
}

/// The sampler settings shared by the Generate and GGUF tabs, validated like
/// the CLI's `sampling_params` (`--temp >= 0`, `--top-p` in [0, 1], a positive
/// repetition penalty).
fn sampling_from(
    temp: f32,
    top_k: usize,
    top_p: f32,
    rep_penalty: f32,
) -> Result<SamplingParams, String> {
    if !(temp.is_finite() && temp >= 0.0) {
        return Err(format!(
            "temperature must be a number >= 0 (0 = greedy), got {temp}"
        ));
    }
    if !(top_p.is_finite() && (0.0..=1.0).contains(&top_p)) {
        return Err(format!("top-p must be a number in [0, 1], got {top_p}"));
    }
    if !(rep_penalty.is_finite() && rep_penalty > 0.0) {
        return Err(format!(
            "repetition penalty must be a positive number, got {rep_penalty}"
        ));
    }
    Ok(SamplingParams {
        temperature: temp,
        top_k,
        top_p,
        repetition_penalty: rep_penalty,
    })
}

/// Stop strings as typed (one per line): empty lines dropped, `\n` `\t` `\\` unescaped.
fn parse_stops(raw: &[String]) -> Vec<String> {
    raw.iter()
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut out = String::new();
            let mut it = s.chars();
            while let Some(c) = it.next() {
                if c != '\\' {
                    out.push(c);
                    continue;
                }
                match it.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('\\') => out.push('\\'),
                    Some(o) => {
                        out.push('\\');
                        out.push(o);
                    }
                    None => out.push('\\'),
                }
            }
            out
        })
        .collect()
}

/// Carried by `resume_unwind` out of the engine's text callback to end a cancelled generation.
struct Cancelled;

/// Generate exactly as `train_transformer generate` does for the same flags:
/// a token budget or stop strings select the token path (`generate_tokens_with`,
/// stop text excluded), otherwise `num_chars` characters. Returns seed +
/// continuation; `on_text` receives only the continuation, as it is produced.
fn generate_once(
    slm: &GenerativeSLM,
    p: &GenParams,
    on_text: &mut dyn FnMut(&str),
) -> Result<String, String> {
    let params = sampling_from(p.temp, p.top_k, p.top_p, p.rep_penalty)?;
    let stops = parse_stops(&p.stop);
    let mut rng = Rng::new(p.gen_seed.unwrap_or_else(time_seed));
    let out = if p.max_tokens > 0 || !stops.is_empty() {
        let max_tokens = if p.max_tokens > 0 {
            p.max_tokens
        } else {
            p.num_chars
        };
        slm.generate_tokens_with(&p.seed_text, max_tokens, &stops, &params, &mut rng, on_text)
            .map(|g| format!("{}{}", p.seed_text, g.text))
    } else {
        slm.generate_stream_with(&p.seed_text, p.num_chars, &params, &mut rng, |f| on_text(f))
    };
    out.map_err(|e| format!("generation failed: {e}"))
}

/// [`generate_once`] that ends as soon as `cancelled()` answers true after a fragment, returning
/// (seed + the text so far, true).
///
/// ponytail: the three decode loops in `ferrum_core` have no stop hook, so the callback unwinds
/// (`resume_unwind` runs no panic hook and prints nothing) and `catch_unwind` stops here; the
/// loops only borrow the model, so nothing is left half-built. Upgrade path: a cancel hook in
/// `generate_stream_core` and the stream types.
pub(crate) fn run_generation(
    slm: &GenerativeSLM,
    p: &GenParams,
    cancelled: &dyn Fn() -> bool,
    on_text: &mut dyn FnMut(&str),
) -> Result<(String, bool), String> {
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    let mut so_far = String::new();
    let run = catch_unwind(AssertUnwindSafe(|| {
        generate_once(slm, p, &mut |frag| {
            so_far.push_str(frag);
            on_text(frag);
            if cancelled() {
                resume_unwind(Box::new(Cancelled));
            }
        })
    }));
    match run {
        Ok(text) => text.map(|t| (t, false)),
        Err(payload) if payload.is::<Cancelled>() => Ok((format!("{}{so_far}", p.seed_text), true)),
        Err(payload) => resume_unwind(payload),
    }
}

/// Load a model and generate text. With `stream = true`, each fragment is
/// emitted as a `gen-fragment` event as it is produced; the full string is also
/// returned. `cancel_job("generate")` ends it after the current fragment and
/// returns the text produced so far.
#[tauri::command]
pub async fn generate_slm<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    params: GenParams,
) -> Result<String, String> {
    let job = state.jobs.start(Job::Generate)?;
    tauri::async_runtime::spawn_blocking(move || gen_inner(app, params, job))
        .await
        .map_err(|e| format!("task error: {e}"))?
}

fn gen_inner<R: Runtime>(app: AppHandle<R>, p: GenParams, job: JobGuard) -> Result<String, String> {
    sampling_from(p.temp, p.top_k, p.top_p, p.rep_penalty)?; // fail before the model loads
    checked_path("model path", &p.model_path, true)?;
    let slm = GenerativeSLM::load(&p.model_path)
        .map_err(|e| format!("cannot load model {}: {e}", p.model_path))?;

    ferrum_core::set_verbose(p.verbose);
    let cancelled = || job.cancelled();
    let result = if p.stream {
        run_generation(&slm, &p, &cancelled, &mut |frag| {
            let _ = app.emit("gen-fragment", frag.to_string());
        })
    } else {
        run_generation(&slm, &p, &cancelled, &mut |_| {})
    };
    ferrum_core::set_verbose(false);
    result.map(|(text, _stopped)| text)
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Evaluation (requirement #5) — perplexity table rows
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalParams {
    pub model_path: String,
    pub text: Option<String>,
    pub text_path: Option<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EvalRow {
    pub model: String,
    pub num_predictions: usize,
    pub cross_entropy: f32,
    pub bits_per_token: f32,
    pub perplexity: f32,
    pub uniform_baseline: f32,
}

/// Score a model against held-out text; returns one table row.
#[tauri::command]
pub async fn evaluate_slm(
    state: State<'_, AppState>,
    params: EvalParams,
) -> Result<EvalRow, String> {
    // One pass of the engine's scorer, which has no stop hook: it can be refused while another
    // evaluation runs, but not cancelled.
    let job = state.jobs.start(Job::Evaluate)?;
    tauri::async_runtime::spawn_blocking(move || {
        let _job = job;
        checked_path("model path", &params.model_path, true)?;
        if let Some(tp) = params.text_path.as_deref().filter(|s| !s.trim().is_empty()) {
            checked_path("held-out text path", tp, true)?;
        }
        let text = match (&params.text, &params.text_path) {
            (Some(t), _) if !t.trim().is_empty() => t.clone(),
            (_, Some(path)) if !path.trim().is_empty() => {
                std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?
            }
            _ => return Err("provide held-out text or a text file to evaluate against".into()),
        };
        let slm = GenerativeSLM::load(&params.model_path)
            .map_err(|e| format!("cannot load model {}: {e}", params.model_path))?;
        let ev = slm
            .evaluate(&text)
            .map_err(|e| format!("evaluation failed: {e}"))?;
        Ok(EvalRow {
            model: file_name(&params.model_path),
            num_predictions: ev.num_predictions,
            cross_entropy: ev.cross_entropy,
            bits_per_token: ev.bits_per_token,
            perplexity: ev.perplexity,
            uniform_baseline: slm.meta.output_dim as f32,
        })
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Model info / load / reload (requirement #6)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub path: String,
    pub bytes: u64,
    pub format: String,
    pub name: String,
    pub task: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub tokenizer: String,
    pub merges: usize,
    pub layers: usize,
}

/// Inspect a saved model file (mirrors `slm_cli info`).
#[tauri::command]
pub fn model_info(path: String) -> Result<ModelInfo, String> {
    checked_path("model path", &path, true)?;
    let raw = std::fs::read(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let slm = GenerativeSLM::from_bytes(&raw).map_err(|e| format!("not a valid model: {e}"))?;
    let m = &slm.meta;
    let (tokenizer, merges) = if m.tokenizer_state.is_empty() {
        (
            format!("character-level ({} chars)", m.class_names.len()),
            0,
        )
    } else {
        let merges = m
            .tokenizer_state
            .split(';')
            .filter(|s| !s.is_empty())
            .count();
        (format!("byte-level BPE ({} tokens)", m.output_dim), merges)
    };
    Ok(ModelInfo {
        path: path.clone(),
        bytes: raw.len() as u64,
        format: ferrum_core::loader::gguf_format(&raw)
            .map_err(|e| format!("not a valid model: {e}"))?,
        name: m.dataset_name.clone(),
        task: format!("{:?}", m.task),
        input_dim: m.input_dim,
        output_dim: m.output_dim,
        tokenizer,
        merges,
        layers: slm.model.len(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 5b. GGUF import & run (Llama/Qwen checkpoints)
// ─────────────────────────────────────────────────────────────────────────────

/// Map a UI quant string to an import precision (`None` = keep f32).
fn parse_quant(q: &str) -> Result<Option<QKind>, String> {
    match q {
        "int4" | "q4" => Ok(Some(QKind::Int4)),
        "int8" | "q8" => Ok(Some(QKind::Int8)),
        "f32" | "none" => Ok(None),
        other => Err(format!("quant must be int4 | int8 | f32 (got '{other}')")),
    }
}

/// Estimate resident bytes for a loaded model from the GGUF directory: the token
/// embedding stays f32, everything else packs to the chosen precision.
fn gguf_resident_bytes(g: &Gguf, prec: Option<QKind>) -> usize {
    let mut total = 0usize;
    for t in &g.tensors {
        let n = t.num_elements();
        let bytes = if t.name == "token_embd.weight" {
            n.saturating_mul(4)
        } else {
            match prec {
                Some(QKind::Int4) => n.div_ceil(2),
                Some(QKind::Int8) => n,
                None => n.saturating_mul(4),
            }
        };
        total = total.saturating_add(bytes);
    }
    total
}

/// Total elements across all tensors. Saturating: a hostile directory's
/// element counts must not overflow the sum (panic in debug, wrap in release).
fn param_count(g: &Gguf) -> u64 {
    g.tensors
        .iter()
        .fold(0u64, |acc, t| acc.saturating_add(t.num_elements() as u64))
}

/// Best-effort available RAM (Linux `/proc/meminfo`); `None` elsewhere.
fn available_memory_bytes() -> Option<usize> {
    // Cross-platform via sysinfo (0.33 reports bytes). The previous /proc/meminfo
    // reader returned None on macOS/Windows, silently disabling the RAM guard on
    // the very desktops the Tauri app also ships to.
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let avail = sys.available_memory();
    (avail > 0).then_some(avail as usize)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufSummary {
    pub path: String,
    pub bytes: u64,
    pub version: u32,
    pub architecture: String,
    pub num_tensors: usize,
    /// Total parameters across all tensors (sum of element counts).
    pub param_count: u64,
    pub model_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub tokenizer: String,
    /// Whether `run_gguf` can execute this file (arch llama/qwen2).
    pub runnable: bool,
    pub note: String,
    pub est_int4_mb: f64,
    pub est_int8_mb: f64,
    pub est_f32_mb: f64,
    pub avail_mb: Option<f64>,
}

/// Inspect a GGUF file without loading its weights (streamed header parse).
#[tauri::command]
pub async fn gguf_info(path: String) -> Result<GgufSummary, String> {
    tauri::async_runtime::spawn_blocking(move || {
        checked_path("GGUF path", &path, true)?;
        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let g = Gguf::open(&path).map_err(|e| format!("cannot open {path}: {e}"))?;
        let arch = g.architecture().unwrap_or("?").to_string();
        let meta_usize = |k: &str| g.meta(k).and_then(|v| v.as_usize()).unwrap_or(0);
        let dim = meta_usize(&format!("{arch}.embedding_length"));
        let n_layers = meta_usize(&format!("{arch}.block_count"));
        let n_heads = meta_usize(&format!("{arch}.attention.head_count"));
        let n_kv = g
            .meta(&format!("{arch}.attention.head_count_kv"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_heads);
        let vocab = g
            .tensor("token_embd.weight")
            .filter(|t| t.dims.len() == 2)
            .map(|t| t.dims[1] as usize)
            .unwrap_or_else(|| meta_usize(&format!("{arch}.vocab_size")));
        let tokenizer = match GgufTokenizer::from_gguf(&g) {
            Ok(t) => format!("{:?} ({} tokens)", t.model(), t.vocab_size()),
            Err(_) => "none in file (use token IDs)".to_string(),
        };
        let runnable = arch == "llama" || arch == "qwen2";
        let note = if !runnable {
            format!("architecture '{arch}' is not runnable (llama / qwen2 only)")
        } else {
            "decode is a few tok/s on CPU; prefer int8 for speed, int4 for memory".to_string()
        };
        let mb = |b: usize| b as f64 / 1e6;
        let param_count = param_count(&g);
        Ok(GgufSummary {
            path: path.clone(),
            bytes,
            version: g.version,
            architecture: arch,
            num_tensors: g.tensors.len(),
            param_count,
            model_dim: dim,
            n_layers,
            n_heads,
            n_kv_heads: n_kv,
            vocab_size: vocab,
            tokenizer,
            runnable,
            note,
            est_int4_mb: mb(gguf_resident_bytes(&g, Some(QKind::Int4))),
            est_int8_mb: mb(gguf_resident_bytes(&g, Some(QKind::Int8))),
            est_f32_mb: mb(gguf_resident_bytes(&g, None)),
            avail_mb: available_memory_bytes().map(mb),
        })
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

fn one_f32() -> f32 {
    1.0
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufRunParams {
    pub model_path: String,
    pub prompt: String,
    pub quant: String,
    pub max_new: usize,
    pub temp: f32,
    pub gen_seed: Option<u64>,
    /// Raw space-separated token IDs (used when the file has no tokenizer).
    pub ids: Option<String>,
    /// Load even if the memory estimate exceeds available RAM.
    pub force: bool,
    /// Optional fine-tune checkpoint (`.flck`) to overlay on the base model.
    /// When set, the model is loaded f32 (a checkpoint holds f32 weights).
    pub resume: Option<String>,
    /// Keep only the `k` highest-probability tokens (`0` disables).
    #[serde(default)]
    pub top_k: usize,
    /// Nucleus sampling threshold (`1.0` disables).
    #[serde(default = "one_f32")]
    pub top_p: f32,
    /// Down-weight already-seen tokens (`1.0` disables).
    #[serde(default = "one_f32")]
    pub rep_penalty: f32,
    /// Disable automatic chat-template formatting for instruct GGUFs.
    #[serde(default)]
    pub raw: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufRunResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated: usize,
    pub seconds: f32,
    pub tokens_per_sec: f32,
    /// A cancel ended the run early; `text` is what was produced so far.
    pub stopped: bool,
}

/// Import a llama/qwen2 GGUF and generate from it (mirrors `slm_cli run-gguf`).
/// `cancel_job("run")` stops it before the next prompt or generated token.
#[tauri::command]
pub async fn run_gguf(
    state: State<'_, AppState>,
    params: GgufRunParams,
) -> Result<GgufRunResult, String> {
    let job = state.jobs.start(Job::Run)?;
    tauri::async_runtime::spawn_blocking(move || gguf_run_inner(params, job))
        .await
        .map_err(|e| format!("task error: {e}"))?
}

/// The error of a run that was cancelled before it produced anything.
const CANCELLED_EARLY: &str = "cancelled before any text was produced";

fn gguf_run_inner(p: GgufRunParams, job: JobGuard) -> Result<GgufRunResult, String> {
    let params = sampling_from(p.temp, p.top_k, p.top_p, p.rep_penalty)?;
    checked_path("model path", &p.model_path, true)?;
    if let Some(r) = p.resume.as_deref().filter(|s| !s.trim().is_empty()) {
        checked_path("checkpoint path", r, true)?;
    }
    // A fine-tune checkpoint holds f32 weights, so applying one forces f32.
    let resume = p.resume.as_ref().filter(|s| !s.trim().is_empty()).cloned();
    let prec = if resume.is_some() {
        None
    } else {
        parse_quant(&p.quant)?
    };
    let g = Gguf::open(&p.model_path).map_err(|e| format!("cannot open {}: {e}", p.model_path))?;

    // Memory guard before the (potentially large) load.
    let est = gguf_resident_bytes(&g, prec);
    if let Some(avail) = available_memory_bytes() {
        if (est as f64) > 0.9 * avail as f64 && !p.force {
            return Err(format!(
                "estimated resident memory ({:.2} GB) exceeds 90% of available ({:.2} GB) — \
                 choose a smaller quant or enable 'load anyway'.",
                est as f64 / 1e9,
                avail as f64 / 1e9
            ));
        }
    }

    let tok = GgufTokenizer::from_gguf(&g).ok();
    if job.cancelled() {
        return Err(CANCELLED_EARLY.into());
    }
    let mut model = g
        .load_llama_prec(prec)
        .map_err(|e| format!("cannot load model: {e}"))?;
    if job.cancelled() {
        return Err(CANCELLED_EARLY.into());
    }

    // Overlay fine-tuned weights from a checkpoint, if requested.
    if let Some(ckpt) = &resume {
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        let mut tr = LlamaTrainer::new(model).map_err(|e| format!("cannot wrap model: {e}"))?;
        tr.load_checkpoint_from_vec(bytes)
            .map_err(|e| format!("cannot apply checkpoint {ckpt}: {e}"))?;
        model = tr.model;
    }

    let mut fmt = None;
    let prompt_ids: Vec<usize> = if let Some(ids) = p.ids.as_ref().filter(|s| !s.trim().is_empty())
    {
        ids.split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect()
    } else if let Some(t) = &tok {
        fmt = if p.raw {
            None
        } else {
            ferrum_core::chat_format::detect(
                g.meta("tokenizer.chat_template").and_then(|v| v.as_str()),
                g.architecture(),
                |piece| tok.as_ref().and_then(|t| t.token_id(piece)).is_some(),
            )
        };
        let mut v = Vec::new();
        if let Some(bos) = t.bos() {
            v.push(bos);
        }
        match fmt {
            Some(f) => v.extend(ferrum_core::chat_format::encode_segments(
                t,
                &ferrum_core::chat_format::render(f, &p.prompt),
            )),
            None => v.extend(t.encode(&p.prompt)),
        }
        v
    } else {
        return Err("this GGUF has no tokenizer; enter space-separated token IDs instead".into());
    };
    if prompt_ids.is_empty() {
        return Err("empty prompt — enter text (with a tokenizer) or token IDs".into());
    }

    let mut rng = Rng::new(p.gen_seed.unwrap_or_else(time_seed));
    let mut stops: Vec<usize> = tok
        .as_ref()
        .and_then(GgufTokenizer::eos)
        .into_iter()
        .collect();
    if let Some(f) = fmt {
        if let Some(id) = tok
            .as_ref()
            .and_then(|t| t.token_id(ferrum_core::chat_format::stop_token(f)))
        {
            stops.push(id);
        }
    }
    let t0 = std::time::Instant::now();
    let stopped = std::cell::Cell::new(false);
    let out = model
        .generate_cancellable(&prompt_ids, p.max_new, &params, &stops, &mut rng, &|| {
            let c = job.cancelled();
            stopped.set(stopped.get() || c);
            c
        })
        .map_err(|e| format!("generation failed: {e}"))?;
    let seconds = t0.elapsed().as_secs_f32();

    let text = match &tok {
        Some(t) => format!("{}{}", p.prompt, t.decode(&out)),
        None => out
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" "),
    };
    Ok(GgufRunResult {
        text,
        prompt_tokens: prompt_ids.len(),
        generated: out.len(),
        tokens_per_sec: out.len() as f32 / seconds.max(1e-6),
        seconds,
        stopped: stopped.get(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// 5b. Fine-tune an imported GGUF (mirrors `slm_cli finetune-gguf`)
// ─────────────────────────────────────────────────────────────────────────────

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
) -> Result<FinetuneResult, String> {
    let job = state.jobs.start(Job::Finetune)?;
    tauri::async_runtime::spawn_blocking(move || finetune_inner(app, params, job))
        .await
        .map_err(|e| format!("task error: {e}"))?
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
    ferrum_core::set_verbose(p.verbose);
    let cleanup = || ferrum_core::set_verbose(false);

    // 1. Open + f32-load the base model.
    let g = Gguf::open(&p.model_path).map_err(|e| {
        cleanup();
        format!("cannot open {}: {e}", p.model_path)
    })?;
    let est = gguf_resident_bytes(&g, None); // f32
    let train_est = est.saturating_mul(4); // weights + grad + Adam m/v
    if let Some(avail) = available_memory_bytes() {
        if (train_est as f64) > 0.9 * avail as f64 && !p.force {
            cleanup();
            return Err(format!(
                "estimated training memory ({:.2} GB) exceeds 90% of available ({:.2} GB) — \
                 fine-tune a smaller model or enable 'train anyway'.",
                train_est as f64 / 1e9,
                avail as f64 / 1e9
            ));
        }
    }
    let tok = GgufTokenizer::from_gguf(&g).map_err(|_| {
        cleanup();
        "this GGUF has no tokenizer; text fine-tuning needs one".to_string()
    })?;
    let model = g.load_llama_prec(None).map_err(|e| {
        cleanup();
        format!("cannot load model: {e}")
    })?;

    // 2. Read + tokenize the corpus.
    let text = std::fs::read_to_string(&p.corpus_path).map_err(|e| {
        cleanup();
        format!("cannot read corpus {}: {e}", p.corpus_path)
    })?;
    if text.trim().is_empty() {
        cleanup();
        return Err(format!("corpus {} is empty", p.corpus_path));
    }
    let mut tokens: Vec<usize> = Vec::new();
    if let Some(bos) = tok.bos() {
        tokens.push(bos);
    }
    tokens.extend(tok.encode(&text));

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
    let mut tr = LlamaTrainer::new(model).map_err(|e| {
        cleanup();
        format!("cannot build trainer: {e}")
    })?;
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

    // The schedule spans the *cumulative* step timeline (after any resume), so a
    // resumed run does not start past total_steps — which would pin the LR at 0.
    if p.warmup > 0 {
        let total = tr.step_count() + steps_per_epoch * p.epochs as u64;
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
    let total_steps = tr.step_count() + steps_per_epoch * p.epochs as u64;
    let mut epochs_run = 0;
    let mut stopped = false;
    for e in 0..p.epochs {
        // A cancel ends the epoch after a step (never the last step of the run).
        let (loss, halted) =
            match tr.finetune_epoch_until(&tokens, seq, batch, &mut rng, threads, &mut |step| {
                step < total_steps && job.cancelled()
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
    };
    let _ = app.emit("finetune-done", result.clone());
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// GGUF export (Export tab): write a loaded — optionally fine-tuned — llama/
// qwen2 model back out as a runnable GGUF at a chosen quantization.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportParams {
    pub model_path: String,
    pub out_path: String,
    /// Output type name, parsed by `GgufQuant::from_str` ("q8_0", "q4_k", "f16", …).
    pub quant: String,
    /// Optional `.flck` fine-tune checkpoint to overlay before export.
    pub resume: Option<String>,
    /// Bypass the RAM guard.
    pub force: bool,
    pub verbose: bool,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub out_path: String,
    pub bytes: u64,
    pub seconds: f64,
    /// Per-type tensor counts of the written file, e.g. "42× Q6_K, 19× F32".
    pub tensor_summary: String,
}

/// Output-buffer size estimate for the RAM guard: the whole encoded file is
/// built in memory before the atomic write, so a lossless f32 target costs a
/// second model's worth; f16 half; quantized targets ≤ ~a third (conservative).
fn export_output_estimate(est_model: usize, quant: ferrum_core::GgufQuant) -> usize {
    use ferrum_core::GgufQuant as Q;
    match quant {
        Q::F32 => est_model,
        Q::F16 => est_model / 2,
        _ => est_model / 3,
    }
}

/// Human name for the GGML tensor-type ids the exporter can emit.
fn ggml_type_name(t: u32) -> String {
    match t {
        0 => "F32".into(),
        1 => "F16".into(),
        2 => "Q4_0".into(),
        3 => "Q4_1".into(),
        6 => "Q5_0".into(),
        7 => "Q5_1".into(),
        8 => "Q8_0".into(),
        9 => "Q8_1".into(),
        12 => "Q4_K".into(),
        13 => "Q5_K".into(),
        14 => "Q6_K".into(),
        other => format!("type{other}"),
    }
}

/// The testable core of the Export tab: everything `export_gguf` does except
/// event emission. `progress` receives one human-readable line per phase
/// (prefixes: "opening", "loading", "applying", "encoding").
pub(crate) fn do_export(
    p: &ExportParams,
    progress: &dyn Fn(&str),
    cancelled: &dyn Fn() -> bool,
) -> Result<ExportResult, String> {
    // Cancel is polled between the phases; the encode-and-write phase is one call into the
    // engine and cannot be interrupted (it writes `<out>.tmp` and renames it when complete).
    let check = || {
        if cancelled() {
            Err(String::from("export cancelled; no file was written"))
        } else {
            Ok(())
        }
    };
    checked_path("output path (.gguf)", &p.out_path, false)?;
    let quant = ferrum_core::GgufQuant::from_str(&p.quant)
        .ok_or_else(|| format!("unknown output type '{}'", p.quant))?;
    checked_path("source GGUF path", &p.model_path, true)?;
    if let Some(r) = p.resume.as_deref().filter(|s| !s.trim().is_empty()) {
        checked_path("checkpoint path", r, true)?;
    }
    // Whitespace-only checkpoint paths mean "none" (parity with `gguf_run_inner`).
    let resume = p.resume.as_deref().map(str::trim).filter(|s| !s.is_empty());

    progress(&format!("opening {} (streamed)…", p.model_path));
    let g = Gguf::open(&p.model_path).map_err(|e| format!("cannot open {}: {e}", p.model_path))?;

    // Memory guard. Two transient peaks, not additive: applying a checkpoint
    // allocates grads + both Adam moments (≈4× the f32 model — freed before the
    // write), and the write itself holds model + encoded output buffer.
    let est_model = gguf_resident_bytes(&g, None);
    let write_peak = est_model.saturating_add(export_output_estimate(est_model, quant));
    let est = if resume.is_some() {
        est_model.saturating_mul(4).max(write_peak)
    } else {
        write_peak
    };
    if let Some(avail) = available_memory_bytes() {
        if (est as f64) > 0.9 * avail as f64 && !p.force {
            return Err(format!(
                "estimated peak memory ({:.2} GB{}) exceeds 90% of available ({:.2} GB) — \
                 pick a smaller output type or tick 'Export anyway'.",
                est as f64 / 1e9,
                if resume.is_some() {
                    ": checkpoint apply needs weights + grads + Adam moments"
                } else {
                    ": f32 model + output buffer"
                },
                avail as f64 / 1e9
            ));
        }
    }

    check()?;
    let t0 = std::time::Instant::now();
    progress("loading weights (f32)…");
    let mut model = g
        .load_llama_prec(None)
        .map_err(|e| format!("cannot load model: {e}"))?;
    check()?;

    if let Some(ckpt) = resume {
        progress(&format!("applying {ckpt}…"));
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        let mut tr = LlamaTrainer::new(model).map_err(|e| format!("cannot wrap model: {e}"))?;
        tr.load_checkpoint_from_vec(bytes)
            .map_err(|e| format!("cannot apply checkpoint: {e}"))?;
        model = tr.model;
        check()?;
    }

    progress(&format!("encoding + writing ({})…", p.quant));
    ferrum_core::write_llama_gguf(&model, &g, quant, &p.out_path)
        .map_err(|e| format!("export failed: {e}"))?;

    // Per-type summary, read back from the written file (also proves it
    // re-opens as valid GGUF).
    let gout =
        Gguf::open(&p.out_path).map_err(|e| format!("written file failed to re-open: {e}"))?;
    let mut counts: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for t in &gout.tensors {
        *counts.entry(t.ggml_type).or_default() += 1;
    }
    let tensor_summary = counts
        .iter()
        .map(|(ty, n)| format!("{n}× {}", ggml_type_name(*ty)))
        .collect::<Vec<_>>()
        .join(", ");

    let bytes = std::fs::metadata(&p.out_path).map(|m| m.len()).unwrap_or(0);
    Ok(ExportResult {
        out_path: p.out_path.clone(),
        bytes,
        seconds: t0.elapsed().as_secs_f64(),
        tensor_summary,
    })
}

/// Export a llama/qwen2 GGUF (optionally with a fine-tune checkpoint applied)
/// to a new GGUF at the requested quantization. Phase progress streams as
/// `export-progress` events (see [`do_export`] for the phases).
#[tauri::command]
pub async fn export_gguf<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    params: ExportParams,
) -> Result<ExportResult, String> {
    let job = state.jobs.start(Job::Export)?;
    tauri::async_runtime::spawn_blocking(move || {
        ferrum_core::set_verbose(params.verbose);
        let emit = |m: &str| {
            let _ = app.emit("export-progress", m.to_string());
        };
        let r = do_export(&params, &emit, &|| job.cancelled());
        ferrum_core::set_verbose(false);
        r
    })
    .await
    .map_err(|e| format!("task error: {e}"))?
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. System monitor (requirement #7)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SysStats {
    pub available: bool,
    pub cpu_total: f32,
    pub per_core: Vec<f32>,
    pub cores: usize,
    pub mem_used: u64,
    pub mem_total: u64,
    pub mem_percent: f32,
    pub ferrum_threads: usize,
}

/// Sample current CPU and memory load. Polled by the frontend on an interval.
#[tauri::command]
pub fn system_stats(state: State<'_, AppState>) -> Result<SysStats, String> {
    // A poisoned lock only means an earlier poll panicked; sysinfo state is still valid.
    let mut sys = state.sys.lock().unwrap_or_else(|e| e.into_inner());
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    let per_core: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
    let mem_total = sys.total_memory();
    let mem_used = sys.used_memory();
    let mem_percent = if mem_total > 0 {
        mem_used as f32 / mem_total as f32 * 100.0
    } else {
        0.0
    };
    Ok(SysStats {
        available: true,
        cpu_total: sys.global_cpu_usage(),
        cores: per_core.len(),
        per_core,
        mem_used,
        mem_total,
        mem_percent,
        ferrum_threads: ferrum_core::num_threads(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (X1): pure command logic, no GUI runtime required.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_sums_saturate_on_hostile_element_counts() {
        let mut b = ferrum_core::GgufBuilder::new();
        b.tensor("token_embd.weight", &[2], 0, vec![0; 8]);
        b.tensor("w", &[3], 0, vec![0; 12]);
        let mut g = Gguf::parse(b.into_bytes()).unwrap();
        assert_eq!(param_count(&g), 5);
        assert_eq!(gguf_resident_bytes(&g, None), 20);
        assert_eq!(gguf_resident_bytes(&g, Some(QKind::Int8)), 8 + 3);
        assert_eq!(gguf_resident_bytes(&g, Some(QKind::Int4)), 8 + 2);
        // Saturated dims (`parse` rejects them, but the sums must not rely
        // on that): no overflow panic, no wrap-around.
        for t in &mut g.tensors {
            t.dims = vec![u64::MAX, 8];
        }
        assert_eq!(param_count(&g), u64::MAX);
        for prec in [None, Some(QKind::Int8), Some(QKind::Int4)] {
            assert_eq!(gguf_resident_bytes(&g, prec), usize::MAX);
        }
    }

    #[test]
    fn model_info_reports_gguf_format_and_rejects_other_files() {
        use ferrum_core::{GenerativeSLM, Rng};
        let dir = std::env::temp_dir().join(format!("ferrum_gui_info_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = "abc abc abc ".repeat(20);
        let slm = GenerativeSLM::train_transformer(
            &corpus,
            4,
            16,
            2,
            1,
            32,
            4,
            0.01,
            4,
            0,
            &mut Rng::new(1),
        )
        .unwrap();
        let model = dir.join("m.gguf");
        slm.save(model.to_str().unwrap()).unwrap();
        let info = model_info(model.to_str().unwrap().to_string()).unwrap();
        assert_eq!(info.format, "GGUF v3 (ferrum, int8)");
        assert!(info.bytes > 0);
        let junk = dir.join("junk.gguf");
        std::fs::write(&junk, b"FINF\x05\x00\x00\x00").unwrap();
        let err = model_info(junk.to_str().unwrap().to_string())
            .err()
            .expect("FINF junk must be rejected");
        assert!(err.contains("old Ferrum .bin (FINF) model"), "{err}");
        assert!(model_info("/nonexistent/m.gguf".into()).is_err());
    }

    // ── G2: download timeouts + URL validation ────────────────────────────────

    #[test]
    fn read_text_file_caps_the_read_and_refuses_endless_files() {
        let dir = std::env::temp_dir().join(format!("ferrum_gui_read_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.txt");
        std::fs::write(&f, "héllo wörld, this is text").unwrap();
        let p = f.to_str().unwrap().to_string();
        assert_eq!(
            read_text_file(p.clone(), None).unwrap(),
            "héllo wörld, this is text"
        );
        assert_eq!(read_text_file(p.clone(), Some(5)).unwrap(), "héll");
        // A cap that splits a multi-byte char decodes lossily instead of failing.
        assert_eq!(read_text_file(p.clone(), Some(2)).unwrap(), "h\u{fffd}");
        assert_eq!(read_text_file(p.clone(), Some(0)).unwrap(), "");
        assert_eq!(
            read_text_file(p.clone(), Some(usize::MAX)).unwrap().len(),
            27
        );
        let e = read_text_file(dir.join("nope").to_str().unwrap().into(), None).unwrap_err();
        assert!(e.contains("does not exist"), "{e}");
        #[cfg(unix)]
        {
            // Regression: std::fs::read on /dev/zero never returns (and eats all RAM);
            // now refused up front by the path check.
            let e = read_text_file("/dev/zero".into(), Some(1000)).unwrap_err();
            assert!(e.contains("not allowed"), "{e}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn http_agent_builds() {
        // Building the timeout-bounded agent must not panic.
        let _ = http_agent();
    }

    #[test]
    fn fetch_text_rejects_non_http_urls() {
        // Validation happens before any network access, so these never hit the
        // wire — they fail fast with a clear message.
        for url in [
            "ftp://example.com/x",
            "file:///etc/passwd",
            "javascript:1",
            "",
        ] {
            let err = fetch_text(url, 1024).unwrap_err();
            assert!(
                err.contains("http://"),
                "unexpected error for {url:?}: {err}"
            );
        }
    }

    // ── file_name helper ──────────────────────────────────────────────────────

    #[test]
    fn file_name_extracts_basename() {
        assert_eq!(file_name("/a/b/model.gguf"), "model.gguf");
        assert_eq!(file_name("model.gguf"), "model.gguf");
    }

    // ── GGUF run params (F2: sampling knobs) ─────────────────────────────────

    #[test]
    fn gguf_run_params_sampling_defaults_off() {
        let p: GgufRunParams = serde_json::from_str(
            r#"{"modelPath":"m","prompt":"p","quant":"int4","maxNew":8,"temp":0.8,"force":false}"#,
        )
        .unwrap();
        assert_eq!((p.top_k, p.top_p, p.rep_penalty), (0, 1.0, 1.0));
    }

    #[test]
    fn available_memory_is_reported_cross_platform() {
        // The RAM guard is only meaningful if this returns a real figure; the old
        // /proc/meminfo reader returned None off Linux, disabling the guard there.
        let avail = available_memory_bytes().expect("available memory should be reported");
        assert!(
            avail > 16 * 1024 * 1024,
            "implausibly small avail: {avail} bytes"
        );
    }

    #[test]
    fn time_seed_is_nonzero() {
        assert_ne!(time_seed(), 0);
    }

    // ── G7: greedy decoding and the full sampling controls ───────────────────

    fn tiny_bpe() -> GenerativeSLM {
        let corpus = "the quick brown fox jumps over the lazy dog. ".repeat(40);
        let cfg = TransformerConfig {
            context_len: 8,
            embed_dim: 16,
            num_heads: 2,
            num_blocks: 1,
            hidden_dim: 32,
            epochs: 2,
            lr: 0.01,
            batch_size: 8,
            vocab_size: 300,
            ..TransformerConfig::default()
        };
        GenerativeSLM::train_transformer_config_threaded(
            &corpus,
            &cfg,
            1,
            &mut Rng::new(3),
            |_, _| {},
        )
        .unwrap()
    }

    fn gen_params(seed_text: &str) -> GenParams {
        serde_json::from_value(serde_json::json!({
            "modelPath": "m", "seedText": seed_text, "numChars": 60, "temp": 0.8,
            "genSeed": 5, "stream": false, "verbose": false
        }))
        .unwrap()
    }

    fn generate_with_gui(slm: &GenerativeSLM, p: &GenParams) -> (String, String) {
        let mut streamed = String::new();
        let (out, stopped) =
            run_generation(slm, p, &|| false, &mut |f| streamed.push_str(f)).unwrap();
        assert!(!stopped);
        (out, streamed)
    }

    #[test]
    fn gen_params_new_fields_default_to_the_old_behaviour() {
        let p = gen_params("x");
        assert_eq!((p.top_k, p.top_p, p.rep_penalty), (0, 1.0, 1.0));
        assert_eq!((p.max_tokens, p.stop.len()), (0, 0));
    }

    #[test]
    fn character_path_equals_the_library_and_streams_the_continuation() {
        let slm = tiny_bpe();
        let p = gen_params("the quick");
        let (out, streamed) = generate_with_gui(&slm, &p);
        let want = slm
            .generate_with(
                "the quick",
                60,
                &SamplingParams::with_temperature(0.8),
                &mut Rng::new(5),
            )
            .unwrap();
        assert_eq!(out, want);
        assert_eq!(format!("the quick{streamed}"), out);
    }

    #[test]
    fn token_path_equals_generate_tokens_for_every_sampling_control() {
        let slm = tiny_bpe();
        let mut p = gen_params("the ");
        p.temp = 0.9;
        p.top_k = 5;
        p.top_p = 0.9;
        p.rep_penalty = 1.2;
        p.max_tokens = 25;
        let (out, streamed) = generate_with_gui(&slm, &p);
        let params = SamplingParams {
            temperature: 0.9,
            top_k: 5,
            top_p: 0.9,
            repetition_penalty: 1.2,
        };
        let g = slm
            .generate_tokens("the ", 25, &[], &params, &mut Rng::new(5))
            .unwrap();
        assert_eq!(out, format!("the {}", g.text));
        assert_eq!(streamed, g.text);
        assert!(g.n_tokens <= 25);
        // Without max_tokens the character budget doubles as the token budget (as in the CLI).
        let mut q = gen_params("the ");
        q.stop = vec!["zzzz-never".into()];
        let g = slm
            .generate_tokens(
                "the ",
                60,
                &["zzzz-never".to_string()],
                &SamplingParams::with_temperature(0.8),
                &mut Rng::new(5),
            )
            .unwrap();
        assert_eq!(generate_with_gui(&slm, &q).0, format!("the {}", g.text));
    }

    #[test]
    fn temperature_zero_is_exact_argmax_whatever_the_seed() {
        let slm = tiny_bpe();
        let mut p = gen_params("the ");
        p.temp = 0.0;
        p.gen_seed = Some(1);
        let a = generate_with_gui(&slm, &p).0;
        p.gen_seed = Some(999);
        let b = generate_with_gui(&slm, &p).0;
        p.gen_seed = None; // time-seeded: still identical
        let c = generate_with_gui(&slm, &p).0;
        assert_eq!(a, b);
        assert_eq!(a, c);
        let want = slm
            .generate_with(
                "the ",
                60,
                &SamplingParams::with_temperature(0.0),
                &mut Rng::new(0),
            )
            .unwrap();
        assert_eq!(a, want);
    }

    #[test]
    fn stop_strings_end_generation_and_are_not_in_the_text() {
        let slm = tiny_bpe();
        let mut p = gen_params("the ");
        p.temp = 0.0;
        let full = generate_with_gui(&slm, &p).0;
        let cont: Vec<char> = full.chars().skip(4).collect();
        assert!(cont.len() > 12, "{full:?}");
        let stop: String = cont[8..10].iter().collect();
        p.stop = vec!["".into(), stop.clone()]; // the empty line is ignored
        let (out, streamed) = generate_with_gui(&slm, &p);
        let at = full[4..].find(&stop).unwrap();
        assert_eq!(out, format!("the {}", &full[4..][..at]));
        assert!(!streamed.contains(&stop));
        assert_eq!(format!("the {streamed}"), out);
    }

    #[test]
    fn stop_escapes_and_validation() {
        let raw = |s: &str| vec![s.to_string()];
        assert_eq!(parse_stops(&raw("a\\nb")), vec!["a\nb"]);
        assert_eq!(parse_stops(&raw("\\t.")), vec!["\t."]);
        assert_eq!(parse_stops(&raw("c:\\\\x")), vec!["c:\\x"]);
        assert_eq!(parse_stops(&raw("\\q")), vec!["\\q"]);
        assert_eq!(parse_stops(&raw("end\\")), vec!["end\\"]);
        assert!(parse_stops(&raw("")).is_empty());

        assert!(sampling_from(0.0, 0, 1.0, 1.0).is_ok());
        for (t, p, r, word) in [
            (-0.1, 1.0, 1.0, "temperature"),
            (f32::NAN, 1.0, 1.0, "temperature"),
            (0.5, 1.5, 1.0, "top-p"),
            (0.5, -0.1, 1.0, "top-p"),
            (0.5, 1.0, 0.0, "repetition"),
        ] {
            let e = sampling_from(t, 0, p, r).unwrap_err();
            assert!(e.contains(word), "{e}");
        }
    }

    // ── G9: cancelling jobs ──────────────────────────────────────────────────

    #[test]
    fn a_cancelled_generation_returns_the_text_so_far_for_both_paths() {
        let slm = tiny_bpe();
        for max_tokens in [0usize, 30] {
            let mut p = gen_params("the ");
            p.temp = 0.0;
            p.max_tokens = max_tokens;
            let (full, _) = generate_with_gui(&slm, &p);
            let polls = std::cell::Cell::new(0);
            let mut streamed = String::new();
            let (part, stopped) = run_generation(
                &slm,
                &p,
                &|| {
                    polls.set(polls.get() + 1);
                    polls.get() >= 3
                },
                &mut |f| streamed.push_str(f),
            )
            .unwrap();
            assert!(stopped, "max_tokens {max_tokens}");
            assert_eq!(part, format!("the {streamed}"));
            assert!(
                full.starts_with(&part) && part.len() < full.len(),
                "{part:?} vs {full:?}"
            );
        }
        // A cancel that never comes changes nothing; other panics still propagate.
        let r = std::panic::catch_unwind(|| {
            let mut p = gen_params("the ");
            p.temp = 0.0;
            run_generation(&tiny_bpe(), &p, &|| false, &mut |_| panic!("boom"))
        });
        assert!(r.is_err());
    }

    #[test]
    fn a_cancelled_export_writes_nothing_and_leaves_no_temp_file() {
        let dir =
            std::env::temp_dir().join(format!("ferrum_gui_exportcancel_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        std::fs::write(&src, crate::test_support::tiny_llama_gguf(8, 2, 2, 16, 32)).unwrap();
        let p = ExportParams {
            model_path: src.to_string_lossy().into_owned(),
            out_path: dst.to_string_lossy().into_owned(),
            quant: "f32".into(),
            resume: None,
            force: false,
            verbose: false,
        };
        // Cancelled at each of the polls in turn: always an error, never a file.
        for after in 0..2 {
            let polls = std::cell::Cell::new(0);
            let e = do_export(&p, &|_| {}, &|| {
                polls.set(polls.get() + 1);
                polls.get() > after
            })
            .unwrap_err();
            assert!(e.contains("cancelled"), "{e}");
            assert!(!dst.exists(), "no output after a cancel at poll {after}");
            assert!(!dir.join("dst.gguf.tmp").exists());
        }
        // Without a cancel the same call completes.
        assert!(do_export(&p, &|_| {}, &|| false).is_ok());
        assert!(dst.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_gguf_run_cancelled_before_it_starts_says_so() {
        let dir = std::env::temp_dir().join(format!("ferrum_gui_runcancel_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("m.gguf");
        std::fs::write(&src, crate::test_support::tiny_llama_gguf(8, 2, 2, 16, 32)).unwrap();
        let p = |max_new: usize| -> GgufRunParams {
            serde_json::from_value(serde_json::json!({
                "modelPath": src, "prompt": "hi", "quant": "int8", "maxNew": max_new, "temp": 0.0,
                "genSeed": 1, "ids": null, "force": false, "resume": null, "raw": true
            }))
            .unwrap()
        };
        let jobs = crate::jobs::Jobs::default();
        let job = jobs.start(Job::Run).unwrap();
        assert!(jobs.cancel(Job::Run));
        assert_eq!(gguf_run_inner(p(6), job).err().unwrap(), CANCELLED_EARLY);
        let r = gguf_run_inner(p(6), jobs.start(Job::Run).unwrap()).unwrap();
        assert!(!r.stopped && r.generated > 0);
        let _ = std::fs::remove_dir_all(&dir);
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

    // ── GGUF export (do_export) ──────────────────────────────────────────────

    use ferrum_core::GgufQuant;

    #[test]
    fn export_output_estimate_matches_cli_heuristic() {
        assert_eq!(export_output_estimate(900, GgufQuant::F32), 900);
        assert_eq!(export_output_estimate(900, GgufQuant::F16), 450);
        assert_eq!(export_output_estimate(900, GgufQuant::Q8_0), 300);
        assert_eq!(export_output_estimate(900, GgufQuant::Q4K), 300);
    }

    #[test]
    fn do_export_rejects_bad_inputs() {
        let base = ExportParams {
            model_path: "/nonexistent.gguf".into(),
            out_path: "/tmp/out.gguf".into(),
            quant: "q8_0".into(),
            resume: None,
            force: false,
            verbose: false,
        };
        let noop = |_: &str| {};

        let mut p = base;
        p.out_path = "  ".into();
        assert!(do_export(&p, &noop, &|| false)
            .unwrap_err()
            .contains("output"));

        p.out_path = "/tmp/out.gguf".into();
        p.quant = "q9_9".into();
        assert!(do_export(&p, &noop, &|| false)
            .unwrap_err()
            .contains("q9_9"));

        // A missing source file is refused by the path check and named.
        p.quant = "q8_0".into();
        assert!(do_export(&p, &noop, &|| false)
            .unwrap_err()
            .contains("does not exist"));

        // Device paths are refused before anything is opened.
        p.out_path = "/dev/null".into();
        assert!(do_export(&p, &noop, &|| false)
            .unwrap_err()
            .contains("not allowed"));
    }

    #[test]
    fn file_commands_refuse_bad_paths() {
        let dir = std::env::temp_dir().join(format!("ferrum_gui_cmdpaths_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap().to_string();

        // read_text_file: device, directory, missing, NUL; a normal file works.
        for bad in [
            "/dev/zero",
            "/proc/self/mem",
            d.as_str(),
            "/nonexistent/x.txt",
            "a\0b",
            "",
        ] {
            assert!(read_text_file(bad.into(), None).is_err(), "{bad:?}");
        }
        let f = dir.join("ok.txt");
        std::fs::write(&f, "hello").unwrap();
        assert_eq!(
            read_text_file(f.to_str().unwrap().into(), None).unwrap(),
            "hello"
        );

        // save_corpus: refuses devices and directories, writes a new file (making parents).
        assert!(save_corpus("x".into(), "/dev/null".into()).is_err());
        assert!(save_corpus("x".into(), d.clone()).is_err());
        assert!(save_corpus("x".into(), String::new()).is_err());
        let out = dir.join("sub/out.txt");
        assert_eq!(
            save_corpus("abc".into(), out.to_str().unwrap().into()).unwrap(),
            3
        );

        // model_info: directory and device are refused with the path message.
        assert!(model_info(d).err().unwrap().contains("directory"));
        assert!(model_info("/dev/null".into())
            .err()
            .unwrap()
            .contains("not allowed"));
    }

    #[test]
    fn do_export_roundtrips_a_tiny_model_with_phases() {
        use ferrum_core::Gguf;
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let src = dir.join(format!("ferrum_gui_export_src_{pid}.gguf"));
        let dst = dir.join(format!("ferrum_gui_export_dst_{pid}.gguf"));
        std::fs::write(&src, crate::test_support::tiny_llama_gguf(8, 2, 2, 16, 32)).unwrap();

        let p = ExportParams {
            model_path: src.to_string_lossy().into_owned(),
            out_path: dst.to_string_lossy().into_owned(),
            quant: "f32".into(),
            resume: None,
            force: false,
            verbose: false,
        };
        let phases = std::cell::RefCell::new(Vec::<String>::new());
        let r = do_export(&p, &|m| phases.borrow_mut().push(m.to_string()), &|| false).unwrap();

        // Phases arrive in order: opening → loading → encoding (no resume).
        let ph = phases.borrow();
        assert!(ph.len() >= 3, "want ≥3 phases, got {ph:?}");
        assert!(ph[0].starts_with("opening"), "{ph:?}");
        assert!(ph[1].starts_with("loading"), "{ph:?}");
        assert!(ph.last().unwrap().starts_with("encoding"), "{ph:?}");
        assert!(!ph.iter().any(|m| m.starts_with("applying")), "{ph:?}");

        // Result: the reported path/size match the real file, and the summary
        // names F32 tensors (the whole fixture is F32).
        assert_eq!(r.out_path, p.out_path);
        assert_eq!(r.bytes, std::fs::metadata(&p.out_path).unwrap().len());
        assert!(r.bytes > 0);
        assert!(r.tensor_summary.contains("F32"), "{}", r.tensor_summary);

        // The output re-opens and re-loads as a runnable llama model.
        let g = Gguf::open(&p.out_path).unwrap();
        assert_eq!(g.architecture(), Some("llama"));
        g.load_llama_prec(None).unwrap();

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }
}
