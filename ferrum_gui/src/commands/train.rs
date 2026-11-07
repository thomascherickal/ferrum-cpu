//! The Train tab: train a transformer SLM in-process with streamed progress.

use crate::infra::error::CmdError;
use crate::infra::jobs::{Job, JobGuard};
use crate::infra::paths::checked_path;
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
    /// Saved precision: "f32", "int8" or "int4" (the CLI's `--quant`); absent means int8.
    #[serde(default = "default_quant")]
    pub quant: String,
}

fn default_quant() -> String {
    "int8".into()
}

/// Save `slm` at the precision `quant` names, as the CLI's `--quant` does.
fn save_as(
    slm: &GenerativeSLM,
    path: &str,
    quant: Option<ferrum_core::QKind>,
) -> Result<(), ferrum_core::InferError> {
    match quant {
        Some(ferrum_core::QKind::Int8) => slm.save(path),
        Some(ferrum_core::QKind::Int4) => slm.save_int4(path),
        None => slm.save_f32(path),
    }
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
) -> Result<TrainResult, CmdError> {
    let job = state.jobs.start(Job::Train)?;
    crate::infra::error::blocking(move || train_inner(app, params, job)).await
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

/// `sha256:<hex>` of the corpus file when the run writes a checkpoint (the CLI's rule), else none.
fn corpus_id(p: &TrainParams) -> Result<Option<String>, String> {
    if p.checkpoint_path.trim().is_empty() {
        return Ok(None);
    }
    slm_cli::hash::file_id(&p.corpus_path)
        .map(Some)
        .map_err(|e| format!("cannot read corpus {}: {e}", p.corpus_path))
}

fn train_inner<R: Runtime>(
    app: AppHandle<R>,
    p: TrainParams,
    job: JobGuard,
) -> Result<TrainResult, String> {
    // ---- shared validation with clear messages ----
    checked_path("output model path", &p.model_path, false)?;
    // Checked before training so a typo costs nothing (`parse_quant` also takes q4/q8/none).
    let quant = super::guard::parse_quant(&p.quant)?;
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
    // streamed from disk instead of read here. Large BPE text files (>= 8 MiB) are also streamed
    // from disk through a temporary TokenFile so training never needs the full corpus in RAM.
    let mut token_input = if ferrum_core::is_gguf_file(&p.corpus_path) {
        Some(open_token_corpus(&p)?)
    } else {
        None
    };
    let stream_disk = token_input.is_none()
        && p.vocab_size >= 256
        && std::fs::metadata(&p.corpus_path)
            .map(|m| m.is_file() && m.len() >= ferrum_core::STREAM_CORPUS_THRESHOLD)
            .unwrap_or(false);
    let corpus = if token_input.is_some() || stream_disk {
        String::new()
    } else {
        let corpus = crate::infra::fmt_err(std::fs::read_to_string(&p.corpus_path), &format!("cannot read corpus {}", p.corpus_path))?;
        if corpus.trim().is_empty() {
            return Err(format!("corpus {} is empty", p.corpus_path));
        }
        if let Err(e) = validate_for_training(&corpus, p.context_len) { return Err(e.to_string()); }
        corpus
    };

    let corpus_id = corpus_id(&p)?;
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
                // The same identity the CLI records, so a checkpoint resumes from either.
                corpus_id,
                ..TransformerConfig::default()
            };
            let val = ValidationConfig {
                val_fraction: p.val_fraction,
                patience: p.patience,
            };
            match (token_input.as_mut(), stream_disk, p.val_fraction > 0.0) {
                (Some((file, tok)), _, true) => GenerativeSLM::train_transformer_tokens_validated(
                    file,
                    tok,
                    &cfg,
                    p.threads,
                    &val,
                    &mut rng,
                    |v| progress(v.epoch, v.train_loss),
                ),
                (Some((file, tok)), _, false) => GenerativeSLM::train_transformer_tokens(
                    file, tok, &cfg, p.threads, &mut rng, progress,
                ),
                (None, true, true) => GenerativeSLM::train_transformer_from_path_validated(
                    std::path::Path::new(&p.corpus_path),
                    &cfg,
                    p.threads,
                    &val,
                    &mut rng,
                    |v| progress(v.epoch, v.train_loss),
                ),
                (None, true, false) => GenerativeSLM::train_transformer_from_path(
                    std::path::Path::new(&p.corpus_path),
                    &cfg,
                    p.threads,
                    &mut rng,
                    progress,
                ),
                (None, false, true) => GenerativeSLM::train_transformer_config_validated(
                    &corpus,
                    &cfg,
                    p.threads,
                    &val,
                    &mut rng,
                    |v| progress(v.epoch, v.train_loss),
                ),
                (None, false, false) => GenerativeSLM::train_transformer_config_threaded(
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
    let saved = save_as(&slm, &p.model_path, quant);
    // Before the `?`: a failed save must not leave the engine's verbose mode on for later jobs.
    ferrum_core::set_verbose(false);
    saved.map_err(|e| format!("cannot save model {}: {e}", p.model_path))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_tests::{app, call, msg, tmp};
    use serde_json::{json, Value};

    /// The fields the Train tab always sends; everything else takes its default.
    fn required() -> Value {
        json!({
            "method": "transformer", "corpusPath": "c.txt", "modelPath": "m.gguf",
            "contextLen": 8, "embedDim": 8, "numHeads": 2, "numBlocks": 1, "hiddenDim": 16,
            "epochs": 1, "lr": 0.01, "batchSize": 4, "vocabSize": 0, "seed": 1, "threads": 1,
            "verbose": false
        })
    }

    fn params(extra: Value) -> TrainParams {
        let mut v = required();
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn omitted_fields_take_the_cli_defaults() {
        let p = params(json!({}));
        assert_eq!(p.quant, "int8", "absent quant saves int8, like --quant");
        assert!(p.qat, "QAT is on unless the form turns it off");
        assert_eq!(
            (p.window_stride, p.max_tokens, p.checkpoint_every),
            (0, 0, 0)
        );
        assert!(p.tokenizer_path.is_empty() && p.checkpoint_path.is_empty());
        assert!(!p.resume && !p.force && p.val_fraction == 0.0);
        let p = params(json!({"qat": false, "quant": "f32"}));
        assert!(!p.qat && p.quant == "f32");
    }

    #[test]
    fn corpus_id_only_when_a_checkpoint_is_written() {
        let dir = tmp("train_corpus_id");
        let corpus = dir.join("c.txt");
        std::fs::write(&corpus, "abc").unwrap();
        let c = corpus.to_string_lossy();
        assert_eq!(corpus_id(&params(json!({"corpusPath": c}))), Ok(None));
        let blank = params(json!({"corpusPath": c, "checkpointPath": "  "}));
        assert_eq!(corpus_id(&blank), Ok(None), "a blank path is no checkpoint");
        let with = params(json!({"corpusPath": c, "checkpointPath": "ck.fckp"}));
        let id = corpus_id(&with).unwrap().unwrap();
        assert_eq!(
            Some(id),
            slm_cli::hash::file_id(&c).ok(),
            "the CLI's identity"
        );
        let gone = params(json!({"corpusPath": dir.join("none.txt"), "checkpointPath": "ck"}));
        assert!(corpus_id(&gone)
            .unwrap_err()
            .starts_with("cannot read corpus "));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The CLI's token-file errors name its flags; the Train tab names its own fields instead.
    #[test]
    fn token_corpus_errors_speak_the_forms_language() {
        let dir = tmp("train_token_corpus");
        let text = "the small brown fox and the small cat ".repeat(4);
        let s = |n: &str| dir.join(n).to_string_lossy().into_owned();
        std::fs::write(s("c.txt"), &text).unwrap();
        let tok = ferrum_core::ByteBpeTokenizer::train(&text, 270).unwrap();
        let other = ferrum_core::ByteBpeTokenizer::train("other words entirely", 260).unwrap();
        std::fs::write(s("tok.gguf"), tok.to_gguf_bytes()).unwrap();
        std::fs::write(s("other.gguf"), other.to_gguf_bytes()).unwrap();
        slm_cli::tokenize::tokenize(
            &slm_cli::tokenize::TokenizeOptions {
                input: s("c.txt"),
                tokenizer: s("tok.gguf"),
                out: s("c.tok.gguf"),
                doc_sep: ferrum_core::DOC_SEP.into(),
                stop: ferrum_core::StopCondition::none(),
            },
            &mut |_| {},
        )
        .unwrap();
        let open = |extra: Value| {
            let mut e = json!({"corpusPath": s("c.tok.gguf"), "vocabSize": 270});
            for (k, v) in extra.as_object().unwrap() {
                e[k] = v.clone();
            }
            open_token_corpus(&params(e)).err()
        };
        let e = open(json!({"tokenizerPath": "  "})).unwrap();
        assert!(
            e.contains("the Tokenizer file field") && !e.contains("--tokenizer"),
            "{e}"
        );
        let e = open(json!({"tokenizerPath": s("tok.gguf"), "vocabSize": 0})).unwrap();
        assert!(
            e.contains("Vocab 0 (character-level)") && !e.contains("--vocab"),
            "{e}"
        );
        let e = open(json!({"tokenizerPath": s("other.gguf")})).unwrap();
        assert!(
            e.contains("run Tokenize corpus again") && !e.contains("`tokenize`"),
            "{e}"
        );
        assert_eq!(open(json!({"tokenizerPath": s("tok.gguf")})), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Regression: a save that failed after training left the engine's verbose mode on.
    #[cfg(unix)]
    #[test]
    fn a_failed_save_turns_verbose_off_and_each_precision_saves() {
        let (_app, w) = app();
        let dir = tmp("train_save");
        let corpus = dir.join("c.txt");
        std::fs::write(&corpus, "abc abd abe ".repeat(8)).unwrap();
        let train = |model: &std::path::Path, quant: &str, verbose: bool| {
            let mut p = required();
            p["corpusPath"] = json!(corpus);
            p["modelPath"] = json!(model);
            p["quant"] = json!(quant);
            p["verbose"] = json!(verbose);
            call(&w, "train_slm", json!({ "params": p }))
        };
        for quant in ["int8", "int4", "f32"] {
            let m = dir.join(format!("{quant}.gguf"));
            train(&m, quant, false).unwrap();
            let slm = GenerativeSLM::load(m.to_str().unwrap()).unwrap();
            assert_eq!(slm.meta.input_dim, 8, "{quant}");
        }
        let locked = dir.join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        let _restore = crate::test_support::restrict(&locked, 0o500, 0o700);
        let e = train(&locked.join("m.gguf"), "int8", true);
        assert!(_restore.is_none() || msg(&e.unwrap_err()).starts_with("cannot save model "));
        assert!(!ferrum_core::is_verbose(), "verbose mode outlived the failed job");
        drop(_restore);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn large_bpe_corpus_streams_from_disk_with_and_without_validation() {
        let (_app, w) = app();
        let dir = tmp("train_stream_disk");
        let corpus = dir.join("large.txt");
        let unit = "the quick brown fox jumps over the lazy dog\n";
        let repeats = (ferrum_core::STREAM_CORPUS_THRESHOLD as usize / unit.len()) + 64;
        std::fs::write(&corpus, unit.repeat(repeats)).unwrap();

        for (i, val_frac) in [0.0_f32, 0.2_f32].into_iter().enumerate() {
            let model = dir.join(format!("m{i}.gguf"));
            let mut p = required();
            p["corpusPath"] = json!(corpus);
            p["modelPath"] = json!(model);
            p["vocabSize"] = json!(264);
            p["maxTokens"] = json!(64);
            p["valFraction"] = json!(val_frac);
            let res = call(&w, "train_slm", json!({ "params": p })).unwrap();
            assert_eq!(res["tokenizer"], "byte-level BPE");
            assert!(model.is_file());
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
