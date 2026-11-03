//! Inspect saved models and run imported GGUFs (Llama/Qwen checkpoints).

use super::generate::{one_f32, parse_stops, sampling_from};
use super::guard::{available_memory_bytes, gguf_resident_bytes, parse_quant, ram_guard};
use super::time_seed;
use crate::error::CmdError;
use crate::jobs::{Job, JobGuard};
use crate::paths::checked_path;
use crate::AppState;
use ferrum_core::llm_stop::StopMatcher;
use ferrum_core::{GenerativeSLM, Gguf, GgufTokenizer, LlamaTrainer, QKind, Rng};
use serde::{Deserialize, Serialize};
use tauri::State;

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
pub fn model_info(path: String) -> Result<ModelInfo, CmdError> {
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

/// Total elements across all tensors. Saturating: a hostile directory's
/// element counts must not overflow the sum (panic in debug, wrap in release).
fn param_count(g: &Gguf) -> u64 {
    g.tensors
        .iter()
        .fold(0u64, |acc, t| acc.saturating_add(t.num_elements() as u64))
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
pub async fn gguf_info(path: String) -> Result<GgufSummary, CmdError> {
    crate::error::blocking(move || {
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
    /// Stop strings, one per entry (as the Generate tab: `\n` `\t` `\\` unescaped, empty
    /// entries dropped); the text ends before the earliest one. Needs a tokenizer.
    #[serde(default)]
    pub stop: Vec<String>,
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
) -> Result<GgufRunResult, CmdError> {
    let job = state.jobs.start(Job::Run)?;
    crate::error::blocking(move || gguf_run_inner(params, job)).await
}

/// The error of a run that was cancelled before it produced anything.
const CANCELLED_EARLY: &str = "cancelled before any text was produced";

/// Stop a run that was cancelled before it produced anything.
fn not_cancelled(job: &JobGuard) -> Result<(), String> {
    if job.cancelled() {
        Err(CANCELLED_EARLY.into())
    } else {
        Ok(())
    }
}

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
    ram_guard(
        "resident memory",
        "",
        est,
        available_memory_bytes(),
        p.force,
        "choose a smaller quant or enable 'load anyway'",
    )?;

    let tok = GgufTokenizer::from_gguf(&g).ok();
    let text_stops = parse_stops(&p.stop);
    if tok.is_none() && !text_stops.is_empty() {
        return Err("stop strings need a tokenizer; this GGUF has none (enter token IDs without stop strings)".into());
    }
    not_cancelled(&job)?;
    let mut model = g
        .load_llama_prec(prec)
        .map_err(|e| format!("cannot load model: {e}"))?;
    not_cancelled(&job)?;

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
    // With no stop strings the matcher never fires: `generate_cancellable` bit for bit.
    let mut matcher = tok
        .as_ref()
        .map(|t| StopMatcher::new(t, prompt_ids.last().copied().unwrap_or(0), &text_stops));
    let out = model
        .generate_until_cancellable(
            &prompt_ids,
            p.max_new,
            &params,
            &stops,
            &mut rng,
            &|| {
                let c = job.cancelled();
                stopped.set(stopped.get() || c);
                c
            },
            &mut |out| matcher.as_mut().is_some_and(|m| m.check(out)),
        )
        .map_err(|e| format!("generation failed: {e}"))?;
    let seconds = t0.elapsed().as_secs_f32();

    let text = match &tok {
        Some(t) => match matcher.as_ref().and_then(|m| m.cut()) {
            Some(cut) => format!("{}{cut}", p.prompt),
            None => format!("{}{}", p.prompt, t.decode(&out)),
        },
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

    #[test]
    fn gguf_run_params_sampling_defaults_off() {
        let p: GgufRunParams = serde_json::from_str(
            r#"{"modelPath":"m","prompt":"p","quant":"int4","maxNew":8,"temp":0.8,"force":false}"#,
        )
        .unwrap();
        assert_eq!((p.top_k, p.top_p, p.rep_penalty), (0, 1.0, 1.0));
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
}
