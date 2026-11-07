//! The Evaluate tab: score a model on held-out text, with the numbers of
//! `train_transformer eval --json` (perplexity, cross-entropy, bits per token,
//! bits per byte, tokens, bytes, seconds).
//!
//! A Ferrum model is scored with `GenerativeSLM::evaluate`; an imported
//! Llama/Qwen GGUF with `LlamaModel::score_rolling`, the same calls `eval` and
//! `eval-gguf` make. (`slm_cli`'s report formatting lives in that binary
//! crate, so the JSON is built here with the same keys.)

use crate::commands::{
    available_memory_bytes, file_name, gguf_resident_bytes, parse_quant, ram_guard,
};
use crate::infra::error::CmdError;
use crate::infra::jobs::Job;
use crate::infra::paths::checked_path;
use crate::AppState;
use ferrum_core::{apply_checkpoint_weights, GenerativeSLM, Gguf, GgufTokenizer};
use serde::{Deserialize, Serialize};
use tauri::State;

/// Default scoring window of an imported model (as `eval-gguf`): long enough for useful context,
/// short enough that a 32k-context model is not scored in one enormous forward pass.
const DEFAULT_WINDOW: usize = 512;

fn f32_quant() -> String {
    "f32".into()
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalParams {
    pub model_path: String,
    /// Pasted held-out text; wins over `text_path` when both are given.
    pub text: Option<String>,
    pub text_path: Option<String>,
    /// An optional second file, scored the same way and reported as the "test" row.
    #[serde(default)]
    pub test_path: Option<String>,
    /// Replace invalid UTF-8 in the files instead of refusing them.
    #[serde(default)]
    pub lossy: bool,
    // Imported Llama/Qwen GGUF only (a Ferrum model ignores these):
    /// In-memory precision: `f32` (default), `int8` or `int4`.
    #[serde(default = "f32_quant")]
    pub quant: String,
    /// Scoring window in tokens; `0` = `min(512, the model's context length)`.
    #[serde(default)]
    pub ctx: usize,
    /// A fine-tune checkpoint (`.flck`) to apply first (forces f32).
    #[serde(default)]
    pub resume: Option<String>,
    /// Load even if the memory estimate exceeds available RAM.
    #[serde(default)]
    pub force: bool,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct EvalRow {
    pub model: String,
    /// `held-out` or `test`.
    pub label: String,
    pub num_predictions: usize,
    pub cross_entropy: f32,
    pub bits_per_token: f32,
    pub perplexity: f32,
    /// The perplexity of a model that learned nothing: its vocabulary size.
    pub uniform_baseline: f32,
    pub bits_per_byte: f64,
    /// Bytes of the scored text.
    pub bytes: usize,
    pub seconds: f64,
    /// Exactly what `eval --json` writes: `perplexity`, `cross_entropy_nats`,
    /// `bits_per_token`, `bits_per_byte`, `tokens`, `bytes`, `seconds`.
    pub report: serde_json::Value,
}

/// One table row from the library's numbers. `bits_per_byte` is the CLI's formula:
/// cross-entropy (nats) x tokens / ln 2 / bytes of text.
fn row(
    model: &str,
    label: &str,
    s: Stats,
    baseline: usize,
    text_bytes: usize,
    seconds: f64,
) -> EvalRow {
    let bits_per_byte =
        f64::from(s.cross_entropy) * s.tokens as f64 / std::f64::consts::LN_2 / text_bytes as f64;
    let report = serde_json::json!({
        "perplexity": s.perplexity, "cross_entropy_nats": s.cross_entropy,
        "bits_per_token": s.bits_per_token, "bits_per_byte": bits_per_byte,
        "tokens": s.tokens, "bytes": text_bytes, "seconds": seconds,
    });
    EvalRow {
        model: model.into(),
        label: label.into(),
        num_predictions: s.tokens,
        cross_entropy: s.cross_entropy,
        bits_per_token: s.bits_per_token,
        perplexity: s.perplexity,
        uniform_baseline: baseline as f32,
        bits_per_byte,
        bytes: text_bytes,
        seconds,
        report,
    }
}

/// The numbers both scorers end in.
struct Stats {
    tokens: usize,
    cross_entropy: f32,
    bits_per_token: f32,
    perplexity: f32,
}

impl Stats {
    /// From a summed negative log-likelihood (nats) over `tokens` predictions, as `eval-gguf`.
    fn from_nll(total_nll: f64, tokens: usize) -> Self {
        let cross_entropy = (total_nll / tokens as f64) as f32;
        Stats {
            tokens,
            cross_entropy,
            bits_per_token: cross_entropy / std::f32::consts::LN_2,
            perplexity: cross_entropy.exp(),
        }
    }
}

/// Read a text file for scoring; refuses devices and (unless `lossy`) invalid UTF-8.
fn read_text(what: &str, path: &str, lossy: bool) -> Result<String, String> {
    checked_path(what, path, true)?;
    let bytes = crate::infra::fmt_err(std::fs::read(path), &format!("cannot read {path}"))?;
    let text = if lossy {
        String::from_utf8_lossy(&bytes).into_owned()
    } else {
        String::from_utf8(bytes).map_err(|e| {
            format!("cannot read {path}: {e} (tick 'Decode invalid UTF-8 lossily' to replace bad bytes)")
        })?
    };
    if text.trim().is_empty() {
        return Err(format!("{what} {path} is empty"));
    }
    Ok(text)
}

fn nonblank(s: &Option<String>) -> Option<&str> {
    s.as_deref().map(str::trim).filter(|s| !s.is_empty())
}

/// Is `path` an imported Llama/Qwen GGUF (as opposed to a Ferrum model)?
fn is_imported(path: &str) -> bool {
    Gguf::open(path)
        .ok()
        .and_then(|g| g.architecture().map(|a| a == "llama" || a == "qwen2"))
        .unwrap_or(false)
}

/// The testable core of the Evaluate tab: everything `evaluate_slm` does without the job slot.
pub(crate) fn evaluate_now(p: &EvalParams) -> Result<Vec<EvalRow>, String> {
    checked_path("model path", &p.model_path, true)?;
    let mut texts: Vec<(&str, String)> = Vec::new();
    match (p.text.as_deref(), nonblank(&p.text_path)) {
        (Some(t), _) if !t.trim().is_empty() => texts.push(("held-out", t.to_string())),
        (_, Some(path)) => {
            texts.push(("held-out", read_text("held-out text path", path, p.lossy)?))
        }
        _ => return Err("provide held-out text or a text file to evaluate against".into()),
    }
    if let Some(path) = nonblank(&p.test_path) {
        texts.push(("test", read_text("test text path", path, p.lossy)?));
    }
    let name = file_name(&p.model_path);
    if is_imported(&p.model_path) {
        return score_llama(p, &name, &texts);
    }
    let slm = crate::infra::fmt_err(GenerativeSLM::load(&p.model_path), &format!("cannot load model {}", p.model_path))?;
    texts
        .iter()
        .map(|(label, text)| {
            let t0 = std::time::Instant::now();
            let ev = crate::infra::fmt_err(slm.evaluate(text), "evaluation failed")?;
            let s = Stats {
                tokens: ev.num_predictions,
                cross_entropy: ev.cross_entropy,
                bits_per_token: ev.bits_per_token,
                perplexity: ev.perplexity,
            };
            let seconds = t0.elapsed().as_secs_f64();
            Ok(row(
                &name,
                label,
                s,
                slm.meta.output_dim,
                text.len(),
                seconds,
            ))
        })
        .collect()
}

/// `eval-gguf` for the GUI: load at the chosen precision (with the same RAM guard), optionally
/// apply a fine-tune checkpoint, then `score_rolling` every text.
fn score_llama(
    p: &EvalParams,
    name: &str,
    texts: &[(&str, String)],
) -> Result<Vec<EvalRow>, String> {
    let resume = nonblank(&p.resume);
    if let Some(r) = resume {
        checked_path("checkpoint path", r, true)?;
    }
    // A checkpoint holds f32 weights, so applying one needs an f32 load.
    let prec = if resume.is_some() {
        None
    } else {
        parse_quant(&p.quant)?
    };
    let g = crate::infra::fmt_err(Gguf::open(&p.model_path), &format!("cannot open {}", p.model_path))?;
    let Ok(tok) = GgufTokenizer::from_gguf(&g) else { return Err("this GGUF has no tokenizer; perplexity needs one".to_string()); };
    let mut est = gguf_resident_bytes(&g, prec);
    if let Some(ckpt) = resume {
        // The model, the checkpoint's bytes and one decoded parameter at a time (as `eval-gguf`).
        let len =
            std::fs::metadata(ckpt).map_or(0, |m| usize::try_from(m.len()).unwrap_or(usize::MAX));
        est = est.saturating_add(len).saturating_add(est / 2);
    }
    let hint = "choose a smaller precision or tick 'load anyway'";
    ram_guard("resident memory", "", est, available_memory_bytes(), p.force, hint)?;
    let mut model = crate::infra::fmt_err(g.load_llama_prec(prec), "cannot load model")?;
    if let Some(ckpt) = resume {
        let bytes = crate::infra::fmt_err(std::fs::read(ckpt), &format!("cannot read checkpoint {ckpt}"))?;
        crate::infra::fmt_err(apply_checkpoint_weights(&mut model, bytes), &format!("cannot apply checkpoint {ckpt}"))?;
    }
    let max_window = model.cfg.context_len;
    let window = if p.ctx == 0 {
        DEFAULT_WINDOW.min(max_window)
    } else {
        p.ctx
    };
    if window < 2 || window > max_window {
        return Err(format!(
            "the scoring window must be between 2 and {max_window} tokens (got {window})"
        ));
    }
    texts
        .iter()
        .map(|(label, text)| {
            // The BOS token (when the tokenizer adds one) lets the first text token be scored too.
            let mut ids: Vec<usize> = tok.prefix_bos().into_iter().collect();
            ids.extend(tok.encode(text));
            if ids.len() < 2 {
                return Err(format!(
                    "the {label} text is too short: scoring needs at least two tokens"
                ));
            }
            let t0 = std::time::Instant::now();
            let score = crate::infra::fmt_err(model.score_rolling(&ids, window), "evaluation failed")?;
            let seconds = t0.elapsed().as_secs_f64();
            let s = Stats::from_nll(-score.logprob, score.n_tokens);
            Ok(row(
                name,
                label,
                s,
                model.cfg.vocab_size,
                text.len(),
                seconds,
            ))
        })
        .collect()
}

/// Score a model (a Ferrum model or an imported Llama/Qwen GGUF) against held-out text and an
/// optional test file; returns one table row per text.
#[tauri::command]
pub async fn evaluate_slm(
    state: State<'_, AppState>,
    params: EvalParams,
) -> Result<Vec<EvalRow>, CmdError> {
    // One pass of the engine's scorer, which has no stop hook: it can be refused while another
    // evaluation runs, but not cancelled.
    let job = state.jobs.start(Job::Evaluate)?;
    crate::infra::error::blocking(move || {
        let _job = job;
        evaluate_now(&params)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{Rng, TransformerConfig};

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_gui_eval_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn tiny_model(dir: &std::path::Path) -> String {
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
        let slm = GenerativeSLM::train_transformer_config_threaded(
            &corpus,
            &cfg,
            1,
            &mut Rng::new(3),
            |_, _| {},
        )
        .unwrap();
        let path = dir.join("m.gguf").to_string_lossy().into_owned();
        slm.save(&path).unwrap();
        path
    }

    fn params(model: &str) -> EvalParams {
        serde_json::from_value(serde_json::json!({
            "modelPath": model, "text": "the lazy dog. the quick brown fox.", "textPath": null
        }))
        .unwrap()
    }

    #[test]
    fn the_old_request_shape_still_parses_with_cli_defaults() {
        let p = params("m");
        assert_eq!((p.lossy, p.ctx, p.force), (false, 0, false));
        assert_eq!(p.quant, "f32");
        assert!(p.test_path.is_none() && p.resume.is_none());
    }

    #[test]
    fn a_ferrum_model_row_equals_the_library_and_the_cli_json() {
        let dir = scratch("slm");
        let model = tiny_model(&dir);
        let text = "the lazy dog. the quick brown fox.";
        let rows = evaluate_now(&params(&model)).unwrap();
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        let ev = GenerativeSLM::load(&model).unwrap().evaluate(text).unwrap();
        assert_eq!(r.label, "held-out");
        assert_eq!(r.model, "m.gguf");
        assert_eq!(r.num_predictions, ev.num_predictions);
        assert_eq!(r.cross_entropy.to_bits(), ev.cross_entropy.to_bits());
        assert_eq!(r.perplexity.to_bits(), ev.perplexity.to_bits());
        assert_eq!(r.bytes, text.len());
        // bits/byte is `eval`'s formula, and the JSON has exactly `eval --json`'s keys.
        let bpb = f64::from(ev.cross_entropy) * ev.num_predictions as f64
            / std::f64::consts::LN_2
            / text.len() as f64;
        assert_eq!(r.bits_per_byte, bpb);
        let keys: Vec<&str> = r
            .report
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let mut want = [
            "perplexity",
            "cross_entropy_nats",
            "bits_per_token",
            "bits_per_byte",
            "tokens",
            "bytes",
            "seconds",
        ];
        want.sort_unstable();
        let mut got = keys.clone();
        got.sort_unstable();
        assert_eq!(got, want);
        assert_eq!(r.report["tokens"], ev.num_predictions);
        assert_eq!(r.report["bytes"], text.len());
        assert_eq!(r.report["bits_per_byte"], bpb);
        assert!(r.seconds >= 0.0 && r.report["seconds"].as_f64().unwrap() == r.seconds);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_test_file_adds_a_second_row_and_lossy_decoding_is_opt_in() {
        let dir = scratch("pair");
        let model = tiny_model(&dir);
        let held = dir.join("held.txt");
        let test = dir.join("test.txt");
        std::fs::write(&held, "the quick brown fox jumps over the lazy dog. ").unwrap();
        std::fs::write(&test, b"the lazy dog \xff jumps over the fox. ").unwrap(); // invalid UTF-8
        let mut p = params(&model);
        p.text = None;
        p.text_path = Some(held.to_string_lossy().into_owned());
        p.test_path = Some(test.to_string_lossy().into_owned());
        let e = evaluate_now(&p).unwrap_err();
        assert!(e.contains("not valid") || e.contains("UTF-8"), "{e}");
        assert!(e.contains("lossily"), "{e}");
        p.lossy = true;
        let rows = evaluate_now(&p).unwrap();
        assert_eq!(
            rows.iter().map(|r| r.label.as_str()).collect::<Vec<_>>(),
            ["held-out", "test"]
        );
        let slm = GenerativeSLM::load(&model).unwrap();
        let lossy = String::from_utf8_lossy(b"the lazy dog \xff jumps over the fox. ").into_owned();
        assert_eq!(
            rows[1].num_predictions,
            slm.evaluate(&lossy).unwrap().num_predictions
        );
        assert_eq!(rows[1].bytes, lossy.len());
        // Pasted text wins over the held-out file; the test file is still scored.
        p.text = Some("the lazy dog.".into());
        let rows = evaluate_now(&p).unwrap();
        assert_eq!(rows[0].bytes, "the lazy dog.".len());
        assert_eq!(rows.len(), 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn evaluate_errors_name_the_problem() {
        let dir = scratch("err");
        let model = tiny_model(&dir);
        let mut p = params(&model);
        p.text = Some("  ".into());
        assert!(evaluate_now(&p)
            .unwrap_err()
            .contains("provide held-out text"));
        p.text_path = Some(dir.join("nope.txt").to_string_lossy().into_owned());
        assert!(evaluate_now(&p).unwrap_err().contains("does not exist"));
        let empty = dir.join("empty.txt");
        std::fs::write(&empty, "  \n").unwrap();
        p.text_path = Some(empty.to_string_lossy().into_owned());
        assert!(evaluate_now(&p).unwrap_err().contains("is empty"));
        p.text_path = None;
        p.text = Some("ok".into());
        p.model_path = dir.join("missing.gguf").to_string_lossy().into_owned();
        assert!(evaluate_now(&p).unwrap_err().contains("does not exist"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_imported_gguf_is_scored_like_eval_gguf() {
        let dir = scratch("llama");
        let src = dir.join("base.gguf");
        std::fs::write(&src, crate::test_support::tiny_llama_gguf(8, 2, 2, 16, 32)).unwrap();
        let text = "hi abc hi abc hi abc hi abc";
        let mut p = params(&src.to_string_lossy());
        p.text = Some(text.into());
        let rows = evaluate_now(&p).unwrap();
        assert_eq!(rows.len(), 1);

        // The library call `eval-gguf` makes: BOS prefix, window min(512, context).
        let g = Gguf::open(src.to_str().unwrap()).unwrap();
        let tok = GgufTokenizer::from_gguf(&g).unwrap();
        let model = g.load_llama_prec(None).unwrap();
        let mut ids: Vec<usize> = tok.prefix_bos().into_iter().collect();
        ids.extend(tok.encode(text));
        let score = model.score_rolling(&ids, 128).unwrap();
        let want = Stats::from_nll(-score.logprob, score.n_tokens);
        let r = &rows[0];
        assert_eq!(r.num_predictions, want.tokens);
        assert_eq!(r.cross_entropy.to_bits(), want.cross_entropy.to_bits());
        assert_eq!(r.perplexity.to_bits(), want.perplexity.to_bits());
        assert_eq!(r.uniform_baseline, 32.0);
        assert_eq!(r.report["tokens"], want.tokens);

        // Another precision and an explicit window run; a bad window or precision is refused.
        p.quant = "int8".into();
        p.ctx = 16;
        assert!(evaluate_now(&p).is_ok());
        p.ctx = 1;
        assert!(evaluate_now(&p).unwrap_err().contains("window"));
        p.ctx = 129;
        assert!(evaluate_now(&p).unwrap_err().contains("between 2 and 128"));
        p.ctx = 0;
        p.quant = "int3".into();
        assert!(evaluate_now(&p).unwrap_err().contains("quant"));
        p.quant = "f32".into();
        p.resume = Some(dir.join("none.flck").to_string_lossy().into_owned());
        assert!(evaluate_now(&p).unwrap_err().contains("does not exist"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_text_of_one_token_cannot_be_scored() {
        let dir = scratch("short");
        let src = dir.join("base.gguf");
        let tiny = crate::test_support::Tiny {
            add_bos: Some(false),
            ..Default::default()
        };
        std::fs::write(&src, tiny.build()).unwrap();
        let mut p = params(&src.to_string_lossy());
        p.text = Some("h".into());
        assert_eq!(
            evaluate_now(&p).unwrap_err(),
            "the held-out text is too short: scoring needs at least two tokens"
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
