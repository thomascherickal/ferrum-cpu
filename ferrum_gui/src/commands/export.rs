//! The Export tab: write a loaded, optionally fine-tuned, llama/qwen2 model
//! back out as a runnable GGUF at a chosen quantisation.

use super::guard::{available_memory_bytes, gguf_resident_bytes};
use crate::error::CmdError;
use crate::jobs::Job;
use crate::paths::checked_path;
use crate::AppState;
use ferrum_core::{Gguf, LlamaTrainer};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Runtime, State};

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
) -> Result<ExportResult, CmdError> {
    let job = state.jobs.start(Job::Export)?;
    crate::error::blocking(move || {
        ferrum_core::set_verbose(params.verbose);
        let emit = |m: &str| {
            let _ = app.emit("export-progress", m.to_string());
        };
        let r = do_export(&params, &emit, &|| job.cancelled());
        ferrum_core::set_verbose(false);
        r
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::GgufQuant;

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
