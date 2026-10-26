//! The `estimate_job` command: resolves file sizes and model facts, reads the
//! live hardware, and calls the pure [`crate::estimate::estimate`].

use crate::estimate::{estimate, Estimate, GgufFacts, JobKind, JobParams, SlmFacts, Throughput};
use crate::hardware;
use crate::paths::checked_path;
use crate::AppState;
use ferrum_core::Gguf;
use tauri::State;

/// Size of an existing regular file the webview named (`None` if unusable).
fn file_len(path: &str) -> Option<u64> {
    if path.trim().is_empty() {
        return None;
    }
    checked_path("path", path, true).ok()?;
    std::fs::metadata(path).ok().map(|m| m.len())
}

fn open(path: &str) -> Option<(Gguf, u64)> {
    let len = file_len(path)?;
    Some((Gguf::open(path).ok()?, len))
}

fn elements(g: &Gguf, name: &str) -> u64 {
    g.tensor(name).map_or(0, |t| t.num_elements() as u64)
}

/// Facts of a Ferrum SLM from its GGUF header (no weights are read).
pub fn slm_facts(path: &str) -> Option<SlmFacts> {
    let (g, file_bytes) = open(path)?;
    let params = g
        .tensors
        .iter()
        .map(|t| t.num_elements() as u64)
        .sum::<u64>();
    let tables = elements(&g, "token_embd.weight") + elements(&g, "position_embd.weight");
    let vocab = g
        .tensor("token_embd.weight")
        .filter(|t| t.dims.len() == 2)
        .map_or(0, |t| t.dims[1] as usize);
    Some(SlmFacts {
        params,
        matmul_params: params.saturating_sub(tables),
        vocab,
        file_bytes,
    })
}

/// Is `path` a Llama/Qwen GGUF (rather than a Ferrum model)?
fn is_imported(path: &str) -> bool {
    open(path).is_some_and(|(g, _)| matches!(g.architecture(), Some("llama" | "qwen2")))
}

/// Facts of an imported Llama/Qwen GGUF from its header.
pub fn gguf_facts(path: &str) -> Option<GgufFacts> {
    let (g, file_bytes) = open(path)?;
    let arch = g.architecture().unwrap_or("llama").to_string();
    let meta = |k: &str| {
        g.meta(&format!("{arch}.{k}"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0)
    };
    Some(GgufFacts {
        params: g.tensors.iter().map(|t| t.num_elements() as u64).sum(),
        embed_params: elements(&g, "token_embd.weight"),
        has_output_head: g.tensor("output.weight").is_some(),
        n_layers: meta("block_count"),
        context_len: meta("context_length"),
        file_bytes,
    })
}

/// Fill in everything the pure estimate needs from the paths in `p`.
pub fn resolve(kind: JobKind, p: &mut JobParams) {
    p.corpus_bytes = file_len(&p.corpus_path).or(p.corpus_bytes);
    p.text_bytes = file_len(&p.text_path).or(p.text_bytes);
    p.test_bytes = file_len(&p.test_path).or(p.test_bytes);
    p.resume_bytes = file_len(&p.resume_path).or(p.resume_bytes);
    match kind {
        // Evaluate takes either kind of model: a Llama/Qwen file is scored like `eval-gguf`.
        JobKind::Evaluate if is_imported(&p.model_path) => p.gguf = gguf_facts(&p.model_path),
        JobKind::Generate | JobKind::Evaluate | JobKind::ModelInfo => {
            p.slm = slm_facts(&p.model_path)
        }
        JobKind::GgufRun | JobKind::GgufFinetune | JobKind::GgufExport | JobKind::GgufInfo => {
            p.gguf = gguf_facts(&p.model_path)
        }
        _ => {}
    }
}

/// Window-free core of [`estimate_job`]: live hardware, resolved files, no Tauri state.
pub fn estimate_now(kind: JobKind, mut params: JobParams, bench: Option<Throughput>) -> Estimate {
    resolve(kind, &mut params);
    estimate(kind, &params, &hardware::detect(), bench.as_ref())
}

/// Tauri command: time and RAM estimate for one button, from the live hardware.
#[tauri::command]
pub async fn estimate_job(
    state: State<'_, AppState>,
    kind: JobKind,
    params: JobParams,
) -> Result<Estimate, String> {
    let bench = *state.bench.lock().unwrap_or_else(|e| e.into_inner());
    tauri::async_runtime::spawn_blocking(move || estimate_now(kind, params, bench))
        .await
        .map_err(|e| format!("task error: {e}"))
}
