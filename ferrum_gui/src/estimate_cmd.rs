//! The `estimate_job` command: resolves file sizes and model facts, reads the
//! live hardware, and calls the pure [`crate::estimate::estimate`].

use crate::error::CmdError;
use crate::estimate::{estimate, Estimate, GgufFacts, JobKind, JobParams, SlmFacts, Throughput};
use crate::hardware;
use crate::paths::{checked_input, checked_path};
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

/// Bytes of every file the Prepare tab would read (`None` while no input is usable).
/// ponytail: walks the folders on every (debounced) form change; a huge tree makes the tooltip slow.
fn input_bytes(p: &JobParams) -> Option<u64> {
    let inputs: Vec<String> = p
        .input_paths
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| checked_input("input", s).is_ok())
        .collect();
    if inputs.is_empty() {
        return None;
    }
    let files =
        slm_cli::prepare::expand_inputs(&inputs, std::path::Path::new(p.out_dir.trim())).ok()?;
    Some(
        files
            .iter()
            .map(|f| std::fs::metadata(f).map_or(0, |m| m.len()))
            .sum(),
    )
}

/// The header facts of a token file, when `path` is one: tokens, documents, vocabulary.
fn token_file_facts(path: &str) -> Option<(u64, u64, usize)> {
    if path.trim().is_empty() || checked_path("path", path, true).is_err() {
        return None;
    }
    if !ferrum_core::is_gguf_file(path) {
        return None;
    }
    let f = ferrum_core::TokenFile::open(path).ok()?;
    Some((f.len() as u64, f.documents(), f.vocab_size()))
}

/// Fill in everything the pure estimate needs from the paths in `p`.
pub fn resolve(kind: JobKind, p: &mut JobParams) {
    p.corpus_bytes = file_len(&p.corpus_path).or(p.corpus_bytes);
    p.text_bytes = file_len(&p.text_path).or(p.text_bytes);
    p.test_bytes = file_len(&p.test_path).or(p.test_bytes);
    p.resume_bytes = file_len(&p.resume_path).or(p.resume_bytes);
    if kind == JobKind::Train {
        if let Some((tokens, docs, vocab)) = token_file_facts(&p.corpus_path) {
            p.corpus_tokens = Some(tokens);
            p.corpus_documents = Some(docs);
            p.corpus_vocab = Some(vocab);
        }
    }
    match kind {
        // Evaluate takes either kind of model: a Llama/Qwen file is scored like `eval-gguf`.
        JobKind::Evaluate if is_imported(&p.model_path) => p.gguf = gguf_facts(&p.model_path),
        JobKind::Generate | JobKind::Evaluate | JobKind::ModelInfo | JobKind::RuntimeExport => {
            p.slm = slm_facts(&p.model_path)
        }
        JobKind::GgufRun | JobKind::GgufFinetune | JobKind::GgufExport | JobKind::GgufInfo => {
            p.gguf = gguf_facts(&p.model_path)
        }
        JobKind::Prepare | JobKind::Tokenize => p.input_bytes = input_bytes(p).or(p.input_bytes),
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
) -> Result<Estimate, CmdError> {
    let bench = *state.bench.lock().unwrap_or_else(|e| e.into_inner());
    crate::error::blocking_ok(move || estimate_now(kind, params, bench)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_input_size_sums_files_and_folders_and_skips_hidden_files_and_the_output() {
        let d = std::env::temp_dir().join(format!("ferrum_gui_estcmd_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("sub/out")).unwrap();
        std::fs::write(d.join("a.txt"), vec![b'a'; 100]).unwrap();
        std::fs::write(d.join("sub/b.txt"), vec![b'b'; 50]).unwrap();
        std::fs::write(d.join("sub/.hidden"), vec![b'h'; 1000]).unwrap();
        std::fs::write(d.join("sub/out/train.txt"), vec![b't'; 1000]).unwrap();
        let p = |inputs: &[String]| JobParams {
            input_paths: inputs.to_vec(),
            out_dir: d.join("sub/out").to_string_lossy().into(),
            ..JobParams::default()
        };
        let a = d.join("a.txt").to_string_lossy().into_owned();
        let sub = d.join("sub").to_string_lossy().into_owned();
        assert_eq!(input_bytes(&p(&[a.clone(), sub.clone()])), Some(150));
        // Unusable entries (blank, missing, a device) are ignored; nothing usable gives no size.
        let junk = [
            "".to_string(),
            "/dev/null".to_string(),
            "/nope/x".to_string(),
        ];
        assert_eq!(input_bytes(&p(&junk)), None);
        assert_eq!(input_bytes(&p(&[a, junk[1].clone()])), Some(100));
        // Through `resolve`, which is what `estimate_job` calls.
        let mut jp = p(&[sub]);
        resolve(JobKind::Prepare, &mut jp);
        assert_eq!(jp.input_bytes, Some(50));
        let _ = std::fs::remove_dir_all(&d);
    }
}
