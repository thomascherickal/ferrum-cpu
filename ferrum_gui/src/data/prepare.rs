//! The Prepare tab: `train_transformer prepare` with the same code. The files
//! are read and written by `slm_cli::prepare` in the backend (a corpus never
//! passes through the webview), so for the same options the splits and the
//! manifest are byte-identical to the CLI's. The manifest `args` are the
//! equivalent CLI flags, so a GUI run can be repeated on the command line.

use crate::infra::error::CmdError;
use crate::infra::jobs::Job;
use crate::infra::paths::{checked_dir, checked_input};
use crate::AppState;
use ferrum_core::{StopCondition, DOC_SEP};
use serde::{Deserialize, Serialize};
use slm_cli::prepare::{prepare, PrepareOptions, PrepareProgress};
use std::collections::BTreeMap;
use tauri::{AppHandle, Emitter, Runtime, State};

fn d_frac() -> f64 {
    0.05
}
fn d_seed() -> u64 {
    42
}
fn d_min() -> usize {
    1
}
fn d_sep() -> String {
    DOC_SEP.to_string()
}

/// The form of the Prepare tab; the defaults are the CLI's.
#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PrepareParams {
    /// Files and folders (folders are walked, hidden files skipped).
    pub inputs: Vec<String>,
    pub out_dir: String,
    #[serde(default = "d_frac")]
    pub val_frac: f64,
    #[serde(default = "d_frac")]
    pub test_frac: f64,
    #[serde(default = "d_seed")]
    pub seed: u64,
    #[serde(default = "d_min")]
    pub min_chars: usize,
    #[serde(default)]
    pub max_chars: usize,
    #[serde(default = "d_sep")]
    pub doc_sep: String,
    #[serde(default)]
    pub dedup: bool,
    #[serde(default)]
    pub nfkc: bool,
    #[serde(default)]
    pub lowercase: bool,
    #[serde(default)]
    pub strip_gutenberg: bool,
    #[serde(default)]
    pub lossy: bool,
}

/// One written split.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SplitSummary {
    pub path: String,
    pub documents: u64,
    pub chars: u64,
    pub bytes: u64,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PrepareResult {
    pub out_dir: String,
    pub manifest_path: String,
    pub files: usize,
    pub train: SplitSummary,
    pub val: SplitSummary,
    pub test: SplitSummary,
    /// Rejected documents by reason (all five reasons, zeros included).
    pub rejected: BTreeMap<String, u64>,
    /// Kept documents that had invalid bytes replaced (only with `lossy`).
    pub lossy_repaired: u64,
    /// The one-line summary `train_transformer prepare` prints.
    pub summary: String,
    pub seconds: f64,
}

/// The CLI flags that mean `p` (what the manifest records as `args`).
pub(crate) fn cli_args(p: &PrepareParams) -> Vec<String> {
    let mut a = Vec::new();
    for i in &p.inputs {
        a.extend(["--input".to_string(), i.clone()]);
    }
    a.extend(["--out".to_string(), p.out_dir.clone()]);
    if p.doc_sep != DOC_SEP {
        a.extend(["--doc-sep".to_string(), p.doc_sep.clone()]);
    }
    for (on, flag) in [
        (p.lossy, "--lossy"),
        (p.nfkc, "--nfkc"),
        (p.lowercase, "--lowercase"),
        (p.strip_gutenberg, "--strip-gutenberg"),
    ] {
        if on {
            a.push(flag.to_string());
        }
    }
    for (flag, v) in [
        ("--min-chars", p.min_chars.to_string()),
        ("--max-chars", p.max_chars.to_string()),
    ] {
        a.extend([flag.to_string(), v]);
    }
    if p.dedup {
        a.push("--dedup".to_string());
    }
    for (flag, v) in [
        ("--val-frac", p.val_frac.to_string()),
        ("--test-frac", p.test_frac.to_string()),
        ("--seed", p.seed.to_string()),
    ] {
        a.extend([flag.to_string(), v]);
    }
    a
}

/// Check the paths and turn the form into library options.
fn options(p: &PrepareParams, stop: StopCondition) -> Result<PrepareOptions, String> {
    let inputs: Vec<String> = p
        .inputs
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if inputs.is_empty() {
        return Err("add at least one input file or folder".into());
    }
    for i in &inputs {
        checked_input("input", i)?;
    }
    checked_dir("output folder", &p.out_dir)?;
    let p = PrepareParams {
        inputs,
        out_dir: p.out_dir.trim().to_string(),
        ..p.clone()
    };
    Ok(PrepareOptions {
        args: cli_args(&p),
        inputs: p.inputs,
        out: p.out_dir,
        doc_sep: p.doc_sep,
        lossy: p.lossy,
        nfkc: p.nfkc,
        lowercase: p.lowercase,
        strip_gutenberg: p.strip_gutenberg,
        min_chars: p.min_chars,
        max_chars: p.max_chars,
        dedup: p.dedup,
        val_frac: p.val_frac,
        test_frac: p.test_frac,
        seed: p.seed,
        stop,
    })
}

/// The CLI's "pass --lossy" advice names a flag the GUI does not have.
fn gui_wording(e: String) -> String {
    e.replace(
        "(pass --lossy to replace invalid bytes)",
        "(tick 'Replace invalid UTF-8 bytes' to repair them instead)",
    )
    .replace(
        "--val-frac and --test-frac",
        "the validation and test fractions",
    )
    .replace("use --doc-sep", "change the document separator")
}

fn split(manifest: &serde_json::Value, name: &str) -> SplitSummary {
    let o = &manifest["outputs"][name];
    SplitSummary {
        path: o["path"].as_str().unwrap_or_default().to_string(),
        documents: o["documents"].as_u64().unwrap_or(0),
        chars: o["chars"].as_u64().unwrap_or(0),
        bytes: o["bytes"].as_u64().unwrap_or(0),
    }
}

/// The testable core: everything `prepare_dataset` does except the job slot and the events.
pub(crate) fn do_prepare(
    p: &PrepareParams,
    stop: StopCondition,
    progress: &mut dyn FnMut(PrepareProgress),
) -> Result<PrepareResult, String> {
    let o = options(p, stop)?;
    let t0 = std::time::Instant::now();
    let r = prepare(&o, progress).map_err(gui_wording)?;
    let files = r.manifest["inputs"].as_array().map_or(0, Vec::len);
    Ok(PrepareResult {
        out_dir: o.out.clone(),
        manifest_path: r.manifest_path.to_string_lossy().into_owned(),
        files,
        train: split(&r.manifest, "train"),
        val: split(&r.manifest, "val"),
        test: split(&r.manifest, "test"),
        summary: r.summary(&o),
        rejected: r.rejected,
        lossy_repaired: r.lossy_repaired,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

/// Clean, deduplicate and split text files into train/val/test with a manifest.
/// Progress streams as `prepare-progress` events; `cancel_job("prepare")` ends
/// the run between documents and leaves every previous output untouched.
#[tauri::command]
pub async fn prepare_dataset<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    params: PrepareParams,
) -> Result<PrepareResult, CmdError> {
    let job = state.jobs.start(Job::Prepare)?;
    crate::infra::error::blocking(move || {
        let mut emit = |p: PrepareProgress| {
            let _ = app.emit(
                "prepare-progress",
                serde_json::json!({
                    "file": p.file, "files": p.files, "path": p.path,
                    "bytesDone": p.bytes_done, "bytesTotal": p.bytes_total, "documents": p.documents,
                }),
            );
        };
        do_prepare(&params, job.stop().clone(), &mut emit)
    })
    .await
}

#[cfg(test)]
mod tests;
