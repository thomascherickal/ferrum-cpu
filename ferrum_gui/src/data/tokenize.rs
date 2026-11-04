//! The Train tab's "Tokenize corpus": `train_transformer tokenize` with the same code
//! (`slm_cli::tokenize`). The text is streamed from disk and the ids are written as they are
//! produced, so a corpus larger than RAM works and nothing passes through the webview. The
//! ids are those of `ByteBpeTokenizer::encode_docs` over the files joined with the separator,
//! and `train` accepts the result in place of the text (detected by content).

use crate::error::CmdError;
use crate::jobs::Job;
use crate::paths::{checked_input, checked_path};
use crate::AppState;
use ferrum_core::{StopCondition, DOC_SEP};
use serde::{Deserialize, Serialize};
use slm_cli::tokenize::{tokenize, TokenizeOptions, TokenizeProgress};
use tauri::{AppHandle, Emitter, Runtime, State};

fn d_sep() -> String {
    DOC_SEP.to_string()
}

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TokenizeParams {
    /// A text file or a folder of them.
    pub input: String,
    /// The tokenizer file (made by a first training run with a Tokenizer file set).
    pub tokenizer_path: String,
    /// The token file to write.
    pub out_path: String,
    #[serde(default = "d_sep")]
    pub doc_sep: String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct TokenizeResult {
    pub out_path: String,
    pub tokens: u64,
    pub documents: u64,
    pub files: usize,
    pub vocab_size: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub seconds: f64,
    /// The line `train_transformer tokenize` prints.
    pub summary: String,
}

/// The CLI's hint names a command the GUI has as a button.
fn gui_wording(e: String) -> String {
    e.trim_start_matches("model format error: ")
        .replace(
            "(`prepare --lossy` repairs it)",
            "(the Prepare tab's 'Replace invalid UTF-8 bytes' repairs it)",
        )
        .replace(
            "create one with `train <corpus.txt> <model.gguf> --tokenizer",
            "create one by training once with the Tokenizer file field set to",
        )
}

/// The testable core: everything `tokenize_corpus` does except the job slot and the events.
pub(crate) fn do_tokenize(
    p: &TokenizeParams,
    stop: StopCondition,
    progress: &mut dyn FnMut(TokenizeProgress),
) -> Result<TokenizeResult, String> {
    let input = p.input.trim();
    if input.is_empty() {
        return Err("add the text file or folder to tokenize".into());
    }
    checked_input("corpus", input)?;
    checked_path("tokenizer path", &p.tokenizer_path, true)?;
    checked_path("token file path", &p.out_path, false)?;
    let o = TokenizeOptions {
        input: input.to_string(),
        tokenizer: p.tokenizer_path.trim().to_string(),
        out: p.out_path.trim().to_string(),
        doc_sep: p.doc_sep.clone(),
        stop,
    };
    let t0 = std::time::Instant::now();
    let r = tokenize(&o, progress).map_err(gui_wording)?;
    let seconds = t0.elapsed().as_secs_f64();
    Ok(TokenizeResult {
        summary: format!(
            "tokenized {} → {}: {} tokens, {} documents, vocab {} ({} bytes in, {} bytes out, {:.1}s)",
            o.input, o.out, r.tokens, r.documents, r.vocab_size, r.bytes_in, r.bytes_out, seconds
        ),
        out_path: o.out,
        tokens: r.tokens,
        documents: r.documents,
        files: r.files,
        vocab_size: r.vocab_size,
        bytes_in: r.bytes_in,
        bytes_out: r.bytes_out,
        seconds,
    })
}

/// Encode a corpus with a tokenizer file into a token file that Train can learn from.
/// Progress streams as `tokenize-progress` events; `cancel_job("tokenize")` ends the run
/// between documents and writes no file.
#[tauri::command]
pub async fn tokenize_corpus<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    params: TokenizeParams,
) -> Result<TokenizeResult, CmdError> {
    let job = state.jobs.start(Job::Tokenize)?;
    crate::error::blocking(move || {
        let mut emit = |p: TokenizeProgress| {
            let _ = app.emit(
                "tokenize-progress",
                serde_json::json!({
                    "file": p.file, "files": p.files, "path": p.path,
                    "bytesDone": p.bytes_done, "bytesTotal": p.bytes_total, "tokens": p.tokens,
                }),
            );
        };
        do_tokenize(&params, job.stop().clone(), &mut emit)
    })
    .await
}

#[cfg(test)]
mod tests;
