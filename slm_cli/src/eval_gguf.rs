//! `eval-gguf`: perplexity and bits per byte of an imported (Llama/Qwen-style) GGUF on a text file.

use crate::args::*;
use crate::eval::{report, Stats};
use crate::memory::*;
use crate::util::*;
use crate::validate::*;
use std::time::Instant;

/// Default scoring window: long enough for useful context, short enough that a 32k-context
/// model is not scored with one enormous (quadratic) forward pass.
const DEFAULT_WINDOW: usize = 512;

pub(crate) fn cmd_eval_gguf(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use ferrum_core::{Gguf, GgufTokenizer, QKind};

    if args.positional.len() < 2 {
        return Err(
            "usage: train_transformer eval-gguf <model.gguf> <text.txt> \
                    [--ctx N] [--quant f32|int8|int4] [--resume ckpt.flck] [--json out.json]"
                .into(),
        );
    }
    require_resume_file(args.value("resume"), ".flck")?;
    if let Some(json) = args.value("json") {
        check_writable(json)?;
    }
    let (path, text_path) = (&args.positional[0], &args.positional[1]);
    let resume = args.value("resume");
    let quant = args.value("quant").unwrap_or("f32");
    let prec = match quant {
        "int4" | "q4" => Some(QKind::Int4),
        "int8" | "q8" => Some(QKind::Int8),
        "f32" | "none" => None,
        other => return Err(format!("--quant must be f32|int8|int4 (got '{other}')").into()),
    };
    // A checkpoint holds f32 weights, so applying one needs an f32 load.
    let prec = if resume.is_some() {
        if prec.is_some() {
            eprintln!("note: --resume applies f32 fine-tuned weights; ignoring --quant {quant}");
        }
        None
    } else {
        prec
    };
    let text = read_corpus(text_path)?;

    eprintln!("Opening {path} (streamed)…");
    let g = Gguf::open(path).map_err(|e| format!("cannot open GGUF {path}: {e}"))?;
    let tok = GgufTokenizer::from_gguf(&g)
        .map_err(|_| "this GGUF has no tokenizer; perplexity needs one")?;
    let mut est = estimate_resident_bytes(&g, prec);
    if let Some(ckpt) = resume {
        est = est.max(resume_peak_bytes(est, file_len(ckpt)));
    }
    if let Some(avail) = available_memory_bytes() {
        if let Some(e) = memory_error(
            "resident memory",
            est,
            avail,
            args.has("force"),
            "try --quant int8 or int4, or pass --force to attempt it anyway.",
        ) {
            return Err(e.into());
        }
    }
    eprintln!("Loading weights…");
    let mut model = g
        .load_llama_prec(prec)
        .map_err(|e| format!("cannot load model: {e}"))?;
    if let Some(ckpt) = resume {
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        ferrum_core::apply_checkpoint_weights(&mut model, bytes)
            .map_err(|e| format!("cannot apply checkpoint {ckpt}: {e}"))?;
        eprintln!("  applied fine-tuned checkpoint {ckpt}");
    }
    let max_window = model.cfg.context_len;
    let window = args.get("ctx", DEFAULT_WINDOW.min(max_window));
    if window < 2 || window > max_window {
        return Err(format!("--ctx must be between 2 and {max_window} (got {window})").into());
    }

    // The BOS token (when the tokenizer adds one) lets the first text token be scored too.
    let mut ids: Vec<usize> = tok.prefix_bos().into_iter().collect();
    ids.extend(tok.encode(&text));
    if ids.len() < 2 {
        return Err("the text is too short: scoring needs at least two tokens".into());
    }
    let t0 = Instant::now();
    let score = model.score_rolling(&ids, window)?;
    let seconds = t0.elapsed().as_secs_f64();
    let stats = Stats::from_nll(-score.logprob, score.n_tokens);
    report(path, text_path, &text, &stats, seconds, args.value("json"))
}
