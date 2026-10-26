//! `run-gguf`: generate from an imported (Llama/Qwen-style) GGUF.

use crate::args::*;
use crate::memory::*;
use crate::util::*;
use crate::validate::*;
use std::time::Instant;

pub(crate) fn cmd_run_gguf(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use ferrum_core::{Gguf, GgufTokenizer, QKind, Rng};

    if args.positional.is_empty() {
        return Err("usage: train_transformer run-gguf <model.gguf> [prompt] \
                    [--quant int4|int8|f32] [--max N] [--temp F] \
                    [--top-k N] [--top-p F] [--rep F] [--ids \"1 2 3\"] [--raw]"
            .into());
    }
    require_resume_file(args.value("resume"))?;
    let params = sampling_params(args, 0.8, "rep")?; // fail before the model loads
                                                     // Parsed up front so a typo fails before a multi-GB load.
    let ids_arg: Option<Vec<usize>> = args
        .value("ids")
        .map(|ids| {
            ids.split_whitespace()
                .map(|s| {
                    s.parse()
                        .map_err(|_| format!("--ids: '{s}' is not a token id"))
                })
                .collect::<Result<_, String>>()
        })
        .transpose()?;
    let path = &args.positional[0];
    let prompt_text = args.positional[1..].join(" ");

    // A fine-tune checkpoint holds f32 weights, so applying one forces an f32
    // load regardless of --quant.
    let resume = args.value("resume").map(String::from);
    let quant = args.value("quant").unwrap_or("int4");
    let prec = if resume.is_some() {
        if quant != "int4" && quant != "f32" && quant != "none" {
            eprintln!("note: --resume applies f32 fine-tuned weights; ignoring --quant {quant}");
        }
        None
    } else {
        match quant {
            "int4" | "q4" => Some(QKind::Int4),
            "int8" | "q8" => Some(QKind::Int8),
            "f32" | "none" => None,
            other => return Err(format!("--quant must be int4|int8|f32 (got '{other}')").into()),
        }
    };

    // Streamed open: parse the header without reading the whole file.
    eprintln!("Opening {path} (streamed)…");
    let g = Gguf::open(path).map_err(|e| format!("cannot open GGUF {path}: {e}"))?;
    eprintln!(
        "  GGUF v{}   architecture = {}",
        g.version,
        g.architecture().unwrap_or("?")
    );

    // Memory guard before the (potentially multi-GB) load.
    let mut est = estimate_resident_bytes(&g, prec);
    eprintln!(
        "  estimated resident ≈ {:.2} GB  ({})",
        est as f64 / 1e9,
        if resume.is_some() {
            "f32, because --resume applies f32 weights".to_string()
        } else {
            format!("--quant {quant}")
        }
    );
    if let Some(ckpt) = &resume {
        // Applying a checkpoint peaks well above the resident model.
        est = est.max(resume_peak_bytes(est, file_len(ckpt)));
        eprintln!(
            "  estimated peak while applying {ckpt} ≈ {:.2} GB",
            est as f64 / 1e9
        );
    }
    if let Some(avail) = available_memory_bytes() {
        eprintln!("  available memory   ≈ {:.2} GB", avail as f64 / 1e9);
        if let Some(e) = memory_error(
            "resident memory",
            est,
            avail,
            args.has("force"),
            "try a smaller --quant, or pass --force to attempt it anyway.",
        ) {
            return Err(e.into());
        }
    }

    // Tokenizer is optional; without one, the prompt must be raw token IDs.
    let tok = GgufTokenizer::from_gguf(&g).ok();
    match &tok {
        Some(t) => eprintln!("  tokenizer = {:?}  (vocab {})", t.model(), t.vocab_size()),
        None => eprintln!("  tokenizer = none in file — supply --ids \"<token ids>\""),
    }

    eprintln!("Loading weights…");
    let t0 = Instant::now();
    let mut model = g
        .load_llama_prec(prec)
        .map_err(|e| format!("cannot load model: {e}"))?;
    eprintln!(
        "  loaded in {:.2}s: {} layers, dim {}, vocab {}, ctx {}",
        t0.elapsed().as_secs_f32(),
        model.cfg.n_layers,
        model.cfg.model_dim,
        model.cfg.vocab_size,
        model.cfg.context_len,
    );

    // Overlay fine-tuned weights from a checkpoint, if requested.
    if let Some(ckpt) = &resume {
        let bytes =
            std::fs::read(ckpt).map_err(|e| format!("cannot read checkpoint {ckpt}: {e}"))?;
        ferrum_core::apply_checkpoint_weights(&mut model, bytes)
            .map_err(|e| format!("cannot apply checkpoint {ckpt}: {e}"))?;
        eprintln!("  applied fine-tuned checkpoint {ckpt}");
    }

    // Build the prompt token IDs (explicit --ids win; else encode the text,
    // auto-wrapping instruct prompts in their chat template unless --raw).
    let mut fmt = None;
    let prompt_ids: Vec<usize> = if let Some(ids) = ids_arg {
        ids
    } else if let Some(t) = &tok {
        fmt = if args.has("raw") {
            None
        } else {
            ferrum_core::chat_format::detect(
                g.meta("tokenizer.chat_template").and_then(|v| v.as_str()),
                g.architecture(),
                |p| tok.as_ref().and_then(|t| t.token_id(p)).is_some(),
            )
        };
        let mut ids = Vec::new();
        if let Some(bos) = t.prefix_bos() {
            ids.push(bos);
        }
        match fmt {
            Some(f) => {
                eprintln!("  chat format = {f:?} (pass --raw to disable)");
                ids.extend(ferrum_core::chat_format::encode_segments(
                    t,
                    &ferrum_core::chat_format::render(f, &prompt_text),
                ));
            }
            None => ids.extend(t.encode(&prompt_text)),
        }
        ids
    } else {
        return Err(
            "this GGUF has no tokenizer; pass --ids \"<space-separated token ids>\"".into(),
        );
    };
    if prompt_ids.is_empty() {
        return Err("empty prompt — provide text (with a tokenizer) or --ids".into());
    }

    let max_new: usize = args.get("max", 64);
    let temp = params.temperature;
    let gen_seed: u64 = args.get("gen-seed", time_seed());
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
    let mut rng = Rng::new(gen_seed);

    eprintln!(
        "\nPrefilling {} prompt tokens, generating up to {max_new} (temp {temp})…",
        prompt_ids.len()
    );
    let t1 = Instant::now();
    let out_ids = model.generate(&prompt_ids, max_new, &params, &stops, &mut rng)?;
    let dt = t1.elapsed().as_secs_f32();

    // stdout carries only the generated text, so `run-gguf ... > out.txt` is clean.
    match &tok {
        Some(t) => println!("{prompt_text}{}", t.decode(&out_ids)),
        None => {
            let ids: Vec<String> = out_ids.iter().map(usize::to_string).collect();
            println!("token ids: {}", ids.join(" "));
        }
    }
    let room = model.cfg.context_len - prompt_ids.len();
    if max_new > room && out_ids.len() == room && out_ids.last().is_none_or(|t| !stops.contains(t))
    {
        eprintln!("stopped: context full at {} tokens", out_ids.len());
    }
    eprintln!(
        "[{} tokens in {:.2}s = {:.1} tok/s]",
        out_ids.len(),
        dt,
        out_ids.len() as f32 / dt.max(1e-6)
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// finetune-gguf: fine-tune an imported llama/qwen2 GGUF on a text corpus
// ─────────────────────────────────────────────────────────────────────────────
