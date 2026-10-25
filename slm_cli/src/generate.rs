//! `run` and `generate`: sample text from a Ferrum model.

use crate::args::*;
use crate::train::*;
use crate::util::*;
use crate::validate::*;
use ferrum_core::{GenerativeSLM, Rng};

pub(crate) fn cmd_run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 3 {
        return Err(
            "usage: train_transformer run <corpus.txt> <model.gguf> <seed text> [options]".into(),
        );
    }
    sampling_params(args, 0.8, "repetition-penalty")?; // fail before training starts
    require_resume_file(args.value("resume"))?; // also when the model exists and training is skipped
    let corpus_path = &args.positional[0];
    let model_path = &args.positional[1];
    let seed_text = args.positional[2..].join(" ");
    let corpus = read_corpus(corpus_path)?;

    let slm = if std::path::Path::new(model_path).exists() && !args.has("force") {
        let slm = GenerativeSLM::load(model_path)?;
        println!("Loaded saved weights from {model_path} — no retraining needed.\n");
        slm
    } else {
        train_and_save(&corpus, corpus_path, model_path, args)?
    };

    let out = generate_text(&slm, &seed_text, args)?;
    if !args.has("stream") {
        println!("{out}");
    }
    Ok(())
}

pub(crate) fn cmd_generate(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 2 {
        return Err("usage: train_transformer generate <model.gguf> <seed text> [options]".into());
    }
    sampling_params(args, 0.8, "repetition-penalty")?; // fail before the model loads
    let model_path = &args.positional[0];
    // Join remaining positionals so multi-word seeds work without quoting.
    let seed_text = args.positional[1..].join(" ");

    let slm = GenerativeSLM::load(model_path)
        .map_err(|e| format!("cannot load model {model_path}: {e}"))?;
    let out = generate_text(&slm, &seed_text, args)?;
    // Streaming already printed the text live; avoid printing it twice.
    if !args.has("stream") {
        println!("{out}");
    }
    Ok(())
}

pub(crate) fn generate_text(
    slm: &GenerativeSLM,
    seed_text: &str,
    args: &Args,
) -> Result<String, Box<dyn std::error::Error>> {
    let num_chars: usize = args.get("chars", 200);
    let params = sampling_params(args, 0.8, "repetition-penalty")?;
    let seed: u64 = args.get("gen-seed", time_seed());

    let stops = args.get_all("stop");
    if args.value("max-tokens").is_some() || !stops.is_empty() {
        // Token-budgeted path: any prompt length, stop strings excluded.
        // ponytail: not streamed live (`generate_tokens` returns the whole
        // continuation); add a callback variant if --stop needs live output.
        let max_tokens: usize = args.get("max-tokens", num_chars);
        let g = slm.generate_tokens(seed_text, max_tokens, &stops, &params, &mut Rng::new(seed))?;
        let out = format!("{seed_text}{}", g.text);
        if args.has("stream") {
            println!("{out}"); // callers skip their own print when streaming
        }
        return Ok(out);
    }

    let mut rng = Rng::new(seed);
    if args.has("stream") {
        // Print the seed, then each generated fragment as it lands, flushing so
        // the terminal shows the completion appearing live.
        use std::io::Write;
        print!("{seed_text}");
        let _ = std::io::stdout().flush();
        let out = slm.generate_stream_with(seed_text, num_chars, &params, &mut rng, |frag| {
            print!("{frag}");
            let _ = std::io::stdout().flush();
        })?;
        println!();
        Ok(out)
    } else {
        Ok(slm.generate_with(seed_text, num_chars, &params, &mut rng)?)
    }
}
