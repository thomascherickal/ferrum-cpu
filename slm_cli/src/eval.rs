//! `eval`: bits per byte / perplexity of a model on a text file.

use crate::args::*;
use crate::util::*;
use ferrum_core::GenerativeSLM;
use std::time::Instant;

pub(crate) fn cmd_eval(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 2 {
        return Err("usage: train_transformer eval <model.gguf> <heldout.txt>".into());
    }
    let model_path = &args.positional[0];
    let text_path = &args.positional[1];

    let slm = GenerativeSLM::load(model_path)
        .map_err(|e| format!("cannot load model {model_path}: {e}"))?;
    let text = read_corpus(text_path)?;

    let t0 = Instant::now();
    let eval = slm.evaluate(&text)?;
    let seconds = t0.elapsed().as_secs_f64();
    let bytes = text.len();
    let bits_per_byte = f64::from(eval.cross_entropy) * eval.num_predictions as f64
        / std::f64::consts::LN_2
        / bytes as f64;
    println!("Model        : {model_path}");
    println!(
        "Held-out text: {text_path}  ({} chars)",
        text.chars().count()
    );
    println!("Predictions  : {}", eval.num_predictions);
    println!("Cross-entropy: {:.4} nats/token", eval.cross_entropy);
    println!("Bits/token   : {:.4}", eval.bits_per_token);
    println!("Perplexity   : {:.4}", eval.perplexity);
    println!("Bits/byte    : {bits_per_byte:.4}");
    println!("Seconds      : {seconds:.2}");
    if let Some(path) = args.value("json") {
        let v = serde_json::json!({
            "perplexity": eval.perplexity, "cross_entropy_nats": eval.cross_entropy,
            "bits_per_token": eval.bits_per_token, "bits_per_byte": bits_per_byte,
            "tokens": eval.num_predictions, "bytes": bytes, "seconds": seconds,
        });
        if let Some(parent) = std::path::Path::new(path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&v)? + "\n")?;
    }
    Ok(())
}
