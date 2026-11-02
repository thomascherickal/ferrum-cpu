//! `eval`: bits per byte / perplexity of a model on a text file.

use crate::args::*;
use crate::util::*;
use ferrum_core::GenerativeSLM;
use std::time::Instant;

/// The numbers `eval` and `eval-gguf` both report.
pub(crate) struct Stats {
    pub(crate) tokens: usize,
    pub(crate) cross_entropy: f32,
    pub(crate) bits_per_token: f32,
    pub(crate) perplexity: f32,
}

impl Stats {
    /// From a summed negative log-likelihood (nats) over `tokens` predictions.
    pub(crate) fn from_nll(total_nll: f64, tokens: usize) -> Self {
        let cross_entropy = (total_nll / tokens as f64) as f32;
        Stats {
            tokens,
            cross_entropy,
            bits_per_token: cross_entropy / std::f32::consts::LN_2,
            perplexity: cross_entropy.exp(),
        }
    }
}

/// Print the report (and write `--json`) in the shared `eval` format.
pub(crate) fn report(
    model_path: &str,
    text_path: &str,
    text: &str,
    eval: &Stats,
    seconds: f64,
    json: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = text.len();
    let bits_per_byte =
        f64::from(eval.cross_entropy) * eval.tokens as f64 / std::f64::consts::LN_2 / bytes as f64;
    println!("Model        : {model_path}");
    println!(
        "Held-out text: {text_path}  ({} chars)",
        text.chars().count()
    );
    println!("Predictions  : {}", eval.tokens);
    println!("Cross-entropy: {:.4} nats/token", eval.cross_entropy);
    println!("Bits/token   : {:.4}", eval.bits_per_token);
    println!("Perplexity   : {:.4}", eval.perplexity);
    println!("Bits/byte    : {bits_per_byte:.4}");
    println!("Seconds      : {seconds:.2}");
    if let Some(path) = json {
        let v = serde_json::json!({
            "perplexity": eval.perplexity, "cross_entropy_nats": eval.cross_entropy,
            "bits_per_token": eval.bits_per_token, "bits_per_byte": bits_per_byte,
            "tokens": eval.tokens, "bytes": bytes, "seconds": seconds,
        });
        std::fs::write(path, serde_json::to_string_pretty(&v)? + "\n")
            .map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    Ok(())
}

pub(crate) fn cmd_eval(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 2 {
        return Err("usage: train_transformer eval <model.gguf> <heldout.txt>".into());
    }
    if let Some(json) = args.value("json") {
        crate::validate::check_writable(json)?;
    }
    let model_path = &args.positional[0];
    let text_path = &args.positional[1];

    let slm = GenerativeSLM::load(model_path)
        .map_err(|e| format!("cannot load model {model_path}: {e}"))?;
    let text = read_corpus(text_path)?;

    let t0 = Instant::now();
    let eval = slm.evaluate(&text)?;
    let seconds = t0.elapsed().as_secs_f64();
    let stats = Stats {
        tokens: eval.num_predictions,
        cross_entropy: eval.cross_entropy,
        bits_per_token: eval.bits_per_token,
        perplexity: eval.perplexity,
    };
    report(
        model_path,
        text_path,
        &text,
        &stats,
        seconds,
        args.value("json"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_nll_converts_nats_to_bits_and_perplexity() {
        let s = Stats::from_nll(4.0 * std::f64::consts::LN_2, 4);
        assert!((s.cross_entropy - std::f32::consts::LN_2).abs() < 1e-6);
        assert!((s.bits_per_token - 1.0).abs() < 1e-6);
        assert!((s.perplexity - 2.0).abs() < 1e-5);
    }

    #[test]
    fn report_writes_the_json_and_names_the_path_when_it_cannot() {
        let d = std::env::temp_dir().join(format!("ferrum_eval_report_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let stats = Stats::from_nll(8.0, 4);
        let ok = d.join("m.json");
        report("m.gguf", "t.txt", "abcd", &stats, 0.5, ok.to_str()).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&ok).unwrap()).unwrap();
        assert_eq!(
            (v["tokens"].as_u64(), v["bytes"].as_u64()),
            (Some(4), Some(4))
        );
        let bad = d.join("no-such-dir/m.json");
        let e = report("m.gguf", "t.txt", "abcd", &stats, 0.5, bad.to_str())
            .unwrap_err()
            .to_string();
        assert!(
            e.starts_with("cannot write ") && e.contains("no-such-dir"),
            "{e}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
