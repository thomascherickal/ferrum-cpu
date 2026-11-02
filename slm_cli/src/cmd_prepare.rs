//! `train_transformer prepare`: parse the flags, run [`slm_cli::prepare`], print the summary.

use crate::args::Args;
use slm_cli::prepare::{prepare, PrepareOptions};

pub(crate) fn cmd_prepare(args: &Args, raw: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let usage =
        "usage: train_transformer prepare --input <file|dir> [--input ...] --out <dir> [options]";
    let inputs = args.get_all("input");
    if inputs.is_empty() {
        return Err(usage.into());
    }
    let out = args.value("out").ok_or(usage)?;
    let o = PrepareOptions {
        inputs,
        out: out.to_string(),
        doc_sep: args
            .value("doc-sep")
            .unwrap_or(ferrum_core::DOC_SEP)
            .to_string(),
        lossy: args.has("lossy"),
        nfkc: args.has("nfkc"),
        lowercase: args.has("lowercase"),
        strip_gutenberg: args.has("strip-gutenberg"),
        min_chars: args.get("min-chars", 1),
        max_chars: args.get("max-chars", 0),
        dedup: args.has("dedup"),
        val_frac: args.get("val-frac", 0.05),
        test_frac: args.get("test-frac", 0.05),
        seed: args.get("seed", 42),
        args: raw.to_vec(),
        stop: ferrum_core::StopCondition::none(),
    };
    let report = prepare(&o, &mut |_| {})?;
    println!("{}", report.summary(&o));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::PREPARE_FLAGS;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn cmd_prepare_requires_input_and_out() {
        let none = Args::parse_for("prepare", &s(&["--out", "o"]), PREPARE_FLAGS).unwrap();
        assert!(cmd_prepare(&none, &[])
            .unwrap_err()
            .to_string()
            .contains("usage: train_transformer prepare"));
        let no_out = Args::parse_for("prepare", &s(&["--input", "x"]), PREPARE_FLAGS).unwrap();
        assert!(cmd_prepare(&no_out, &[])
            .unwrap_err()
            .to_string()
            .contains("usage: train_transformer prepare"));
    }

    #[test]
    fn cmd_prepare_runs_the_pipeline_and_prints_the_summary() {
        let d = std::env::temp_dir().join(format!("ferrum_cmd_prepare_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("c.txt"), "one document<|endoftext|>another document").unwrap();
        let (input, out) = (d.join("c.txt"), d.join("out"));
        let raw = s(&[
            "--input",
            input.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--val-frac",
            "0",
            "--test-frac",
            "0",
        ]);
        let a = Args::parse_for("prepare", &raw, PREPARE_FLAGS).unwrap();
        cmd_prepare(&a, &raw).unwrap();
        assert!(out.join("train.txt").is_file() && out.join("manifest.json").is_file());
        let _ = std::fs::remove_dir_all(&d);
    }
}
