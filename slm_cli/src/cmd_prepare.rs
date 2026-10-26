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
}
