//! Flag-value validation shared by `train`, `generate`, `run-gguf` and `finetune-gguf`.

use crate::args::*;

/// A learning rate must be a positive finite number; anything else trains garbage.
pub(crate) fn check_lr(lr: f32) -> Result<(), Box<dyn std::error::Error>> {
    if lr.is_finite() && lr > 0.0 {
        Ok(())
    } else {
        Err(format!("--lr must be a positive, finite number (got {lr})").into())
    }
}

/// Regularisation / clipping flags: dropout in [0, 1), weight decay and gradient clip finite and
/// >= 0. NaN or out-of-range values otherwise train garbage that is only noticed at the end.
pub(crate) fn check_regularizers(dropout: f32, weight_decay: f32, clip: f32) -> Result<(), String> {
    if !(dropout.is_finite() && (0.0..1.0).contains(&dropout)) {
        return Err(format!("--dropout must be in [0, 1), got {dropout}"));
    }
    if !(weight_decay.is_finite() && weight_decay >= 0.0) {
        return Err(format!(
            "--weight_decay must be a finite number >= 0, got {weight_decay}"
        ));
    }
    if !(clip.is_finite() && clip >= 0.0) {
        return Err(format!("--clip must be a finite number >= 0, got {clip}"));
    }
    Ok(())
}

/// `--tokens` is a budget: a finite number >= 0 (0 = use --epochs). `nan`, `-5` and `inf` parse
/// but would mean "no budget" or "train forever".
pub(crate) fn check_token_budget(tokens: f64) -> Result<(), String> {
    if tokens.is_finite() && tokens >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "--tokens must be a finite number >= 0, got {tokens}"
        ))
    }
}

/// `--temp` must be a finite number >= 0 (0 = greedy). `nan`/`inf` parse as floats, and a
/// negative or NaN value would silently turn into greedy / near-greedy decoding.
pub(crate) fn check_temp(temp: f32) -> Result<(), String> {
    if temp.is_finite() && temp >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "--temp must be a finite number >= 0 (0 = greedy), got {temp}"
        ))
    }
}

/// The sampler flags shared by `generate`/`run` and `run-gguf`, validated: `--temp` (default
/// `temp_default`), `--top-k`, `--top-p` in [0, 1] and a positive finite repetition penalty
/// read from `rep_flag` (a value <= 0 or NaN would silently disable it).
pub(crate) fn sampling_params(
    args: &Args,
    temp_default: f32,
    rep_flag: &str,
) -> Result<ferrum_core::SamplingParams, String> {
    let params = ferrum_core::SamplingParams {
        temperature: args.get("temp", temp_default),
        top_k: args.get("top-k", 0),
        top_p: args.get("top-p", 1.0),
        repetition_penalty: args.get(rep_flag, 1.0),
        ..Default::default()
    };
    check_temp(params.temperature)?;
    if !(params.top_p.is_finite() && (0.0..=1.0).contains(&params.top_p)) {
        return Err(format!(
            "--top-p must be a number in [0, 1], got {}",
            params.top_p
        ));
    }
    if !(params.repetition_penalty.is_finite() && params.repetition_penalty > 0.0) {
        return Err(format!(
            "--{rep_flag} must be a positive, finite number, got {}",
            params.repetition_penalty
        ));
    }
    Ok(params)
}

pub(crate) use slm_cli::outputs::{check_distinct_paths, check_writable};

/// Most data-parallel shards / worker threads a run may ask for; more is never useful.
pub(crate) const MAX_THREADS: usize = 1024;

/// `--epochs 0` trains nothing yet saves a model; with a `--tokens` budget the epoch count is
/// ignored, so `train` only checks it when there is no budget (`hint_tokens`).
pub(crate) fn check_epochs(epochs: usize, hint_tokens: bool) -> Result<(), String> {
    if epochs >= 1 {
        return Ok(());
    }
    let hint = if hint_tokens { ", or set --tokens" } else { "" };
    Err(format!("--epochs must be at least 1{hint} (got {epochs})"))
}

/// `--batch` must be at least 1 and `--threads` (0 = auto) at most [`MAX_THREADS`].
pub(crate) fn check_batch_threads(batch: usize, threads: usize) -> Result<(), String> {
    if batch == 0 {
        return Err("--batch must be at least 1 (got 0)".into());
    }
    if threads > MAX_THREADS {
        return Err(format!(
            "--threads must be at most {MAX_THREADS} (got {threads}); 0 = auto"
        ));
    }
    Ok(())
}

/// `--resume` must name an existing file; checked before any multi-GB model
/// load. Blank means "none" (as in `export-gguf`). `ext` names the checkpoint kind the command
/// resumes: `.fckp` for train/run, `.flck` for the llama fine-tune commands.
pub(crate) fn require_resume_file(resume: Option<&str>, ext: &str) -> Result<(), String> {
    match resume.map(str::trim).filter(|s| !s.is_empty()) {
        Some(p) if !std::path::Path::new(p).is_file() => Err(format!(
            "checkpoint {p} does not exist (--resume needs an existing {ext} file)"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn require_resume_file_names_the_missing_checkpoint() {
        assert!(require_resume_file(None, ".fckp").is_ok());
        assert!(
            require_resume_file(Some("   "), ".fckp").is_ok(),
            "blank = none"
        );
        let e = require_resume_file(Some("/nonexistent/x.flck"), ".flck").unwrap_err();
        assert!(
            e.contains("/nonexistent/x.flck")
                && e.contains("does not exist")
                && e.contains("needs an existing .flck file"),
            "{e}"
        );
        let dir = std::env::temp_dir().join(format!("ferrum_resume_chk_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(
            require_resume_file(Some(dir.to_str().unwrap()), ".flck").is_err(),
            "a directory is not a file"
        );
        let f = dir.join("c.flck");
        std::fs::write(&f, b"x").unwrap();
        assert!(require_resume_file(Some(f.to_str().unwrap()), ".flck").is_ok());
    }

    #[test]
    fn sampling_flags_are_validated_with_per_command_names() {
        let spec_g = spec_for("generate").unwrap();
        let spec_r = spec_for("run-gguf").unwrap();
        let g = |a: &[&str]| Args::parse(&s(a), spec_g).unwrap();
        let r = |a: &[&str]| Args::parse(&s(a), spec_r).unwrap();
        // Defaults, and explicit valid values (0 = greedy, top-p 0 and 1 are the edges).
        let p = sampling_params(&g(&[]), 0.8, "repetition-penalty").unwrap();
        assert_eq!(
            (p.temperature, p.top_k, p.top_p, p.repetition_penalty),
            (0.8, 0, 1.0, 1.0)
        );
        let ok = g(&[
            "--temp",
            "0",
            "--top-p",
            "0",
            "--top-k",
            "5",
            "--repetition-penalty",
            "1.3",
        ]);
        let p = sampling_params(&ok, 0.8, "repetition-penalty").unwrap();
        assert_eq!(
            (p.temperature, p.top_p, p.top_k, p.repetition_penalty),
            (0.0, 0.0, 5, 1.3)
        );
        assert!(sampling_params(&g(&["--top-p", "1"]), 0.8, "repetition-penalty").is_ok());
        // run-gguf names its flag --rep.
        let p = sampling_params(&r(&["--rep", "2"]), 0.8, "rep").unwrap();
        assert_eq!(p.repetition_penalty, 2.0);
        // Bad values name the flag.
        for (a, flag) in [
            (&["--temp", "-0.5"][..], "--temp"),
            (&["--temp", "nan"], "--temp"),
            (&["--temp", "inf"], "--temp"),
            (&["--top-p", "1.5"], "--top-p"),
            (&["--top-p", "-0.1"], "--top-p"),
            (&["--top-p", "nan"], "--top-p"),
            (&["--repetition-penalty", "0"], "--repetition-penalty"),
            (&["--repetition-penalty", "-1"], "--repetition-penalty"),
            (&["--repetition-penalty", "nan"], "--repetition-penalty"),
            (&["--repetition-penalty", "inf"], "--repetition-penalty"),
        ] {
            let e = sampling_params(&g(a), 0.8, "repetition-penalty").unwrap_err();
            assert!(e.contains(flag), "{a:?}: {e}");
        }
        let e = sampling_params(&r(&["--rep", "0"]), 0.8, "rep").unwrap_err();
        assert!(e.contains("--rep "), "{e}");
        assert!(check_temp(0.0).is_ok() && check_temp(1e9).is_ok());
        assert!(check_temp(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn epoch_batch_and_thread_counts_reject_nonsense() {
        assert!(check_epochs(1, true).is_ok() && check_epochs(100, false).is_ok());
        let e = check_epochs(0, true).unwrap_err();
        assert!(e.contains("--epochs") && e.contains("--tokens"), "{e}");
        let e = check_epochs(0, false).unwrap_err();
        assert!(e.contains("--epochs") && !e.contains("--tokens"), "{e}");
        assert!(check_batch_threads(1, 0).is_ok() && check_batch_threads(16, 1024).is_ok());
        assert!(check_batch_threads(0, 0).unwrap_err().contains("--batch"));
        let e = check_batch_threads(16, 1025).unwrap_err();
        assert!(e.contains("--threads") && e.contains("1025"), "{e}");
    }

    #[test]
    fn regularizer_and_budget_flags_reject_nonsense() {
        assert!(check_regularizers(0.0, 0.0, 0.0).is_ok());
        assert!(check_regularizers(0.5, 0.01, 1.0).is_ok());
        for (d, w, c, flag) in [
            (1.0, 0.0, 0.0, "--dropout"),
            (-0.1, 0.0, 0.0, "--dropout"),
            (f32::NAN, 0.0, 0.0, "--dropout"),
            (0.0, -1.0, 0.0, "--weight_decay"),
            (0.0, f32::NAN, 0.0, "--weight_decay"),
            (0.0, f32::INFINITY, 0.0, "--weight_decay"),
            (0.0, 0.0, -1.0, "--clip"),
            (0.0, 0.0, f32::NAN, "--clip"),
        ] {
            let e = check_regularizers(d, w, c).unwrap_err();
            assert!(e.contains(flag), "{flag}: {e}");
        }
        assert!(check_token_budget(0.0).is_ok() && check_token_budget(5e7).is_ok());
        for t in [f64::NAN, -5.0, f64::INFINITY] {
            assert!(check_token_budget(t).unwrap_err().contains("--tokens"));
        }
    }
}
