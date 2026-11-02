//! Graceful stop for `train` and `finetune-gguf`: `--max-seconds N` and the stop file
//! `<output>.stop`, both folded into one [`StopCondition`].

use crate::args::Args;
use ferrum_core::{StopCause, StopCondition};
use std::time::Duration;

/// `--max-seconds` as a budget: a finite number > 0 that fits a `Duration` (fractions allowed).
fn max_seconds(args: &Args) -> Result<Option<Duration>, String> {
    if args.value("max-seconds").is_none() {
        return Ok(None);
    }
    let s: f64 = args.get("max-seconds", 0.0);
    if s.is_finite() && s > 0.0 {
        if let Ok(d) = Duration::try_from_secs_f64(s) {
            return Ok(Some(d));
        }
    }
    Err(format!(
        "--max-seconds must be a positive, finite number of seconds (got {s})"
    ))
}

/// The run's stop condition: the `--max-seconds` deadline (started now) plus the stop file
/// `<anchor>.stop`, where `anchor` is the output path the run writes.
pub(crate) fn condition(args: &Args, anchor: &str) -> Result<StopCondition, String> {
    let mut c = StopCondition::none().with_stop_file(format!("{anchor}.stop"));
    if let Some(d) = max_seconds(args)? {
        c = c.with_deadline(d);
    }
    Ok(c)
}

/// The one-line stop report; `ckpt` is the checkpoint to resume from, if one was written.
fn announcement(cause: StopCause, step: u64, ckpt: Option<&str>) -> String {
    match ckpt {
        Some(p) => format!("stopped early at step {step}: {cause}; resume with --resume {p}"),
        None => format!(
            "stopped early at step {step}: {cause}; no checkpoint was configured \
             (pass --checkpoint to make a stopped run resumable)"
        ),
    }
}

/// Print the one-line stop report.
pub(crate) fn announce(cause: StopCause, step: u64, ckpt: Option<&str>) {
    println!("{}", announcement(cause, step, ckpt));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{spec_for, Args};

    fn args(extra: &[&str]) -> Args {
        let raw: Vec<String> = extra.iter().map(|s| s.to_string()).collect();
        Args::parse(&raw, spec_for("train").unwrap()).unwrap()
    }

    #[test]
    fn max_seconds_accepts_fractions_and_rejects_nonsense() {
        assert_eq!(max_seconds(&args(&[])), Ok(None));
        assert_eq!(
            max_seconds(&args(&["--max-seconds", "0.25"])),
            Ok(Some(Duration::from_millis(250)))
        );
        for bad in ["0", "-1", "nan", "inf", "1e30"] {
            let e = max_seconds(&args(&["--max-seconds", bad])).unwrap_err();
            assert!(e.contains("--max-seconds"), "{bad}: {e}");
        }
    }

    #[test]
    fn condition_watches_the_stop_file_next_to_the_output() {
        let c = condition(&args(&[]), "out/model.gguf").unwrap();
        assert_eq!(c.stop_file().unwrap().to_str(), Some("out/model.gguf.stop"));
        assert_eq!(c.should_stop(), None);
        let t = condition(&args(&["--max-seconds", "3600"]), "m").unwrap();
        assert_eq!(t.should_stop(), None);
    }

    #[test]
    fn the_stop_report_names_the_checkpoint_or_says_there_is_none() {
        assert_eq!(
            announcement(StopCause::StopFile, 7, Some("m.ckpt")),
            "stopped early at step 7: stop file found; resume with --resume m.ckpt"
        );
        let none = announcement(StopCause::TimeBudget, 3, None);
        assert!(none.starts_with("stopped early at step 3: time budget reached; no checkpoint"));
        assert!(none.ends_with("(pass --checkpoint to make a stopped run resumable)"));
    }
}
