//! Graceful stop for `train` and `finetune-gguf`: `--max-seconds N`, the stop file
//! `<output>.stop` and the interrupt flag ([`crate::interrupt`]), all folded into one
//! [`StopCondition`]; [`report`] is the one place a stopped run is reported.

use crate::args::Args;
use ferrum_core::{StopCause, StopCondition};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
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

/// The run's stop condition: the `--max-seconds` deadline (started now), the stop file
/// `<anchor>.stop`, where `anchor` is the output path the run writes, and the installed
/// interrupt flag, if any.
pub(crate) fn condition(args: &Args, anchor: &str) -> Result<StopCondition, String> {
    condition_with(args, anchor, crate::interrupt::installed())
}

/// [`condition`] with the interrupt flag given (`None`: no interrupt source).
fn condition_with(
    args: &Args,
    anchor: &str,
    interrupt: Option<Arc<AtomicBool>>,
) -> Result<StopCondition, String> {
    let mut c = StopCondition::none().with_stop_file(format!("{anchor}.stop"));
    if let Some(d) = max_seconds(args)? {
        c = c.with_deadline(d);
    }
    if let Some(f) = interrupt {
        c = c.with_cancel(f);
    }
    Ok(c)
}

/// The one-line stop report; `ckpt` is the checkpoint to resume from, if one was written.
/// In the CLI the only cancel source is the interrupt flag, so `Cancelled` means Ctrl-C.
fn announcement(cause: StopCause, step: u64, ckpt: Option<&str>) -> String {
    match (cause, ckpt) {
        (StopCause::Cancelled, Some(p)) => format!(
            "interrupted at step {step}: checkpoint written to {p}; resume with --resume {p}"
        ),
        (StopCause::Cancelled, None) => format!(
            "interrupted at step {step}: no checkpoint was configured; nothing saved to resume \
             from (pass --checkpoint to make an interrupted run resumable)"
        ),
        (_, Some(p)) => {
            format!("stopped early at step {step}: {cause}; resume with --resume {p}")
        }
        (_, None) => format!(
            "stopped early at step {step}: {cause}; no checkpoint was configured \
             (pass --checkpoint to make a stopped run resumable)"
        ),
    }
}

/// After a run: if it `stopped` early, remove the stop file and print the one-line stop
/// report. Returns the cause, `None` for a run that completed.
pub(crate) fn report(
    cond: &StopCondition,
    stopped: Option<(StopCause, u64)>,
    ckpt: Option<&str>,
) -> Option<StopCause> {
    let (cause, step) = stopped?;
    cond.clear_stop_file();
    println!("{}", announcement(cause, step, ckpt));
    Some(cause)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{spec_for, Args};
    use std::sync::atomic::Ordering;

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
        assert_eq!(
            announcement(StopCause::Cancelled, 5, Some("m.ckpt")),
            "interrupted at step 5: checkpoint written to m.ckpt; resume with --resume m.ckpt"
        );
        let none = announcement(StopCause::Cancelled, 2, None);
        assert!(
            none.starts_with("interrupted at step 2: no checkpoint was configured; nothing saved"),
            "{none}"
        );
    }

    #[test]
    fn the_interrupt_flag_is_one_more_source_and_wins_a_tie() {
        let dir = std::env::temp_dir().join(format!("ferrum_cli_stop_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let anchor = dir.join("m.gguf").to_str().unwrap().to_string();
        let flag = Arc::new(AtomicBool::new(false));
        let c = condition_with(
            &args(&["--max-seconds", "3600"]),
            &anchor,
            Some(flag.clone()),
        )
        .unwrap();
        assert_eq!(c.should_stop(), None);
        flag.store(true, Ordering::Relaxed);
        assert_eq!(c.should_stop(), Some(StopCause::Cancelled));
        // With every source firing at once the fixed order decides: flag, file, deadline.
        std::fs::write(format!("{anchor}.stop"), b"").unwrap();
        let all = condition_with(
            &args(&["--max-seconds", "1e-9"]),
            &anchor,
            Some(flag.clone()),
        )
        .unwrap();
        assert_eq!(all.should_stop(), Some(StopCause::Cancelled));
        flag.store(false, Ordering::Relaxed);
        assert_eq!(all.should_stop(), Some(StopCause::StopFile));
        all.clear_stop_file();
        assert_eq!(all.should_stop(), Some(StopCause::TimeBudget));
        // Without a flag (no handler installed) only the file and the deadline count.
        let plain = condition_with(&args(&[]), &anchor, None).unwrap();
        assert_eq!(plain.should_stop(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn report_is_silent_for_a_finished_run_and_consumes_the_stop_file() {
        let dir = std::env::temp_dir().join(format!("ferrum_cli_report_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let anchor = dir.join("m.gguf").to_str().unwrap().to_string();
        let c = condition(&args(&[]), &anchor).unwrap();
        std::fs::write(format!("{anchor}.stop"), b"").unwrap();
        assert_eq!(report(&c, None, Some("m.ckpt")), None);
        assert!(
            c.stop_file().unwrap().exists(),
            "a finished run leaves the file alone"
        );
        let got = report(&c, Some((StopCause::StopFile, 4)), Some("m.ckpt"));
        assert_eq!(got, Some(StopCause::StopFile));
        assert!(!c.stop_file().unwrap().exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
