//! The `--max-seconds` stop deadline shared by the Train and Fine-tune tabs.

/// The run's stop condition: `base` (the cancel flag of the job) plus the `--max-seconds` deadline,
/// validated with the CLI's rule and message.
pub(super) fn stop_condition(
    base: &ferrum_core::StopCondition,
    max_seconds: Option<f64>,
) -> Result<ferrum_core::StopCondition, String> {
    let Some(s) = max_seconds else {
        return Ok(base.clone());
    };
    if s.is_finite() && s > 0.0 {
        if let Ok(d) = std::time::Duration::try_from_secs_f64(s) {
            return Ok(base.clone().with_deadline(d));
        }
    }
    Err(format!(
        "--max-seconds must be a positive, finite number of seconds (got {s})"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_condition_mirrors_the_cli_budget_rule() {
        let base = ferrum_core::StopCondition::none();
        assert_eq!(stop_condition(&base, None).unwrap(), base);
        assert_ne!(
            stop_condition(&base, Some(0.25)).unwrap(),
            base,
            "a deadline was added"
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e300] {
            let e = stop_condition(&base, Some(bad)).unwrap_err();
            assert!(
                e.starts_with("--max-seconds must be a positive, finite number of seconds (got "),
                "{e}"
            );
        }
    }
}
