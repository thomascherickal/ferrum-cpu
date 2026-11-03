//! Helpers shared by the commands that load GGUF models: the quantisation
//! parser and the RAM guard's size estimates.

use ferrum_core::QKind;

/// Map a UI quant string to an import precision (`None` = keep f32).
pub(crate) fn parse_quant(q: &str) -> Result<Option<QKind>, String> {
    ferrum_core::memory::import_precision(q)
        .ok_or_else(|| format!("quant must be int4 | int8 | f32 (got '{q}')"))
}

/// Estimate resident bytes for a loaded model from the GGUF directory: the token
/// embedding stays f32, everything else packs to the chosen precision.
pub(crate) use ferrum_core::memory::resident_bytes as gguf_resident_bytes;

/// Best-effort available RAM (Linux `/proc/meminfo`); `None` elsewhere.
pub(crate) fn available_memory_bytes() -> Option<usize> {
    // Cross-platform via sysinfo (0.33 reports bytes). The previous /proc/meminfo
    // reader returned None on macOS/Windows, silently disabling the RAM guard on
    // the very desktops the Tauri app also ships to.
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let avail = sys.available_memory();
    (avail > 0).then_some(avail as usize)
}

/// The RAM guard every GGUF load shares: refuse when `est` bytes exceed 90% of `avail` (unknown
/// RAM never refuses) unless `force`. `what` names the estimate ("resident memory"), `detail`
/// goes inside its parentheses after the size, `advice` ends the sentence.
pub(crate) fn ram_guard(
    what: &str,
    detail: &str,
    est: usize,
    avail: Option<usize>,
    force: bool,
    advice: &str,
) -> Result<(), String> {
    match avail {
        Some(avail) if (est as f64) > 0.9 * avail as f64 && !force => Err(format!(
            "estimated {what} ({:.2} GB{detail}) exceeds 90% of available ({:.2} GB) — {advice}.",
            est as f64 / 1e9,
            avail as f64 / 1e9
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ram_guard_refuses_over_90_percent_unless_forced_or_unknown() {
        let g = |est, avail, force| ram_guard("peak memory", ": why", est, avail, force, "do this");
        assert_eq!(
            g(3_000_000_000, Some(2_000_000_000), false).unwrap_err(),
            "estimated peak memory (3.00 GB: why) exceeds 90% of available (2.00 GB) — do this."
        );
        // Exactly 90% passes; forcing and unknown RAM never refuse.
        assert!(g(1_800_000_000, Some(2_000_000_000), false).is_ok());
        assert!(g(3_000_000_000, Some(2_000_000_000), true).is_ok());
        assert!(g(usize::MAX, None, false).is_ok());
    }

    #[test]
    fn available_memory_is_reported_cross_platform() {
        // The RAM guard is only meaningful if this returns a real figure; the old
        // /proc/meminfo reader returned None off Linux, disabling the guard there.
        let avail = available_memory_bytes().expect("available memory should be reported");
        assert!(
            avail > 16 * 1024 * 1024,
            "implausibly small avail: {avail} bytes"
        );
    }
}
