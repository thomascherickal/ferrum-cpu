//! Pre-flight memory guard for training allocations.
use crate::error::{InferError, Result};

/// Memory the machine can still hand out (`MemAvailable`), where the OS says.
pub(super) fn available_memory_bytes() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: usize = info
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    kb.checked_mul(1024)
}

/// Training peaks above 16 GB, or above three quarters of the available RAM,
/// are refused up front: `--batch`/`--context` far beyond the machine would
/// otherwise abort inside the allocator mid-run. (A function, not a `usize`
/// constant: 16 GB does not fit a 32-bit `usize`, where the cap is `usize::MAX`.)
fn max_train_bytes() -> usize {
    usize::try_from(16u64 << 30).unwrap_or(usize::MAX)
}

pub(super) fn check_train_memory(need: usize, avail: Option<usize>) -> Result<()> {
    let max = max_train_bytes();
    let cap = avail.map_or(max, |a| (a / 4 * 3).min(max));
    if need > cap {
        return Err(InferError::DimMismatch(format!(
            "training would need about {:.1} GB (weights, optimizer state, activations) but only {:.1} GB \
             may be used; lower --batch, --context, --embed or --blocks",
            need as f64 / 1e9,
            cap as f64 / 1e9
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{available_memory_bytes, check_train_memory, max_train_bytes};

    #[test]
    fn check_train_memory_caps_at_16_gb_and_three_quarters_of_available() {
        assert!(check_train_memory(1 << 30, None).is_ok());
        assert!(check_train_memory(max_train_bytes(), None).is_ok());
        let e = check_train_memory(max_train_bytes() + 1, None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("lower --batch"), "{e}");
        // Available RAM below the 16 GB cap tightens it to 3/4 of what is free.
        assert!(check_train_memory(7 << 30, Some(10 << 30)).is_ok());
        assert!(check_train_memory(8 << 30, Some(10 << 30)).is_err());
        // Plenty of RAM never lifts the 16 GB cap.
        assert!(check_train_memory(17 << 30, Some(usize::MAX)).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn available_memory_is_reported_on_linux() {
        assert!(available_memory_bytes().is_some_and(|b| b > (1 << 20)));
    }
}
