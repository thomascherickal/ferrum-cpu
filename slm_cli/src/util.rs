//! Small helpers shared by several subcommands.

use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

/// Lowercase hex sha256 of `bytes`.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub(crate) fn time_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE)
        | 1
}

pub(crate) fn read_corpus(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let corpus =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read corpus {path}: {e}"))?;
    if corpus.trim().is_empty() {
        return Err(format!("corpus {path} is empty").into());
    }
    Ok(corpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_seed_is_never_zero() {
        // `| 1` keeps an RNG seeded from it away from the all-zero state.
        assert_eq!(time_seed() & 1, 1);
    }

    #[test]
    fn read_corpus_returns_text_and_names_the_problem_otherwise() {
        let d = std::env::temp_dir().join(format!("ferrum_util_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let good = d.join("good.txt");
        std::fs::write(&good, "hello\n").unwrap();
        assert_eq!(read_corpus(good.to_str().unwrap()).unwrap(), "hello\n");
        let blank = d.join("blank.txt");
        std::fs::write(&blank, " \n\t\n").unwrap();
        let e = read_corpus(blank.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(e.contains("is empty"), "{e}");
        let e = read_corpus(d.join("nope.txt").to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(e.starts_with("cannot read corpus "), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
