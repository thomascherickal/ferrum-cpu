//! Small helpers shared by several subcommands.

use std::time::{SystemTime, UNIX_EPOCH};

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
