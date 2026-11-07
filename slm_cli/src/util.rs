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

/// Read up to `max_bytes` from `path` (truncated cleanly at a UTF-8 char boundary) so `--sample`
/// can pick a seed from a disk-streamed text corpus without loading the whole file into RAM.
pub(crate) fn read_corpus_prefix(
    path: &str,
    max_bytes: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|e| format!("cannot read corpus {path}: {e}"))?;
    let mut buf = Vec::with_capacity(max_bytes.min(1 << 16));
    f.take(max_bytes as u64)
        .read_to_end(&mut buf)
        .map_err(|e| format!("cannot read corpus {path}: {e}"))?;
    let valid = match std::str::from_utf8(&buf) {
        Ok(s) => s,
        Err(e) => std::str::from_utf8(&buf[..e.valid_up_to()]).unwrap_or(""),
    };
    Ok(valid.to_owned())
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
        let multi = d.join("multi.txt");
        std::fs::write(&multi, "hello🌸world").unwrap();
        // Cut inside the 4-byte emoji: `read_corpus_prefix` truncates cleanly at the char boundary.
        assert_eq!(
            read_corpus_prefix(multi.to_str().unwrap(), 7).unwrap(),
            "hello"
        );
        assert_eq!(
            read_corpus_prefix(multi.to_str().unwrap(), 64).unwrap(),
            "hello🌸world"
        );
        assert!(read_corpus_prefix(d.join("nope.txt").to_str().unwrap(), 16)
            .unwrap_err()
            .to_string()
            .starts_with("cannot read corpus "));
        let _ = std::fs::remove_dir_all(&d);
    }
}
