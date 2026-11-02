//! sha256 as lowercase hex: how Ferrum pins inputs, outputs and tokenizers.

use sha2::{Digest, Sha256};

/// Lowercase hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `sha256:<hex>` of the file at `path`, read in 64 KiB chunks so a large corpus is never
/// held in memory. This is the corpus identity a checkpoint records (`TransformerConfig::corpus_id`).
pub fn file_id(path: &str) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!(
        "sha256:{}",
        h.finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn file_id_streams_the_file_and_matches_sha256_hex() {
        let p = std::env::temp_dir().join(format!("ferrum_file_id_{}", std::process::id()));
        let data = vec![7u8; 200_000]; // several chunks
        std::fs::write(&p, &data).unwrap();
        let id = super::file_id(p.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(id, format!("sha256:{}", super::sha256_hex(&data)));
        assert!(super::file_id("/nonexistent/ferrum_corpus").is_err());
    }

    #[test]
    fn matches_the_known_digest_of_abc() {
        assert_eq!(
            super::sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
