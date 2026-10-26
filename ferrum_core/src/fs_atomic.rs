//! Crash-safe file replacement shared by every writer of a model file.
use crate::error::{InferError, Result};
use std::io::Write;

/// Write `bytes` to `path` via a same-directory temp file + rename, so a crash or full disk
/// mid-write can never leave a truncated file at `path` and an existing file stays intact until
/// the new one is complete (`rename` replaces an existing destination on Unix and Windows).
/// Errors name `path`; a failed write leaves no temp file behind.
pub(crate) fn atomic_write(path: &str, bytes: &[u8]) -> Result<()> {
    atomic_write_with(path, |f| f.write_all(bytes))
}

/// [`atomic_write`] for content produced incrementally: `fill` writes (and may seek in) the temp
/// file, which is renamed over `path` only if `fill` succeeds. Same guarantees.
pub(crate) fn atomic_write_with(
    path: &str,
    fill: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
) -> Result<()> {
    let tmp = format!("{path}.tmp");
    let io = |e: std::io::Error| InferError::Io(format!("cannot write {path}: {e}"));
    std::fs::File::create(&tmp)
        .and_then(|mut f| fill(&mut f))
        .and_then(|()| std::fs::rename(&tmp, path))
        .map_err(|e| {
            let _ = std::fs::remove_file(&tmp); // best-effort cleanup
            io(e)
        })
}

#[cfg(test)]
mod tests {
    use super::atomic_write;

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_fsatomic_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn writes_exact_bytes_replaces_and_leaves_no_temp() {
        let d = scratch("ok");
        let p = d.join("m.gguf");
        let p = p.to_str().unwrap();
        atomic_write(p, b"old").unwrap();
        atomic_write(p, b"new bytes").unwrap();
        assert_eq!(std::fs::read(p).unwrap(), b"new bytes");
        assert!(!std::path::Path::new(&format!("{p}.tmp")).exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn failure_keeps_the_old_file_and_names_the_path() {
        let d = scratch("fail");
        let p = d.join("m.gguf");
        let p = p.to_str().unwrap();
        atomic_write(p, b"old").unwrap();
        // A directory squatting on the temp name makes the write fail.
        std::fs::create_dir(format!("{p}.tmp")).unwrap();
        let e = atomic_write(p, b"new").unwrap_err().to_string();
        assert!(e.contains(&format!("cannot write {p}")), "{e}");
        assert_eq!(std::fs::read(p).unwrap(), b"old", "old file must survive");
        // A destination that is a directory fails at the rename and cleans its temp file.
        let dir_dest = d.join("dest");
        std::fs::create_dir(&dir_dest).unwrap();
        let q = dir_dest.to_str().unwrap();
        assert!(atomic_write(q, b"x").is_err());
        assert!(!std::path::Path::new(&format!("{q}.tmp")).exists());
        let _ = std::fs::remove_dir_all(&d);
    }
}
