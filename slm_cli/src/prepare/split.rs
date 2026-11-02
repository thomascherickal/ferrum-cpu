//! One output split (`train.txt`, `val.txt`, `test.txt`): a `*.part` file that
//! is hashed and counted as it is written.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Documents in a file are joined by this line.
const JOIN: &str = "\n<|endoftext|>\n";

pub(super) fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Where `name` is written until the whole run has succeeded.
pub(super) fn part(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.part"))
}

pub(super) struct Split {
    path: PathBuf,
    tmp: PathBuf,
    w: BufWriter<std::fs::File>,
    hash: Sha256,
    bytes: u64,
    pub(super) docs: u64,
    pub(super) chars: u64,
}

impl Split {
    pub(super) fn create(dir: &Path, name: &str) -> Result<Self, String> {
        let path = dir.join(format!("{name}.txt"));
        let tmp = part(dir, &format!("{name}.txt"));
        let file = std::fs::File::create(&tmp)
            .map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
        Ok(Split {
            w: BufWriter::new(file),
            path,
            tmp,
            hash: Sha256::new(),
            bytes: 0,
            docs: 0,
            chars: 0,
        })
    }
    pub(super) fn write(&mut self, doc: &str) -> Result<(), String> {
        if self.docs > 0 {
            self.put(JOIN.as_bytes())?;
        }
        self.put(doc.as_bytes())?;
        self.docs += 1;
        self.chars += doc.chars().count() as u64;
        Ok(())
    }
    fn put(&mut self, b: &[u8]) -> Result<(), String> {
        self.w
            .write_all(b)
            .map_err(|e| format!("cannot write {}: {e}", self.tmp.display()))?;
        self.hash.update(b);
        self.bytes += b.len() as u64;
        Ok(())
    }
    pub(super) fn finish(mut self) -> Result<Value, String> {
        self.w
            .flush()
            .map_err(|e| format!("cannot write {}: {e}", self.tmp.display()))?;
        Ok(json!({
            "path": self.path.to_string_lossy(),
            "bytes": self.bytes,
            "sha256": hex(&self.hash.finalize()),
            "documents": self.docs,
            "chars": self.chars,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_split_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A split whose `.part` file is a symlink to /dev/full: it opens, and every flushed write
    /// fails with "No space left on device".
    #[cfg(target_os = "linux")]
    fn full_disk_split(tag: &str) -> (PathBuf, Split) {
        let d = tmp(tag);
        std::os::unix::fs::symlink("/dev/full", part(&d, "train.txt")).unwrap();
        let s = Split::create(&d, "train").unwrap();
        (d, s)
    }

    #[test]
    fn a_split_joins_documents_counts_them_and_hashes_the_bytes() {
        let d = tmp("ok");
        let mut s = Split::create(&d, "train").unwrap();
        s.write("ab").unwrap();
        s.write("cd").unwrap();
        let v = s.finish().unwrap();
        let want = format!("ab{JOIN}cd");
        assert_eq!(v["bytes"], want.len());
        assert_eq!(
            (v["documents"].as_u64(), v["chars"].as_u64()),
            (Some(2), Some(4))
        );
        assert_eq!(v["sha256"], hex(&Sha256::digest(want.as_bytes())));
        assert_eq!(
            std::fs::read_to_string(part(&d, "train.txt")).unwrap(),
            want
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn create_names_the_part_file_when_it_cannot_be_made() {
        let e = Split::create(Path::new("/nonexistent-ferrum-dir"), "train")
            .err()
            .unwrap();
        assert!(
            e.starts_with("cannot create ") && e.contains("train.txt.part"),
            "{e}"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_failed_write_and_a_failed_flush_name_the_part_file() {
        // A document bigger than the BufWriter buffer is written through at once.
        let (d, mut s) = full_disk_split("write");
        let e = s.write(&"x".repeat(1 << 20)).unwrap_err();
        assert!(
            e.starts_with("cannot write ") && e.contains("train.txt.part"),
            "{e}"
        );
        // A small one stays buffered, and the failure surfaces when `finish` flushes.
        let (_, mut s) = full_disk_split("flush");
        s.write("small").unwrap();
        let e = s.finish().unwrap_err();
        assert!(
            e.starts_with("cannot write ") && e.contains("train.txt.part"),
            "{e}"
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
