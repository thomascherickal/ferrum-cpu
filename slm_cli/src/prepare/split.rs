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
