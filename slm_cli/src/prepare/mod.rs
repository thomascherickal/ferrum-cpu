//! `prepare`: stream raw text, clean it per document, drop duplicates and
//! out-of-range lengths, split train/val/test by a seeded content hash, and
//! write a manifest that pins every input and output by sha256 — a
//! deterministic, DVC-friendly data stage.
//!
//! [`prepare`] is the whole pipeline: no argument parsing, no printing. The
//! `prepare` subcommand and the GUI's Datasets tab both call it, so they write
//! the same bytes for the same options.

mod input;
mod split;

pub use input::{expand_inputs, for_each_raw_doc};

use ferrum_core::{
    clean_corpus, gutenberg_bodies, normalize_nfkc, squeeze_spaces, strip_invisible, CleanOptions,
    StopCondition,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use split::{hex, part, Split};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// Everything a run needs. The fields mirror the `prepare` flags.
pub struct PrepareOptions {
    pub inputs: Vec<String>,
    pub out: String,
    pub doc_sep: String,
    pub lossy: bool,
    pub nfkc: bool,
    pub lowercase: bool,
    pub strip_gutenberg: bool,
    pub min_chars: usize,
    pub max_chars: usize,
    pub dedup: bool,
    pub val_frac: f64,
    pub test_frac: f64,
    pub seed: u64,
    /// The raw argument list, recorded in the manifest.
    pub args: Vec<String>,
    /// Polled between documents; a stop ends the run with nothing written.
    pub stop: StopCondition,
}

/// Where a run is, reported through the `progress` callback of [`prepare`]: at
/// the start of every input file and about every [`PROGRESS_EVERY`] documents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepareProgress {
    /// 1-based index of the file being read.
    pub file: usize,
    pub files: usize,
    pub path: String,
    /// Bytes of the inputs read so far and in total (an estimate inside a file).
    pub bytes_done: u64,
    pub bytes_total: u64,
    /// Raw documents seen so far (before the filters).
    pub documents: u64,
}

/// A progress report about every this many raw documents.
pub const PROGRESS_EVERY: u64 = 2048;

/// What a finished run wrote.
#[derive(Clone, Debug)]
pub struct PrepareReport {
    pub out: PathBuf,
    pub manifest_path: PathBuf,
    /// Kept documents per split: train, val, test.
    pub documents: [u64; 3],
    /// Rejected documents by reason (`duplicate`, `empty`, `invalid_utf8`, `too_long`, `too_short`).
    pub rejected: BTreeMap<String, u64>,
    /// Kept documents that had invalid bytes replaced (`lossy` only).
    pub lossy_repaired: u64,
    /// The manifest exactly as written to `manifest.json`.
    pub manifest: Value,
}

impl PrepareReport {
    /// The one-line summary the CLI prints.
    pub fn summary(&self, o: &PrepareOptions) -> String {
        let docs: Vec<String> = ["train", "val", "test"]
            .iter()
            .zip(self.documents)
            .map(|(s, n)| format!("{s}={n}"))
            .collect();
        let repaired = if self.lossy_repaired > 0 {
            format!("  lossy_repaired={}", self.lossy_repaired)
        } else {
            String::new()
        };
        format!(
            "prepared {} → {}  ({})  rejected {}{repaired}",
            o.inputs.join(" "),
            o.out,
            docs.join(" "),
            rejected_summary(&self.rejected)
        )
    }
}

/// The message of a run ended by `PrepareOptions::stop`.
pub const CANCELLED: &str = "prepare cancelled; no output files were changed";

/// The files a run writes.
const OUTPUTS: [&str; 4] = ["train.txt", "val.txt", "test.txt", "manifest.json"];

/// `duplicate=3 too_short=1` (non-zero reasons only), or `none`.
fn rejected_summary(rejected: &BTreeMap<String, u64>) -> String {
    let parts: Vec<String> = rejected
        .iter()
        .filter(|(_, n)| **n != 0)
        .map(|(k, n)| format!("{k}={n}"))
        .collect();
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join(" ")
    }
}

/// Run the whole pipeline. On any error (or a stop) the previous outputs are
/// left untouched and no `*.part` file remains.
pub fn prepare(
    o: &PrepareOptions,
    progress: &mut dyn FnMut(PrepareProgress),
) -> Result<PrepareReport, String> {
    if !(0.0..1.0).contains(&o.val_frac)
        || !(0.0..1.0).contains(&o.test_frac)
        || o.val_frac + o.test_frac >= 1.0
    {
        return Err("--val-frac and --test-frac must be in [0, 1) and sum to less than 1".into());
    }
    let out = Path::new(&o.out);
    // An input that is one of this run's own outputs is refused here, untouched.
    let files = crate::util::err_str(expand_inputs(&o.inputs, out))?;
    crate::util::fmt_err(
        std::fs::create_dir_all(out),
        &format!("cannot create directory {}", out.display()),
    )?;
    // Everything is written to `*.part` and renamed into place only after every input was read
    // and validated, so a failed run leaves the previous outputs untouched.
    let r = build(o, out, files, progress);
    if r.is_err() {
        for n in OUTPUTS {
            let _ = std::fs::remove_file(part(out, n));
        }
    }
    r
}

/// Move the finished `*.part` files into place. The manifest is the commit marker: an old one is
/// removed first and the new one is renamed last, so a `manifest.json` next to the splits always
/// describes a complete set.
// ponytail: the data files are not rolled back; a crash or failed rename can leave a partial set
// WITHOUT a manifest, never a manifest over a partial set. A staging directory swap would make the
// set atomic if it ever matters.
fn commit_outputs(
    out: &Path,
    rename: &dyn Fn(&Path, &Path) -> std::io::Result<()>,
) -> Result<(), String> {
    let manifest = out.join("manifest.json");
    match std::fs::remove_file(&manifest) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("cannot remove {}: {e}", manifest.display())),
    }
    for n in OUTPUTS {
        let (from, to) = (part(out, n), out.join(n));
        crate::util::fmt_err(
            rename(&from, &to),
            &format!("cannot rename {} to {}", from.display(), to.display()),
        )?;
    }
    Ok(())
}

fn build(
    o: &PrepareOptions,
    out: &Path,
    files: Vec<PathBuf>,
    progress: &mut dyn FnMut(PrepareProgress),
) -> Result<PrepareReport, String> {
    let mut splits = [
        Split::create(out, "train")?,
        Split::create(out, "val")?,
        Split::create(out, "test")?,
    ];
    let val_cut = (o.val_frac * 10_000.0).round() as u64;
    let test_cut = val_cut + (o.test_frac * 10_000.0).round() as u64;
    let clean_opts = CleanOptions {
        strip_gutenberg: false,
        lowercase: o.lowercase,
        collapse_whitespace: true,
        normalize_punctuation: true,
        strip_control_chars: true,
        max_chars: None,
    };
    let mut rejected: BTreeMap<&str, u64> = [
        "duplicate",
        "empty",
        "invalid_utf8",
        "too_long",
        "too_short",
    ]
    .map(|k| (k, 0))
    .into();
    let mut lossy_repaired = 0u64; // kept documents that had bytes replaced (--lossy)
    let mut seen: HashSet<[u8; 16]> = HashSet::new();
    let mut uniq: HashSet<char> = HashSet::new();
    let (mut lines, mut words) = (0u64, 0u64);
    let mut inputs = Vec::new();
    let sizes: Vec<u64> = files
        .iter()
        .map(|p| std::fs::metadata(p).map_or(0, |m| m.len()))
        .collect();
    let bytes_total: u64 = sizes.iter().sum();
    let (mut bytes_before, mut raw_docs) = (0u64, 0u64);

    for (i, path) in files.iter().enumerate() {
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        let report = |bytes_done: u64, documents: u64| PrepareProgress {
            file: i + 1,
            files: files.len(),
            path: path.to_string_lossy().into_owned(),
            bytes_done,
            bytes_total,
            documents,
        };
        progress(report(bytes_before, raw_docs));
        for_each_raw_doc(path, &o.doc_sep, &mut hasher, &mut total, |raw, offset| {
            // ponytail: polled between documents, so one huge document is read to its end first.
            if o.stop.should_stop().is_some() {
                return Err(CANCELLED.to_string());
            }
            raw_docs += 1;
            if raw_docs % PROGRESS_EVERY == 0 {
                progress(report(bytes_before + offset, raw_docs));
            }
            let text = match std::str::from_utf8(raw) {
                Ok(s) => std::borrow::Cow::Borrowed(s),
                Err(_) if o.lossy => {
                    lossy_repaired += 1;
                    String::from_utf8_lossy(raw)
                }
                Err(e) => {
                    return Err(format!(
                        "{}: invalid UTF-8 at byte {} (pass --lossy to replace invalid bytes)",
                        path.display(),
                        offset + e.valid_up_to() as u64
                    ))
                }
            };
            // Every line ending is a newline: CRLF, and the lone CR of classic Mac files
            // (which `strip_invisible` would otherwise delete, gluing words together).
            let text = if text.contains('\r') {
                std::borrow::Cow::Owned(text.replace("\r\n", "\n").replace('\r', "\n"))
            } else {
                text
            };
            let books = if o.strip_gutenberg {
                gutenberg_bodies(&text)
            } else {
                vec![text.as_ref()]
            };
            for book in books {
                let mut doc = strip_invisible(book);
                if o.nfkc {
                    doc = normalize_nfkc(&doc);
                }
                let doc = squeeze_spaces(&clean_corpus(&doc, &clean_opts));
                let doc = doc.trim();
                let n = doc.chars().count();
                let reason = if n == 0 {
                    Some("empty")
                } else if n < o.min_chars {
                    Some("too_short")
                } else if o.max_chars > 0 && n > o.max_chars {
                    Some("too_long")
                } else {
                    None
                };
                if let Some(r) = reason {
                    *rejected.entry(r).or_default() += 1;
                    continue;
                }
                if o.dedup {
                    let d = Sha256::digest(doc.as_bytes());
                    let key: [u8; 16] = std::array::from_fn(|i| d[i]);
                    if !seen.insert(key) {
                        *rejected.entry("duplicate").or_default() += 1;
                        continue;
                    }
                }
                let mut h = Sha256::new();
                h.update(o.seed.to_le_bytes());
                h.update(doc.as_bytes());
                let d = h.finalize();
                let bucket = u64::from_be_bytes(std::array::from_fn(|i| d[i])) % 10_000;
                let which = if bucket < val_cut {
                    1
                } else if bucket < test_cut {
                    2
                } else {
                    0
                };
                if which == 0 {
                    uniq.extend(doc.chars());
                    lines += doc.lines().count() as u64;
                    words += doc.split_whitespace().count() as u64;
                }
                splits[which].write(doc)?;
            }
            Ok(())
        })?;
        bytes_before += sizes[i];
        inputs.push(json!({"path": path.to_string_lossy(), "bytes": total, "sha256": hex(&hasher.finalize())}));
    }
    progress(PrepareProgress {
        file: files.len(),
        files: files.len(),
        path: files
            .last()
            .map_or(String::new(), |p| p.to_string_lossy().into_owned()),
        bytes_done: bytes_total,
        bytes_total,
        documents: raw_docs,
    });

    let [train, val, test] = splits;
    let total_docs = train.docs + val.docs + test.docs;
    if train.docs == 0 {
        return Err(format!(
            "train split is empty: {total_docs} documents found, need at least 3 (one file is one \
             document unless it contains the separator {:?}; use --doc-sep or add more files)",
            o.doc_sep
        ));
    }
    let documents = [train.docs, val.docs, test.docs];
    let train = train.finish()?;
    let train_stats = json!({
        "bytes": train["bytes"], "chars": train["chars"], "lines": lines,
        "unique_chars": uniq.len(), "words": words,
    });
    let mut outputs = Map::new();
    outputs.insert("test".into(), test.finish()?);
    outputs.insert("train".into(), train);
    outputs.insert("val".into(), val.finish()?);
    let mut manifest = json!({
        "args": o.args,
        "ferrum_version": env!("CARGO_PKG_VERSION"),
        "inputs": inputs,
        "outputs": outputs,
        "rejected": rejected,
        "train_stats": train_stats,
    });
    // Only when non-zero, so manifests of clean runs keep their exact bytes.
    if lossy_repaired > 0 {
        manifest["lossy_repaired"] = json!(lossy_repaired);
    }
    let mpart = part(out, "manifest.json");
    let content = serde_json::to_string_pretty(&manifest).unwrap_or_default() + "\n";
    write_manifest(&mpart, &content)?;
    commit_outputs(out, &|from, to| std::fs::rename(from, to))?;
    Ok(PrepareReport {
        out: out.to_path_buf(),
        manifest_path: out.join("manifest.json"),
        documents,
        rejected: rejected
            .into_iter()
            .map(|(k, n)| (k.to_string(), n))
            .collect(),
        lossy_repaired,
        manifest,
    })
}

fn write_manifest(mpart: &Path, content: &str) -> Result<(), String> {
    match std::fs::write(mpart, content) {
        Ok(()) => Ok(()),
        Err(e) => Err(format!("cannot write {}: {e}", mpart.display())),
    }
}

#[cfg(test)]
mod tests;
