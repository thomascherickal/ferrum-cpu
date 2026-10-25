//! `train_transformer prepare`: stream raw text, clean it per document, drop
//! duplicates and out-of-range lengths, split train/val/test by a seeded
//! content hash, and write a manifest that pins every input and output by
//! sha256 — a deterministic, DVC-friendly data stage.

use crate::args::Args;
use ferrum_core::{
    clean_corpus, gutenberg_bodies, normalize_nfkc, squeeze_spaces, strip_invisible, CleanOptions,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const JOIN: &str = "\n<|endoftext|>\n";

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

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
}

struct Split {
    path: PathBuf,
    w: BufWriter<std::fs::File>,
    hash: Sha256,
    bytes: u64,
    docs: u64,
    chars: u64,
}

impl Split {
    fn create(dir: &Path, name: &str) -> std::io::Result<Self> {
        let path = dir.join(format!("{name}.txt"));
        Ok(Split {
            w: BufWriter::new(std::fs::File::create(&path)?),
            path,
            hash: Sha256::new(),
            bytes: 0,
            docs: 0,
            chars: 0,
        })
    }
    fn write(&mut self, doc: &str) -> std::io::Result<()> {
        if self.docs > 0 {
            self.put(JOIN.as_bytes())?;
        }
        self.put(doc.as_bytes())?;
        self.docs += 1;
        self.chars += doc.chars().count() as u64;
        Ok(())
    }
    fn put(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.w.write_all(b)?;
        self.hash.update(b);
        self.bytes += b.len() as u64;
        Ok(())
    }
    fn finish(mut self) -> std::io::Result<Value> {
        self.w.flush()?;
        Ok(json!({
            "path": self.path.to_string_lossy(),
            "bytes": self.bytes,
            "sha256": hex(&self.hash.finalize()),
            "documents": self.docs,
            "chars": self.chars,
        }))
    }
}

/// Every regular, non-hidden file under each input (directories walked
/// recursively; FIFOs, devices and sockets inside them are skipped), sorted by
/// path. A non-regular file named directly is an error: `/dev/zero` or a pipe
/// would otherwise be read forever.
///
/// Anything under `outdir` (this run's own outputs and manifest, e.g. `--out`
/// inside `--input`) is skipped when found by walking a directory and an error
/// when named directly: re-ingesting the outputs would feed them back in (and
/// they are truncated before any input is read).
fn expand_inputs(inputs: &[String], outdir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let outdir = std::fs::canonicalize(outdir).unwrap_or_else(|_| outdir.to_path_buf());
    let inside_out = |p: &Path| std::fs::canonicalize(p).is_ok_and(|c| c.starts_with(&outdir));
    fn walk(
        p: &Path,
        top: bool,
        inside_out: &dyn Fn(&Path) -> bool,
        out: &mut Vec<PathBuf>,
    ) -> std::io::Result<()> {
        if p.is_dir() {
            if !top && inside_out(p) {
                return Ok(());
            }
            for e in std::fs::read_dir(p)? {
                let e = e?.path();
                if !e
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
                {
                    walk(&e, false, inside_out, out)?;
                }
            }
        } else if p.is_file() {
            if inside_out(p) {
                if top {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("input {} is inside --out (its own output)", p.display()),
                    ));
                }
                return Ok(());
            }
            out.push(p.to_path_buf());
        } else if top {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("input {} is not a regular file or directory", p.display()),
            ));
        }
        Ok(())
    }
    let mut out = Vec::new();
    for i in inputs {
        let p = Path::new(i);
        if !p.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("input {i} does not exist"),
            ));
        }
        walk(p, true, &inside_out, &mut out)?;
    }
    out.sort();
    Ok(out)
}

/// Bytes read per `read` call (memory is the largest document plus this).
const CHUNK: usize = 1 << 20;

/// Stream `path` as raw documents split on `sep` (byte-exact). An empty
/// separator yields the whole file as one document. Calls `f(bytes, offset)`.
///
/// Reads fixed-size chunks (not lines), so a file with no newlines costs one
/// chunk of buffer rather than doubling the document, and walks each chunk's
/// separators once (no per-document memmove of the rest of the chunk).
fn for_each_raw_doc(
    path: &Path,
    sep: &str,
    hasher: &mut Sha256,
    total: &mut u64,
    mut f: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<(), String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let sep = sep.as_bytes();
    let mut doc: Vec<u8> = Vec::new();
    let mut doc_start = 0u64;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
        };
        hasher.update(&buf[..n]);
        *total += n as u64;
        let old_len = doc.len();
        doc.extend_from_slice(&buf[..n]);
        if sep.is_empty() {
            continue;
        }
        // A separator can only complete inside the bytes just appended (it may
        // start up to `sep.len() - 1` bytes before them).
        let mut from = old_len.saturating_sub(sep.len() - 1);
        let mut consumed = 0;
        while let Some(i) = doc[from..]
            .windows(sep.len())
            .position(|w| w == sep)
            .map(|i| from + i)
        {
            f(&doc[consumed..i], doc_start)?;
            doc_start += (i + sep.len() - consumed) as u64;
            consumed = i + sep.len();
            from = consumed;
        }
        doc.drain(..consumed);
    }
    f(&doc, doc_start)
}

/// `duplicate=3 too_short=1` (non-zero reasons only), or `none`.
fn rejected_summary(rejected: &Value) -> String {
    let parts: Vec<String> = rejected
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(_, n)| n.as_u64() != Some(0))
        .map(|(k, n)| format!("{k}={n}"))
        .collect();
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join(" ")
    }
}

pub fn run(o: &PrepareOptions) -> Result<(), Box<dyn std::error::Error>> {
    if !(0.0..1.0).contains(&o.val_frac)
        || !(0.0..1.0).contains(&o.test_frac)
        || o.val_frac + o.test_frac >= 1.0
    {
        return Err("--val-frac and --test-frac must be in [0, 1) and sum to less than 1".into());
    }
    let out = Path::new(&o.out);
    std::fs::create_dir_all(out)?;
    // A failed run must not leave a stale manifest beside truncated splits.
    match std::fs::remove_file(out.join("manifest.json")) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
        _ => {}
    }
    // Resolve inputs before any split is created (and so truncated): an input that is one of
    // this run's own outputs is refused here, untouched.
    let files = expand_inputs(&o.inputs, out)?;
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
    let mut seen: HashSet<[u8; 16]> = HashSet::new();
    let mut uniq: HashSet<char> = HashSet::new();
    let (mut lines, mut words) = (0u64, 0u64);
    let mut inputs = Vec::new();

    for path in files {
        let mut hasher = Sha256::new();
        let mut total = 0u64;
        for_each_raw_doc(&path, &o.doc_sep, &mut hasher, &mut total, |raw, offset| {
            let text = match std::str::from_utf8(raw) {
                Ok(s) => std::borrow::Cow::Borrowed(s),
                Err(_) if o.lossy => {
                    *rejected.get_mut("invalid_utf8").unwrap() += 1;
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
                    *rejected.get_mut(r).unwrap() += 1;
                    continue;
                }
                if o.dedup {
                    let key: [u8; 16] = Sha256::digest(doc.as_bytes())[..16].try_into().unwrap();
                    if !seen.insert(key) {
                        *rejected.get_mut("duplicate").unwrap() += 1;
                        continue;
                    }
                }
                let mut h = Sha256::new();
                h.update(o.seed.to_le_bytes());
                h.update(doc.as_bytes());
                let bucket = u64::from_be_bytes(h.finalize()[..8].try_into().unwrap()) % 10_000;
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
                splits[which].write(doc).map_err(|e| e.to_string())?;
            }
            Ok(())
        })?;
        inputs.push(json!({"path": path.to_string_lossy(), "bytes": total, "sha256": hex(&hasher.finalize())}));
    }

    let [train, val, test] = splits;
    let train = train.finish()?;
    let train_stats = json!({
        "bytes": train["bytes"], "chars": train["chars"], "lines": lines,
        "unique_chars": uniq.len(), "words": words,
    });
    let mut outputs = Map::new();
    outputs.insert("test".into(), test.finish()?);
    outputs.insert("train".into(), train);
    outputs.insert("val".into(), val.finish()?);
    let manifest = json!({
        "args": o.args,
        "ferrum_version": env!("CARGO_PKG_VERSION"),
        "inputs": inputs,
        "outputs": outputs,
        "rejected": rejected,
        "train_stats": train_stats,
    });
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    let docs: Vec<String> = ["train", "val", "test"]
        .iter()
        .map(|s| format!("{s}={}", manifest["outputs"][s]["documents"]))
        .collect();
    println!(
        "prepared {} → {}  ({})  rejected {}",
        o.inputs.join(" "),
        o.out,
        docs.join(" "),
        rejected_summary(&manifest["rejected"])
    );
    Ok(())
}

pub(crate) fn cmd_prepare(args: &Args, raw: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let usage =
        "usage: train_transformer prepare --input <file|dir> [--input ...] --out <dir> [options]";
    let inputs = args.get_all("input");
    if inputs.is_empty() {
        return Err(usage.into());
    }
    let out = args.value("out").ok_or(usage)?;
    run(&PrepareOptions {
        inputs,
        out: out.to_string(),
        doc_sep: args
            .value("doc-sep")
            .unwrap_or(ferrum_core::DOC_SEP)
            .to_string(),
        lossy: args.has("lossy"),
        nfkc: args.has("nfkc"),
        lowercase: args.has("lowercase"),
        strip_gutenberg: args.has("strip-gutenberg"),
        min_chars: args.get("min-chars", 1),
        max_chars: args.get("max-chars", 0),
        dedup: args.has("dedup"),
        val_frac: args.get("val-frac", 0.05),
        test_frac: args.get("test-frac", 0.05),
        seed: args.get("seed", 42),
        args: raw.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::PREPARE_FLAGS;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn cmd_prepare_requires_input_and_out() {
        let none = Args::parse_for("prepare", &s(&["--out", "o"]), PREPARE_FLAGS).unwrap();
        assert!(cmd_prepare(&none, &[])
            .unwrap_err()
            .to_string()
            .contains("usage: train_transformer prepare"));
        let no_out = Args::parse_for("prepare", &s(&["--input", "x"]), PREPARE_FLAGS).unwrap();
        assert!(cmd_prepare(&no_out, &[])
            .unwrap_err()
            .to_string()
            .contains("usage: train_transformer prepare"));
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("ferrum_prep_unit_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// (documents, offsets, hash hex, total) from `for_each_raw_doc`.
    fn split(path: &Path, sep: &str) -> (Vec<Vec<u8>>, Vec<u64>, String, u64) {
        let (mut docs, mut offs) = (Vec::new(), Vec::new());
        let (mut h, mut total) = (Sha256::new(), 0);
        for_each_raw_doc(path, sep, &mut h, &mut total, |d, o| {
            docs.push(d.to_vec());
            offs.push(o);
            Ok(())
        })
        .unwrap();
        (docs, offs, hex(&h.finalize()), total)
    }

    #[test]
    fn separator_straddling_the_chunk_boundary_is_found_once() {
        let sep = "<|endoftext|>";
        let d = tmpdir("straddle");
        // The separator starts 5 bytes before the end of the first chunk.
        let mut data = vec![b'a'; CHUNK - 5];
        data.extend_from_slice(sep.as_bytes());
        data.extend_from_slice(b"tail");
        let p = d.join("f.txt");
        std::fs::write(&p, &data).unwrap();
        let (docs, offs, hash, total) = split(&p, sep);
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].len(), CHUNK - 5);
        assert_eq!(docs[1], b"tail");
        assert_eq!(offs, vec![0, (CHUNK - 5 + sep.len()) as u64]);
        assert_eq!(total, data.len() as u64);
        assert_eq!(hash, hex(&Sha256::digest(&data)));
        // A separator split across *every* possible chunk-edge offset still counts once.
        for back in 1..sep.len() {
            let mut data = vec![b'b'; CHUNK - back];
            data.extend_from_slice(sep.as_bytes());
            std::fs::write(&p, &data).unwrap();
            let (docs, _, _, _) = split(&p, sep);
            assert_eq!(docs.len(), 2, "back={back}");
            assert_eq!(docs[0].len(), CHUNK - back);
            assert!(docs[1].is_empty());
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_single_huge_line_with_many_separators_splits_in_linear_time() {
        // 60k documents on one 800 KB line: draining the line per separator was quadratic.
        let sep = "<|endoftext|>";
        let d = tmpdir("manysep");
        let data = format!("doc{sep}").repeat(60_000);
        let p = d.join("f.txt");
        std::fs::write(&p, &data).unwrap();
        let t = std::time::Instant::now();
        let (docs, offs, _, _) = split(&p, sep);
        assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
        assert_eq!(docs.len(), 60_001); // 60k separators -> 60k docs + the empty tail
        assert!(docs[..60_000].iter().all(|x| x == b"doc"));
        assert_eq!(offs[1], (3 + sep.len()) as u64);
        assert_eq!(offs[60_000], data.len() as u64);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn newline_free_file_is_one_document_and_empty_file_is_one_empty_document() {
        let d = tmpdir("nonl");
        let data = vec![b'x'; 2 * CHUNK + 17];
        let p = d.join("f.txt");
        std::fs::write(&p, &data).unwrap();
        let (docs, offs, hash, total) = split(&p, "");
        assert_eq!((docs.len(), docs[0].len(), offs[0]), (1, data.len(), 0));
        assert_eq!(
            (total, hash),
            (data.len() as u64, hex(&Sha256::digest(&data)))
        );
        std::fs::write(&p, b"").unwrap();
        let (docs, _, _, total) = split(&p, "<|endoftext|>");
        assert_eq!((docs, total), (vec![Vec::<u8>::new()], 0));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn callback_errors_and_missing_files_propagate() {
        let d = tmpdir("err");
        let p = d.join("f.txt");
        std::fs::write(&p, b"a<|endoftext|>b").unwrap();
        let (mut h, mut t) = (Sha256::new(), 0);
        let e = for_each_raw_doc(&p, "<|endoftext|>", &mut h, &mut t, |_, _| {
            Err("stop".into())
        })
        .unwrap_err();
        assert_eq!(e, "stop");
        let e = for_each_raw_doc(&d.join("nope"), "", &mut h, &mut t, |_, _| Ok(())).unwrap_err();
        assert!(e.contains("cannot open"), "{e}");
        // A directory opens on Linux but cannot be read: a clean error, not a hang.
        let e = for_each_raw_doc(&d, "", &mut h, &mut t, |_, _| Ok(()));
        assert!(e.is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[cfg(unix)]
    #[test]
    fn non_regular_inputs_are_rejected_or_skipped_not_read_forever() {
        // Named directly: refused (reading /dev/zero would never end).
        let e =
            expand_inputs(&["/dev/zero".to_string()], Path::new("/nonexistent-out")).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        assert!(e.to_string().contains("/dev/zero"), "{e}");
        // Inside a walked directory: skipped silently, regular files kept, hidden ignored.
        let d = tmpdir("walk");
        std::fs::write(d.join("a.txt"), b"a").unwrap();
        std::fs::write(d.join(".hidden"), b"h").unwrap();
        if std::process::Command::new("mkfifo")
            .arg(d.join("pipe"))
            .status()
            .is_ok_and(|s| s.success())
        {
            let got = expand_inputs(&[d.to_string_lossy().into()], Path::new("/nonexistent-out"))
                .unwrap();
            assert_eq!(got, vec![d.join("a.txt")]);
        }
        // A dangling symlink is not a regular file: skipped inside a directory.
        std::os::unix::fs::symlink(d.join("missing"), d.join("dangling")).unwrap();
        assert_eq!(
            expand_inputs(&[d.to_string_lossy().into()], Path::new("/nonexistent-out")).unwrap(),
            vec![d.join("a.txt")]
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn expand_inputs_skips_the_output_dir_and_refuses_files_inside_it() {
        let d = tmpdir("outskip");
        let out = d.join("out");
        std::fs::create_dir_all(out.join("nested")).unwrap();
        std::fs::write(d.join("a.txt"), b"a").unwrap();
        std::fs::write(out.join("train.txt"), b"t").unwrap();
        std::fs::write(out.join("nested/x.txt"), b"x").unwrap();
        let dir = d.to_string_lossy().to_string();
        // Walking the parent finds only the real input.
        assert_eq!(
            expand_inputs(std::slice::from_ref(&dir), &out).unwrap(),
            vec![d.join("a.txt")]
        );
        // The output dir itself as the input: everything in it is an output, so nothing remains.
        assert!(expand_inputs(&[out.to_string_lossy().into()], &out)
            .unwrap()
            .is_empty());
        // Naming an output file directly is an error that names the problem.
        let e = expand_inputs(&[out.join("train.txt").to_string_lossy().into()], &out).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        assert!(e.to_string().contains("inside --out"), "{e}");
        // A missing out dir (cannot canonicalize) filters nothing.
        let got = expand_inputs(&[dir], &d.join("not-yet")).unwrap();
        assert_eq!(got.len(), 3);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn rejected_summary_lists_nonzero_reasons_in_key_order() {
        let none = json!({"duplicate": 0, "empty": 0});
        assert_eq!(rejected_summary(&none), "none");
        let some = json!({"duplicate": 3, "empty": 0, "too_short": 1});
        assert_eq!(rejected_summary(&some), "duplicate=3 too_short=1");
        assert_eq!(rejected_summary(&Value::Null), "none");
    }
}
