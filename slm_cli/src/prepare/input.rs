//! Reading the inputs of a corpus run: which files, and their documents streamed
//! in fixed-size chunks. Shared by `prepare` and `tokenize`.

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Every regular, non-hidden file under each input (directories walked
/// recursively; FIFOs, devices and sockets inside them are skipped), sorted by
/// path. A non-regular file named directly is an error: `/dev/zero` or a pipe
/// would otherwise be read forever.
///
/// Anything under `outdir` (this run's own outputs and manifest, e.g. `--out`
/// inside `--input`) is skipped when found by walking a directory and an error
/// when named directly: re-ingesting the outputs would feed them back in.
pub fn expand_inputs(inputs: &[String], outdir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let outdir = std::fs::canonicalize(outdir).unwrap_or(outdir.to_path_buf());
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
            let ctx = |e: std::io::Error| {
                std::io::Error::new(e.kind(), format!("cannot read {}: {e}", p.display()))
            };
            for e in std::fs::read_dir(p).map_err(ctx)? {
                let e = e.map_err(ctx)?.path();
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
pub fn for_each_raw_doc(
    path: &Path,
    sep: &str,
    hasher: &mut Sha256,
    total: &mut u64,
    mut f: impl FnMut(&[u8], u64) -> Result<(), String>,
) -> Result<(), String> {
    let mut file = crate::util::fmt_err(
        std::fs::File::open(path),
        &format!("cannot open {}", path.display()),
    )?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepare::split::hex;

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

    fn noop_doc(_: &[u8], _: u64) -> Result<(), String> {
        Ok(())
    }

    #[test]
    fn callback_errors_and_missing_files_propagate() {
        assert!(noop_doc(&[], 0).is_ok());
        let d = tmpdir("err");
        let p = d.join("f.txt");
        std::fs::write(&p, b"a<|endoftext|>b").unwrap();
        let (mut h, mut t) = (Sha256::new(), 0);
        let e = for_each_raw_doc(&p, "<|endoftext|>", &mut h, &mut t, |_, _| {
            Err("stop".into())
        })
        .unwrap_err();
        assert_eq!(e, "stop");
        let e = for_each_raw_doc(&d.join("nope"), "", &mut h, &mut t, noop_doc).unwrap_err();
        assert!(e.contains("cannot open"), "{e}");
        // A directory opens on Linux but cannot be read: a clean error, not a hang.
        let e = for_each_raw_doc(&d, "", &mut h, &mut t, noop_doc);
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
        // A unix socket is another non-regular file; std can make one without any tool.
        let _socket = std::os::unix::net::UnixListener::bind(d.join("sock")).unwrap();
        let got =
            expand_inputs(&[d.to_string_lossy().into()], Path::new("/nonexistent-out")).unwrap();
        assert_eq!(got, vec![d.join("a.txt")]);
        // A dangling symlink is not a regular file: skipped inside a directory.
        std::os::unix::fs::symlink(d.join("missing"), d.join("dangling")).unwrap();
        assert_eq!(
            expand_inputs(&[d.to_string_lossy().into()], Path::new("/nonexistent-out")).unwrap(),
            vec![d.join("a.txt")]
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_directory_is_reported_with_its_path() {
        use std::os::unix::fs::PermissionsExt;
        let d = tmpdir("noread");
        let locked = d.join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let r = expand_inputs(
            &[locked.to_string_lossy().into()],
            Path::new("/nonexistent-out"),
        );
        // Restore first so the cleanup works even if the assertion fails.
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        let _ = std::fs::remove_dir_all(&d);
        // ponytail: a process that bypasses permissions (root) reads the directory anyway.
        if let Err(e) = r {
            assert!(e.to_string().contains("cannot read ") && e.to_string().contains("locked"));
        }
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
}
