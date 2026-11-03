//! Corpus text commands: download, clean, save and read text files.

use crate::error::CmdError;
use crate::paths::checked_path;
use crate::progress::Throttle;
use ferrum_core::{clean_corpus, corpus_stats, CleanOptions};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::Path;
use tauri::{AppHandle, Emitter, Runtime};

/// HTTP client with bounded connect/read timeouts so a hung or slow host can
/// never block a download task indefinitely (G2).
pub(crate) fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout_read(std::time::Duration::from_secs(30))
        .timeout_write(std::time::Duration::from_secs(30))
        .build()
}

/// Synchronous core of [`download_text`]: validate the URL, fetch with timeouts,
/// and return at most `cap` bytes decoded as UTF-8. Separated from the Tauri
/// command so it is unit-testable without an `AppHandle`.
fn fetch_text(url: &str, cap: usize) -> Result<String, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("URL must start with http:// or https://".to_string());
    }
    let cap = cap.max(1);
    let resp = http_agent()
        .get(url)
        .call()
        .map_err(|e| format!("download failed: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .take(cap as u64)
        .read_to_end(&mut buf)
        .map_err(|e| format!("read failed: {e}"))?;
    if buf.is_empty() {
        return Err("downloaded resource was empty".to_string());
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Download a text resource (e.g. a Project Gutenberg `.txt`) and return it as a
/// UTF-8 string (lossily decoded), capped at `max_bytes` (default 8 MiB). Uses an
/// HTTP client with connect/read timeouts so a slow host cannot hang the task.
#[tauri::command]
pub async fn download_text(url: String, max_bytes: Option<usize>) -> Result<String, CmdError> {
    crate::error::blocking(move || {
        let cap = max_bytes.unwrap_or(8 * 1024 * 1024);
        fetch_text(&url, cap)
    })
    .await
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanOpts {
    pub strip_gutenberg: bool,
    pub lowercase: bool,
    pub collapse_whitespace: bool,
    pub normalize_punctuation: bool,
    pub strip_control_chars: bool,
    pub max_chars: Option<usize>,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CleanResult {
    pub cleaned: String,
    pub preview: String,
    pub chars: usize,
    pub bytes: usize,
    pub lines: usize,
    pub words: usize,
    pub unique_chars: usize,
}

/// Bytes of raw text cleaned between two progress checks.
const CLEAN_CHUNK: usize = 256 * 1024;

/// [`clean_corpus`] in line-aligned chunks, calling `progress(bytes, documents)` after each
/// (`bytes` = text consumed, counted after Gutenberg stripping; `documents` = non-empty lines). The result equals `clean_corpus(raw, options)`: Gutenberg
/// stripping runs first and whitespace collapsing and the length cap run last, both over the
/// whole text; the per-character steps in between are local, so chunks do not change them.
fn clean_with_progress(
    raw: &str,
    options: &CleanOptions,
    chunk_bytes: usize,
    progress: &mut dyn FnMut(u64, u64),
) -> String {
    let flags = |gut, col, max| CleanOptions {
        strip_gutenberg: gut,
        lowercase: false,
        collapse_whitespace: col,
        normalize_punctuation: false,
        strip_control_chars: false,
        max_chars: max,
    };
    let stripped;
    let body = if options.strip_gutenberg {
        stripped = clean_corpus(raw, &flags(true, false, None));
        stripped.as_str()
    } else {
        raw
    };
    let per_char = CleanOptions {
        strip_gutenberg: false,
        collapse_whitespace: false,
        max_chars: None,
        ..options.clone()
    };
    let mut out = String::with_capacity(body.len());
    let (mut start, mut documents) = (0usize, 0u64);
    while start < body.len() {
        let mut end = (start + chunk_bytes).min(body.len());
        while !body.is_char_boundary(end) {
            end += 1;
        }
        // Cut after a newline (or at the end) so a chunk holds whole lines.
        end = body[end..].find('\n').map_or(body.len(), |i| end + i + 1);
        let chunk = &body[start..end];
        documents += chunk.lines().filter(|l| !l.trim().is_empty()).count() as u64;
        out.push_str(&clean_corpus(chunk, &per_char));
        start = end;
        progress(start as u64, documents);
    }
    if options.collapse_whitespace || options.max_chars.is_some() {
        out = clean_corpus(
            &out,
            &flags(false, options.collapse_whitespace, options.max_chars),
        );
    }
    out
}

/// Clean raw text into a training-ready corpus and report statistics. Does not touch
/// disk. Progress streams as `clean-progress {bytes, documents}` events, at most one per
/// 100 ms, and a final one with the totals.
#[tauri::command]
pub async fn clean_text<R: Runtime>(
    app: AppHandle<R>,
    raw: String,
    opts: CleanOpts,
) -> Result<CleanResult, CmdError> {
    crate::error::blocking(move || {
        let mut th = Throttle::wall();
        clean_text_with(&raw, opts, &mut th, &mut |bytes, documents| {
            let _ = app.emit(
                "clean-progress",
                serde_json::json!({ "bytes": bytes, "documents": documents }),
            );
        })
    })
    .await
}

fn clean_text_with(
    raw: &str,
    opts: CleanOpts,
    throttle: &mut Throttle,
    emit: &mut dyn FnMut(u64, u64),
) -> Result<CleanResult, String> {
    let options = CleanOptions {
        strip_gutenberg: opts.strip_gutenberg,
        lowercase: opts.lowercase,
        collapse_whitespace: opts.collapse_whitespace,
        normalize_punctuation: opts.normalize_punctuation,
        strip_control_chars: opts.strip_control_chars,
        max_chars: opts.max_chars.filter(|&n| n > 0),
    };
    let mut last = (0, 0);
    let cleaned = clean_with_progress(raw, &options, CLEAN_CHUNK, &mut |b, d| {
        last = (b, d);
        if throttle.ready() {
            emit(b, d);
        }
    });
    // The final event always goes out, with the totals.
    emit(last.0, last.1);
    if cleaned.trim().is_empty() {
        return Err("nothing left after cleaning — check the source and options".into());
    }
    let st = corpus_stats(&cleaned);
    let preview: String = cleaned.chars().take(4000).collect();
    Ok(CleanResult {
        preview,
        chars: st.chars,
        bytes: st.bytes,
        lines: st.lines,
        words: st.words,
        unique_chars: st.unique_chars,
        cleaned,
    })
}

/// Write a corpus to disk (creating parent directories), returning bytes written.
#[tauri::command]
pub fn save_corpus(text: String, path: String) -> Result<usize, CmdError> {
    checked_path("output file path", &path, false)?;
    if let Some(parent) = Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {parent:?}: {e}"))?;
        }
    }
    std::fs::write(&path, text.as_bytes()).map_err(|e| format!("cannot write {path}: {e}"))?;
    Ok(text.len())
}

/// Read a UTF-8 text file (lossy), capped at `max_bytes` (default 16 MiB).
#[tauri::command]
pub fn read_text_file(path: String, max_bytes: Option<usize>) -> Result<String, CmdError> {
    // Devices and FIFOs are refused up front; the cap still bounds huge regular files.
    checked_path("file path", &path, true)?;
    let cap = max_bytes.unwrap_or(16 * 1024 * 1024) as u64;
    let file = std::fs::File::open(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut data = Vec::new();
    file.take(cap)
        .read_to_end(&mut data)
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    Ok(String::from_utf8_lossy(&data).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(all: bool, max: Option<usize>) -> CleanOpts {
        CleanOpts {
            strip_gutenberg: all,
            lowercase: all,
            collapse_whitespace: all,
            normalize_punctuation: all,
            strip_control_chars: all,
            max_chars: max,
        }
    }

    const SAMPLE: &str = "\u{feff}Intro\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n\
        Hello\u{a0}\u{201c}World\u{201d}   \r\n\r\n\r\n\r\nSecond \u{2014} line\x01  \n\n\n\nThird…\n\
        *** END OF THE PROJECT GUTENBERG EBOOK X ***\nTail";

    #[test]
    fn chunked_cleaning_equals_clean_corpus_for_every_chunk_size_and_option_mix() {
        for all in [false, true] {
            for max in [None, Some(40)] {
                let o = opts(all, max);
                let options = CleanOptions {
                    strip_gutenberg: o.strip_gutenberg,
                    lowercase: o.lowercase,
                    collapse_whitespace: o.collapse_whitespace,
                    normalize_punctuation: o.normalize_punctuation,
                    strip_control_chars: o.strip_control_chars,
                    max_chars: max,
                };
                for chunk in [1, 7, 40, 1 << 20] {
                    let got = clean_with_progress(SAMPLE, &options, chunk, &mut |_, _| {});
                    assert_eq!(got, clean_corpus(SAMPLE, &options), "{all} {max:?} {chunk}");
                }
            }
        }
        // Nothing to clean: no chunk, no progress, empty result.
        let mut calls = 0;
        let o = CleanOptions::default();
        assert_eq!(clean_with_progress("", &o, 8, &mut |_, _| calls += 1), "");
        assert_eq!(calls, 0);
    }

    #[test]
    fn clean_emits_progress() {
        let raw = "alpha\n\nbeta gamma\ndelta\n".repeat(50);
        // A clock 30 ms later at every check: a throttled stream, then the final event.
        let mut ms = 0;
        let mut th = Throttle::new(|| {
            ms += 30;
            ms
        });
        let mut ev: Vec<(u64, u64)> = Vec::new();
        let mut o = opts(false, None);
        o.collapse_whitespace = true;
        let r = clean_text_with(&raw, o, &mut th, &mut |b, d| ev.push((b, d))).unwrap();
        assert!(r.words > 0);
        // One event per chunk would be 1 here (the text is one 256 KiB chunk): the chunk's
        // event and the final one carry the totals and the 150 non-empty lines.
        let last = *ev.last().unwrap();
        assert_eq!(last, (raw.len() as u64, 150));
        assert!(ev.windows(2).all(|w| w[0].0 <= w[1].0), "{ev:?}");
    }

    #[test]
    fn clean_progress_is_throttled_but_the_final_event_always_goes_out() {
        // 4 lines, 1-byte chunks => many checks, all at the same instant.
        let mut th = Throttle::new(|| 0);
        let mut ev = Vec::new();
        let options = CleanOptions::default();
        let mut sink = |b, d| ev.push((b, d));
        let mut last = (0, 0);
        clean_with_progress("a\nb\nc\nd\n", &options, 1, &mut |b, d| {
            last = (b, d);
            if th.ready() {
                sink(b, d);
            }
        });
        sink(last.0, last.1);
        // The burst passed one event (the first); the final one is extra and carries the totals.
        assert_eq!(ev, [(2, 1), (8, 4)]);
    }

    #[test]
    fn an_empty_result_is_still_an_error_after_the_final_event() {
        let mut th = Throttle::new(|| 0);
        let mut ev = Vec::new();
        let e = clean_text_with("   \n", opts(false, None), &mut th, &mut |b, d| {
            ev.push((b, d))
        })
        .unwrap_err();
        assert!(e.contains("nothing left"), "{e}");
        assert_eq!(ev.last(), Some(&(4, 0)));
    }

    #[test]
    fn read_text_file_caps_the_read_and_refuses_endless_files() {
        let dir = std::env::temp_dir().join(format!("ferrum_gui_read_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.txt");
        std::fs::write(&f, "héllo wörld, this is text").unwrap();
        let p = f.to_str().unwrap().to_string();
        assert_eq!(
            read_text_file(p.clone(), None).unwrap(),
            "héllo wörld, this is text"
        );
        assert_eq!(read_text_file(p.clone(), Some(5)).unwrap(), "héll");
        // A cap that splits a multi-byte char decodes lossily instead of failing.
        assert_eq!(read_text_file(p.clone(), Some(2)).unwrap(), "h\u{fffd}");
        assert_eq!(read_text_file(p.clone(), Some(0)).unwrap(), "");
        assert_eq!(
            read_text_file(p.clone(), Some(usize::MAX)).unwrap().len(),
            27
        );
        let e = read_text_file(dir.join("nope").to_str().unwrap().into(), None).unwrap_err();
        assert!(e.contains("does not exist"), "{e}");
        #[cfg(unix)]
        {
            // Regression: std::fs::read on /dev/zero never returns (and eats all RAM);
            // now refused up front by the path check.
            let e = read_text_file("/dev/zero".into(), Some(1000)).unwrap_err();
            assert!(e.contains("not allowed"), "{e}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn http_agent_builds() {
        // Building the timeout-bounded agent must not panic.
        let _ = http_agent();
    }

    #[test]
    fn fetch_text_rejects_non_http_urls() {
        // Validation happens before any network access, so these never hit the
        // wire — they fail fast with a clear message.
        for url in [
            "ftp://example.com/x",
            "file:///etc/passwd",
            "javascript:1",
            "",
        ] {
            let err = fetch_text(url, 1024).unwrap_err();
            assert!(
                err.contains("http://"),
                "unexpected error for {url:?}: {err}"
            );
        }
    }

    #[test]
    fn fetch_text_returns_the_body_up_to_the_cap_and_refuses_failures() {
        use crate::test_support::http_server;
        let (url, _rx) = http_server("200 OK", "héllo wörld".as_bytes().to_vec(), 1);
        assert_eq!(fetch_text(&url, 1024).unwrap(), "héllo wörld");
        // The cap cuts the body (lossily when it splits a character); a cap of 0 still reads a byte.
        let (url, _rx) = http_server("200 OK", "héllo".as_bytes().to_vec(), 2);
        assert_eq!(fetch_text(&url, 2).unwrap(), "h\u{fffd}");
        assert_eq!(fetch_text(&url, 0).unwrap(), "h");
        let (url, _rx) = http_server("200 OK", Vec::new(), 1);
        assert_eq!(
            fetch_text(&url, 8).unwrap_err(),
            "downloaded resource was empty"
        );
        let (url, _rx) = http_server("404 Not Found", b"no".to_vec(), 1);
        assert!(fetch_text(&url, 8)
            .unwrap_err()
            .starts_with("download failed:"));
        // The command applies the 8 MiB default and reports the same errors as the core.
        let (url, _rx) = http_server("200 OK", b"text".to_vec(), 1);
        let got = tauri::async_runtime::block_on(download_text(url, None)).unwrap();
        assert_eq!(got, "text");
    }

    #[test]
    fn save_corpus_reports_a_folder_or_file_it_cannot_make() {
        let dir = std::env::temp_dir().join(format!("ferrum_gui_savefail_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let blocker = dir.join("file");
        std::fs::write(&blocker, "x").unwrap();
        // A parent that is a regular file cannot become a folder.
        let e = save_corpus(
            "x".into(),
            blocker.join("sub/out.txt").to_str().unwrap().into(),
        )
        .unwrap_err();
        assert!(
            e.contains("cannot create") || e.contains("cannot access"),
            "{e}"
        );
        // A read-only folder refuses the file itself.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let ro = dir.join("ro");
            std::fs::create_dir(&ro).unwrap();
            std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o500)).unwrap();
            let e =
                save_corpus("x".into(), ro.join("out.txt").to_str().unwrap().into()).unwrap_err();
            assert!(e.contains("cannot write"), "{e}");
            std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_commands_refuse_bad_paths() {
        let dir = std::env::temp_dir().join(format!("ferrum_gui_cmdpaths_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap().to_string();

        // read_text_file: device, directory, missing, NUL; a normal file works.
        for bad in [
            "/dev/zero",
            "/proc/self/mem",
            d.as_str(),
            "/nonexistent/x.txt",
            "a\0b",
            "",
        ] {
            assert!(read_text_file(bad.into(), None).is_err(), "{bad:?}");
        }
        let f = dir.join("ok.txt");
        std::fs::write(&f, "hello").unwrap();
        assert_eq!(
            read_text_file(f.to_str().unwrap().into(), None).unwrap(),
            "hello"
        );

        // save_corpus: refuses devices and directories, writes a new file (making parents).
        assert!(save_corpus("x".into(), "/dev/null".into()).is_err());
        assert!(save_corpus("x".into(), d.clone()).is_err());
        assert!(save_corpus("x".into(), String::new()).is_err());
        let out = dir.join("sub/out.txt");
        assert_eq!(
            save_corpus("abc".into(), out.to_str().unwrap().into()).unwrap(),
            3
        );
    }
}
