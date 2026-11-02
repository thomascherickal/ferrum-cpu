//! Corpus text commands: download, clean, save and read text files.

use crate::error::CmdError;
use crate::paths::checked_path;
use ferrum_core::{clean_corpus, corpus_stats, CleanOptions};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::Path;

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

#[derive(Serialize)]
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

/// Clean raw text into a training-ready corpus and report statistics. Pure;
/// does not touch disk.
#[tauri::command]
pub fn clean_text(raw: String, opts: CleanOpts) -> Result<CleanResult, CmdError> {
    let options = CleanOptions {
        strip_gutenberg: opts.strip_gutenberg,
        lowercase: opts.lowercase,
        collapse_whitespace: opts.collapse_whitespace,
        normalize_punctuation: opts.normalize_punctuation,
        strip_control_chars: opts.strip_control_chars,
        max_chars: opts.max_chars.filter(|&n| n > 0),
    };
    let cleaned = clean_corpus(&raw, &options);
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
