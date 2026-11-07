//! URL builders, path sanitisation, and streamed HTTP download into `<dest>.part`.

use super::auth::{basic_auth_header, KaggleCreds};
use crate::infra::http::http_agent;
use crate::infra::progress::{Progress, Tap};
use serde::Serialize;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Hard cap on a single download (bytes) so a mislabeled huge file cannot fill
/// the disk unbounded.
pub(super) const DOWNLOAD_CAP: usize = 512 * 1024 * 1024;

/// Where the files come from. The commands use [`HOSTS`]; tests point it at a local server.
pub(super) struct Hosts<'a> {
    pub hf: &'a str,
    pub kaggle: &'a str,
}

pub(super) const HOSTS: Hosts = Hosts {
    hf: "https://huggingface.co",
    kaggle: "https://www.kaggle.com",
};

/// Public HuggingFace dataset file URL (`resolve` endpoint).
pub fn hf_resolve_url(base: &str, repo: &str, file: &str, revision: &str) -> String {
    let rev = if revision.is_empty() {
        "main"
    } else {
        revision
    };
    format!("{base}/datasets/{repo}/resolve/{rev}/{file}")
}

/// Kaggle single-file download URL (requires Basic auth).
pub fn kaggle_file_url(base: &str, owner_slug: &str, file: &str) -> String {
    format!("{base}/api/v1/datasets/download/{owner_slug}/{file}")
}

/// The leaf file name of a repo path, stripped of any directory components so a
/// crafted `file` cannot escape `dest_dir`. Falls back to `fallback`.
pub fn safe_basename(file: &str, fallback: &str) -> String {
    let name = file
        .rsplit(['/', '\\'])
        .next()
        .filter(|s| !s.is_empty() && *s != "." && *s != "..")
        .unwrap_or(fallback);
    name.to_string()
}

/// Build the on-disk destination path inside `dest_dir` for `file`.
pub fn dest_path(dest_dir: &str, file: &str, fallback: &str) -> PathBuf {
    Path::new(dest_dir).join(safe_basename(file, fallback))
}

/// Where this download came from, returned to the UI.
#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct DownloadResult {
    pub path: String,
    pub bytes: u64,
    pub source: String,
}

/// Stream `url` (with optional Basic / Bearer auth) to `dest`, capped at
/// [`DOWNLOAD_CAP`] (over-cap is an error). Goes through `<dest>.part`. Validates
/// the URL scheme and creates parent directories.
pub(super) fn download_to_file(
    url: &str,
    basic: Option<&KaggleCreds>,
    bearer: Option<&str>,
    dest: &Path,
    progress: &mut Progress,
) -> Result<u64, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("URL must start with http:// or https://".to_string());
    }
    if let Some(parent) = dest.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {parent:?}: {e}"))?;
        }
    }
    let mut req = http_agent().get(url);
    if let Some(c) = basic {
        req = req.set("Authorization", &basic_auth_header(&c.username, &c.key));
    }
    if let Some(tok) = bearer {
        if !tok.is_empty() {
            req = req.set("Authorization", &format!("Bearer {tok}"));
        }
    }
    let resp = req.call().map_err(|e| format!("download failed: {e}"))?;
    let total = resp.header("Content-Length").and_then(|v| v.parse().ok());
    stream_to_dest(
        resp.into_reader(),
        dest,
        DOWNLOAD_CAP as u64,
        total,
        progress,
    )
}

/// Copy `reader` into `<dest>.part`, then rename over `dest` once complete.
/// More than `cap` bytes, an empty body, or any I/O error removes the `.part`
/// file and leaves `dest` untouched. Memory use is the copy buffer only.
pub(super) fn stream_to_dest(
    reader: impl Read,
    dest: &Path,
    cap: u64,
    total: Option<u64>,
    progress: &mut Progress,
) -> Result<u64, String> {
    let mut part_name = dest.as_os_str().to_owned();
    part_name.push(".part");
    let part = PathBuf::from(part_name);
    let result = (|| {
        let mut file =
            std::fs::File::create(&part).map_err(|e| format!("cannot create {part:?}: {e}"))?;
        let tap = Tap::new(reader, total, &mut *progress);
        let n = std::io::copy(&mut tap.take(cap + 1), &mut file)
            .map_err(|e| format!("write failed: {e}"))?;
        if n > cap {
            return Err(format!("download exceeds the {cap} byte limit"));
        }
        if n == 0 {
            return Err("downloaded resource was empty (check the dataset id / file path)".into());
        }
        file.sync_all().map_err(|e| format!("write failed: {e}"))?;
        std::fs::rename(&part, dest).map_err(|e| format!("cannot move into place: {e}"))?;
        progress.finish(n);
        Ok(n)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infra::progress::Throttle;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_ds_dl_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    struct FailingReader(usize);
    impl Read for FailingReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.0 == 0 {
                return Err(std::io::Error::other("connection reset"));
            }
            let n = self.0.min(buf.len());
            buf[..n].fill(b'x');
            self.0 -= n;
            Ok(n)
        }
    }

    struct Chunks {
        chunk: usize,
        reads: usize,
    }
    impl Read for Chunks {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.reads == 0 {
                return Ok(0);
            }
            self.reads -= 1;
            let n = self.chunk.min(buf.len());
            buf[..n].fill(b'x');
            Ok(n)
        }
    }

    #[test]
    fn hf_and_kaggle_urls_are_well_formed() {
        assert_eq!(
            hf_resolve_url(
                HOSTS.hf,
                "roneneldan/TinyStories",
                "TinyStories-valid.txt",
                "main"
            ),
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
        );
        assert!(hf_resolve_url(HOSTS.hf, "a/b", "f.txt", "").ends_with("/resolve/main/f.txt"));
        assert_eq!(
            kaggle_file_url(
                HOSTS.kaggle,
                "kingburrito666/shakespeare-plays",
                "Shakespeare_data.csv"
            ),
            "https://www.kaggle.com/api/v1/datasets/download/kingburrito666/shakespeare-plays/Shakespeare_data.csv"
        );
    }

    #[test]
    fn safe_basename_and_dest_path_strip_traversal() {
        assert_eq!(
            safe_basename("wikitext-2-raw/wiki.train.raw", "fb"),
            "wiki.train.raw"
        );
        assert_eq!(safe_basename("a/b/c.txt", "fb"), "c.txt");
        assert_eq!(safe_basename("../../etc/passwd", "fb"), "passwd");
        assert_eq!(safe_basename("", "fallback"), "fallback");
        assert_eq!(safe_basename("..", "fallback"), "fallback");
        assert_eq!(
            dest_path("/tmp/out", "sub/dir/file.txt", "fb"),
            Path::new("/tmp/out/file.txt")
        );
    }

    #[test]
    fn download_rejects_non_http_urls_before_network() {
        let dest = std::env::temp_dir().join("ferrum_ds_test_noop.bin");
        let err = download_to_file(
            "ftp://example.com/x",
            None,
            None,
            &dest,
            &mut Progress::silent(),
        )
        .unwrap_err();
        assert!(err.contains("http://"));
    }

    #[test]
    fn stream_complete_over_cap_and_midway_errors_behave_atomically() {
        let d = scratch("stream");
        let dest = d.join("f.txt");
        assert_eq!(
            stream_to_dest(&b"hello"[..], &dest, 10, None, &mut Progress::silent()).unwrap(),
            5
        );
        assert_eq!(std::fs::read(&dest).unwrap(), b"hello");
        assert!(!d.join("f.txt.part").exists());
        assert_eq!(
            stream_to_dest(&b"0123456789"[..], &dest, 10, None, &mut Progress::silent()).unwrap(),
            10
        );

        let err_cap = stream_to_dest(
            &b"0123456789A"[..],
            &dest,
            10,
            None,
            &mut Progress::silent(),
        )
        .unwrap_err();
        assert!(err_cap.contains("exceeds the 10 byte limit"), "{err_cap}");
        assert_eq!(std::fs::read(&dest).unwrap(), b"0123456789");

        std::fs::write(&dest, b"old").unwrap();
        assert!(stream_to_dest(
            FailingReader(100),
            &dest,
            1000,
            None,
            &mut Progress::silent()
        )
        .is_err());
        assert!(stream_to_dest(&b""[..], &dest, 10, None, &mut Progress::silent()).is_err());
        assert_eq!(std::fs::read(&dest).unwrap(), b"old");
        assert!(!d.join("f.txt.part").exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_download_sends_its_credentials_and_lands_in_the_folder() {
        use crate::test_support::http_server;
        let d = scratch("dl");
        let (url, rx) = http_server("200 OK", b"corpus".to_vec(), 3);
        let kaggle = KaggleCreds {
            username: "user".into(),
            key: "pass".into(),
        };
        let dest = d.join("deep/er/a.txt");
        assert_eq!(
            download_to_file(&url, Some(&kaggle), None, &dest, &mut Progress::silent()).unwrap(),
            6
        );
        assert_eq!(std::fs::read(&dest).unwrap(), b"corpus");
        let sent = rx.recv().unwrap().to_lowercase();
        assert!(sent.contains("authorization: basic dxnlcjpwyxnz"), "{sent}");
        download_to_file(
            &url,
            None,
            Some("tok"),
            &d.join("b.txt"),
            &mut Progress::silent(),
        )
        .unwrap();
        assert!(rx
            .recv()
            .unwrap()
            .to_lowercase()
            .contains("authorization: bearer tok"));
        download_to_file(
            &url,
            None,
            Some(""),
            &d.join("c.txt"),
            &mut Progress::silent(),
        )
        .unwrap();
        assert!(!rx.recv().unwrap().to_lowercase().contains("authorization"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn download_emits_monotonic_progress_and_handles_failures() {
        use crate::test_support::http_server;
        let d = scratch("prog");
        let dest = d.join("a.txt");
        let mut ms = 0u64;
        let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
        let n = {
            let throttle = Throttle::new(|| {
                ms += 40;
                ms
            });
            let mut p = Progress::new(throttle, |b, t| seen.push((b, t)));
            stream_to_dest(
                Chunks {
                    chunk: 1000,
                    reads: 20,
                },
                &dest,
                1 << 20,
                Some(20_000),
                &mut p,
            )
            .unwrap()
        };
        assert_eq!(n, 20_000);
        assert!(seen.len() < 20 / 2 + 2, "{seen:?}");
        assert!(seen.windows(2).all(|w| w[0].0 < w[1].0), "{seen:?}");
        assert_eq!(*seen.last().unwrap(), (20_000, Some(20_000)));

        let mut fail_seen: Vec<(u64, Option<u64>)> = Vec::new();
        let mut p = Progress::new(Throttle::new(|| 0), |b, t| fail_seen.push((b, t)));
        let r = stream_to_dest(
            Chunks {
                chunk: 10,
                reads: 3,
            },
            &d.join("x"),
            5,
            None,
            &mut p,
        );
        assert!(r.unwrap_err().contains("exceeds"));
        drop(p);
        assert_eq!(fail_seen, [(6, None)]);

        let (url, _rx) = http_server("200 OK", b"corpus".to_vec(), 1);
        let mut http_seen: Vec<(u64, Option<u64>)> = Vec::new();
        let mut p = Progress::new(Throttle::new(|| 0), |b, t| http_seen.push((b, t)));
        download_to_file(&url, None, None, &d.join("h.txt"), &mut p).unwrap();
        drop(p);
        assert_eq!(http_seen.first(), Some(&(6, Some(6))));
        assert_eq!(http_seen.last(), Some(&(6, Some(6))));

        let (url404, _rx) = http_server("404 Not Found", b"gone".to_vec(), 1);
        let e = download_to_file(
            &url404,
            None,
            None,
            &d.join("nf.txt"),
            &mut Progress::silent(),
        )
        .unwrap_err();
        assert!(e.starts_with("download failed:"), "{e}");

        std::fs::write(d.join("file"), "x").unwrap();
        let e = download_to_file(
            &url404,
            None,
            None,
            &d.join("file/sub/y.txt"),
            &mut Progress::silent(),
        )
        .unwrap_err();
        assert!(e.starts_with("cannot create"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }
}
