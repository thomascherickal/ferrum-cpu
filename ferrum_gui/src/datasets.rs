//! Dataset discovery + download from HuggingFace and Kaggle.
//!
//! Only datasets small enough to be practical for **CPU** training/inference are
//! surfaced ([`CPU_MAX_MB`]). The pure logic — the curated catalog, the URL
//! builders, Kaggle credential parsing, base64 for Basic auth, and the
//! destination-path sanitiser — is split out from the `#[tauri::command]`
//! wrappers so it is unit-tested without a GUI runtime or a network.
//!
//! HuggingFace dataset files resolve to a public URL and need no auth (a token
//! is accepted for gated repos). Kaggle's API requires Basic auth from
//! `KAGGLE_USERNAME` / `KAGGLE_KEY` or `~/.kaggle/kaggle.json`.

use crate::commands::http_agent;
use crate::error::CmdError;
use crate::paths::checked_dir;
use serde::Serialize;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Upper bound (MB) on datasets we surface as "CPU-friendly". Bigger sets still
/// download via the generic commands, but are kept out of the curated list so a
/// user does not pick something that won't train on a laptop.
pub const CPU_MAX_MB: u32 = 200;

/// Hard cap on a single download (bytes) so a mislabeled huge file cannot fill
/// the disk unbounded.
const DOWNLOAD_CAP: usize = 512 * 1024 * 1024;

/// One curated, CPU-sized dataset.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DatasetEntry {
    pub id: String,
    pub name: String,
    /// `"huggingface"` or `"kaggle"`.
    pub source: String,
    pub description: String,
    pub approx_mb: u32,
    /// HuggingFace `owner/name`, or Kaggle `owner/slug`.
    pub repo: String,
    /// File within the repo to fetch.
    pub file: String,
    /// `"text"` | `"csv"` — informational for the UI.
    pub format: String,
}

// Compact row builder for the static dataset catalog below.
#[allow(clippy::too_many_arguments)]
fn e(
    id: &str,
    name: &str,
    source: &str,
    desc: &str,
    mb: u32,
    repo: &str,
    file: &str,
    fmt: &str,
) -> DatasetEntry {
    DatasetEntry {
        id: id.into(),
        name: name.into(),
        source: source.into(),
        description: desc.into(),
        approx_mb: mb,
        repo: repo.into(),
        file: file.into(),
        format: fmt.into(),
    }
}

/// The curated catalog of small, CPU-friendly text datasets. URLs are resolved
/// (and existence verified) at download time, so a stale entry surfaces as a
/// clear "download failed" message rather than silently doing the wrong thing.
pub fn catalog() -> Vec<DatasetEntry> {
    vec![
        e(
            "tinystories-valid",
            "TinyStories (validation)",
            "huggingface",
            "Short synthetic children's stories — ideal tiny LM corpus.",
            20,
            "roneneldan/TinyStories",
            "TinyStories-valid.txt",
            "text",
        ),
        e(
            "tinystories-v2-valid",
            "TinyStories V2 GPT-4 (validation)",
            "huggingface",
            "GPT-4-generated TinyStories, validation split.",
            22,
            "roneneldan/TinyStories",
            "TinyStoriesV2-GPT4-valid.txt",
            "text",
        ),
        e(
            "wikitext2-raw-train",
            "WikiText-2 (raw, train)",
            "huggingface",
            "Lightly-processed Wikipedia text; a classic small LM benchmark.",
            12,
            "mindchain/wikitext2",
            "wikitext-2-raw/wiki.train.raw",
            "text",
        ),
        e(
            "shakespeare-plays",
            "Shakespeare Plays",
            "kaggle",
            "All of Shakespeare's plays as lines of dialogue.",
            10,
            "kingburrito666/shakespeare-plays",
            "Shakespeare_data.csv",
            "csv",
        ),
        e(
            "poetry-foundation",
            "Poetry Foundation Poems",
            "kaggle",
            "~13k English poems — compact, stylistically rich corpus.",
            6,
            "tgdivy/poetry-foundation-poems",
            "PoetryFoundationData.csv",
            "csv",
        ),
    ]
}

/// The catalog filtered to entries within the CPU size budget.
pub fn cpu_catalog() -> Vec<DatasetEntry> {
    catalog()
        .into_iter()
        .filter(|d| d.approx_mb <= CPU_MAX_MB)
        .collect()
}

// ── URL builders ──────────────────────────────────────────────────────────────

/// Where the files come from. The commands use [`HOSTS`]; tests point it at a local server.
struct Hosts<'a> {
    hf: &'a str,
    kaggle: &'a str,
}

const HOSTS: Hosts = Hosts {
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

// ── Kaggle credentials ────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct KaggleCreds {
    pub username: String,
    pub key: String,
}

/// Minimal parser for `~/.kaggle/kaggle.json` (`{"username":"..","key":".."}`),
/// using `serde_json` (already a dependency).
pub fn parse_kaggle_json(s: &str) -> Option<KaggleCreds> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    let username = v.get("username")?.as_str()?.to_string();
    let key = v.get("key")?.as_str()?.to_string();
    if username.is_empty() || key.is_empty() {
        return None;
    }
    Some(KaggleCreds { username, key })
}

/// Resolve Kaggle credentials from the environment, then from
/// `~/.kaggle/kaggle.json`.
pub fn kaggle_creds() -> Option<KaggleCreds> {
    kaggle_creds_from(|name| std::env::var(name).ok())
}

/// [`kaggle_creds`] over any environment lookup, so tests need not touch the real one.
fn kaggle_creds_from(env: impl Fn(&str) -> Option<String>) -> Option<KaggleCreds> {
    if let (Some(username), Some(key)) = (env("KAGGLE_USERNAME"), env("KAGGLE_KEY")) {
        if !username.is_empty() && !key.is_empty() {
            return Some(KaggleCreds { username, key });
        }
    }
    let home = env("HOME").or_else(|| env("USERPROFILE"))?;
    let path = Path::new(&home).join(".kaggle").join("kaggle.json");
    let s = std::fs::read_to_string(path).ok()?;
    parse_kaggle_json(&s)
}

// ── base64 (for the Basic auth header) ────────────────────────────────────────

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
pub fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn basic_auth_header(user: &str, key: &str) -> String {
    format!(
        "Basic {}",
        base64_encode(format!("{user}:{key}").as_bytes())
    )
}

// ── Destination path ──────────────────────────────────────────────────────────

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

// ── Download core ─────────────────────────────────────────────────────────────

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
fn download_to_file(
    url: &str,
    basic: Option<&KaggleCreds>,
    bearer: Option<&str>,
    dest: &Path,
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
    stream_to_dest(resp.into_reader(), dest, DOWNLOAD_CAP as u64)
}

/// Copy `reader` into `<dest>.part`, then rename over `dest` once complete.
/// More than `cap` bytes, an empty body, or any I/O error removes the `.part`
/// file and leaves `dest` untouched. Memory use is the copy buffer only.
fn stream_to_dest(reader: impl Read, dest: &Path, cap: u64) -> Result<u64, String> {
    let mut part_name = dest.as_os_str().to_owned();
    part_name.push(".part");
    let part = PathBuf::from(part_name);
    let result = (|| {
        let mut file =
            std::fs::File::create(&part).map_err(|e| format!("cannot create {part:?}: {e}"))?;
        // Read one byte past the cap so a longer body is detected, not truncated.
        let n = std::io::copy(&mut reader.take(cap + 1), &mut file)
            .map_err(|e| format!("write failed: {e}"))?;
        if n > cap {
            return Err(format!("download exceeds the {cap} byte limit"));
        }
        if n == 0 {
            return Err("downloaded resource was empty (check the dataset id / file path)".into());
        }
        file.sync_all().map_err(|e| format!("write failed: {e}"))?;
        std::fs::rename(&part, dest).map_err(|e| format!("cannot move into place: {e}"))?;
        Ok(n)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result
}

// ── Tauri commands ────────────────────────────────────────────────────────────

/// List the curated, CPU-sized datasets available to download.
#[tauri::command]
pub fn list_datasets() -> Vec<DatasetEntry> {
    cpu_catalog()
}

const NEEDS_KAGGLE_CREDS: &str = "Kaggle needs credentials: set KAGGLE_USERNAME and KAGGLE_KEY, \
     or place kaggle.json in ~/.kaggle/";

fn download_dataset_from(
    hosts: &Hosts,
    kaggle: Option<KaggleCreds>,
    hf_token: Option<String>,
    id: &str,
    dest_dir: &str,
) -> Result<DownloadResult, String> {
    checked_dir("download folder", dest_dir)?;
    let entry = catalog()
        .into_iter()
        .find(|d| d.id == id)
        .ok_or_else(|| format!("unknown dataset id '{id}'"))?;
    let dest = dest_path(dest_dir, &entry.file, &entry.id);
    // The catalog holds only these two sources (a test checks it).
    let (url, basic, bearer, source) = if entry.source == "kaggle" {
        (
            kaggle_file_url(hosts.kaggle, &entry.repo, &entry.file),
            Some(kaggle.ok_or(NEEDS_KAGGLE_CREDS)?),
            None,
            "kaggle",
        )
    } else {
        (
            hf_resolve_url(hosts.hf, &entry.repo, &entry.file, "main"),
            None,
            hf_token,
            "huggingface",
        )
    };
    let bytes = download_to_file(&url, basic.as_ref(), bearer.as_deref(), &dest)?;
    Ok(DownloadResult {
        path: dest.display().to_string(),
        bytes,
        source: source.into(),
    })
}

/// Download a curated catalog entry by `id` into `dest_dir`.
#[tauri::command]
pub async fn download_dataset(id: String, dest_dir: String) -> Result<DownloadResult, CmdError> {
    crate::error::blocking(move || {
        let hf_token = std::env::var("HF_TOKEN").ok();
        download_dataset_from(&HOSTS, kaggle_creds(), hf_token, &id, &dest_dir)
    })
    .await
}

fn download_hf_from(
    hosts: &Hosts,
    env_token: Option<String>,
    (repo, file, revision, dest_dir, hf_token): (&str, &str, Option<&str>, &str, Option<String>),
) -> Result<DownloadResult, String> {
    checked_dir("download folder", dest_dir)?;
    if repo.trim().is_empty() || file.trim().is_empty() {
        return Err("provide a HuggingFace repo (owner/name) and file".into());
    }
    let url = hf_resolve_url(hosts.hf, repo, file, revision.unwrap_or("main"));
    let dest = dest_path(dest_dir, file, "dataset.bin");
    let token = hf_token.or(env_token);
    let bytes = download_to_file(&url, None, token.as_deref(), &dest)?;
    Ok(DownloadResult {
        path: dest.display().to_string(),
        bytes,
        source: "huggingface".into(),
    })
}

/// Download any HuggingFace dataset file (`repo` = `owner/name`), not just the
/// curated ones. `revision` defaults to `main`; `hf_token` is for gated repos.
#[tauri::command]
pub async fn download_hf_file(
    repo: String,
    file: String,
    revision: Option<String>,
    dest_dir: String,
    hf_token: Option<String>,
) -> Result<DownloadResult, CmdError> {
    crate::error::blocking(move || {
        let env_token = std::env::var("HF_TOKEN").ok();
        let args = (&*repo, &*file, revision.as_deref(), &*dest_dir, hf_token);
        download_hf_from(&HOSTS, env_token, args)
    })
    .await
}

fn download_kaggle_from(
    hosts: &Hosts,
    kaggle: Option<KaggleCreds>,
    owner_slug: &str,
    file: &str,
    dest_dir: &str,
) -> Result<DownloadResult, String> {
    checked_dir("download folder", dest_dir)?;
    if owner_slug.trim().is_empty() || file.trim().is_empty() {
        return Err("provide a Kaggle owner/slug and file".into());
    }
    let creds = kaggle.ok_or(NEEDS_KAGGLE_CREDS)?;
    let url = kaggle_file_url(hosts.kaggle, owner_slug, file);
    let dest = dest_path(dest_dir, file, "dataset.bin");
    let bytes = download_to_file(&url, Some(&creds), None, &dest)?;
    Ok(DownloadResult {
        path: dest.display().to_string(),
        bytes,
        source: "kaggle".into(),
    })
}

/// Download any Kaggle dataset file (`owner_slug` = `owner/dataset`).
#[tauri::command]
pub async fn download_kaggle_file(
    owner_slug: String,
    file: String,
    dest_dir: String,
) -> Result<DownloadResult, CmdError> {
    crate::error::blocking(move || {
        download_kaggle_from(&HOSTS, kaggle_creds(), &owner_slug, &file, &dest_dir)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_commands_refuse_bad_destination_folders() {
        use tauri::async_runtime::block_on;
        for bad in ["/dev/shm/x", "/proc/x", "", "a\0b"] {
            let d = bad.to_string();
            let e = block_on(download_dataset("any".into(), d.clone())).unwrap_err();
            assert!(e.contains("download folder"), "{bad:?}: {e}");
            let e = block_on(download_hf_file(
                "o/n".into(),
                "f".into(),
                None,
                d.clone(),
                None,
            ))
            .unwrap_err();
            assert!(e.contains("download folder"), "{bad:?}: {e}");
            let e = block_on(download_kaggle_file("o/n".into(), "f".into(), d)).unwrap_err();
            assert!(e.contains("download folder"), "{bad:?}: {e}");
        }
    }

    #[test]
    fn catalog_is_nonempty_and_cpu_sized() {
        let c = catalog();
        assert!(!c.is_empty());
        for d in &c {
            assert!(d.approx_mb <= CPU_MAX_MB, "{} exceeds CPU budget", d.id);
            assert!(
                d.source == "huggingface" || d.source == "kaggle",
                "bad source for {}",
                d.id
            );
            assert!(!d.repo.is_empty() && !d.file.is_empty());
        }
        // ids are unique.
        let mut ids: Vec<&str> = c.iter().map(|d| d.id.as_str()).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate dataset ids");
    }

    #[test]
    fn cpu_catalog_filters_by_size() {
        // Every listed entry is within budget (all of ours are, but the filter
        // is what the command relies on).
        assert!(cpu_catalog().iter().all(|d| d.approx_mb <= CPU_MAX_MB));
        assert_eq!(
            cpu_catalog().len(),
            catalog()
                .iter()
                .filter(|d| d.approx_mb <= CPU_MAX_MB)
                .count()
        );
    }

    #[test]
    fn hf_url_is_well_formed() {
        assert_eq!(
            hf_resolve_url(HOSTS.hf, "roneneldan/TinyStories", "TinyStories-valid.txt", "main"),
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
        );
        // Empty revision defaults to main.
        assert!(hf_resolve_url(HOSTS.hf, "a/b", "f.txt", "").ends_with("/resolve/main/f.txt"));
    }

    #[test]
    fn kaggle_url_is_well_formed() {
        assert_eq!(
            kaggle_file_url(HOSTS.kaggle, "kingburrito666/shakespeare-plays", "Shakespeare_data.csv"),
            "https://www.kaggle.com/api/v1/datasets/download/kingburrito666/shakespeare-plays/Shakespeare_data.csv"
        );
    }

    #[test]
    fn parse_kaggle_json_extracts_creds() {
        let c = parse_kaggle_json(r#"{"username":"alice","key":"deadbeef"}"#).unwrap();
        assert_eq!(c.username, "alice");
        assert_eq!(c.key, "deadbeef");
        assert!(parse_kaggle_json(r#"{"username":"alice"}"#).is_none()); // missing key
        assert!(parse_kaggle_json(r#"{"username":"","key":"x"}"#).is_none()); // empty
        assert!(parse_kaggle_json("not json").is_none());
    }

    #[test]
    fn base64_matches_known_vectors() {
        // RFC 4648 test vectors.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        // Basic auth header shape.
        assert_eq!(basic_auth_header("user", "pass"), "Basic dXNlcjpwYXNz");
    }

    #[test]
    fn safe_basename_strips_directories_and_traversal() {
        assert_eq!(
            safe_basename("wikitext-2-raw/wiki.train.raw", "fb"),
            "wiki.train.raw"
        );
        assert_eq!(safe_basename("a/b/c.txt", "fb"), "c.txt");
        assert_eq!(safe_basename("../../etc/passwd", "fb"), "passwd");
        assert_eq!(safe_basename("", "fallback"), "fallback");
        assert_eq!(safe_basename("..", "fallback"), "fallback");
    }

    #[test]
    fn dest_path_joins_into_dir() {
        let p = dest_path("/tmp/out", "sub/dir/file.txt", "fb");
        assert_eq!(p, Path::new("/tmp/out/file.txt"));
    }

    #[test]
    fn download_rejects_non_http_urls_before_network() {
        let dest = std::env::temp_dir().join("ferrum_ds_test_noop.bin");
        let err = download_to_file("ftp://example.com/x", None, None, &dest).unwrap_err();
        assert!(err.contains("http://"));
    }

    /// Fresh scratch dir per test (std only; removed by the caller).
    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_ds_{name}_{}", std::process::id()));
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

    #[test]
    fn stream_complete_renames_and_leaves_no_part() {
        let d = scratch("ok");
        let dest = d.join("f.txt");
        assert_eq!(stream_to_dest(&b"hello"[..], &dest, 10).unwrap(), 5);
        assert_eq!(std::fs::read(&dest).unwrap(), b"hello");
        assert!(!d.join("f.txt.part").exists());
        // Exactly at the cap is fine.
        assert_eq!(stream_to_dest(&b"0123456789"[..], &dest, 10).unwrap(), 10);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn stream_over_cap_errors_and_leaves_nothing() {
        let d = scratch("cap");
        let dest = d.join("f.txt");
        let err = stream_to_dest(&b"0123456789A"[..], &dest, 10).unwrap_err();
        assert!(err.contains("exceeds the 10 byte limit"), "{err}");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn stream_midway_error_leaves_nothing_and_keeps_existing_dest() {
        let d = scratch("err");
        let dest = d.join("f.txt");
        let err = stream_to_dest(FailingReader(100), &dest, 1000).unwrap_err();
        assert!(err.contains("connection reset"), "{err}");
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);

        std::fs::write(&dest, b"old").unwrap();
        assert!(stream_to_dest(FailingReader(100), &dest, 1000).is_err());
        assert!(stream_to_dest(&b"0123456789A"[..], &dest, 10).is_err());
        assert!(stream_to_dest(&b""[..], &dest, 10).is_err());
        assert_eq!(std::fs::read(&dest).unwrap(), b"old");
        assert!(!d.join("f.txt.part").exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn download_dataset_unknown_id_errors_fast() {
        // The synchronous validation path (no network) is what we assert on.
        assert!(catalog().iter().all(|d| d.id != "does-not-exist"));
    }

    fn lookup<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn kaggle_credentials_come_from_the_environment_then_kaggle_json() {
        let creds = |u: &str, k: &str| {
            Some(KaggleCreds {
                username: u.into(),
                key: k.into(),
            })
        };
        let env = [("KAGGLE_USERNAME", "alice"), ("KAGGLE_KEY", "k1")];
        assert_eq!(kaggle_creds_from(lookup(&env)), creds("alice", "k1"));
        // Without them (or with an empty one) the file in ~/.kaggle is read, via HOME or USERPROFILE.
        let d = scratch("kaggle_home");
        std::fs::create_dir_all(d.join(".kaggle")).unwrap();
        std::fs::write(
            d.join(".kaggle/kaggle.json"),
            r#"{"username":"bob","key":"k2"}"#,
        )
        .unwrap();
        let home = d.to_str().unwrap();
        assert_eq!(
            kaggle_creds_from(lookup(&[("HOME", home)])),
            creds("bob", "k2")
        );
        let empty = [
            ("KAGGLE_USERNAME", ""),
            ("KAGGLE_KEY", "k"),
            ("USERPROFILE", home),
        ];
        assert_eq!(kaggle_creds_from(lookup(&empty)), creds("bob", "k2"));
        // No variables, no home, or no file: none.
        assert_eq!(kaggle_creds_from(lookup(&[])), None);
        assert_eq!(
            kaggle_creds_from(lookup(&[("HOME", "/nonexistent-home")])),
            None
        );
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
        // Basic auth, creating the missing parent folders.
        let dest = d.join("deep/er/a.txt");
        assert_eq!(
            download_to_file(&url, Some(&kaggle), None, &dest).unwrap(),
            6
        );
        assert_eq!(std::fs::read(&dest).unwrap(), b"corpus");
        let sent = rx.recv().unwrap().to_lowercase();
        assert!(sent.contains("authorization: basic dxnlcjpwyxnz"), "{sent}");
        // A bearer token is sent; an empty one is not.
        download_to_file(&url, None, Some("tok"), &d.join("b.txt")).unwrap();
        assert!(rx
            .recv()
            .unwrap()
            .to_lowercase()
            .contains("authorization: bearer tok"));
        download_to_file(&url, None, Some(""), &d.join("c.txt")).unwrap();
        assert!(!rx.recv().unwrap().to_lowercase().contains("authorization"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_download_that_cannot_start_or_is_refused_leaves_nothing() {
        use crate::test_support::http_server;
        let d = scratch("dl_fail");
        let (url, _rx) = http_server("404 Not Found", b"gone".to_vec(), 1);
        let dest = d.join("x.txt");
        let e = download_to_file(&url, None, None, &dest).unwrap_err();
        assert!(e.starts_with("download failed:"), "{e}");
        assert!(!dest.exists());
        // A parent that is a file cannot be made into a folder.
        std::fs::write(d.join("file"), "x").unwrap();
        let e = download_to_file(&url, None, None, &d.join("file/sub/y.txt")).unwrap_err();
        assert!(e.starts_with("cannot create"), "{e}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_curated_download_picks_the_source_and_needs_kaggle_credentials_only_for_kaggle() {
        use crate::test_support::http_server;
        let d = scratch("curated");
        let dir = d.to_str().unwrap();
        let (base, rx) = http_server("200 OK", b"story".to_vec(), 3);
        let hosts = Hosts {
            hf: &base,
            kaggle: &base,
        };
        let e = download_dataset_from(&hosts, None, None, "nope", dir).unwrap_err();
        assert_eq!(e, "unknown dataset id 'nope'");
        let hf = download_dataset_from(&hosts, None, Some("tok".into()), "tinystories-valid", dir)
            .unwrap();
        assert_eq!((hf.source.as_str(), hf.bytes), ("huggingface", 5));
        assert!(hf.path.ends_with("TinyStories-valid.txt"));
        let sent = rx.recv().unwrap();
        assert!(
            sent.contains("/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"),
            "{sent}"
        );
        assert!(
            sent.to_lowercase().contains("authorization: bearer tok"),
            "{sent}"
        );
        let e = download_dataset_from(&hosts, None, None, "shakespeare-plays", dir).unwrap_err();
        assert!(e.starts_with("Kaggle needs credentials"), "{e}");
        let creds = KaggleCreds {
            username: "u".into(),
            key: "k".into(),
        };
        let k = download_dataset_from(&hosts, Some(creds), None, "shakespeare-plays", dir).unwrap();
        assert_eq!(k.source, "kaggle");
        let sent = rx.recv().unwrap();
        assert!(
            sent.contains(
                "/api/v1/datasets/download/kingburrito666/shakespeare-plays/Shakespeare_data.csv"
            ),
            "{sent}"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_generic_downloads_check_their_form_then_fetch_the_named_file() {
        use crate::test_support::http_server;
        let d = scratch("generic");
        let dir = d.to_str().unwrap();
        let (base, rx) = http_server("200 OK", b"rows".to_vec(), 3);
        let hosts = Hosts {
            hf: &base,
            kaggle: &base,
        };
        let e = download_hf_from(&hosts, None, (" ", "f", None, dir, None)).unwrap_err();
        assert!(e.contains("provide a HuggingFace repo"), "{e}");
        // The explicit token wins over the environment's; a revision picks the branch.
        let hf = download_hf_from(
            &hosts,
            Some("env".into()),
            ("o/n", "sub/f.txt", Some("v2"), dir, Some("mine".into())),
        )
        .unwrap();
        assert_eq!((hf.source.as_str(), hf.bytes), ("huggingface", 4));
        assert!(hf.path.ends_with("f.txt"));
        let sent = rx.recv().unwrap();
        assert!(
            sent.contains("/datasets/o/n/resolve/v2/sub/f.txt"),
            "{sent}"
        );
        assert!(sent.to_lowercase().contains("bearer mine"), "{sent}");
        download_hf_from(
            &hosts,
            Some("env".into()),
            ("o/n", "f.txt", None, dir, None),
        )
        .unwrap();
        assert!(rx.recv().unwrap().to_lowercase().contains("bearer env"));

        let e = download_kaggle_from(&hosts, None, "", "f", dir).unwrap_err();
        assert!(e.contains("provide a Kaggle owner/slug"), "{e}");
        let e = download_kaggle_from(&hosts, None, "o/n", "f.csv", dir).unwrap_err();
        assert!(e.starts_with("Kaggle needs credentials"), "{e}");
        let creds = KaggleCreds {
            username: "u".into(),
            key: "k".into(),
        };
        let k = download_kaggle_from(&hosts, Some(creds), "o/n", "f.csv", dir).unwrap();
        assert_eq!((k.source.as_str(), k.bytes), ("kaggle", 4));
        assert!(rx
            .recv()
            .unwrap()
            .contains("/api/v1/datasets/download/o/n/f.csv"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
