//! Dataset discovery + download from HuggingFace and Kaggle.
//!
//! Only datasets small enough to be practical for **CPU** training/inference are
//! surfaced ([`CPU_MAX_MB`]). Credential parsing and Base64 live in [`auth`];
//! URL construction, path sanitisation, and streamed `.part` writes live in
//! [`download`].

mod auth;
mod download;

pub use auth::{kaggle_creds, KaggleCreds};
pub use download::{dest_path, hf_resolve_url, kaggle_file_url, DownloadResult};

use crate::infra::error::CmdError;
use crate::infra::paths::checked_dir;
use crate::infra::progress::Progress;
use download::{download_to_file, Hosts, HOSTS};
use serde::Serialize;
use tauri::{AppHandle, Runtime};

/// Upper bound (MB) on datasets we surface as "CPU-friendly". Bigger sets still
/// download via the generic commands, but are kept out of the curated list so a
/// user does not pick something that won't train on a laptop.
pub const CPU_MAX_MB: u32 = 200;

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

/// The curated catalog of small, CPU-friendly text datasets.
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
    progress: &mut Progress,
) -> Result<DownloadResult, String> {
    checked_dir("download folder", dest_dir)?;
    let entry = catalog()
        .into_iter()
        .find(|d| d.id == id)
        .ok_or_else(|| format!("unknown dataset id '{id}'"))?;
    let dest = dest_path(dest_dir, &entry.file, &entry.id);
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
    let bytes = download_to_file(&url, basic.as_ref(), bearer.as_deref(), &dest, progress)?;
    Ok(DownloadResult {
        path: dest.display().to_string(),
        bytes,
        source: source.into(),
    })
}

/// Download a curated catalog entry by `id` into `dest_dir`.
#[tauri::command]
pub async fn download_dataset<R: Runtime>(
    app: AppHandle<R>,
    id: String,
    dest_dir: String,
) -> Result<DownloadResult, CmdError> {
    crate::infra::error::blocking(move || {
        let hf_token = std::env::var("HF_TOKEN").ok();
        let mut progress = Progress::events(app, id.clone());
        download_dataset_from(
            &HOSTS,
            kaggle_creds(),
            hf_token,
            &id,
            &dest_dir,
            &mut progress,
        )
    })
    .await
}

fn download_hf_from(
    hosts: &Hosts,
    env_token: Option<String>,
    (repo, file, revision, dest_dir, hf_token): (&str, &str, Option<&str>, &str, Option<String>),
    progress: &mut Progress,
) -> Result<DownloadResult, String> {
    checked_dir("download folder", dest_dir)?;
    if repo.trim().is_empty() || file.trim().is_empty() {
        return Err("provide a HuggingFace repo (owner/name) and file".into());
    }
    let url = hf_resolve_url(hosts.hf, repo, file, revision.unwrap_or("main"));
    let dest = dest_path(dest_dir, file, "dataset.bin");
    let token = hf_token.or(env_token);
    let bytes = download_to_file(&url, None, token.as_deref(), &dest, progress)?;
    Ok(DownloadResult {
        path: dest.display().to_string(),
        bytes,
        source: "huggingface".into(),
    })
}

/// Download any HuggingFace dataset file (`repo` = `owner/name`), not just the
/// curated ones. `revision` defaults to `main`; `hf_token` is for gated repos.
#[tauri::command]
pub async fn download_hf_file<R: Runtime>(
    app: AppHandle<R>,
    repo: String,
    file: String,
    revision: Option<String>,
    dest_dir: String,
    hf_token: Option<String>,
) -> Result<DownloadResult, CmdError> {
    crate::infra::error::blocking(move || {
        let env_token = std::env::var("HF_TOKEN").ok();
        let args = (&*repo, &*file, revision.as_deref(), &*dest_dir, hf_token);
        let mut progress = Progress::events(app, format!("{repo}/{file}"));
        download_hf_from(&HOSTS, env_token, args, &mut progress)
    })
    .await
}

fn download_kaggle_from(
    hosts: &Hosts,
    kaggle: Option<KaggleCreds>,
    owner_slug: &str,
    file: &str,
    dest_dir: &str,
    progress: &mut Progress,
) -> Result<DownloadResult, String> {
    checked_dir("download folder", dest_dir)?;
    if owner_slug.trim().is_empty() || file.trim().is_empty() {
        return Err("provide a Kaggle owner/slug and file".into());
    }
    let creds = kaggle.ok_or(NEEDS_KAGGLE_CREDS)?;
    let url = kaggle_file_url(hosts.kaggle, owner_slug, file);
    let dest = dest_path(dest_dir, file, "dataset.bin");
    let bytes = download_to_file(&url, Some(&creds), None, &dest, progress)?;
    Ok(DownloadResult {
        path: dest.display().to_string(),
        bytes,
        source: "kaggle".into(),
    })
}

/// Download any Kaggle dataset file (`owner_slug` = `owner/dataset`).
#[tauri::command]
pub async fn download_kaggle_file<R: Runtime>(
    app: AppHandle<R>,
    owner_slug: String,
    file: String,
    dest_dir: String,
) -> Result<DownloadResult, CmdError> {
    crate::infra::error::blocking(move || {
        let mut progress = Progress::events(app, format!("{owner_slug}/{file}"));
        download_kaggle_from(
            &HOSTS,
            kaggle_creds(),
            &owner_slug,
            &file,
            &dest_dir,
            &mut progress,
        )
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ferrum_ds_mod_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn download_commands_refuse_bad_destination_folders() {
        use tauri::async_runtime::block_on;
        let app = tauri::test::mock_app();
        let h = app.handle();
        for bad in ["/dev/shm/x", "/proc/x", "", "a\0b"] {
            let d = bad.to_string();
            let e = block_on(download_dataset(h.clone(), "any".into(), d.clone())).unwrap_err();
            assert!(e.contains("download folder"), "{bad:?}: {e}");
            let e = block_on(download_hf_file(
                h.clone(),
                "o/n".into(),
                "f".into(),
                None,
                d.clone(),
                None,
            ))
            .unwrap_err();
            assert!(e.contains("download folder"), "{bad:?}: {e}");
            let e =
                block_on(download_kaggle_file(h.clone(), "o/n".into(), "f".into(), d)).unwrap_err();
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
        let mut ids: Vec<&str> = c.iter().map(|d| d.id.as_str()).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate dataset ids");
        assert!(cpu_catalog().iter().all(|d| d.approx_mb <= CPU_MAX_MB));
        assert_eq!(cpu_catalog(), list_datasets());
    }

    #[test]
    fn the_curated_and_generic_downloads_pick_the_source_and_check_credentials() {
        use crate::test_support::http_server;
        let d = scratch("curated_generic");
        let dir = d.to_str().unwrap();
        let (base, rx) = http_server("200 OK", b"story".to_vec(), 6);
        let hosts = Hosts {
            hf: &base,
            kaggle: &base,
        };
        let e = download_dataset_from(&hosts, None, None, "nope", dir, &mut Progress::silent())
            .unwrap_err();
        assert_eq!(e, "unknown dataset id 'nope'");
        let hf = download_dataset_from(
            &hosts,
            None,
            Some("tok".into()),
            "tinystories-valid",
            dir,
            &mut Progress::silent(),
        )
        .unwrap();
        assert_eq!((hf.source.as_str(), hf.bytes), ("huggingface", 5));
        assert!(hf.path.ends_with("TinyStories-valid.txt"));
        let sent = rx.recv().unwrap();
        assert!(
            sent.contains("/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt")
        );
        assert!(sent.to_lowercase().contains("authorization: bearer tok"));

        let e = download_dataset_from(
            &hosts,
            None,
            None,
            "shakespeare-plays",
            dir,
            &mut Progress::silent(),
        )
        .unwrap_err();
        assert!(e.starts_with("Kaggle needs credentials"), "{e}");
        let creds = KaggleCreds {
            username: "u".into(),
            key: "k".into(),
        };
        let k = download_dataset_from(
            &hosts,
            Some(creds.clone()),
            None,
            "shakespeare-plays",
            dir,
            &mut Progress::silent(),
        )
        .unwrap();
        assert_eq!(k.source, "kaggle");
        let sent = rx.recv().unwrap();
        assert!(sent.contains(
            "/api/v1/datasets/download/kingburrito666/shakespeare-plays/Shakespeare_data.csv"
        ));

        let e = download_hf_from(
            &hosts,
            None,
            (" ", "f", None, dir, None),
            &mut Progress::silent(),
        )
        .unwrap_err();
        assert!(e.contains("provide a HuggingFace repo"), "{e}");
        let hf2 = download_hf_from(
            &hosts,
            Some("env".into()),
            ("o/n", "sub/f.txt", Some("v2"), dir, Some("mine".into())),
            &mut Progress::silent(),
        )
        .unwrap();
        assert_eq!((hf2.source.as_str(), hf2.bytes), ("huggingface", 5));
        assert!(hf2.path.ends_with("f.txt"));
        let sent = rx.recv().unwrap();
        assert!(sent.contains("/datasets/o/n/resolve/v2/sub/f.txt"));
        assert!(sent.to_lowercase().contains("bearer mine"));

        download_hf_from(
            &hosts,
            Some("env".into()),
            ("o/n", "f.txt", None, dir, None),
            &mut Progress::silent(),
        )
        .unwrap();
        assert!(rx.recv().unwrap().to_lowercase().contains("bearer env"));

        let e =
            download_kaggle_from(&hosts, None, "", "f", dir, &mut Progress::silent()).unwrap_err();
        assert!(e.contains("provide a Kaggle owner/slug"), "{e}");
        let e = download_kaggle_from(&hosts, None, "o/n", "f.csv", dir, &mut Progress::silent())
            .unwrap_err();
        assert!(e.starts_with("Kaggle needs credentials"), "{e}");
        let k2 = download_kaggle_from(
            &hosts,
            Some(creds),
            "o/n",
            "f.csv",
            dir,
            &mut Progress::silent(),
        )
        .unwrap();
        assert_eq!((k2.source.as_str(), k2.bytes), ("kaggle", 5));
        assert!(rx
            .recv()
            .unwrap()
            .contains("/api/v1/datasets/download/o/n/f.csv"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
