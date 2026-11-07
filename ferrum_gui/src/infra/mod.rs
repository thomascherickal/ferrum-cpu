//! Shared plumbing every command uses: structured errors, job slots and cancel, progress
//! events, path helpers, the HTTP client.

pub(crate) mod error;
pub(crate) mod http;
pub(crate) mod jobs;
pub(crate) mod paths;
pub(crate) mod progress;

/// Lock a mutex, recovering the inner value if an earlier holder panicked.
pub(crate) fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

/// Format an `Err(e)` as `"{prefix}: {e}"` without creating a per-call closure.
pub(crate) fn fmt_err<T, E: std::fmt::Display>(res: Result<T, E>, prefix: &str) -> Result<T, String> {
    match res {
        Ok(v) => Ok(v),
        Err(e) => Err(format!("{prefix}: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_and_fmt_err_handle_ok_and_err_cases() {
        let m = std::sync::Arc::new(std::sync::Mutex::new(7));
        assert_eq!(*lock(&m), 7);
        let m2 = m.clone();
        let _ = std::thread::spawn(move || {
            let _g = m2.lock().unwrap();
            panic!("poison");
        })
        .join();
        assert_eq!(*lock(&m), 7);
        assert_eq!(fmt_err::<i32, &str>(Ok(3), "p"), Ok(3));
        assert_eq!(fmt_err::<i32, &str>(Err("bad"), "p"), Err("p: bad".into()));
    }

    #[test]
    fn the_group_exposes_the_shared_plumbing() {
        assert_eq!(error::classify("training cancelled"), "cancelled");
        assert!(error::CODES.iter().any(|(code, _)| *code == "cancelled"));
        assert!(paths::checked_path("p", "", false).is_err());
        const { assert!(progress::INTERVAL_MS > 0 && paths::MAX_PATH_BYTES > 0) };
        for name in [
            std::any::type_name::<jobs::Jobs>(),
            std::any::type_name::<error::CmdError>(),
        ] {
            assert!(name.starts_with("ferrum_gui_lib::infra::"), "{name}");
        }
    }

    /// `infra` is the bottom layer: it names no group above it.
    #[test]
    fn the_group_depends_on_no_other_group() {
        let hits = crate::test_support::sources_using(
            "infra",
            &["commands", "data", "serving", "sizing", "evaluate"],
        );
        assert!(hits.is_empty(), "{hits:?}");
    }

    /// The layering tests' scanner (`test_support::sources_using`) finds real dependencies only.
    #[test]
    fn the_layering_scanner_finds_real_dependencies_only() {
        let hits = crate::test_support::sources_using("commands", &["infra"]);
        assert!(
            hits.iter().any(|h| h.ends_with(": crate::infra")),
            "{hits:?}"
        );
        assert!(crate::test_support::sources_using("infra", &["nowhere"]).is_empty());
    }
}
