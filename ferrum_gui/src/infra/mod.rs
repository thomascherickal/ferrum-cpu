//! Shared plumbing every command uses: structured errors, job slots and cancel, progress
//! events, path helpers, the HTTP client.

pub(crate) mod error;
pub(crate) mod http;
pub(crate) mod jobs;
pub(crate) mod paths;
pub(crate) mod progress;

#[cfg(test)]
mod tests {
    use super::*;

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
