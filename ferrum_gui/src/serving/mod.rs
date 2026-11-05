//! Running a model for other programs: the in-process HTTP server and the Ollama / LM Studio export.

pub(crate) mod runtime_export;
pub(crate) mod serve;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_group_exposes_the_server_and_the_runtime_export() {
        for name in [
            std::any::type_name::<serve::ServeParams>(),
            std::any::type_name::<serve::Serve>(),
            std::any::type_name::<runtime_export::RuntimeParams>(),
        ] {
            assert!(name.starts_with("ferrum_gui_lib::serving::"), "{name}");
        }
        assert!(
            !serve::ServeStatus::default().running,
            "a fresh status reports no server"
        );
    }

    /// Dependencies point `commands` -> `serving` -> `infra` (CLAUDE.md, Layout).
    #[test]
    fn the_group_depends_only_on_infra() {
        let hits = crate::test_support::sources_using("serving", &["commands", "data", "sizing"]);
        assert!(hits.is_empty(), "{hits:?}");
    }
}
