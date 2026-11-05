//! Corpus handling: dataset downloads, `prepare`, and token-file writing.

pub(crate) mod datasets;
pub(crate) mod prepare;
pub(crate) mod tokenize;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_group_exposes_its_three_concerns() {
        assert!(!datasets::catalog().is_empty());
        assert!(datasets::cpu_catalog().len() <= datasets::catalog().len());
        for name in [
            std::any::type_name::<prepare::PrepareParams>(),
            std::any::type_name::<tokenize::TokenizeParams>(),
        ] {
            assert!(name.starts_with("ferrum_gui_lib::data::"), "{name}");
        }
    }

    /// Dependencies point `commands` -> `data` -> `infra` (CLAUDE.md, Layout): this group
    /// never names the command layer or a sibling group.
    #[test]
    fn the_group_depends_only_on_infra() {
        let hits = crate::test_support::sources_using("data", &["commands", "serving", "sizing"]);
        assert!(hits.is_empty(), "{hits:?}");
    }
}
