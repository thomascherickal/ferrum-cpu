//! Library half of `slm_cli`: the logic of `train_transformer` subcommands that
//! other front ends (the GUI) reuse instead of copying. The binary parses
//! arguments and prints; everything here takes plain options and returns data.
//!
//! This is a library of the CLI package rather than a `ferrum_core` module because
//! the pipeline needs `sha2` (split hashing, manifest) and `serde_json` (manifest),
//! and `ferrum_core` may not gain dependencies.

#![forbid(unsafe_code)]

pub mod file_kind;
pub mod hash;
pub mod outputs;
pub mod prepare;
pub mod runtime_export;
pub mod token_corpus;
pub mod tokenize;

/// The library is the surface `ferrum_gui` builds on: these pin that every promised module is
/// public and that its entry points keep their shape.
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    #[allow(clippy::type_complexity)] // the spelled-out fn types are what this test pins
    fn every_module_is_reachable_with_its_entry_point_signatures() {
        let _: fn(&[u8]) -> String = hash::sha256_hex;
        let _: fn(&str) -> std::io::Result<String> = hash::file_id;
        let _: fn(&str) -> Result<file_kind::FileKind, String> = file_kind::detect;
        let _: fn(&str) -> Result<(), String> = outputs::check_writable;
        let _: fn(&[(&str, &str)]) -> Result<(), String> = outputs::check_distinct_paths;
        let _: fn(
            &prepare::PrepareOptions,
            &mut dyn FnMut(prepare::PrepareProgress),
        ) -> Result<prepare::PrepareReport, String> = prepare::prepare;
        let _: fn(
            &tokenize::TokenizeOptions,
            &mut dyn FnMut(tokenize::TokenizeProgress),
        ) -> Result<tokenize::TokenizeReport, String> = tokenize::tokenize;
        let _: fn(
            &str,
            Option<&str>,
            bool,
        )
            -> Result<(ferrum_core::TokenFile, ferrum_core::ByteBpeTokenizer), String> =
            token_corpus::open_token_file;
        let _: fn(&Path) -> Option<String> = runtime_export::plain_gguf_warning;
        let _: fn(&Path) -> Result<String, String> = runtime_export::modelfile_for;
        assert!(tokenize::PROGRESS_EVERY > 0 && !tokenize::CANCELLED.is_empty());
    }

    #[test]
    fn library_entry_points_answer_trivial_inputs() {
        assert_eq!(
            hash::sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let missing = "/nonexistent/ferrum/lib-surface.gguf";
        assert_eq!(file_kind::detect(missing), Ok(file_kind::FileKind::Model));
        assert!(hash::file_id(missing).is_err());
        assert!(outputs::check_distinct_paths(&[("a", "x.gguf"), ("b", "x.gguf")]).is_err());
        assert!(runtime_export::plain_gguf_warning(Path::new(missing)).is_none());
    }
}
