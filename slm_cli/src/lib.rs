//! Library half of `slm_cli`: the logic of `train_transformer` subcommands that
//! other front ends (the GUI) reuse instead of copying. The binary parses
//! arguments and prints; everything here takes plain options and returns data.
//!
//! This is a library of the CLI package rather than a `ferrum_core` module because
//! the pipeline needs `sha2` (split hashing, manifest) and `serde_json` (manifest),
//! and `ferrum_core` may not gain dependencies.

#![forbid(unsafe_code)]

pub mod prepare;
pub mod runtime_export;
