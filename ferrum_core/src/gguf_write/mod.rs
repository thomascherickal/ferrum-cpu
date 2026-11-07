//! Pure-`std` GGUF *writer* — serialize a Ferrum `LlamaModel` back to a GGUF v3
//! file that runs in the wider ecosystem (llama.cpp / ollama / LM Studio).
//!
//! The reader in [`crate::gguf`] is the specification: every block encoder here
//! is the exact inverse of the matching `dequant_*` decoder, and is verified by
//! round-tripping through [`crate::gguf::Gguf`].
#![warn(missing_docs)]

mod blocks;
mod builder;
mod kquant;
mod llama;
mod quant;
#[cfg(test)]
mod test_support;

pub(crate) use self::blocks::encode_tensor;
pub use self::blocks::f32_to_f16;
pub use self::builder::GgufBuilder;
pub use self::llama::{llama_gguf_bytes, write_llama_gguf};
pub use self::quant::{size_label, GgufQuant};
