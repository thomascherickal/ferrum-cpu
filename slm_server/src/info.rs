//! What `/v1/info` reports about the loaded model.

use ferrum_core::{GenerativeSLM, Gguf, GgufTokenizer, LlamaModel, QKind, TokModel};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize)]
pub struct Info {
    pub params: usize,
    pub context_len: usize,
    pub vocab: usize,
    pub quant: String,
    pub tokenizer: String,
    pub model_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Describe `model`, loaded from `bytes` (hashed for `model_sha256`).
pub fn info_for(model: &GenerativeSLM, bytes: &[u8]) -> Info {
    Info {
        params: model.model.num_params(),
        context_len: model.meta.input_dim,
        vocab: model.meta.output_dim,
        quant: ferrum_core::loader::gguf_quant(bytes).unwrap_or_else(|_| "unknown".into()),
        tokenizer: if model.meta.tokenizer_state.is_empty() {
            "char".into()
        } else {
            "bpe".into()
        },
        model_sha256: sha256_hex(bytes),
    }
}

/// Describe an imported model: its parameter count (every tensor of the file), context and
/// vocabulary, the in-memory precision it computes at (`int4` / `int8` / `f32`, the
/// `--quant` it was loaded with), and its tokenizer family (`bpe` / `spm`).
pub(crate) fn info_for_imported(
    g: &Gguf,
    model: &LlamaModel,
    tok: &GgufTokenizer,
    prec: Option<QKind>,
    model_sha256: String,
) -> Info {
    Info {
        params: g.tensors.iter().map(|t| t.num_elements()).sum(),
        context_len: model.cfg.context_len,
        vocab: model.cfg.vocab_size,
        quant: match prec {
            Some(QKind::Int4) => "int4",
            Some(QKind::Int8) => "int8",
            None => "f32",
        }
        .into(),
        tokenizer: match tok.model() {
            TokModel::Bpe => "bpe",
            TokModel::Spm => "spm",
        }
        .into(),
        model_sha256,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{llama, model};

    #[test]
    fn info_for_an_imported_model() {
        let g = Gguf::parse(llama::tiny_llama(&[])).unwrap();
        let tok = GgufTokenizer::from_gguf(&g).unwrap();
        for (prec, name) in [
            (Some(QKind::Int4), "int4"),
            (Some(QKind::Int8), "int8"),
            (None, "f32"),
        ] {
            let m = g.load_llama_prec(prec).unwrap();
            let i = info_for_imported(&g, &m, &tok, prec, "ab".into());
            assert_eq!(
                (
                    i.context_len,
                    i.vocab,
                    i.quant.as_str(),
                    i.tokenizer.as_str()
                ),
                (16, 32, name, "bpe")
            );
            // 32x8 embedding and head, 2 x (2 norms of 8, 4 attention 8x8, 3 FFN 8x16), a norm.
            assert_eq!(i.params, 2 * 256 + 2 * (16 + 256 + 384) + 8);
            assert_eq!(i.model_sha256, "ab");
        }
        let g = Gguf::parse(llama::tiny_llama_spm()).unwrap();
        let tok = GgufTokenizer::from_gguf(&g).unwrap();
        let m = g.load_llama_prec(None).unwrap();
        assert_eq!(
            info_for_imported(&g, &m, &tok, None, String::new()).tokenizer,
            "spm"
        );
    }

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn info_params_is_the_real_parameter_count() {
        let m = model(0);
        let i = info_for(&m, &m.to_bytes_quantized().unwrap());
        assert_eq!(i.params, m.model.num_params());
        assert!(i.params > 0);
    }

    #[test]
    fn info_for_bpe_model() {
        let m = model(270);
        let bytes = m.to_bytes_quantized().unwrap();
        let i = info_for(&m, &bytes);
        assert_eq!(
            (i.context_len, i.vocab, i.tokenizer.as_str()),
            (8, m.meta.output_dim, "bpe")
        );
        assert!(i.vocab > 256);
        assert_eq!(i.model_sha256.len(), 64);
        assert_eq!(i.model_sha256, sha256_hex(&bytes));
    }

    #[test]
    fn info_for_char_model() {
        let m = model(0);
        let i = info_for(&m, &m.to_bytes_quantized().unwrap());
        assert_eq!(i.tokenizer, "char");
    }

    #[test]
    fn info_quant_falls_back_to_unknown_on_non_gguf_bytes() {
        let m = model(0);
        let i = info_for(&m, b"not a model file");
        assert_eq!(i.quant, "unknown");
        assert_eq!(i.model_sha256, sha256_hex(b"not a model file"));
    }

    #[test]
    fn info_quant_reported_for_gguf_bytes() {
        let m = model(0);
        let bytes = m.to_bytes_quantized().unwrap();
        let want = ferrum_core::loader::gguf_quant(&bytes).unwrap();
        assert_eq!(info_for(&m, &bytes).quant, want);
    }
}
