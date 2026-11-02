//! Model files: Ferrum models are stored as GGUF (weights + metadata +
//! tokenizer in one file). Weights are dequantized to f32 in memory.
//!
//! A BPE transformer SLM is written in llama.cpp's `gpt2` layout, so
//! llama.cpp / Ollama / LM Studio run it as-is. Every other model uses the
//! generic `ferrum` layout (`ferrum.format = 1`):
//!
//! | Key / tensor | Type | Meaning |
//! |---|---|---|
//! | `general.architecture` | string | `"ferrum"` |
//! | `general.name` | string | `ModelMetadata.dataset_name` |
//! | `ferrum.format` | u32 | `1` |
//! | `ferrum.quant` | string | `"f32"`, `"int8"` or `"int4"` (matrix precision) |
//! | `ferrum.metadata` | string | `ModelMetadata::to_json()` |
//! | `ferrum.normalizer` | string | `Normalizer::encode()` (empty for SLMs) |
//! | `ferrum.layer_count` | u32 | number of layers |
//! | `ferrum.layer.{i}.type` | string | `linear`, `activation`, `flatten`, `embedding`, `layernorm`, `transformer_block` |
//! | `ferrum.layer.{i}.{dim}` | u32 | `linear`: `in`, `out`; `activation`: `kind`; `embedding`: `vocab`, `seq`, `dim`; `layernorm`: `dim`; `transformer_block`: `context`, `heads`, `dim`, `hidden` |
//! | `l{i}.{param}` | F32 / Q8_0 / Q4_0 | GGML dims `[cols, rows]` for a row-major `[rows, cols]` matrix, `[n]` for a vector; 2-D weights quantized at the requested precision when GGML ne0 % 32 == 0, otherwise F32 |
//!
//! Parameter names: `linear`: `weight` `[in,out]`, `bias`; `embedding`:
//! `token`, `pos`; `layernorm`: `gamma`, `beta`; `transformer_block`: `ln1.*`,
//! `q.*`, `k.*`, `v.*`, `out.*`, `ln2.*`, `ffn1.*`, `ffn2.*` (each
//! `weight`/`bias`, or `gamma`/`beta` for the norms).
//!
//! Quantization: matrices whose GGML row length is a multiple of 32 are
//! stored as Q8_0 / Q4_0 when requested; vectors are always F32. The pre-GGUF
//! `.bin` model format is no longer readable.

mod consistency;
mod decode;
mod encode;
mod generic;
mod gpt2;
mod info;
#[cfg(test)]
mod test_support;

pub(crate) use self::consistency::check_slm_consistency;
pub use self::decode::{from_gguf, from_gguf_vec};
pub use self::encode::to_gguf;
pub use self::info::{gguf_format, gguf_quant};

use crate::error::{InferError, Result};
use crate::gguf::{GGML_F32, GGML_Q4_0, GGML_Q8_0};
use crate::meta::{ModelMetadata, Normalizer};
use crate::model::Sequential;
use crate::quant::q8_0_eligible;

/// Multiply two dimension fields read from an untrusted file, rejecting
/// overflow (relevant on 32-bit/wasm targets where usize is 32 bits).
fn mul_dims(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| InferError::Format(format!("dimension product {a}×{b} overflows")))
}

/// On-disk weight precision selected at save time. In GGUF, `Int8` is Q8_0
/// and `Int4` is Q4_0 (matrices whose GGML row length is a multiple of 32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QPrec {
    /// Full f32.
    F32,
    /// int8 (GGUF Q8_0).
    Int8,
    /// int4 (GGUF Q4_0, ~8× smaller than f32).
    Int4,
}

/// Serialize with full f32 weights (Ferrum GGUF).
pub fn to_bytes(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata) -> Result<Vec<u8>> {
    to_gguf(model, norm, meta, QPrec::F32)
}

/// Serialize with int8 (Q8_0) matrices, ≈4× smaller.
pub fn to_bytes_quantized(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
) -> Result<Vec<u8>> {
    to_gguf(model, norm, meta, QPrec::Int8)
}

/// Serialize with int4 (Q4_0) matrices, ≈8× smaller.
pub fn to_bytes_quantized_int4(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
) -> Result<Vec<u8>> {
    to_gguf(model, norm, meta, QPrec::Int4)
}

/// Load a model from GGUF bytes; see [`from_gguf`].
pub fn from_bytes(bytes: &[u8]) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    from_gguf(bytes)
}

/// [`from_bytes`] taking ownership of the bytes (no copy of the file); see [`from_gguf_vec`].
pub fn from_vec(bytes: Vec<u8>) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    from_gguf_vec(bytes)
}

pub fn save(model: &Sequential, norm: &Normalizer, meta: &ModelMetadata, path: &str) -> Result<()> {
    std::fs::write(path, to_bytes(model, norm, meta)?)?;
    Ok(())
}

/// Like [`save`] with int8 matrices.
pub fn save_quantized(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    path: &str,
) -> Result<()> {
    std::fs::write(path, to_bytes_quantized(model, norm, meta)?)?;
    Ok(())
}

/// Like [`save`] with int4 matrices.
pub fn save_quantized_int4(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    path: &str,
) -> Result<()> {
    std::fs::write(path, to_bytes_quantized_int4(model, norm, meta)?)?;
    Ok(())
}

pub fn load(path: &str) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    from_gguf(&std::fs::read(path)?)
}

/// GPT-2's LayerNorm epsilon, which Ferrum's LayerNorm also uses.
const LN_EPS: f32 = 1e-5;

impl QPrec {
    fn name(self) -> &'static str {
        match self {
            QPrec::F32 => "f32",
            QPrec::Int8 => "int8",
            QPrec::Int4 => "int4",
        }
    }
    /// GGML type for a matrix whose GGML rows hold `ne0` values.
    fn ggml_type(self, ne0: usize) -> u32 {
        match self {
            QPrec::Int8 if q8_0_eligible(ne0) => GGML_Q8_0,
            QPrec::Int4 if q8_0_eligible(ne0) => GGML_Q4_0,
            _ => GGML_F32,
        }
    }
}

fn bad(msg: String) -> InferError {
    InferError::Format(msg)
}

#[cfg(test)]
mod tests {
    use super::{
        from_bytes, from_vec, load, save, save_quantized, save_quantized_int4, to_bytes,
        to_bytes_quantized, to_bytes_quantized_int4, QPrec,
    };
    use crate::error::{InferError, Result};
    use crate::gguf::{MetaValue, GGML_F32, GGML_Q4_0, GGML_Q8_0};
    use crate::layer::{Flatten, LayerNorm, Linear, TransformerBlock};
    use crate::loader::decode::{from_gguf, from_gguf_vec};
    use crate::loader::encode::to_gguf;
    use crate::loader::info::gguf_quant;
    use crate::loader::test_support::{
        arch, bpe_parts, fwd, gguf_corpus, load_err_of, make_bundle, make_embedding_bundle,
        make_embedding_bundle_big, max_abs_diff, parse, trained_slm, DummyLayer, FINF_HINT,
    };
    use crate::meta::{ModelMetadata, Normalizer, TaskType};
    use crate::model::Sequential;
    use crate::slm::GenerativeSLM;
    use crate::tensor::Tensor;

    #[test]
    fn roundtrip_preserves_outputs() {
        let (model, norm, meta) = make_bundle();
        let raw = Tensor::row(vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();
        let before = model.forward(&norm.transform(&raw).unwrap()).unwrap();
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (m2, n2, _) = from_bytes(&bytes).unwrap();
        let after = m2.forward(&n2.transform(&raw).unwrap()).unwrap();
        for (a, b) in before.data.iter().zip(&after.data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn save_and_load_use_gguf_files() {
        let (model, norm, meta) = make_bundle();
        let dir = std::env::temp_dir();
        let path = |n: &str| {
            dir.join(format!("ferrum_loader_{n}_{}.gguf", std::process::id()))
                .to_str()
                .unwrap()
                .to_string()
        };
        type Save = fn(&Sequential, &Normalizer, &ModelMetadata, &str) -> Result<()>;
        let saves: [(&str, Save, &str); 3] = [
            ("f32", save, "f32"),
            ("i8", save_quantized, "int8"),
            ("i4", save_quantized_int4, "int4"),
        ];
        for (name, f, quant) in saves {
            let p = path(name);
            f(&model, &norm, &meta, &p).unwrap();
            let bytes = std::fs::read(&p).unwrap();
            assert_eq!(&bytes[..4], b"GGUF");
            assert_eq!(gguf_quant(&bytes).unwrap(), quant);
            let (m2, _, meta2) = load(&p).unwrap();
            assert_eq!(m2.len(), model.len());
            assert_eq!(meta2.class_names, meta.class_names);
            std::fs::remove_file(&p).unwrap();
        }
        assert!(matches!(load(&path("missing")), Err(InferError::Io(_))));
        assert!(save(&model, &norm, &meta, "/nonexistent-dir/x.gguf").is_err());
    }

    #[test]
    fn metadata_survives_roundtrip() {
        let (model, norm, meta) = make_bundle();
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (_, _, m2) = from_bytes(&bytes).unwrap();
        assert_eq!(m2.task, TaskType::Classification);
        assert_eq!(m2.class_names, vec!["X", "Y", "Z"]);
        assert_eq!(m2.feature_names, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn embedding_bundle_roundtrips() {
        let (model, norm, meta) = make_embedding_bundle();
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (m2, _, m2meta) = from_bytes(&bytes).unwrap();
        assert_eq!(m2meta.task, TaskType::TransformerSLM);
        assert_eq!(m2meta.class_names, vec!["a", "b", "c", "d", "e"]);
        // Forward should still work
        let x = Tensor::matrix(1, 4, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        assert!(m2.forward(&x).is_ok());
    }

    #[test]
    fn quantized_roundtrip_is_smaller_and_close() {
        let (model, norm, meta) = make_bundle();
        let raw = Tensor::row(vec![5.1f32, 3.5, 1.4, 0.2]).unwrap();
        let before = model.forward(&norm.transform(&raw).unwrap()).unwrap();

        let full = to_bytes(&model, &norm, &meta).unwrap();
        let quant = to_bytes_quantized(&model, &norm, &meta).unwrap();
        // make_bundle's largest tensor is 32 values (< QUANT_MIN_LEN), so use
        // a bigger Linear to actually exercise the int8 path.
        let big = Linear::new(
            64,
            64,
            (0..64 * 64)
                .map(|i| (i as f32 * 0.37).sin() * 0.1)
                .collect(),
            vec![0.0; 64],
        )
        .unwrap();
        let big_model = Sequential::new().with(Box::new(big));
        let big_norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let big_full = to_bytes(&big_model, &big_norm, &meta).unwrap();
        let big_quant = to_bytes_quantized(&big_model, &big_norm, &meta).unwrap();
        let (quant_len, full_len) = (big_quant.len() as f32, big_full.len() as f32);
        assert!(
            quant_len < full_len * 0.35,
            "quantized {quant_len} not ≈4× smaller than {full_len}"
        );

        // Small model: everything stays f32 and the outputs match closely.
        assert!(quant.len() <= full.len() + model.len() * 8);
        let (m2, n2, _) = from_bytes(&quant).unwrap();
        let after = m2.forward(&n2.transform(&raw).unwrap()).unwrap();
        for (a, b) in before.data.iter().zip(&after.data) {
            assert!((a - b).abs() < 1e-6);
        }

        // Big model: int8 error is bounded by scale/2 per weight.
        let x = Tensor::row(vec![0.1f32; 64]).unwrap();
        let y_full = big_model.forward(&x).unwrap();
        let (m3, _, _) = from_bytes(&big_quant).unwrap();
        let y_quant = m3.forward(&x).unwrap();
        for (a, b) in y_full.data.iter().zip(&y_quant.data) {
            assert!((a - b).abs() < 0.05, "quantized output drifted: {a} vs {b}");
        }
    }

    #[test]
    fn int4_roundtrip_is_smaller_and_close() {
        // A Linear big enough to quantize: int4 is far smaller and close.
        let (in_f, out_f) = (96usize, 96usize);
        let w: Vec<f32> = (0..in_f * out_f)
            .map(|i| (i as f32 * 0.011).sin() * 0.4)
            .collect();
        let big = Linear::new(in_f, out_f, w, vec![0.0; out_f]).unwrap();
        let model = Sequential::new().with(Box::new(big));
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let (_, _, meta) = make_bundle();

        let x = Tensor::matrix(
            1,
            in_f,
            (0..in_f).map(|i| (i as f32 * 0.03).cos()).collect(),
        )
        .unwrap();
        let y_f32 = model.forward(&x).unwrap();

        let bytes = to_bytes_quantized_int4(&model, &norm, &meta).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        // ~8× smaller than the f32 file for the weight payload.
        let f32_bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (int4_len, f32_len) = (bytes.len() as f32, f32_bytes.len() as f32);
        assert!(
            int4_len < f32_len * 0.4,
            "int4 file {int4_len} not far smaller than f32 {f32_len}"
        );

        let (loaded, _, _) = from_bytes(&bytes).unwrap();
        let y_q = loaded.forward(&x).unwrap();
        let mae: f32 = y_f32
            .data
            .iter()
            .zip(&y_q.data)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / out_f as f32;
        // 0.4 tolerance: the sinusoid spans ~5 periods per 32-element input
        // block, so every Q4_0 block scale is near the global max and the
        // asymmetric -8..7 grid clips the opposite-sign extreme to 7/8*amax.
        // Measured MAE ~0.27 (old per-row int4: ~0.05); 0.4 still catches sign
        // errors (~1.2) and scrambled weights.
        assert!(mae < 0.4, "int4 forward drifted too far: mae={mae}");
    }

    #[test]
    fn int4_transformer_block_roundtrips_and_runs() {
        let (model, norm, meta) = make_embedding_bundle_big();
        let bytes = to_bytes_quantized_int4(&model, &norm, &meta).unwrap();
        let (loaded, _, _) = from_bytes(&bytes).unwrap();
        // Forward still runs and is finite.
        let x = Tensor::matrix(1, meta.input_dim, vec![1.0; meta.input_dim]).unwrap();
        let y = loaded.forward(&x).unwrap();
        assert!(y.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantized_transformer_roundtrips() {
        // Every SLM layer type through the int8 writer/reader.
        let (model, norm, meta) = make_embedding_bundle();
        let bytes = to_bytes_quantized(&model, &norm, &meta).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        let (m2, _, m2meta) = from_bytes(&bytes).unwrap();
        assert_eq!(m2meta.task, TaskType::TransformerSLM);
        let x = Tensor::matrix(1, 4, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        assert!(m2.forward(&x).is_ok());
    }

    #[test]
    fn flatten_roundtrips() {
        let lin = Linear::new(
            8,
            3,
            (0..24).map(|i| i as f32 * 0.01).collect(),
            vec![0.0; 3],
        )
        .unwrap();
        let model = Sequential::new()
            .with(Box::new(Flatten::new()))
            .with(Box::new(lin));
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let (_, _, meta) = make_bundle();

        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");

        let (m2, _, _) = from_bytes(&bytes).unwrap();
        assert_eq!(m2.len(), 2);
        assert_eq!(m2.layers()[0].name(), "Flatten");
        let x = Tensor::matrix(2, 4, vec![0.1; 8]).unwrap(); // [2,4] → [1,8]
        let before = model.forward(&x).unwrap();
        let after = m2.forward(&x).unwrap();
        assert_eq!(before.shape, vec![1, 3]);
        for (a, b) in before.data.iter().zip(&after.data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn to_bytes_unknown_layer_error() {
        let mut model = Sequential::new();
        model.push(Box::new(DummyLayer));
        // The stand-in layer passes its input through (it only lacks a serialised form).
        let x = Tensor::matrix(1, 2, vec![1.0, 2.0]).unwrap();
        assert_eq!(model.forward(&x).unwrap().data, x.data);
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let meta = ModelMetadata {
            dataset_name: "test".into(),
            task: TaskType::Classification,
            feature_names: vec![],
            feature_ranges: vec![],
            class_names: vec![],
            target_name: "".into(),
            target_range: [0.0, 0.0],
            input_dim: 0,
            output_dim: 0,
            tokenizer_state: String::new(),
        };
        assert!(matches!(
            to_bytes(&model, &norm, &meta),
            Err(InferError::Format(_))
        ));
    }

    #[test]
    fn layernorm_and_transformerblock_roundtrip() {
        let context_len = 4;
        let num_heads = 2;
        let embedding_dim = 8;
        let hidden_dim = 16;
        let c = embedding_dim;
        let h = hidden_dim;

        let ln = LayerNorm::new(embedding_dim, vec![1.0; c], vec![0.0; c]).unwrap();
        let tb = TransformerBlock::new(
            context_len,
            num_heads,
            embedding_dim,
            vec![1.0; c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![0.1; c * c],
            vec![0.0; c],
            vec![1.0; c],
            vec![0.0; c],
            vec![0.1; c * h],
            vec![0.0; h],
            vec![0.1; h * c],
            vec![0.0; c],
        )
        .unwrap();

        let mut model = Sequential::new();
        model.push(Box::new(ln));
        model.push(Box::new(tb));

        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let meta = ModelMetadata {
            dataset_name: "tb_test".into(),
            task: TaskType::TransformerSLM,
            feature_names: vec![],
            feature_ranges: vec![],
            class_names: vec![],
            target_name: "".into(),
            target_range: [0.0, 0.0],
            input_dim: embedding_dim,
            output_dim: embedding_dim,
            tokenizer_state: String::new(),
        };

        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (m2, _, m2meta) = from_bytes(&bytes).unwrap();
        assert_eq!(m2.len(), 2);
        assert_eq!(m2meta.task, TaskType::TransformerSLM);

        let input = Tensor::matrix(4, embedding_dim, vec![0.5; 4 * embedding_dim]).unwrap();
        let out1 = model.forward(&input).unwrap();
        let out2 = m2.forward(&input).unwrap();
        for (a, b) in out1.data.iter().zip(&out2.data) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn f32_gguf_roundtrip_is_exact() {
        let slm = trained_slm(false);
        let bytes = to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::F32).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        assert_eq!(arch(&bytes), "gpt2");
        let (m2, n2, meta2) = from_gguf(&bytes).unwrap();
        assert_eq!(fwd(&slm.model, &slm.meta), fwd(&m2, &meta2));
        assert_eq!(
            meta2.to_json(),
            slm.meta.to_json(),
            "tokenizer state survives"
        );
        assert_eq!(n2.encode(), "");
        assert_eq!(gguf_quant(&bytes).unwrap(), "f32");
    }

    #[test]
    fn qat_model_reloads_bit_identically_from_int8_gguf() {
        let slm = trained_slm(true);
        let bytes = to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::Int8).unwrap();
        let g = parse(&bytes);
        assert_eq!(g.tensor("token_embd.weight").unwrap().ggml_type, GGML_Q8_0);
        assert_eq!(
            g.tensor("position_embd.weight").unwrap().ggml_type,
            GGML_F32
        );
        let (m2, n2, meta2) = from_gguf(&bytes).unwrap();
        let back = GenerativeSLM::new(m2, n2, meta2);
        let text = &gguf_corpus()[..1200]; // > context + 1 tokens, kept short for speed
        assert_eq!(slm.evaluate(text).unwrap(), back.evaluate(text).unwrap());
        assert_eq!(gguf_quant(&bytes).unwrap(), "int8");
    }

    #[test]
    fn int4_and_ineligible_dims() {
        // embed 48 / hidden 80: no GGML row length is a multiple of 32.
        let (m, n, meta) = bpe_parts(48, 80, false);
        let bytes = to_gguf(&m, &n, &meta, QPrec::Int8).unwrap();
        let g = parse(&bytes);
        assert!(g.tensors.iter().all(|t| t.ggml_type == GGML_F32));
        assert_eq!(g.meta("general.file_type"), Some(&MetaValue::U32(0)));
        assert_eq!(gguf_quant(&bytes).unwrap(), "int8");
        let (m2, _, meta2) = from_gguf(&bytes).unwrap();
        assert_eq!(fwd(&m, &meta), fwd(&m2, &meta2));

        let (m, n, meta) = bpe_parts(64, 128, true);
        let bytes = to_gguf(&m, &n, &meta, QPrec::Int4).unwrap();
        let g = parse(&bytes);
        assert_eq!(
            g.tensor("blk.0.ffn_down.weight").unwrap().ggml_type,
            GGML_Q4_0
        );
        assert_eq!(
            g.tensor("position_embd.weight").unwrap().ggml_type,
            GGML_F32
        );
        assert_eq!(g.meta("general.file_type"), Some(&MetaValue::U32(2)));
        assert_eq!(gguf_quant(&bytes).unwrap(), "int4");
        let (m2, _, meta2) = from_gguf(&bytes).unwrap();
        let err = max_abs_diff(&fwd(&m, &meta), &fwd(&m2, &meta2));
        assert!(err < 0.1, "int4 forward drifted {err}");
    }

    /// The owning loaders parse the buffer in place and agree with the borrowing ones.
    #[test]
    fn owned_loaders_match_the_borrowing_ones() {
        let slm = trained_slm(false);
        let bytes = to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::F32).unwrap();
        let (m1, n1, meta1) = from_bytes(&bytes).unwrap();
        let (m2, n2, meta2) = from_vec(bytes.clone()).unwrap();
        assert_eq!((n1.means, n1.stds), (n2.means, n2.stds));
        assert_eq!(meta1.class_names, meta2.class_names);
        assert_eq!(
            to_gguf(
                &m1,
                &Normalizer {
                    means: vec![],
                    stds: vec![]
                },
                &meta1,
                QPrec::F32
            )
            .unwrap(),
            to_gguf(
                &m2,
                &Normalizer {
                    means: vec![],
                    stds: vec![]
                },
                &meta2,
                QPrec::F32
            )
            .unwrap()
        );
        assert!(load_err_of(from_gguf_vec(b"FINF\x05\x00\x00\x00".to_vec())).contains(FINF_HINT));
        assert!(from_vec(Vec::new()).is_err());
    }
}
