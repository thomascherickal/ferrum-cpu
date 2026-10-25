//! Writing: GgufOut accumulator and the model-to-GGUF entry point.
use crate::error::Result;
use crate::gguf::{transpose_2d, MetaValue, GGML_F32};
use crate::gguf_write::{encode_tensor, size_label, GgufBuilder};
use crate::loader::gpt2::gpt2_parts;
use crate::loader::QPrec;
use crate::meta::{ModelMetadata, Normalizer};
use crate::model::Sequential;

/// Accumulates one model's GGUF keys and tensors.
pub(super) struct GgufOut {
    pub(super) b: GgufBuilder,
    pub(super) prec: QPrec,
    /// Whether any tensor was actually stored quantized.
    pub(super) quantized: bool,
}

impl GgufOut {
    pub(super) fn str(&mut self, key: &str, v: &str) {
        self.b.meta(key, MetaValue::String(v.into()));
    }
    pub(super) fn u32(&mut self, key: &str, v: usize) {
        self.b.meta(key, MetaValue::U32(v as u32));
    }
    /// An F32 tensor with GGML dims `dims`.
    pub(super) fn f32(&mut self, name: &str, dims: &[usize], data: &[f32]) {
        let dims: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        let bytes = encode_tensor(data, GGML_F32).expect("F32 encodes any length");
        self.b.tensor(name, &dims, GGML_F32, bytes);
    }
    /// A 1-D tensor (bias, norm): always F32.
    pub(super) fn vec(&mut self, name: &str, data: &[f32]) {
        self.f32(name, &[data.len()], data);
    }
    /// `rows` GGML rows of `ne0` values (already in GGML order), quantized by
    /// the precision when `ne0 % 32 == 0`.
    pub(super) fn rows(&mut self, name: &str, data: &[f32], rows: usize, ne0: usize) {
        let ty = self.prec.ggml_type(ne0);
        self.quantized |= ty != GGML_F32;
        let bytes = encode_tensor(data, ty).expect("ne0 % 32 == 0 for quantized types");
        self.b.tensor(name, &[ne0 as u64, rows as u64], ty, bytes);
    }
    /// A Ferrum row-major `[rows_in, cols_out]` weight. GGML's `ne0` is the
    /// input dim, so GGML row `o` is column `o`: the matrix is transposed.
    pub(super) fn matrix(&mut self, name: &str, w: &[f32], rows_in: usize, cols_out: usize) {
        self.rows(name, &transpose_2d(w, rows_in, cols_out), cols_out, rows_in);
    }
}

/// Serialize a model as GGUF: llama.cpp's `gpt2` layout for a BPE transformer
/// SLM, the generic `ferrum` layout otherwise. `prec` picks F32 / Q8_0 / Q4_0
/// for the matrices (see the section comment for the rule).
pub fn to_gguf(
    model: &Sequential,
    norm: &Normalizer,
    meta: &ModelMetadata,
    prec: QPrec,
) -> Result<Vec<u8>> {
    let gpt2 = gpt2_parts(model, meta);
    let mut o = GgufOut {
        b: GgufBuilder::new(),
        prec,
        quantized: false,
    };
    let mut stored = meta.clone();
    if gpt2.is_some() {
        stored.tokenizer_state.clear(); // lives in tokenizer.ggml.*
    }
    o.str(
        "general.architecture",
        if gpt2.is_some() { "gpt2" } else { "ferrum" },
    );
    o.str("general.name", &meta.dataset_name);
    if let Some(label) = size_label(model.num_params() as u64) {
        o.str("general.size_label", &label);
    }
    o.u32("ferrum.format", 1);
    o.str("ferrum.quant", prec.name());
    o.str("ferrum.metadata", &stored.to_json());
    o.str("ferrum.normalizer", &norm.encode());
    match &gpt2 {
        Some(p) => o.gpt2(p, meta)?,
        None => o.generic(model)?,
    }
    // general.file_type: 0 = all F32, 7 = mostly Q8_0, 2 = mostly Q4_0
    // (honest: 0 when no tensor could be quantized).
    let ft = match (o.quantized, prec) {
        (true, QPrec::Int8) => 7,
        (true, QPrec::Int4) => 2,
        _ => 0,
    };
    o.u32("general.file_type", ft as usize);
    o.u32("general.quantization_version", 2);
    Ok(o.b.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::to_gguf;
    use crate::loader::test_support::{bpe_parts, gguf_corpus, make_bundle, parse, DummyLayer};
    use crate::loader::QPrec;
    use crate::model::Sequential;
    use crate::tokenizer::ByteBpeTokenizer;

    #[test]
    fn parameterless_model_has_no_size_label() {
        let (_, norm, meta) = make_bundle();
        let bytes = to_gguf(&Sequential::new(), &norm, &meta, QPrec::F32).unwrap();
        assert!(parse(&bytes).meta("general.size_label").is_none());
    }

    #[test]
    fn to_gguf_rejects_inconsistent_models() {
        let (m, n, meta) = bpe_parts(32, 64, true);
        let mut other = meta.clone();
        other.tokenizer_state = ByteBpeTokenizer::train(&gguf_corpus(), 280)
            .unwrap()
            .encode_state();
        assert!(to_gguf(&m, &n, &other, QPrec::F32)
            .unwrap_err()
            .to_string()
            .contains("tokenizer has"));
        let mut broken = meta.clone();
        broken.tokenizer_state = "x".into();
        assert!(to_gguf(&m, &n, &broken, QPrec::F32).is_err());

        let model = Sequential::new().with(Box::new(DummyLayer));
        let (_, n, meta) = make_bundle();
        assert!(to_gguf(&model, &n, &meta, QPrec::F32)
            .unwrap_err()
            .to_string()
            .contains("unknown layer type"));
    }
}
