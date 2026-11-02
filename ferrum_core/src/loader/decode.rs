//! Reading: typed GGUF accessors and the GGUF-to-model entry points.
use crate::error::Result;
use crate::gguf::{transpose_2d, Gguf, MetaValue};
use crate::layer::{LayerNorm, Linear};
use crate::loader::bad;
use crate::loader::generic::generic_from;
use crate::loader::gpt2::gpt2_from;
use crate::meta::{ModelMetadata, Normalizer};
use crate::model::Sequential;

/// Typed, error-reporting reads from a parsed GGUF.
pub(super) struct GgufIn<'a>(pub(super) &'a Gguf);

impl GgufIn<'_> {
    pub(super) fn usize(&self, key: &str) -> Result<usize> {
        self.0
            .meta(key)
            .and_then(MetaValue::as_usize)
            .ok_or_else(|| bad(format!("GGUF key '{key}' missing or not an integer")))
    }
    pub(super) fn str(&self, key: &str) -> Result<&str> {
        self.0
            .meta(key)
            .and_then(MetaValue::as_str)
            .ok_or_else(|| bad(format!("GGUF key '{key}' missing or not a string")))
    }
    pub(super) fn f32(&self, key: &str) -> Result<f32> {
        self.0
            .meta(key)
            .and_then(MetaValue::as_f32)
            .ok_or_else(|| bad(format!("GGUF key '{key}' missing or not a number")))
    }
    pub(super) fn array<T>(
        &self,
        key: &str,
        item: impl Fn(&MetaValue) -> Option<T>,
    ) -> Result<Vec<T>> {
        let m = self.0.meta(key);
        let n = m.and_then(MetaValue::array_len);
        n.and_then(|n| (0..n).map(|i| item(&m?.array_get(i)?)).collect())
            .ok_or_else(|| bad(format!("GGUF key '{key}' missing or of the wrong type")))
    }
    /// Tensor `name` dequantized to f32, after checking its GGML dims.
    pub(super) fn tensor(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
        let t = self
            .0
            .tensor(name)
            .ok_or_else(|| bad(format!("missing tensor '{name}'")))?;
        if !t.dims.iter().copied().eq(dims.iter().map(|&d| d as u64)) {
            return Err(bad(format!(
                "tensor '{name}' has dims {:?}, expected {dims:?}",
                t.dims
            )));
        }
        self.0.dequantize(t)
    }
    pub(super) fn vec(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        self.tensor(name, &[n])
    }
    pub(super) fn rows(&self, name: &str, rows: usize, ne0: usize) -> Result<Vec<f32>> {
        self.tensor(name, &[ne0, rows])
    }
    /// Inverse of [`GgufOut::matrix`]: back to Ferrum's `[rows_in, cols_out]`.
    pub(super) fn matrix(&self, name: &str, rows_in: usize, cols_out: usize) -> Result<Vec<f32>> {
        let g = self.rows(name, cols_out, rows_in)?;
        Ok(transpose_2d(&g, cols_out, rows_in))
    }
    pub(super) fn linear(
        &self,
        w: &str,
        b: &str,
        rows_in: usize,
        cols_out: usize,
    ) -> Result<Linear> {
        Linear::new(
            rows_in,
            cols_out,
            self.matrix(w, rows_in, cols_out)?,
            self.vec(b, cols_out)?,
        )
    }
    pub(super) fn layernorm(&self, g: &str, b: &str, d: usize) -> Result<LayerNorm> {
        LayerNorm::new(d, self.vec(g, d)?, self.vec(b, d)?)
    }
    /// `ferrum.normalizer`; absent (foreign file) or empty = identity.
    pub(super) fn normalizer(&self) -> Result<Normalizer> {
        match self.0.meta("ferrum.normalizer").and_then(MetaValue::as_str) {
            None | Some("") => Ok(Normalizer {
                means: vec![],
                stds: vec![],
            }),
            Some(s) => Normalizer::decode(s),
        }
    }
}

/// Load a model written by `to_gguf`, or a foreign GPT-2 GGUF (e.g. one
/// converted by llama.cpp). Weights are dequantized to f32 in memory.
// ponytail: Q8_0/Q4_0 matrices are expanded to f32 (4×/8× the file's size in
// RAM); keep them packed once a Q8_0 kernel lands, if large models need it.
pub fn from_gguf(bytes: &[u8]) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    from_gguf_vec(bytes.to_vec())
}

/// [`from_gguf`] taking ownership of the bytes: a file just read from disk is
/// parsed in place instead of copied (the borrowed form peaks one file's size
/// higher).
pub fn from_gguf_vec(bytes: Vec<u8>) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    let g = parse_gguf_vec(bytes)?;
    let r = GgufIn(&g);
    match g.architecture() {
        Some("gpt2") => gpt2_from(&r),
        Some("ferrum") => generic_from(&r),
        other => Err(bad(format!(
            "not a Ferrum-loadable model: architecture {other:?} (need gpt2 or ferrum)"
        ))),
    }
}

/// Parse GGUF bytes, with a pointed error for the removed FINF `.bin` format.
pub(super) fn parse_gguf(bytes: &[u8]) -> Result<Gguf> {
    reject_finf(bytes)?;
    Gguf::parse(bytes.to_vec())
}

fn parse_gguf_vec(bytes: Vec<u8>) -> Result<Gguf> {
    reject_finf(&bytes)?;
    Gguf::parse(bytes)
}

fn reject_finf(bytes: &[u8]) -> Result<()> {
    if bytes.starts_with(b"FINF") {
        return Err(bad(
            "old Ferrum .bin (FINF) model — this format was removed; retrain to produce a .gguf"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::from_gguf;
    use crate::gguf::{MetaValue, NumArray};
    use crate::gguf_write::GgufBuilder;
    use crate::loader::encode::to_gguf;
    use crate::loader::test_support::{
        arch, bpe_parts, load_err, load_err_of, make_bundle, make_embedding_bundle_big, rebuild,
        GgufEdit, Kv, Tensors, FINF_HINT,
    };
    use crate::loader::QPrec;

    #[test]
    fn finf_input_gets_removed_format_hint() {
        assert!(load_err_of(from_gguf(b"FINF\x05\x00\x00\x00")).contains(FINF_HINT));
    }

    #[test]
    fn rejects_foreign_and_truncated_gguf() {
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        assert!(load_err(&b.into_bytes()).contains("not a Ferrum-loadable model"));
        assert!(load_err(&GgufBuilder::new().into_bytes()).contains("not a Ferrum-loadable model"));
        assert!(load_err_of(from_gguf(b"FINF\x05\x00\x00\x00")).contains(FINF_HINT));
        for prec in [QPrec::Int8, QPrec::F32] {
            let (m, n, meta) = bpe_parts(32, 64, true);
            let good = to_gguf(&m, &n, &meta, prec).unwrap();
            // Every cut point through the header, then a stride through the data.
            let cuts = (0..good.len()).filter(|&c| c < 16384 || c % 97 == 0);
            for cut in cuts {
                assert!(
                    from_gguf(&good[..cut]).is_err(),
                    "truncated at {cut} must error"
                );
            }
        }
    }

    /// Two tensors sharing bytes are rejected at parse time for both codecs
    /// (the same file loads fine before the edit).
    #[test]
    fn from_gguf_rejects_overlapping_tensors() {
        let (m, n, meta) = bpe_parts(32, 64, true);
        let gpt2 = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let (model, norm, emeta) = make_embedding_bundle_big();
        let generic = to_gguf(&model, &norm, &emeta, QPrec::F32).unwrap();
        for (bytes, arch_name) in [(gpt2, "gpt2"), (generic, "ferrum")] {
            assert_eq!(arch(&bytes), arch_name);
            from_gguf(&bytes).unwrap();
            let mut bad = bytes.clone();
            crate::gguf::patch_tensor_offset(&mut bad, 1, 0); // onto tensor 0
            let e = load_err(&bad);
            assert!(e.contains("overlap"), "{arch_name}: {e}");
        }
    }

    #[test]
    fn from_gguf_reports_bad_fields() {
        let s = |x: &str| MetaValue::String(x.into());
        let check = |bytes: &[u8], why: &str, want: &str, edit: &dyn Fn(&mut Kv, &mut Tensors)| {
            let bad = rebuild(bytes, |kv, ts| edit(kv, ts));
            let e = load_err(&bad);
            assert!(e.contains(want), "{why}: {e:?} lacks {want:?}");
        };
        let (m, n, meta) = bpe_parts(32, 64, true);
        let gpt2 = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let set_tokens = |kv: &mut Kv, f: &dyn Fn(&mut Vec<MetaValue>)| {
            // (A missing key leaves the file valid, and `check` then fails on "expected an error".)
            let key = "tokenizer.ggml.tokens";
            if let Some(MetaValue::Array(t)) = kv.get_mut(key) {
                f(t)
            }
        };
        // Without a token table the edit is a no-op.
        set_tokens(&mut Kv::new(), &Vec::clear);
        let cases: Vec<(&str, &str, GgufEdit)> = vec![
            (
                "missing epsilon",
                "layer_norm_epsilon",
                Box::new(|kv, _| {
                    kv.remove("gpt2.attention.layer_norm_epsilon");
                }),
            ),
            (
                "epsilon not a number",
                "layer_norm_epsilon",
                Box::new(move |kv, _| {
                    kv.insert("gpt2.attention.layer_norm_epsilon".into(), s("x"));
                }),
            ),
            (
                "epsilon is not Ferrum's",
                "uses 0.00001",
                Box::new(|kv, _| {
                    kv.insert(
                        "gpt2.attention.layer_norm_epsilon".into(),
                        MetaValue::F32(1e-6),
                    );
                }),
            ),
            (
                "missing dim",
                "gpt2.embedding_length",
                Box::new(|kv, _| {
                    kv.remove("gpt2.embedding_length");
                }),
            ),
            (
                "dim not an int",
                "gpt2.block_count",
                Box::new(move |kv, _| {
                    kv.insert("gpt2.block_count".into(), s("2"));
                }),
            ),
            (
                "no tokens",
                "tokenizer.ggml.tokens",
                Box::new(|kv, _| {
                    kv.remove("tokenizer.ggml.tokens");
                }),
            ),
            (
                "tokens not strings",
                "tokenizer.ggml.tokens",
                Box::new(move |kv, _| set_tokens(kv, &|t| *t = vec![MetaValue::U32(3)])),
            ),
            (
                "no token types",
                "tokenizer.ggml.token_type",
                Box::new(|kv, _| {
                    kv.remove("tokenizer.ggml.token_type");
                }),
            ),
            (
                "token types not i32",
                "tokenizer.ggml.token_type",
                Box::new(move |kv, _| {
                    kv.insert(
                        "tokenizer.ggml.token_type".into(),
                        MetaValue::Array(vec![s("1")]),
                    );
                }),
            ),
            (
                "type count",
                "token types",
                Box::new(|kv, _| {
                    if let Some(MetaValue::Nums(NumArray::I32(t))) =
                        kv.get_mut("tokenizer.ggml.token_type")
                    {
                        t.pop();
                    }
                }),
            ),
            (
                "merge result missing",
                "no unique token",
                Box::new(move |kv, _| {
                    if let Some(MetaValue::Array(mg)) = kv.get_mut("tokenizer.ggml.merges") {
                        mg[0] = s("q z");
                    }
                }),
            ),
            (
                "merge result reused",
                "no unique token",
                Box::new(|kv, _| {
                    if let Some(MetaValue::Array(mg)) = kv.get_mut("tokenizer.ggml.merges") {
                        mg[1] = mg[0].clone();
                    }
                }),
            ),
            (
                "special not control",
                "not control",
                Box::new(|kv, _| {
                    if let Some(MetaValue::Nums(NumArray::I32(t))) =
                        kv.get_mut("tokenizer.ggml.token_type")
                    {
                        *t.last_mut().unwrap() = 1;
                    }
                }),
            ),
            (
                "missing tensor",
                "missing tensor 'blk.1.ffn_up.bias'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "blk.1.ffn_up.bias")),
            ),
            (
                "wrong dims",
                "position_embd.weight",
                Box::new(|_, ts| {
                    for t in ts.iter_mut().filter(|t| t.0 == "position_embd.weight") {
                        t.1 = vec![t.1[1], t.1[0]];
                    }
                }),
            ),
            (
                "bad metadata",
                "missing task",
                Box::new(move |kv, _| {
                    kv.insert("ferrum.metadata".into(), s("not json"));
                }),
            ),
            (
                "bad normalizer",
                "bad token",
                Box::new(move |kv, _| {
                    kv.insert("ferrum.normalizer".into(), s("x"));
                }),
            ),
            (
                "heads do not divide the width",
                "divisible by num_heads",
                Box::new(|kv, _| {
                    kv.insert("gpt2.attention.head_count".into(), MetaValue::U32(3));
                }),
            ),
            // Round 3: every metadata dim is tied to a (file-bounded) tensor
            // before it sizes anything, so huge values fail on a shape check.
            (
                "huge context",
                "position_embd.weight",
                Box::new(|kv, _| {
                    kv.insert("gpt2.context_length".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                "zero width",
                "token_embd.weight",
                Box::new(|kv, _| {
                    kv.insert("gpt2.embedding_length".into(), MetaValue::U32(0));
                }),
            ),
            (
                "huge width",
                "token_embd.weight",
                Box::new(|kv, _| {
                    kv.insert("gpt2.embedding_length".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                "huge ffn",
                "ffn_up.weight",
                Box::new(|kv, _| {
                    kv.insert("gpt2.feed_forward_length".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                "huge block count",
                "missing tensor 'blk.2.attn_qkv.weight'",
                Box::new(|kv, _| {
                    kv.insert("gpt2.block_count".into(), MetaValue::U64(1 << 40));
                }),
            ),
        ];
        for (why, want, edit) in &cases {
            check(&gpt2, why, want, edit.as_ref());
        }

        let (model, norm, cmeta) = make_bundle();
        let tab = to_gguf(&model, &norm, &cmeta, QPrec::F32).unwrap();
        let (model, norm, emeta) = make_embedding_bundle_big();
        let big = to_gguf(&model, &norm, &emeta, QPrec::F32).unwrap();
        let generic: Vec<(&[u8], &str, &str, GgufEdit)> = vec![
            (
                &tab,
                "format",
                "ferrum.format",
                Box::new(|kv, _| {
                    kv.insert("ferrum.format".into(), MetaValue::U32(2));
                }),
            ),
            (
                &tab,
                "no metadata",
                "ferrum.metadata",
                Box::new(|kv, _| {
                    kv.remove("ferrum.metadata");
                }),
            ),
            (
                &tab,
                "no layer count",
                "ferrum.layer_count",
                Box::new(|kv, _| {
                    kv.remove("ferrum.layer_count");
                }),
            ),
            (
                &tab,
                "type not a string",
                "ferrum.layer.0.type",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.0.type".into(), MetaValue::U32(0));
                }),
            ),
            (
                &tab,
                "unknown type",
                "unknown layer type 'conv'",
                Box::new(move |kv, _| {
                    kv.insert("ferrum.layer.0.type".into(), s("conv"));
                }),
            ),
            (
                &tab,
                "bad activation",
                "bad activation kind 99",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.kind".into(), MetaValue::U32(99));
                }),
            ),
            (
                &tab,
                "activation > u8",
                "bad activation kind 300",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.kind".into(), MetaValue::U32(300));
                }),
            ),
            (
                &tab,
                "missing bias",
                "missing tensor 'l2.bias'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "l2.bias")),
            ),
            (
                &big,
                "missing block tensor",
                "missing tensor 'l1.ffn2.weight'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "l1.ffn2.weight")),
            ),
            (
                &big,
                "missing embedding",
                "missing tensor 'l0.pos'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "l0.pos")),
            ),
            (
                &big,
                "missing layernorm",
                "missing tensor 'l2.gamma'",
                Box::new(|_, ts| ts.retain(|t| t.0 != "l2.gamma")),
            ),
            (
                &big,
                "huge block context",
                "block context × dim",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.context".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                &big,
                "block context × dim overflows",
                "overflows",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.context".into(), MetaValue::U64(u64::MAX));
                }),
            ),
            (
                &big,
                "huge embedding seq",
                "l0.pos",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.0.seq".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                &tab,
                "huge linear width",
                "l0.weight",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.0.out".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                &tab,
                "huge layer count",
                "ferrum.layer.4.type",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer_count".into(), MetaValue::U64(1 << 40));
                }),
            ),
            (
                &big,
                "block heads do not divide the width",
                "divisible by num_heads",
                Box::new(|kv, _| {
                    kv.insert("ferrum.layer.1.heads".into(), MetaValue::U32(3));
                }),
            ),
        ];
        for (bytes, why, want, edit) in &generic {
            check(bytes, why, want, edit.as_ref());
        }
    }
}
