//! The generic `ferrum` architecture codec for every other model.
use crate::activation::Activation;
use crate::error::Result;
use crate::gguf::bounded;
use crate::layer::{
    ActivationLayer, Embedding, Flatten, Layer, LayerNorm, Linear, TransformerBlock,
};
use crate::loader::decode::GgufIn;
use crate::loader::encode::GgufOut;
use crate::loader::{bad, mul_dims};
use crate::meta::{ModelMetadata, Normalizer};
use crate::model::Sequential;

impl GgufOut {
    pub(super) fn generic(&mut self, model: &Sequential) -> Result<()> {
        self.u32("ferrum.layer_count", model.len());
        for (i, layer) in model.layers().iter().enumerate() {
            let k = |s: &str| format!("ferrum.layer.{i}.{s}");
            let t = |s: &str| format!("l{i}.{s}");
            let any = layer.as_any();
            let ty = if let Some(lin) = any.downcast_ref::<Linear>() {
                let (a, b) = (lin.in_features(), lin.out_features());
                self.u32(&k("in"), a);
                self.u32(&k("out"), b);
                self.matrix(&t("weight"), &lin.weight_f32(), a, b);
                self.vec(&t("bias"), &lin.bias.data);
                "linear"
            } else if let Some(act) = any.downcast_ref::<ActivationLayer>() {
                self.u32(&k("kind"), act.kind.tag() as usize);
                "activation"
            } else if any.downcast_ref::<Flatten>().is_some() {
                "flatten"
            } else if let Some(emb) = any.downcast_ref::<Embedding>() {
                let (v, s, d) = (emb.vocab_size(), emb.max_seq_len(), emb.embedding_dim());
                self.u32(&k("vocab"), v);
                self.u32(&k("seq"), s);
                self.u32(&k("dim"), d);
                self.rows(&t("token"), &emb.token_weight.data, v, d);
                self.f32(&t("pos"), &[d, s], &emb.pos_weight.data);
                "embedding"
            } else if let Some(ln) = any.downcast_ref::<LayerNorm>() {
                self.u32(&k("dim"), ln.dim());
                self.vec(&t("gamma"), &ln.gamma.data);
                self.vec(&t("beta"), &ln.beta.data);
                "layernorm"
            } else if let Some(tb) = any.downcast_ref::<TransformerBlock>() {
                let (c, h) = (tb.embedding_dim(), tb.hidden_dim());
                self.u32(&k("context"), tb.context_len());
                self.u32(&k("heads"), tb.num_heads());
                self.u32(&k("dim"), c);
                self.u32(&k("hidden"), h);
                let lins = [
                    ("q", &tb.q_proj, c, c),
                    ("k", &tb.k_proj, c, c),
                    ("v", &tb.v_proj, c, c),
                    ("out", &tb.out_proj, c, c),
                    ("ffn1", &tb.ffn1, c, h),
                    ("ffn2", &tb.ffn2, h, c),
                ];
                for (p, ln) in [("ln1", &tb.ln1), ("ln2", &tb.ln2)] {
                    self.vec(&t(&format!("{p}.gamma")), &ln.gamma.data);
                    self.vec(&t(&format!("{p}.beta")), &ln.beta.data);
                }
                for (p, lin, a, b) in lins {
                    self.matrix(&t(&format!("{p}.weight")), &lin.weight_f32(), a, b);
                    self.vec(&t(&format!("{p}.bias")), &lin.bias.data);
                }
                "transformer_block"
            } else {
                return Err(bad(format!("unknown layer type: {}", layer.name())));
            };
            self.str(&k("type"), ty);
        }
        Ok(())
    }
}

pub(super) fn generic_from(r: &GgufIn) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    let format = r.usize("ferrum.format")?;
    if format != 1 {
        return Err(bad(format!("unsupported ferrum.format {format} (need 1)")));
    }
    let meta = ModelMetadata::from_json(r.str("ferrum.metadata")?)?;
    let mut model = Sequential::new();
    for i in 0..r.usize("ferrum.layer_count")? {
        let k = |s: &str| r.usize(&format!("ferrum.layer.{i}.{s}"));
        let t = |s: &str| format!("l{i}.{s}");
        let layer: Box<dyn Layer> = match r.str(&format!("ferrum.layer.{i}.type"))? {
            "linear" => {
                let (a, b) = (k("in")?, k("out")?);
                Box::new(r.linear(&t("weight"), &t("bias"), a, b)?)
            }
            "activation" => {
                let tag = k("kind")?;
                let act = u8::try_from(tag)
                    .ok()
                    .and_then(Activation::from_tag)
                    .ok_or_else(|| bad(format!("bad activation kind {tag}")))?;
                Box::new(ActivationLayer::new(act))
            }
            "flatten" => Box::new(Flatten::new()),
            "embedding" => {
                let (v, s, d) = (k("vocab")?, k("seq")?, k("dim")?);
                let token = r.rows(&t("token"), v, d)?;
                Box::new(Embedding::new(v, s, d, token, r.rows(&t("pos"), s, d)?)?)
            }
            "layernorm" => Box::new(r.layernorm(&t("gamma"), &t("beta"), k("dim")?)?),
            "transformer_block" => {
                let (ctx, heads, c, h) = (k("context")?, k("heads")?, k("dim")?, k("hidden")?);
                // No tensor carries a block's context, yet it sizes the KV cache
                // (2 · ctx · dim floats): bound ctx · dim by the file. A real
                // model's F32 [dim, ctx] position table alone takes 4 · ctx · dim
                // bytes, so valid files pass.
                bounded(mul_dims(ctx, c)?, r.0.file_len(), "block context × dim")?;
                let lin = |p: &str, a: usize, b: usize| {
                    r.linear(&t(&format!("{p}.weight")), &t(&format!("{p}.bias")), a, b)
                };
                let ln =
                    |p: &str| r.layernorm(&t(&format!("{p}.gamma")), &t(&format!("{p}.beta")), c);
                Box::new(TransformerBlock::from_parts(
                    ctx,
                    heads,
                    c,
                    ln("ln1")?,
                    lin("q", c, c)?,
                    lin("k", c, c)?,
                    lin("v", c, c)?,
                    lin("out", c, c)?,
                    ln("ln2")?,
                    lin("ffn1", c, h)?,
                    lin("ffn2", h, c)?,
                )?)
            }
            other => return Err(bad(format!("unknown layer type '{other}'"))),
        };
        model.push(layer);
    }
    Ok((model, r.normalizer()?, meta))
}

#[cfg(test)]
mod tests {
    use crate::gguf::{MetaValue, GGML_F32, GGML_Q4_0, GGML_Q8_0};
    use crate::gguf_write::size_label;
    use crate::layer::{Flatten, Linear};
    use crate::loader::decode::from_gguf;
    use crate::loader::encode::to_gguf;
    use crate::loader::test_support::{
        make_bundle, make_embedding_bundle_big, max_abs_diff, parse, vals,
    };
    use crate::loader::QPrec;
    use crate::model::Sequential;
    use crate::tensor::Tensor;

    #[test]
    fn tabular_mlp_roundtrips_with_normalizer() {
        let (model, norm, meta) = make_bundle();
        let bytes = to_gguf(&model, &norm, &meta, QPrec::F32).unwrap();
        let g = parse(&bytes);
        assert_eq!(g.architecture(), Some("ferrum"));
        assert_eq!(g.meta("ferrum.layer_count"), Some(&MetaValue::U32(4)));
        assert_eq!(
            g.meta("ferrum.layer.0.type"),
            Some(&MetaValue::String("linear".into()))
        );
        assert_eq!(g.meta("ferrum.layer.1.kind"), Some(&MetaValue::U32(1)));
        assert_eq!(g.tensor("l0.weight").unwrap().dims, vec![4, 8]);
        let (m2, n2, meta2) = from_gguf(&bytes).unwrap();
        assert_eq!(n2.encode(), norm.encode());
        assert_eq!(meta2.to_json(), meta.to_json());
        let x = Tensor::row(vec![0.1, -0.2, 0.3, 0.9]).unwrap();
        assert_eq!(
            model.forward(&x).unwrap().data,
            m2.forward(&x).unwrap().data
        );
    }

    #[test]
    fn generic_codec_covers_every_layer_kind() {
        // Flatten + Linear.
        let lin = Linear::new(8, 3, vals(24, 3), vals(3, 4)).unwrap();
        let model = Sequential::new()
            .with(Box::new(Flatten::new()))
            .with(Box::new(lin));
        let (_, norm, meta) = make_bundle();
        let bytes = to_gguf(&model, &norm, &meta, QPrec::F32).unwrap();
        let (m2, _, _) = from_gguf(&bytes).unwrap();
        assert_eq!(m2.layers()[0].name(), "Flatten");
        let x = Tensor::matrix(2, 4, vals(8, 5)).unwrap();
        assert_eq!(
            model.forward(&x).unwrap().data,
            m2.forward(&x).unwrap().data
        );

        // A char-level transformer (no tokenizer state) → generic codec with
        // Embedding / TransformerBlock / LayerNorm, at every precision.
        let (model, norm, meta) = make_embedding_bundle_big();
        let x = Tensor::matrix(1, 4, vec![0.0, 3.0, 9.0, 1.0]).unwrap();
        let want = model.forward(&x).unwrap().data;
        for (prec, ty, tol) in [
            (QPrec::F32, GGML_F32, 0.0),
            (QPrec::Int8, GGML_Q8_0, 0.05),
            (QPrec::Int4, GGML_Q4_0, 0.1),
        ] {
            let bytes = to_gguf(&model, &norm, &meta, prec).unwrap();
            let g = parse(&bytes);
            assert_eq!(g.architecture(), Some("ferrum"));
            assert_eq!(
                g.meta("general.size_label"),
                size_label(model.num_params() as u64)
                    .map(MetaValue::String)
                    .as_ref()
            );
            assert_eq!(g.tensor("l1.ffn1.weight").unwrap().dims, vec![64, 128]);
            assert_eq!(g.tensor("l1.ffn2.weight").unwrap().ggml_type, ty);
            assert_eq!(g.tensor("l0.token").unwrap().ggml_type, ty);
            assert_eq!(g.tensor("l0.pos").unwrap().ggml_type, GGML_F32);
            assert_eq!(g.tensor("l1.ln2.beta").unwrap().ggml_type, GGML_F32);
            let (m2, _, meta2) = from_gguf(&bytes).unwrap();
            assert_eq!(meta2.to_json(), meta.to_json());
            let err = max_abs_diff(&want, &m2.forward(&x).unwrap().data);
            assert!(err <= tol, "{prec:?}: {err}");
        }
    }
}
