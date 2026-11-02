//! The llama.cpp-compatible gpt2 codec used for BPE transformer SLMs.
use crate::activation::Activation;
use crate::error::Result;
use crate::gguf::{transpose_2d, MetaValue, NumArray};
use crate::layer::{ActivationLayer, Embedding, LayerNorm, Linear, TransformerBlock};
use crate::loader::decode::GgufIn;
use crate::loader::encode::GgufOut;
use crate::loader::{bad, mul_dims, LN_EPS};
use crate::meta::{ModelMetadata, Normalizer, TaskType};
use crate::model::Sequential;
use crate::tokenizer::{byte_to_unicode, ByteBpeTokenizer};
use std::collections::HashMap;

/// The layers of a model the GPT-2 codec can carry.
pub(super) struct Gpt2Parts<'a> {
    pub(super) emb: &'a Embedding,
    pub(super) blocks: Vec<&'a TransformerBlock>,
    pub(super) lnf: &'a LayerNorm,
    pub(super) head: &'a Linear,
}

/// `Some` when `model` is a BPE transformer SLM in exactly GPT-2's shape:
/// `Embedding, TransformerBlock×N (N ≥ 1, uniform), LayerNorm, Linear (no
/// bias, vocab wide), Softmax`.
pub(super) fn gpt2_parts<'a>(model: &'a Sequential, meta: &ModelMetadata) -> Option<Gpt2Parts<'a>> {
    let l = model.layers();
    if meta.task != TaskType::TransformerSLM || meta.tokenizer_state.is_empty() || l.len() < 5 {
        return None;
    }
    let n = l.len();
    let emb = l[0].as_any().downcast_ref::<Embedding>()?;
    let blocks = l[1..n - 3]
        .iter()
        .map(|b| b.as_any().downcast_ref::<TransformerBlock>())
        .collect::<Option<Vec<_>>>()?;
    let lnf = l[n - 3].as_any().downcast_ref::<LayerNorm>()?;
    let head = l[n - 2].as_any().downcast_ref::<Linear>()?;
    let soft = l[n - 1].as_any().downcast_ref::<ActivationLayer>()?;
    let (d, b0) = (emb.embedding_dim(), blocks[0]);
    let uniform = blocks.iter().all(|b| {
        b.context_len() == emb.max_seq_len()
            && b.embedding_dim() == d
            && b.num_heads() == b0.num_heads()
            && b.hidden_dim() == b0.hidden_dim()
    });
    let fits = uniform
        && soft.kind == Activation::Softmax
        && lnf.dim() == d
        && head.in_features() == d
        && head.out_features() == emb.vocab_size()
        && head.bias.data.iter().all(|&x| x == 0.0);
    fits.then_some(Gpt2Parts {
        emb,
        blocks,
        lnf,
        head,
    })
}

impl GgufOut {
    pub(super) fn gpt2(&mut self, p: &Gpt2Parts, meta: &ModelMetadata) -> Result<()> {
        let tok = ByteBpeTokenizer::from_state(&meta.tokenizer_state)?;
        let (v, d, ctx) = (
            p.emb.vocab_size(),
            p.emb.embedding_dim(),
            p.emb.max_seq_len(),
        );
        if tok.vocab_size() != v {
            return Err(bad(format!(
                "tokenizer has {} tokens but the model vocab is {v}",
                tok.vocab_size()
            )));
        }
        let b0 = p.blocks[0];
        for (k, x) in [
            ("context_length", ctx),
            ("embedding_length", d),
            ("block_count", p.blocks.len()),
            ("feed_forward_length", b0.hidden_dim()),
            ("attention.head_count", b0.num_heads()),
        ] {
            self.u32(&format!("gpt2.{k}"), x);
        }
        self.b
            .meta("gpt2.attention.layer_norm_epsilon", MetaValue::F32(LN_EPS));
        self.str("tokenizer.ggml.model", "gpt2");
        self.str("tokenizer.ggml.pre", "gpt-2");
        let strings =
            |v: Vec<String>| MetaValue::Array(v.into_iter().map(MetaValue::String).collect());
        self.b
            .meta("tokenizer.ggml.tokens", strings(tok.gguf_tokens()));
        self.b
            .meta("tokenizer.ggml.merges", strings(tok.gguf_merges()));
        self.b.meta(
            "tokenizer.ggml.token_type",
            NumArray::I32(tok.gguf_token_types()).into_meta(),
        );
        // Without these ids llama.cpp defaults bos/eos to token 11, so refuse.
        let eos = tok.eos_id().ok_or_else(|| {
            bad(format!(
                "the tokenizer has no {} token; llama.cpp would default eos to token 11. \
                 Add it with add_special_token before exporting",
                crate::tokenizer::TOK_EOS
            ))
        })?;
        self.u32("tokenizer.ggml.bos_token_id", eos);
        self.u32("tokenizer.ggml.eos_token_id", eos);
        self.b
            .meta("tokenizer.ggml.add_bos_token", MetaValue::Bool(false));
        self.b
            .meta("tokenizer.ggml.add_eos_token", MetaValue::Bool(false));

        let tok_rows = &p.emb.token_weight.data;
        self.rows("token_embd.weight", tok_rows, v, d);
        self.f32("position_embd.weight", &[d, ctx], &p.emb.pos_weight.data);
        for (i, b) in p.blocks.iter().enumerate() {
            let n = |s: &str| format!("blk.{i}.{s}");
            let (q, k, vp) = (&b.q_proj, &b.k_proj, &b.v_proj);
            self.vec(&n("attn_norm.weight"), &b.ln1.gamma.data);
            self.vec(&n("attn_norm.bias"), &b.ln1.beta.data);
            let qkv: Vec<f32> = [q, k, vp]
                .iter()
                .flat_map(|l| transpose_2d(&l.weight_f32(), d, d))
                .collect();
            self.rows(&n("attn_qkv.weight"), &qkv, 3 * d, d);
            let qkv_b = [&q.bias.data[..], &k.bias.data, &vp.bias.data].concat();
            self.vec(&n("attn_qkv.bias"), &qkv_b);
            self.matrix(&n("attn_output.weight"), &b.out_proj.weight_f32(), d, d);
            self.vec(&n("attn_output.bias"), &b.out_proj.bias.data);
            self.vec(&n("ffn_norm.weight"), &b.ln2.gamma.data);
            self.vec(&n("ffn_norm.bias"), &b.ln2.beta.data);
            let h = b.hidden_dim();
            self.matrix(&n("ffn_up.weight"), &b.ffn1.weight_f32(), d, h);
            self.vec(&n("ffn_up.bias"), &b.ffn1.bias.data);
            self.matrix(&n("ffn_down.weight"), &b.ffn2.weight_f32(), h, d);
            self.vec(&n("ffn_down.bias"), &b.ffn2.bias.data);
        }
        self.vec("output_norm.weight", &p.lnf.gamma.data);
        self.vec("output_norm.bias", &p.lnf.beta.data);
        // Tied head: omitted, llama.cpp falls back to token_embd.
        let head = transpose_2d(&p.head.weight_f32(), d, v);
        if head != *tok_rows {
            self.rows("output.weight", &head, v, d);
        }
        Ok(())
    }
}

/// Map a GGUF BPE vocab onto Ferrum's id layout (256 byte tokens in byte
/// order, one token per merge in rank order, then control tokens), so a
/// foreign GPT-2 vocab (which orders its byte tokens differently) loads too.
/// Returns the tokenizer and `perm[ferrum_id] = file_id`.
fn ferrum_vocab(
    tokens: &[String],
    merges: &[String],
    types: &[i32],
) -> Result<(ByteBpeTokenizer, Vec<usize>)> {
    if types.len() != tokens.len() {
        return Err(bad(format!(
            "gguf vocab: {} tokens but {} token types",
            tokens.len(),
            types.len()
        )));
    }
    let id: HashMap<&str, usize> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();
    let mut used = vec![false; tokens.len()];
    let mut perm = Vec::with_capacity(tokens.len());
    // Tokens never hold a raw space (byte_to_unicode maps it to 'Ġ'), so the
    // merge's single space is the separator and dropping it gives the result.
    let wanted = (0..=255u8)
        .map(|b| byte_to_unicode(b).to_string())
        .chain(merges.iter().map(|m| m.replacen(' ', "", 1)));
    for w in wanted {
        match id.get(w.as_str()) {
            Some(&i) if !used[i] => {
                used[i] = true;
                perm.push(i);
            }
            _ => return Err(bad(format!("gguf vocab: no unique token {w:?}"))),
        }
    }
    perm.extend((0..tokens.len()).filter(|&i| !used[i]));
    let ordered: Vec<String> = perm.iter().map(|&i| tokens[i].clone()).collect();
    let ordered_types: Vec<i32> = perm.iter().map(|&i| types[i]).collect();
    let tok = ByteBpeTokenizer::from_gguf_vocab(&ordered, merges, &ordered_types)?;
    Ok((tok, perm))
}

/// Reorder `ne0`-wide rows: output row `i` is input row `perm[i]`.
fn permute_rows(data: &[f32], perm: &[usize], ne0: usize) -> Vec<f32> {
    perm.iter()
        .flat_map(|&i| &data[i * ne0..(i + 1) * ne0])
        .copied()
        .collect()
}

pub(super) fn gpt2_from(r: &GgufIn) -> Result<(Sequential, Normalizer, ModelMetadata)> {
    let k = |s: &str| r.usize(&format!("gpt2.{s}"));
    let (ctx, d, n_layer) = (
        k("context_length")?,
        k("embedding_length")?,
        k("block_count")?,
    );
    let (ff, heads) = (k("feed_forward_length")?, k("attention.head_count")?);
    // llama.cpp requires the key; Ferrum's LayerNorm is fixed at 1e-5.
    let eps = r.f32("gpt2.attention.layer_norm_epsilon")?;
    if (eps - LN_EPS).abs() > 1e-9 {
        return Err(bad(format!(
            "gpt2.attention.layer_norm_epsilon is {eps}; Ferrum's LayerNorm uses {LN_EPS}"
        )));
    }
    let string = |v: &MetaValue| v.as_str().map(String::from);
    let tokens = r.array("tokenizer.ggml.tokens", string)?;
    let merges = r.array("tokenizer.ggml.merges", string)?;
    let types = r.array("tokenizer.ggml.token_type", |v| match v {
        MetaValue::I32(x) => Some(*x),
        _ => None,
    })?;
    let (tok, perm) = ferrum_vocab(&tokens, &merges, &types)?;
    let v = perm.len();

    let tok_rows = permute_rows(&r.rows("token_embd.weight", v, d)?, &perm, d);
    // No `output.weight`: the head is the token embedding (weight tying).
    let tied = r.0.tensor("output.weight").is_none();
    let head_rows = if tied {
        tok_rows.clone()
    } else {
        permute_rows(&r.rows("output.weight", v, d)?, &perm, d)
    };
    let pos = r.rows("position_embd.weight", ctx, d)?;
    let mut model = Sequential::new().with(Box::new(Embedding::new(v, ctx, d, tok_rows, pos)?));
    for i in 0..n_layer {
        let n = |s: &str| format!("blk.{i}.{s}");
        let qkv = r.rows(&n("attn_qkv.weight"), mul_dims(3, d)?, d)?;
        let qkv_b = r.vec(&n("attn_qkv.bias"), 3 * d)?;
        // GGML rows Q, then K, then V; each block of d rows is [out, in].
        let proj = |j: usize| {
            let w = &qkv[j * d * d..(j + 1) * d * d];
            Linear::new(
                d,
                d,
                transpose_2d(w, d, d),
                qkv_b[j * d..(j + 1) * d].to_vec(),
            )
        };
        model.push(Box::new(TransformerBlock::from_parts(
            ctx,
            heads,
            d,
            r.layernorm(&n("attn_norm.weight"), &n("attn_norm.bias"), d)?,
            proj(0)?,
            proj(1)?,
            proj(2)?,
            r.linear(&n("attn_output.weight"), &n("attn_output.bias"), d, d)?,
            r.layernorm(&n("ffn_norm.weight"), &n("ffn_norm.bias"), d)?,
            r.linear(&n("ffn_up.weight"), &n("ffn_up.bias"), d, ff)?,
            r.linear(&n("ffn_down.weight"), &n("ffn_down.bias"), ff, d)?,
        )?));
    }
    model.push(Box::new(r.layernorm(
        "output_norm.weight",
        "output_norm.bias",
        d,
    )?));
    let head_w = transpose_2d(&head_rows, v, d);
    model.push(Box::new(if tied {
        Linear::tied_head(d, v, head_w)?
    } else {
        Linear::without_bias(d, v, head_w)?
    }));
    model.push(Box::new(ActivationLayer::new(Activation::Softmax)));

    let mut meta = match r.0.meta("ferrum.metadata").and_then(MetaValue::as_str) {
        Some(json) => ModelMetadata::from_json(json)?,
        // A foreign GPT-2 file: rebuild what Ferrum's trainer would record.
        None => ModelMetadata {
            dataset_name: r.str("general.name").unwrap_or("gpt2").into(),
            task: TaskType::TransformerSLM,
            feature_names: (0..ctx).map(|i| format!("c_{i}")).collect(),
            feature_ranges: vec![[0.0, v as f32]; ctx],
            class_names: vec![],
            target_name: "next_char".into(),
            target_range: [0.0, v as f32],
            input_dim: ctx,
            output_dim: v,
            tokenizer_state: String::new(),
        },
    };
    meta.tokenizer_state = tok.encode_state();
    Ok((model, r.normalizer()?, meta))
}

#[cfg(test)]
mod tests {
    use crate::activation::Activation;
    use crate::gguf::{MetaValue, GGML_F32, GGML_Q8_0};
    use crate::gguf_write::size_label;
    use crate::layer::{ActivationLayer, Embedding, LayerNorm, Linear, TransformerBlock};
    use crate::loader::decode::from_gguf;
    use crate::loader::encode::to_gguf;
    use crate::loader::info::gguf_quant;
    use crate::loader::test_support::{
        arch, block, bpe_layers, bpe_parts, empty_norm, fwd, load_err, parse, rebuild, seq, vals,
        LayerEdit, Layers,
    };
    use crate::loader::QPrec;
    use crate::meta::{ModelMetadata, TaskType};
    use crate::slm::GenerativeSLM;
    use crate::tokenizer::ByteBpeTokenizer;

    #[test]
    fn gpt2_layout_matches_llama_cpp() {
        for tie in [true, false] {
            let (m, n, meta) = bpe_parts(64, 128, tie);
            let bytes = to_gguf(&m, &n, &meta, QPrec::Int8).unwrap();
            let g = parse(&bytes);
            let tok = ByteBpeTokenizer::from_state(&meta.tokenizer_state).unwrap();
            let (v, d, ctx, ff) = (tok.vocab_size() as u64, 64u64, 8u64, 128u64);
            let u = |k: &str| g.meta(k).cloned();
            let s = |x: &str| Some(MetaValue::String(x.into()));
            assert_eq!(u("general.architecture"), s("gpt2"));
            assert_eq!(u("general.name"), s("gguf test"));
            assert_eq!(
                u("general.size_label"),
                size_label(m.num_params() as u64).map(MetaValue::String)
            );
            assert_eq!(u("gpt2.context_length"), Some(MetaValue::U32(8)));
            assert_eq!(u("gpt2.embedding_length"), Some(MetaValue::U32(64)));
            assert_eq!(u("gpt2.block_count"), Some(MetaValue::U32(2)));
            assert_eq!(u("gpt2.feed_forward_length"), Some(MetaValue::U32(128)));
            assert_eq!(u("gpt2.attention.head_count"), Some(MetaValue::U32(4)));
            assert_eq!(
                u("gpt2.attention.layer_norm_epsilon"),
                Some(MetaValue::F32(1e-5))
            );
            assert_eq!(u("tokenizer.ggml.model"), s("gpt2"));
            assert_eq!(u("tokenizer.ggml.pre"), s("gpt-2"));
            let strs = |x: Vec<String>| {
                Some(MetaValue::Array(
                    x.into_iter().map(MetaValue::String).collect(),
                ))
            };
            assert_eq!(u("tokenizer.ggml.tokens"), strs(tok.gguf_tokens()));
            assert_eq!(u("tokenizer.ggml.merges"), strs(tok.gguf_merges()));
            assert_eq!(
                u("tokenizer.ggml.token_type"),
                Some(crate::gguf::NumArray::I32(tok.gguf_token_types()).into_meta())
            );
            let eos = tok.eos_id().unwrap() as u32;
            assert_eq!(u("tokenizer.ggml.bos_token_id"), Some(MetaValue::U32(eos)));
            assert_eq!(u("tokenizer.ggml.eos_token_id"), Some(MetaValue::U32(eos)));
            assert_eq!(
                u("tokenizer.ggml.add_bos_token"),
                Some(MetaValue::Bool(false))
            );
            assert_eq!(
                u("tokenizer.ggml.add_eos_token"),
                Some(MetaValue::Bool(false))
            );
            assert_eq!(u("general.file_type"), Some(MetaValue::U32(7)));
            assert_eq!(u("general.quantization_version"), Some(MetaValue::U32(2)));
            assert_eq!(u("ferrum.format"), Some(MetaValue::U32(1)));
            assert_eq!(u("ferrum.quant"), s("int8"));
            assert_eq!(u("ferrum.normalizer"), s(""));
            let mut no_tok = meta.clone();
            no_tok.tokenizer_state.clear();
            assert_eq!(u("ferrum.metadata"), s(&no_tok.to_json()));

            let mut want: Vec<(String, Vec<u64>, u32)> = vec![
                ("token_embd.weight".into(), vec![d, v], GGML_Q8_0),
                ("position_embd.weight".into(), vec![d, ctx], GGML_F32),
            ];
            for i in 0..2 {
                let b = |x: &str| format!("blk.{i}.{x}");
                want.extend([
                    (b("attn_norm.weight"), vec![d], GGML_F32),
                    (b("attn_norm.bias"), vec![d], GGML_F32),
                    (b("attn_qkv.weight"), vec![d, 3 * d], GGML_Q8_0),
                    (b("attn_qkv.bias"), vec![3 * d], GGML_F32),
                    (b("attn_output.weight"), vec![d, d], GGML_Q8_0),
                    (b("attn_output.bias"), vec![d], GGML_F32),
                    (b("ffn_norm.weight"), vec![d], GGML_F32),
                    (b("ffn_norm.bias"), vec![d], GGML_F32),
                    (b("ffn_up.weight"), vec![d, ff], GGML_Q8_0),
                    (b("ffn_up.bias"), vec![ff], GGML_F32),
                    (b("ffn_down.weight"), vec![ff, d], GGML_Q8_0),
                    (b("ffn_down.bias"), vec![d], GGML_F32),
                ]);
            }
            want.push(("output_norm.weight".into(), vec![d], GGML_F32));
            want.push(("output_norm.bias".into(), vec![d], GGML_F32));
            if !tie {
                want.push(("output.weight".into(), vec![d, v], GGML_Q8_0));
            }
            let got: Vec<(String, Vec<u64>, u32)> = g
                .tensors
                .iter()
                .map(|t| (t.name.clone(), t.dims.clone(), t.ggml_type))
                .collect();
            assert_eq!(got, want, "tie={tie}");

            // Q|K|V rows of attn_qkv are the transposed q/k/v projections.
            let f32_bytes = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
            let gf = parse(&f32_bytes);
            let qkv = gf
                .dequantize(gf.tensor("blk.1.attn_qkv.weight").unwrap())
                .unwrap();
            let tb = m.layers()[2]
                .as_any()
                .downcast_ref::<TransformerBlock>()
                .unwrap();
            let dd = 64 * 64;
            for (j, p) in [&tb.q_proj, &tb.k_proj, &tb.v_proj].iter().enumerate() {
                let t = crate::gguf::transpose_2d(&p.weight.data, 64, 64);
                assert_eq!(&qkv[j * dd..(j + 1) * dd], &t[..]);
            }
            let (m2, _, meta2) = from_gguf(&f32_bytes).unwrap();
            assert_eq!(fwd(&m, &meta), fwd(&m2, &meta2), "f32 exact, tie={tie}");
            // The reloaded head is bias-free, so the parameter count survives;
            // a tied file omits `output.weight`, so its head is not counted
            // again (the fixture `m` carries an untied head Linear).
            let shared = if tie { (v * d) as usize } else { 0 };
            assert_eq!(m2.num_params(), m.num_params() - shared, "tie={tie}");
        }
    }

    #[test]
    fn gpt2_codec_needs_the_exact_bpe_slm_shape() {
        let arch_of = |layers: Layers, meta: &ModelMetadata| {
            arch(&to_gguf(&seq(layers), &empty_norm(), meta, QPrec::F32).unwrap())
        };
        let (_, meta) = bpe_layers(32, 64, true);
        let v = meta.output_dim;
        let base = || bpe_layers(32, 64, true).0;
        assert_eq!(arch_of(base(), &meta), "gpt2");

        let mut cls = meta.clone();
        cls.task = TaskType::Classification;
        assert_eq!(arch_of(base(), &cls), "ferrum", "not an SLM");
        let mut chars = meta.clone();
        chars.tokenizer_state.clear();
        assert_eq!(arch_of(base(), &chars), "ferrum", "char-level");

        let ln = || -> Box<dyn crate::layer::Layer> {
            Box::new(LayerNorm::new(32, vals(32, 1), vals(32, 2)).unwrap())
        };
        let variants: Vec<(&str, LayerEdit)> = vec![
            (
                "too few layers",
                Box::new(|l: &mut Layers| {
                    l.remove(1);
                    l.remove(1);
                }),
            ),
            (
                "first is not Embedding",
                Box::new(move |l: &mut Layers| l[0] = ln()),
            ),
            (
                "middle is not a block",
                Box::new(move |l: &mut Layers| l[2] = ln()),
            ),
            (
                "no final LayerNorm",
                Box::new(|l: &mut Layers| l[3] = block(8, 32, 64, 50)),
            ),
            (
                "head is not Linear",
                Box::new(move |l: &mut Layers| l[4] = ln()),
            ),
            (
                "last is not an activation",
                Box::new(move |l: &mut Layers| l[5] = ln()),
            ),
            (
                "last is not Softmax",
                Box::new(|l: &mut Layers| l[5] = Box::new(ActivationLayer::new(Activation::ReLU))),
            ),
            (
                "head has a bias",
                Box::new(move |l: &mut Layers| {
                    l[4] = Box::new(Linear::new(32, v, vals(32 * v, 7), vals(v, 8)).unwrap())
                }),
            ),
            (
                "head width is not the vocab",
                Box::new(move |l: &mut Layers| {
                    l[4] = Box::new(Linear::new(32, 3, vals(96, 7), vec![0.0; 3]).unwrap())
                }),
            ),
            (
                "blocks differ",
                Box::new(|l: &mut Layers| l[2] = block(8, 32, 96, 50)),
            ),
            (
                "block context differs",
                Box::new(|l: &mut Layers| {
                    l[2] = Box::new(
                        TransformerBlock::from_parts(
                            4,
                            4,
                            32,
                            LayerNorm::new(32, vals(32, 1), vals(32, 2)).unwrap(),
                            Linear::new(32, 32, vals(1024, 3), vals(32, 4)).unwrap(),
                            Linear::new(32, 32, vals(1024, 5), vals(32, 6)).unwrap(),
                            Linear::new(32, 32, vals(1024, 7), vals(32, 8)).unwrap(),
                            Linear::new(32, 32, vals(1024, 9), vals(32, 10)).unwrap(),
                            LayerNorm::new(32, vals(32, 11), vals(32, 12)).unwrap(),
                            Linear::new(32, 64, vals(2048, 13), vals(64, 14)).unwrap(),
                            Linear::new(64, 32, vals(2048, 15), vals(32, 16)).unwrap(),
                        )
                        .unwrap(),
                    )
                }),
            ),
        ];
        for (why, edit) in variants {
            let mut l = base();
            edit(&mut l);
            let bytes = to_gguf(&seq(l), &empty_norm(), &meta, QPrec::F32).unwrap();
            assert_eq!(arch(&bytes), "ferrum", "{why}");
            from_gguf(&bytes).expect(why);
        }
    }

    #[test]
    fn gpt2_without_eos_is_refused() {
        // A tokenizer with merges but no special tokens.
        let (layers, mut meta) = bpe_layers(32, 64, true);
        let tok = ByteBpeTokenizer::from_state(&meta.tokenizer_state).unwrap();
        let plain = tok.encode_state().split('\n').next().unwrap().to_string();
        let plain_tok = ByteBpeTokenizer::from_state(&plain).unwrap();
        assert_eq!(plain_tok.eos_id(), None);
        // Model vocab must match: drop the trailing special rows.
        let v = plain_tok.vocab_size();
        let emb = layers[0].as_any().downcast_ref::<Embedding>().unwrap();
        let d = 32;
        let token = emb.token_weight.data[..v * d].to_vec();
        let mut l = layers;
        l[0] = Box::new(Embedding::new(v, 8, d, token.clone(), vals(8 * d, 2)).unwrap());
        l[4] = Box::new(
            Linear::new(d, v, crate::gguf::transpose_2d(&token, v, d), vec![0.0; v]).unwrap(),
        );
        meta.tokenizer_state = plain;
        meta.output_dim = v;
        let model = seq(l);
        let err = to_gguf(&model, &empty_norm(), &meta, QPrec::F32).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("<|endoftext|>") && msg.contains("token 11"),
            "{msg}"
        );
    }

    #[test]
    fn foreign_vocab_order_is_remapped_onto_ferrum_ids() {
        // Simulate a foreign GPT-2 file: same tokens, but the 256 byte tokens
        // rotated by one (a permutation that is not its own inverse). The loader must map them
        // back and permute the embedding rows to match.
        let (m, n, meta) = bpe_parts(32, 64, false);
        let bytes = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let d = 32;
        let foreign = rebuild(&bytes, |kv, ts| {
            if let Some(MetaValue::Array(tokens)) = kv.get_mut("tokenizer.ggml.tokens") {
                tokens[..256].rotate_left(1);
            }
            for (name, _, data) in ts.iter_mut() {
                if name == "token_embd.weight" || name == "output.weight" {
                    let mut rows: Vec<Vec<f32>> = data.chunks(d).map(|r| r.to_vec()).collect();
                    rows[..256].rotate_left(1);
                    *data = rows.concat();
                }
            }
            kv.remove("ferrum.metadata");
            kv.remove("ferrum.normalizer");
            kv.remove("general.name");
        });
        let (m2, n2, meta2) = from_gguf(&foreign).unwrap();
        assert_eq!(fwd(&m, &meta), fwd(&m2, &meta2));
        assert_eq!(meta2.tokenizer_state, meta.tokenizer_state);
        // Without ferrum.metadata the metadata is rebuilt from gpt2.* keys.
        assert_eq!(meta2.task, TaskType::TransformerSLM);
        assert_eq!(meta2.input_dim, 8);
        assert_eq!(meta2.output_dim, meta.output_dim);
        assert_eq!(meta2.dataset_name, "gpt2");
        assert_eq!(meta2.feature_names.len(), 8);
        assert_eq!(n2.encode(), "");
        // ...and gguf_quant falls back to general.file_type.
        assert_eq!(gguf_quant(&foreign).unwrap(), "f32");
    }

    /// Round 3, finding 2: a foreign gpt2 file with `embedding_length = 0`,
    /// a 2^40 context and zero-size `[0, n]` tensors used to load and then
    /// allocate feature names / ranges by the context (TBs from ~2 KB). Such
    /// tensors are illegal now (only 1-D `[0]` may be empty), so it fails at
    /// parse, fast.
    #[test]
    fn zero_width_gpt2_with_huge_context_fails_fast() {
        let (m, n, meta) = bpe_parts(32, 64, true);
        let gpt2 = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let v = parse(&gpt2).tensor("token_embd.weight").unwrap().dims[1];
        let bad = rebuild(&gpt2, |kv, ts| {
            kv.insert("gpt2.embedding_length".into(), MetaValue::U32(0));
            kv.insert("gpt2.block_count".into(), MetaValue::U32(0));
            kv.insert("gpt2.context_length".into(), MetaValue::U64(1 << 40));
            kv.remove("ferrum.metadata");
            *ts = vec![
                ("token_embd.weight".into(), vec![0, v], vec![]),
                ("position_embd.weight".into(), vec![0, 1 << 40], vec![]),
                ("output_norm.weight".into(), vec![0], vec![]),
                ("output_norm.bias".into(), vec![0], vec![]),
            ];
        });
        let t = std::time::Instant::now();
        let e = load_err(&bad);
        assert!(e.contains("only a 1-D [0]"), "{e}");
        assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
    }

    /// Load `bytes` as a GPT-2 GGUF and greedily continue `prompt` by `chars` characters.
    fn greedy_continuation(bytes: &[u8], prompt: &str, chars: usize) -> String {
        let (m, _, meta) = from_gguf(bytes).unwrap();
        let slm = GenerativeSLM::new(m, empty_norm(), meta);
        let greedy = crate::slm::SamplingParams {
            top_k: 1,
            ..crate::slm::SamplingParams::with_temperature(1.0)
        };
        slm.generate_continuation_with(prompt, chars, &greedy, &mut crate::rng::Rng::new(0))
            .unwrap()
    }

    #[test]
    fn greedy_continuation_is_deterministic_on_a_ferrum_written_gpt2_file() {
        // Covers the helper of the ignored foreign-file test below on a file we can always build.
        let (m, n, meta) = bpe_parts(32, 64, true);
        let bytes = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let a = greedy_continuation(&bytes, "ab", 6);
        assert_eq!(a.chars().count(), 6);
        assert_eq!(a, greedy_continuation(&bytes, "ab", 6));
    }

    /// Load the GPT-2 GGUF at `path` and check that greedy continuation of `prompt` is `want`.
    fn check_foreign(path: &str, prompt: &str, want: &str) -> Result<(), String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let got = greedy_continuation(&bytes, prompt, want.chars().count());
        if got == want {
            Ok(())
        } else {
            Err(format!("{path}: got {got:?}, want {want:?}"))
        }
    }

    #[test]
    fn check_foreign_reports_a_missing_file_a_mismatch_and_a_match() {
        let missing = check_foreign("/nonexistent/gpt2.gguf", "ab", "x").unwrap_err();
        assert!(missing.contains("/nonexistent/gpt2.gguf"), "{missing}");
        let (m, n, meta) = bpe_parts(32, 64, true);
        let bytes = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let path = std::env::temp_dir().join(format!("ferrum_foreign_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let path = path.to_str().unwrap();
        let want = greedy_continuation(&bytes, "ab", 6);
        assert_eq!(check_foreign(path, "ab", &want), Ok(()));
        let e = check_foreign(path, "ab", "not what it says").unwrap_err();
        assert!(e.contains("want \"not what it says\""), "{e}");
        let _ = std::fs::remove_file(path);
    }

    /// The real GPT-2 124M f32 GGUF lives under `.ferrum-dev/` (not in the repo). The test takes
    /// one branch on every machine and never skips: when the file is there it must load and
    /// continue exactly as llama.cpp and HF do greedily (FACTS.md), and when it is not the check
    /// must fail with the read error.
    #[test]
    fn loads_foreign_gpt2_gguf_when_present() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../.ferrum-dev/gpt2gguf/gpt2-f32.gguf"
        );
        let r = check_foreign(path, "Hello, my name is", " John. I'm a writer");
        assert_eq!(r.is_ok(), std::path::Path::new(path).exists(), "{r:?}");
    }

    #[test]
    fn a_gpt2_file_without_its_final_layernorm_is_refused() {
        let (m, n, meta) = bpe_parts(32, 64, true);
        let bytes = to_gguf(&m, &n, &meta, QPrec::F32).unwrap();
        let bad = rebuild(&bytes, |_, ts| {
            ts.retain(|(name, _, _)| name != "output_norm.weight")
        });
        let e = load_err(&bad);
        assert!(e.contains("output_norm.weight"), "{e}");
    }
}
