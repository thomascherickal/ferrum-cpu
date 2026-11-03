//! Llama-family import: metadata to LlamaConfig, tensors to LlamaModel.
use crate::error::Result;
use crate::gguf::cursor::fmt;
use crate::gguf::header::bounded;
use crate::gguf::Gguf;
use crate::layer::Linear;
use crate::llm::{
    Attention, FeedForward, LlamaBlock, LlamaConfig, LlamaModel, RmsNorm, RopeScaling, RopeType,
};
use crate::quant::{QKind, QWeight};

/// A `[n_in, n_out]` [`Linear`] from f32 weights, packed to `prec` when given.
fn make_linear(
    n_in: usize,
    n_out: usize,
    w: Vec<f32>,
    bias: Vec<f32>,
    prec: Option<QKind>,
) -> Result<Linear> {
    match prec {
        Some(kind) => {
            Linear::quantized(n_in, n_out, QWeight::from_f32(&w, n_in, n_out, kind), bias)
        }
        None => Linear::new(n_in, n_out, w, bias),
    }
}

impl Gguf {
    fn meta_usize(&self, key: &str) -> Result<usize> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_usize())
            .ok_or_else(|| fmt(&format!("GGUF missing/invalid metadata '{key}'")))
    }

    fn meta_opt_usize(&self, key: &str) -> Option<usize> {
        self.metadata.get(key).and_then(|v| v.as_usize())
    }

    fn meta_f32_or(&self, key: &str, default: f32) -> f32 {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f32())
            .unwrap_or(default)
    }

    /// RoPE scaling as llama.cpp reads it: the `rope_freqs.weight` factors (Llama 3.x) and
    /// `linear` scaling (`{arch}.rope.scaling.factor`, legacy `{arch}.rope.scale_linear`; a
    /// factor with no type is linear, as in llama.cpp). Every other type (`yarn`, `longrope`,
    /// ...) and a non-1 `attn_factor` are refused: running them unscaled would be silently wrong.
    fn rope_scaling(&self, arch: &str, rope_dim: usize) -> Result<RopeScaling> {
        let key = |k: &str| format!("{arch}.rope.{k}");
        let factor = self
            .metadata
            .get(&key("scaling.factor"))
            .or_else(|| self.metadata.get(&key("scale_linear")))
            .and_then(|v| v.as_f32());
        let ty = match self.metadata.get(&key("scaling.type")) {
            Some(v) => v
                .as_str()
                .ok_or_else(|| fmt(&format!("'{}' is not a string", key("scaling.type"))))?,
            None => "linear",
        };
        // llama.cpp: a factor of 0 (or none) means no scaling.
        let factor = factor.filter(|&f| f != 0.0).unwrap_or(1.0);
        let linear_factor = match ty {
            "linear" => factor,
            "none" if factor == 1.0 => 1.0,
            "none" => {
                return Err(fmt(&format!(
                    "'{}' is \"none\" but the scaling factor is {factor}",
                    key("scaling.type")
                )))
            }
            other => {
                return Err(fmt(&format!(
                    "RoPE scaling type '{other}' is not supported: Ferrum implements \
                     none, linear and rope_freqs.weight (yarn and longrope are refused)"
                )))
            }
        };
        let attn = self.meta_f32_or(&key("scaling.attn_factor"), 1.0);
        if attn != 1.0 {
            return Err(fmt(&format!(
                "RoPE scaling attn_factor {attn} is not supported (only 1.0)"
            )));
        }
        let freq_factors = match self.tensor("rope_freqs.weight") {
            Some(t) if t.dims != [(rope_dim / 2) as u64] => {
                return Err(fmt(&format!(
                    "rope_freqs.weight has dims {:?}, expected [{}] (rope_dim / 2)",
                    t.dims,
                    rope_dim / 2
                )))
            }
            Some(t) => self.dequantize(t)?,
            None => Vec::new(),
        };
        Ok(RopeScaling {
            freq_factors,
            linear_factor,
        })
    }

    fn dequant_named(&self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensor(name)
            .ok_or_else(|| fmt(&format!("GGUF tensor '{name}' not found")))?;
        self.dequantize(t)
    }

    /// Import a 2-D weight tensor as a Ferrum [`Linear`], transposing GGUF's
    /// `[n_out, n_in]` storage into Ferrum's `[n_in, n_out]`. With `prec =
    /// Some(kind)` the matrix is packed to in-memory int4/int8; with `prec =
    /// None` it is kept full **f32** — the precision-preserving import (G-Q) that
    /// avoids re-quantizing an already-quantized GGUF onto Ferrum's coarser
    /// per-row grid, at the cost of resident RAM. An optional bias is loaded if
    /// present. The tensor must be exactly `[n_in, n_out]` (the shape the
    /// metadata implies), checked before anything is allocated, which ties
    /// every metadata dim the model later sizes buffers by to real weights.
    fn linear_from(
        &self,
        name: &str,
        bias_name: Option<&str>,
        prec: Option<QKind>,
        n_in: usize,
        n_out: usize,
    ) -> Result<Linear> {
        let t = self
            .tensor(name)
            .ok_or_else(|| fmt(&format!("GGUF tensor '{name}' not found")))?
            .clone();
        if t.dims != [n_in as u64, n_out as u64] {
            return Err(fmt(&format!(
                "tensor '{name}' has dims {:?}, expected [{n_in}, {n_out}] from the metadata",
                t.dims
            )));
        }
        let flat = self.dequantize(&t)?; // row-major [n_out, n_in]
        let w = transpose_2d(&flat, n_out, n_in); // → [n_in, n_out]
        let bias = bias_name.and_then(|b| self.tensor(b)).cloned();
        let has_bias = bias.is_some();
        let bias = match bias {
            Some(bt) => self.dequantize(&bt)?,
            None => vec![0.0; n_out],
        };
        if bias.len() != n_out {
            return Err(fmt(&format!("bias for '{name}' has wrong length")));
        }
        let lin = make_linear(n_in, n_out, w, bias, prec)?;
        // No bias tensor in the file: the zero bias is a placeholder, never a parameter.
        Ok(if has_bias { lin } else { lin.into_bias_free() })
    }

    /// Build a runnable [`LlamaModel`] from a `llama`- or `qwen2`-architecture
    /// GGUF, packing weight matrices to `kind` (int4/int8) in memory.
    ///
    /// RoPE is [`RopeType::Norm`] for llama (llama.cpp's conversion permutes Q/K
    /// for it) and [`RopeType::Neox`] for qwen2 (as in llama.cpp). Other architectures are rejected. Bit-exact parity with
    /// llama.cpp on a real checkpoint is not asserted (it needs the actual file),
    /// but the metadata/tensor mapping, the transpose, and the dequant are all
    /// unit-covered, and the resulting model's cached decode matches its own full
    /// forward.
    pub fn load_llama(&self, kind: QKind) -> Result<LlamaModel> {
        self.load_llama_prec(Some(kind))
    }

    /// Like [`Self::load_llama`] but with explicit import precision: `Some(kind)`
    /// packs weights to in-memory int4/int8 (smaller, doubly-quantized);
    /// `None` keeps them full f32 (G-Q: no second quantization, larger RAM).
    pub fn load_llama_prec(&self, prec: Option<QKind>) -> Result<LlamaModel> {
        let arch = self
            .architecture()
            .ok_or_else(|| fmt("GGUF missing general.architecture"))?
            .to_string();
        if arch != "llama" && arch != "qwen2" {
            return Err(fmt(&format!(
                "architecture '{arch}' is not supported by load_llama (llama / qwen2 only)"
            )));
        }
        // llama.cpp: qwen2 is a NEOX-rope arch (its Q/K are not permuted at conversion).
        let rope_type = if arch == "qwen2" {
            RopeType::Neox
        } else {
            RopeType::Norm
        };
        let dim = self.meta_usize(&format!("{arch}.embedding_length"))?;
        // Each layer needs nine tensors, so the tensor count bounds the
        // layer count before it sizes `blocks`.
        let n_layers = bounded(
            self.meta_usize(&format!("{arch}.block_count"))?,
            self.tensors.len(),
            "block_count",
        )?;
        let n_heads = self.meta_usize(&format!("{arch}.attention.head_count"))?;
        let n_kv = self
            .meta_opt_usize(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads);
        let ffn = self.meta_usize(&format!("{arch}.feed_forward_length"))?;
        let head_dim = self
            .meta_opt_usize(&format!("{arch}.attention.key_length"))
            .unwrap_or(dim / n_heads.max(1));
        let rope_dim = self
            .meta_opt_usize(&format!("{arch}.rope.dimension_count"))
            .unwrap_or(head_dim);
        let rope_base = self.meta_f32_or(&format!("{arch}.rope.freq_base"), 10000.0);
        let rope_scaling = self.rope_scaling(&arch, rope_dim)?;
        let eps = self.meta_f32_or(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5);
        let ctx = self
            .meta_opt_usize(&format!("{arch}.context_length"))
            .unwrap_or(2048);

        // Token embedding: GGUF `[dim, vocab]` (ne) → row-major `[vocab, dim]`,
        // which is exactly the lookup layout LlamaModel wants. Kept f32.
        let emb_t = self
            .tensor("token_embd.weight")
            .ok_or_else(|| fmt("GGUF missing token_embd.weight"))?;
        let vocab = if emb_t.dims.len() == 2 {
            emb_t.dims[1] as usize
        } else {
            self.meta_usize(&format!("{arch}.vocab_size"))?
        };
        // The (file-bounded) embedding must be exactly [dim, vocab]: this ties
        // both to real data before the tied LM head allocates by them.
        let emb_ok = match emb_t.dims[..] {
            [d, v] => d == dim as u64 && v == vocab as u64,
            _ => vocab.checked_mul(dim) == Some(emb_t.num_elements()),
        };
        if !emb_ok || vocab == 0 || dim == 0 {
            return Err(fmt("token_embd.weight size does not match vocab × dim"));
        }
        let tok_emb = self.dequantize(emb_t)?;
        // Every other metadata dim must match the weights it describes (checked
        // in `linear_from` / `norm` before allocating), so the buffers the
        // model later sizes by them are bounded by the file too.
        let mul = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| fmt("GGUF attention dims overflow"))
        };
        let (q_dim, kv_dim) = (mul(n_heads, head_dim)?, mul(n_kv, head_dim)?);
        let norm = |name: &str| -> Result<RmsNorm> {
            let w = self.dequant_named(name)?;
            if w.len() != dim {
                return Err(fmt(&format!("tensor '{name}' length is not {dim}")));
            }
            Ok(RmsNorm::new(w, eps))
        };

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            // Ferrum loads biases only for q/k/v (the qwen2 convention). A file
            // carrying a bias on any other projection would run silently wrong
            // with that bias dropped — refuse loudly instead.
            for stem in ["attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
                let bname = format!("{p}.{stem}.bias");
                if self.tensor(&bname).is_some() {
                    return Err(fmt(&format!(
                        "GGUF tensor '{bname}' is not supported: Ferrum loads biases \
                         only for attn_q/attn_k/attn_v"
                    )));
                }
            }
            let attn_norm = norm(&format!("{p}.attn_norm.weight"))?;
            // Qwen2 carries q/k/v biases; Llama does not (loaded only if present).
            let qkv = |s: &str, n_out: usize| {
                self.linear_from(
                    &format!("{p}.attn_{s}.weight"),
                    Some(&format!("{p}.attn_{s}.bias")),
                    prec,
                    dim,
                    n_out,
                )
            };
            let (wq, wk, wv) = (qkv("q", q_dim)?, qkv("k", kv_dim)?, qkv("v", kv_dim)?);
            let wo =
                self.linear_from(&format!("{p}.attn_output.weight"), None, prec, q_dim, dim)?;
            let attn = Attention::new(
                wq, wk, wv, wo, n_heads, n_kv, head_dim, rope_dim, rope_base, rope_type,
            )?;
            let ffn_norm = norm(&format!("{p}.ffn_norm.weight"))?;
            let lin = |s: &str, n_in: usize, n_out: usize| {
                self.linear_from(&format!("{p}.{s}.weight"), None, prec, n_in, n_out)
            };
            let gate = lin("ffn_gate", dim, ffn)?;
            let up = lin("ffn_up", dim, ffn)?;
            let down = lin("ffn_down", ffn, dim)?;
            blocks.push(LlamaBlock {
                attn_norm,
                attn,
                ffn_norm,
                ffn: FeedForward::new(gate, up, down),
            });
        }

        let final_norm = norm("output_norm.weight")?;
        // Same rule for the LM head: a bias there is never loaded, so refuse it.
        if self.tensor("output.bias").is_some() {
            return Err(fmt(
                "GGUF tensor 'output.bias' is not supported: Ferrum loads biases \
                 only for attn_q/attn_k/attn_v",
            ));
        }
        // LM head: explicit `output.weight`, or tied to the token embedding.
        let tied_head = self.tensor("output.weight").is_none();
        let lm_head = if !tied_head {
            self.linear_from("output.weight", None, prec, dim, vocab)?
        } else {
            let w = transpose_2d(&tok_emb, vocab, dim); // [vocab,dim] → [dim,vocab]
            make_linear(dim, vocab, w, vec![0.0; vocab], prec)?.into_bias_free()
        };

        let cfg = LlamaConfig {
            vocab_size: vocab,
            model_dim: dim,
            n_layers,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            ffn_dim: ffn,
            rope_dim,
            rope_base,
            rope_type,
            rope_scaling: RopeScaling::default(),
            norm_eps: eps,
            context_len: ctx,
        };
        let mut model = LlamaModel {
            cfg,
            tok_emb,
            blocks,
            final_norm,
            lm_head,
            tied_head,
        };
        model.set_rope_scaling(rope_scaling)?;
        Ok(model)
    }
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
pub(crate) fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::transpose_2d;
    use crate::gguf::format::{align_up, DEFAULT_ALIGNMENT, GGML_F32, GGUF_MAGIC};
    use crate::gguf::test_support::{
        kv_str, l_file, l_tensors, llama_err, llama_file, parse_and_open, put_u32, put_u64,
        synth_llama, synth_llama_cfg, TBuf, L_META,
    };
    use crate::gguf::{Gguf, MetaValue};
    use crate::llm::RopeType;
    use crate::quant::QKind;

    #[test]
    fn transpose_2d_is_correct() {
        // [[1,2,3],[4,5,6]] (2×3) → [[1,4],[2,5],[3,6]] (3×2)
        let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(transpose_2d(&m, 2, 3), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn loads_synthetic_llama_and_runs() {
        let g = Gguf::parse(synth_llama()).unwrap();
        assert_eq!(g.architecture(), Some("llama"));
        let model = g.load_llama(QKind::Int8).unwrap();
        assert_eq!(model.cfg.vocab_size, 10);
        assert_eq!(model.cfg.model_dim, 8);
        assert_eq!(model.blocks.len(), 2);

        // Full forward produces finite [seq, vocab] logits.
        let tokens = [1usize, 4, 2, 7, 0];
        let logits = model.forward_tokens(&tokens).unwrap();
        assert_eq!(logits.shape, vec![tokens.len(), model.cfg.vocab_size]);
        assert!(logits.data.iter().all(|v| v.is_finite()));

        // The imported model's KV-cached decode matches its own full forward.
        let mut cache = crate::llm::LlamaCache::new(model.blocks.len());
        let (_, vocab) = logits.matrix_dims().unwrap();
        for (ti, &tok) in tokens.iter().enumerate() {
            let step = model.forward_one(tok, &mut cache).unwrap();
            for (v, &s) in step.iter().enumerate() {
                assert!(
                    (s - logits.data[ti * vocab + v]).abs() < 1e-2,
                    "imported model cached vs full mismatch at {ti},{v}"
                );
            }
        }

        // Generation runs end to end.
        let out = model
            .generate(
                &[1, 2],
                6,
                &crate::llm::SamplingParams::with_temperature(0.8),
                &[],
                &mut crate::rng::Rng::new(1),
            )
            .unwrap();
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|&t| t < model.cfg.vocab_size));
    }

    #[test]
    fn load_llama_without_a_token_embedding_is_refused() {
        llama_err(
            &l_file("token_embd.weight", None, &[]),
            "missing token_embd.weight",
        );
    }

    #[test]
    fn load_llama_rejects_a_rope_dimension_wider_than_the_head() {
        // dim 2 over 2 heads gives head_dim 1; a rope dimension of 2 cannot fit in it.
        let g = l_file(
            "",
            None,
            &[("attention.head_count", 2), ("rope.dimension_count", 2)],
        );
        llama_err(&g, "rope_dim");
    }

    #[test]
    fn linear_from_applies_transpose_so_y_equals_w_times_x() {
        // A single GGUF weight [n_out, n_in]; the imported Ferrum Linear must
        // compute y = W·x (so the transpose is right). int8 keeps error tiny.
        let (n_in, n_out) = (4usize, 3usize);
        // GGUF row-major [n_out, n_in].
        let w_gguf: Vec<f32> = (0..n_out * n_in).map(|i| (i as f32 * 0.1) - 0.5).collect();
        let mut t = TBuf::new();
        t.add("w", &[n_in as u64, n_out as u64], &w_gguf);
        let mut meta = Vec::new();
        kv_str(&mut meta, "general.architecture", "llama");
        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 1);
        put_u64(&mut bytes, 1);
        bytes.extend_from_slice(&meta);
        bytes.extend_from_slice(&t.infos);
        let pad = align_up(bytes.len(), DEFAULT_ALIGNMENT as usize) - bytes.len();
        bytes.extend(std::iter::repeat_n(0u8, pad));
        bytes.extend_from_slice(&t.data);

        let g = Gguf::parse(bytes).unwrap();
        let lin = g
            .linear_from("w", None, Some(QKind::Int8), n_in, n_out)
            .unwrap();
        use crate::layer::Layer;
        let x = vec![0.3f32, -0.7, 0.2, 0.9];
        let y = lin
            .forward(&crate::tensor::Tensor::matrix(1, n_in, x.clone()).unwrap())
            .unwrap();
        // Reference y[o] = Σ_i w_gguf[o*n_in + i] · x[i].
        for o in 0..n_out {
            let want: f32 = (0..n_in).map(|i| w_gguf[o * n_in + i] * x[i]).sum();
            let got = y.data[o];
            assert!((got - want).abs() < 0.05, "y[{o}]={got} want {want}");
        }
    }

    #[test]
    fn loads_tied_embedding_llama() {
        // No output.weight → the LM head is tied to the token embedding.
        let g = Gguf::parse(synth_llama_cfg("llama", false, false)).unwrap();
        assert!(g.tensor("output.weight").is_none());
        let model = g.load_llama(QKind::Int8).unwrap();
        let logits = model.forward_tokens(&[1, 2, 3]).unwrap();
        assert_eq!(logits.shape, vec![3, model.cfg.vocab_size]);
        assert!(logits.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn loads_qwen2_with_attention_biases() {
        let g = Gguf::parse(synth_llama_cfg("qwen2", true, true)).unwrap();
        assert_eq!(g.architecture(), Some("qwen2"));
        let model = g.load_llama(QKind::Int4).unwrap();
        let logits = model.forward_tokens(&[0, 5, 9]).unwrap();
        assert!(logits.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn qwen2_uses_neox_rope_and_llama_uses_norm() {
        // llama.cpp: LLM_ARCH_QWEN2 -> LLAMA_ROPE_TYPE_NEOX, LLM_ARCH_LLAMA -> NORM.
        let q = Gguf::parse(synth_llama_cfg("qwen2", false, false)).unwrap();
        let l = Gguf::parse(synth_llama_cfg("llama", false, false)).unwrap();
        let (mq, ml) = (
            q.load_llama(QKind::Int8).unwrap(),
            l.load_llama(QKind::Int8).unwrap(),
        );
        assert_eq!(mq.cfg.rope_type, RopeType::Neox);
        assert_eq!(ml.cfg.rope_type, RopeType::Norm);
        // and the attention actually rotates that way: same weights, different logits
        let (a, b) = (
            mq.forward_tokens(&[0, 5, 9]).unwrap(),
            ml.forward_tokens(&[0, 5, 9]).unwrap(),
        );
        assert_ne!(a.data, b.data);
    }

    /// synth_llama (head_dim = rope_dim = 4) with extra `llama.rope.*` metadata.
    fn with_rope_meta(kv: &[(&str, MetaValue)]) -> Gguf {
        let mut g = Gguf::parse(synth_llama()).unwrap();
        for (k, v) in kv {
            g.metadata.insert(format!("llama.rope.{k}"), v.clone());
        }
        g
    }

    #[test]
    fn rope_scaling_metadata_sets_the_linear_factor_as_llama_cpp_reads_it() {
        use crate::llm::RopeScaling;
        let lin = |kv: &[(&str, MetaValue)]| {
            let m = with_rope_meta(kv).load_llama_prec(None).unwrap();
            assert!(m.cfg.rope_scaling.freq_factors.is_empty());
            m.cfg.rope_scaling.linear_factor
        };
        let (s, f) = (MetaValue::String, MetaValue::F32);
        // No scaling metadata: unscaled.
        let plain = with_rope_meta(&[]).load_llama_prec(None).unwrap();
        assert_eq!(plain.cfg.rope_scaling, RopeScaling::default());
        assert_eq!(
            lin(&[
                ("scaling.type", s("linear".into())),
                ("scaling.factor", f(4.0))
            ]),
            4.0
        );
        // A factor without a type is linear (llama.cpp's default type); the legacy key too.
        assert_eq!(lin(&[("scaling.factor", f(4.0))]), 4.0);
        assert_eq!(lin(&[("scale_linear", f(2.0))]), 2.0);
        // Factor 0 means none; "none" alone or with factor 1 is unscaled; attn_factor 1 is fine.
        assert_eq!(lin(&[("scaling.factor", f(0.0))]), 1.0);
        assert_eq!(lin(&[("scaling.type", s("none".into()))]), 1.0);
        assert_eq!(
            lin(&[
                ("scaling.type", s("none".into())),
                ("scaling.factor", f(1.0))
            ]),
            1.0
        );
        assert_eq!(lin(&[("scaling.attn_factor", f(1.0))]), 1.0);
        // And the scaling reaches the forward pass.
        let scaled = with_rope_meta(&[("scaling.factor", f(4.0))])
            .load_llama_prec(None)
            .unwrap();
        let toks = [1usize, 4, 2, 7, 0];
        assert_ne!(
            plain.forward_tokens(&toks).unwrap().data,
            scaled.forward_tokens(&toks).unwrap().data
        );
    }

    #[test]
    fn unsupported_rope_scaling_is_refused_never_run_unscaled() {
        let (s, f) = (MetaValue::String, MetaValue::F32);
        for ty in ["yarn", "longrope", "llama3", "dynamic"] {
            let g = with_rope_meta(&[("scaling.type", s(ty.into())), ("scaling.factor", f(4.0))]);
            llama_err(&g, &format!("RoPE scaling type '{ty}' is not supported"));
        }
        llama_err(
            &with_rope_meta(&[("scaling.type", MetaValue::U32(1))]),
            "'llama.rope.scaling.type' is not a string",
        );
        llama_err(
            &with_rope_meta(&[
                ("scaling.type", s("none".into())),
                ("scaling.factor", f(2.0)),
            ]),
            "is \"none\" but the scaling factor is 2",
        );
        llama_err(
            &with_rope_meta(&[("scaling.attn_factor", f(0.5))]),
            "attn_factor 0.5 is not supported",
        );
        llama_err(
            &with_rope_meta(&[("scaling.factor", f(-2.0))]),
            "finite and > 0",
        );
    }

    #[test]
    fn rope_freqs_of_the_wrong_length_is_refused() {
        // rope_dim 4 needs 2 factors. The check fires on the directory entry, before any data.
        let mut g = Gguf::parse(synth_llama()).unwrap();
        g.tensors.push(crate::gguf::TensorInfo {
            name: "rope_freqs.weight".into(),
            dims: vec![3],
            ggml_type: GGML_F32,
            offset: 0,
        });
        llama_err(&g, "rope_freqs.weight has dims [3], expected [2]");
    }

    #[test]
    fn load_llama_errors_on_missing_metadata_and_tensors() {
        // arch=llama but no shape metadata and no tensors.
        let mut bytes = Vec::new();
        put_u32(&mut bytes, GGUF_MAGIC);
        put_u32(&mut bytes, 3);
        put_u64(&mut bytes, 0); // 0 tensors
        put_u64(&mut bytes, 1); // 1 metadata kv
        kv_str(&mut bytes, "general.architecture", "llama");
        // Pad to the alignment boundary so the (empty) data section starts in-bounds.
        let pad = align_up(bytes.len(), DEFAULT_ALIGNMENT as usize) - bytes.len();
        bytes.extend(std::iter::repeat_n(0u8, pad));
        let g = Gguf::parse(bytes).unwrap();
        assert!(
            g.load_llama(QKind::Int8).is_err(),
            "missing embedding_length must error"
        );
    }

    #[test]
    fn load_llama_f32_precision_skips_requantization() {
        // prec = None keeps weights f32 (no second quantization on import).
        let g = Gguf::parse(synth_llama()).unwrap();
        let model = g.load_llama_prec(None).unwrap();
        let logits = model.forward_tokens(&[1, 2, 3]).unwrap();
        assert_eq!(logits.shape, vec![3, model.cfg.vocab_size]);
        assert!(logits.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_size_embedding_attack_is_rejected_at_parse() {
        // Finding 1: token_embd [0, 2^40] with no data used to pass the size
        // check (0 == 0) and then size a 4 TiB tied LM head by its vocab.
        let mut b = crate::gguf_write::GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        for (k, v) in [
            ("embedding_length", 0),
            ("block_count", 0),
            ("attention.head_count", 1),
            ("feed_forward_length", 1),
        ] {
            b.meta(&format!("llama.{k}"), MetaValue::U32(v));
        }
        b.tensor("token_embd.weight", &[0, 1 << 40], GGML_F32, vec![]);
        b.tensor("output_norm.weight", &[0], GGML_F32, vec![]);
        let (p, o) = parse_and_open(&b.into_bytes());
        for e in [p.err().unwrap(), o.err().unwrap()] {
            assert!(e.to_string().contains("only a 1-D [0]"), "{e}");
        }
        // linear_from's zero-size [0, 2^40] weight: same rejection.
        let mut b = crate::gguf_write::GgufBuilder::new();
        b.tensor("w", &[0, 1 << 40], GGML_F32, vec![]);
        assert!(Gguf::parse(b.into_bytes()).is_err());
    }

    #[test]
    fn huge_block_count_is_bounded_by_the_tensor_count() {
        // Finding 3: block_count = 2^40 (U64) with one tiny valid tensor used
        // to abort in Vec::with_capacity.
        let g = llama_file(
            &[
                ("embedding_length", 1),
                ("attention.head_count", 1),
                ("feed_forward_length", 1),
            ],
            &[("block_count", 1 << 40)],
            &[("token_embd.weight", &[1, 1])],
        );
        llama_err(&g, "block_count");
    }

    #[test]
    fn metadata_dims_must_match_the_weights_they_describe() {
        // Control: the consistent file loads, tied head and all.
        let m = l_file("", None, &[]).load_llama_prec(None).unwrap();
        assert_eq!(
            (m.cfg.vocab_size, m.cfg.model_dim, m.blocks.len()),
            (3, 2, 1)
        );
        // A head count / key length that does not match attn_q (it would size
        // the q/context buffers at decode time) - including one whose product
        // with head_dim overflows.
        llama_err(
            &l_file("", None, &[("attention.head_count", 1 << 20)]),
            "attn_q",
        );
        llama_err(
            &l_file("", None, &[("attention.key_length", 1 << 20)]),
            "attn_q",
        );
        let ts = l_tensors();
        let ts: Vec<(&str, &[u64])> = ts.iter().map(|(n, d)| (*n, &d[..])).collect();
        let meta: Vec<(&str, u32)> = L_META
            .into_iter()
            .filter(|(k, _)| *k != "attention.head_count")
            .collect();
        let huge = &[
            ("attention.head_count", 1 << 40),
            ("attention.key_length", 1 << 40),
        ];
        llama_err(&llama_file(&meta, huge, &ts), "overflow");
        let huge_kv = &[
            ("attention.head_count_kv", 1 << 40),
            ("attention.key_length", 1 << 40),
        ];
        llama_err(&llama_file(&L_META, huge_kv, &ts), "overflow");
        // Missing block_count; each other projection / norm checked too.
        let no_blocks: Vec<(&str, u32)> = L_META
            .into_iter()
            .filter(|(k, _)| *k != "block_count")
            .collect();
        llama_err(&llama_file(&no_blocks, &[], &ts), "block_count");
        for (name, dims) in [
            ("blk.0.attn_v.weight", Some(vec![2, 3])),
            ("blk.0.attn_output.weight", Some(vec![3, 2])),
            ("blk.0.ffn_up.weight", Some(vec![2, 3])),
            ("blk.0.attn_norm.weight", None),
            ("blk.0.ffn_norm.weight", None),
        ] {
            let short = name.trim_start_matches("blk.0.");
            llama_err(&l_file(name, dims, &[]), short.trim_end_matches(".weight"));
        }
        // An embedding of the right shape that cannot be decoded.
        let mut b = crate::gguf_write::GgufBuilder::new();
        b.meta("general.architecture", MetaValue::String("llama".into()));
        for (k, v) in L_META {
            b.meta(&format!("llama.{k}"), MetaValue::U32(v));
        }
        b.tensor("token_embd.weight", &[2, 3], 99, vec![0; 24]);
        llama_err(
            &Gguf::parse(b.into_bytes()).unwrap(),
            "unsupported for sizing",
        );
        // A KV head count that does not match attn_k.
        llama_err(
            &l_file("", None, &[("attention.head_count_kv", 2)]),
            "attn_k",
        );
        // feed_forward_length not matching the FFN weights.
        llama_err(
            &l_file("", None, &[("feed_forward_length", 1 << 20)]),
            "ffn_gate",
        );
        llama_err(
            &l_file("blk.0.ffn_down.weight", Some(vec![2, 3]), &[]),
            "ffn_down",
        );
        // embedding_length not matching token_embd, or a 2-D [1, 6] reshaping.
        llama_err(
            &l_file("", None, &[("embedding_length", 1 << 20)]),
            "vocab × dim",
        );
        llama_err(
            &l_file("token_embd.weight", Some(vec![1, 6]), &[]),
            "vocab × dim",
        );
        // Norm weights must be `dim` long.
        llama_err(
            &l_file("blk.0.attn_norm.weight", Some(vec![3]), &[]),
            "attn_norm",
        );
        llama_err(
            &l_file("output_norm.weight", Some(vec![1]), &[]),
            "output_norm",
        );
        // An explicit LM head must be [dim, vocab].
        let mut ts = l_tensors();
        ts.push(("output.weight", vec![2, 4]));
        let ts: Vec<(&str, &[u64])> = ts.iter().map(|(n, d)| (*n, &d[..])).collect();
        llama_err(&llama_file(&L_META, &[], &ts), "output.weight");
    }

    #[test]
    fn one_d_embedding_needs_matching_nonzero_vocab_and_dim() {
        let ts = |n: u64| {
            let mut v = l_tensors();
            v[0].1 = vec![n];
            v
        };
        let file = |n: u64, vocab: u32, dim: u32| {
            let t = ts(n);
            let t: Vec<(&str, &[u64])> = t.iter().map(|(n, d)| (*n, &d[..])).collect();
            let mut meta = L_META.to_vec();
            meta[0].1 = dim;
            meta.push(("vocab_size", vocab));
            llama_file(&meta, &[], &t)
        };
        // [6] = 3 × 2 loads; vocab_size from the metadata.
        assert_eq!(
            file(6, 3, 2).load_llama_prec(None).unwrap().cfg.vocab_size,
            3
        );
        // Wrong product, and the empty [0] with dim 0 claiming 2^30 tokens
        // (0 × anything == 0, so only the non-zero check catches it).
        llama_err(&file(6, 4, 2), "vocab × dim");
        llama_err(&file(0, 1 << 30, 0), "vocab × dim");
        // vocab × dim not matching (overflowing usize on 32-bit targets).
        llama_err(&file(6, u32::MAX, u32::MAX), "vocab × dim");
    }

    #[test]
    fn attention_bias_of_the_wrong_length_is_rejected() {
        let mut ts = l_tensors();
        ts.push(("blk.0.attn_q.bias", vec![3])); // dim is 2
        let ts: Vec<(&str, &[u64])> = ts.iter().map(|(n, d)| (*n, &d[..])).collect();
        let g = llama_file(&L_META, &[], &ts);
        llama_err(&g, "wrong length");
    }
}
