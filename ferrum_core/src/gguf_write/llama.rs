//! Serialize a LlamaModel to a GGUF file.
use crate::error::{InferError, Result};
use crate::fs_atomic::atomic_write;
use crate::gguf::{Gguf, MetaValue, GGML_F32};
use crate::gguf_write::blocks::encode_tensor;
use crate::gguf_write::builder::GgufBuilder;
use crate::gguf_write::quant::{plan_tensor_type, GgufQuant};
use crate::layer::Linear;
use crate::llm::LlamaModel;

/// Serialize a `LlamaModel` to GGUF bytes, carrying the source GGUF's metadata
/// (hyperparameters + tokenizer) forward verbatim. See module docs.
pub fn llama_gguf_bytes(model: &LlamaModel, source: &Gguf, quant: GgufQuant) -> Result<Vec<u8>> {
    let arch = source
        .architecture()
        .ok_or_else(|| InferError::Format("source GGUF missing general.architecture".into()))?
        .to_string();
    if arch != "llama" && arch != "qwen2" {
        return Err(InferError::Format(format!(
            "GGUF export supports llama/qwen2 only (source architecture = '{arch}')"
        )));
    }

    let mut b = GgufBuilder::new();

    // 1. Metadata: copy every source key verbatim, except the file-type hints we
    //    refresh to match the new target, and `general.alignment`: the builder
    //    always lays out tensor data at `DEFAULT_ALIGNMENT` (32), so carrying a
    //    source alignment (e.g. 64) forward would leave the output metadata
    //    claiming an alignment the data isn't actually packed at, and the
    //    reader would then compute every tensor offset from the wrong value
    //    (silent corruption). Dropping the key makes the reader fall back to
    //    the same default the builder used.
    for (k, v) in &source.metadata {
        if k == "general.file_type"
            || k == "general.quantization_version"
            || k == "general.alignment"
        {
            continue;
        }
        b.meta(k, v.clone());
    }
    // `general.file_type` / `general.quantization_version` are refreshed after
    // tensor emission, once we know whether the requested type was actually
    // used (block-alignment fallbacks can turn a k-quant request into F16).

    // 2. Tensors.
    let cfg = &model.cfg;
    // token_embd: Ferrum [vocab, dim] is already raw order; dims [dim, vocab].
    add_tensor(
        &mut b,
        "token_embd.weight",
        &model.tok_emb,
        cfg.model_dim, // block axis = fastest dim = model_dim
        &[cfg.model_dim as u64, cfg.vocab_size as u64],
        true,
        quant,
    );

    for (i, blk) in model.blocks.iter().enumerate() {
        let p = format!("blk.{i}");
        add_norm(
            &mut b,
            &format!("{p}.attn_norm.weight"),
            &blk.attn_norm.weight,
        );
        add_linear(&mut b, &p, "attn_q", &blk.attn.wq, source, true, quant);
        add_linear(&mut b, &p, "attn_k", &blk.attn.wk, source, true, quant);
        add_linear(&mut b, &p, "attn_v", &blk.attn.wv, source, true, quant);
        add_linear(
            &mut b,
            &p,
            "attn_output",
            &blk.attn.wo,
            source,
            false,
            quant,
        );
        add_norm(
            &mut b,
            &format!("{p}.ffn_norm.weight"),
            &blk.ffn_norm.weight,
        );
        add_linear(&mut b, &p, "ffn_gate", &blk.ffn.gate, source, false, quant);
        add_linear(&mut b, &p, "ffn_up", &blk.ffn.up, source, false, quant);
        add_linear(&mut b, &p, "ffn_down", &blk.ffn.down, source, false, quant);
    }

    add_norm(&mut b, "output_norm.weight", &model.final_norm.weight);
    // Emit an explicit LM head only if the source had one (else tying is kept).
    if source.tensor("output.weight").is_some() {
        add_weight_2d(&mut b, "output.weight", &model.lm_head, quant);
    }

    // 3. File-type hint, decided from what was *actually* emitted: if every
    //    matrix fell back (no tensor of the requested type exists), stamping
    //    the requested ftype would mislead external tools — stamp the
    //    fallback's instead.
    let used_requested = b.tensors.iter().any(|t| t.ggml_type == quant.ggml_type());
    let effective = if used_requested {
        quant
    } else {
        GgufQuant::F16
    };
    b.meta("general.file_type", MetaValue::U32(effective.file_type()));
    b.meta("general.quantization_version", MetaValue::U32(2));

    Ok(b.into_bytes())
}

/// Write `llama_gguf_bytes` to a file (atomically — see `crate::fs_atomic`).
pub fn write_llama_gguf(
    model: &LlamaModel,
    source: &Gguf,
    quant: GgufQuant,
    path: &str,
) -> Result<()> {
    let bytes = llama_gguf_bytes(model, source, quant)?;
    atomic_write(path, &bytes)
}

/// Transpose row-major `[rows, cols]` → `[cols, rows]`.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// A 1-D norm weight — always F32.
fn add_norm(b: &mut GgufBuilder, name: &str, weight: &[f32]) {
    // encode_tensor(F32) is infallible for any length; unwrap is safe here.
    let bytes = encode_tensor(weight, GGML_F32).unwrap();
    b.tensor(name, &[weight.len() as u64], GGML_F32, bytes);
}

/// A projection weight (+ optional bias) named `{prefix}.{stem}.weight`.
///
/// `can_have_bias` must be `true` only for `attn_q`/`attn_k`/`attn_v`: those are
/// the only stems [`crate::gguf::Gguf::load_llama_prec`] ever loads a bias for
/// (it passes `bias_name: None` when building the `attn_output`/`ffn_*`
/// `Linear`s). For those stems `lin.bias.data` is therefore always a zero-filled
/// placeholder, not source data — emitting it would silently synthesize a bogus
/// all-zeros bias tensor, so callers for those stems must pass `false`.
fn add_linear(
    b: &mut GgufBuilder,
    prefix: &str,
    stem: &str,
    lin: &Linear,
    source: &Gguf,
    can_have_bias: bool,
    quant: GgufQuant,
) {
    let wname = format!("{prefix}.{stem}.weight");
    add_weight_2d(b, &wname, lin, quant);
    // Emit the bias only if this stem can carry one *and* the source had it.
    if can_have_bias {
        let bname = format!("{prefix}.{stem}.bias");
        if source.tensor(&bname).is_some() {
            let bias = &lin.bias.data;
            let bytes = encode_tensor(bias, GGML_F32).unwrap();
            b.tensor(&bname, &[bias.len() as u64], GGML_F32, bytes);
        }
    }
}

/// Emit a 2-D weight: Ferrum `[n_in, n_out]` → raw `[n_out, n_in]`, dims
/// `[n_in, n_out]`, quantized per policy.
fn add_weight_2d(b: &mut GgufBuilder, name: &str, lin: &Linear, quant: GgufQuant) {
    let (n_in, n_out) = (lin.in_features(), lin.out_features());
    let w = lin.weight_f32(); // [n_in, n_out]
    let raw = transpose(&w, n_in, n_out); // [n_out, n_in]
    add_tensor(
        b,
        name,
        &raw,
        n_in,
        &[n_in as u64, n_out as u64],
        true,
        quant,
    );
}

/// Encode `raw` (already in GGUF byte order) as a tensor of `dims`, choosing the
/// type via `plan_tensor_type` on `block_axis` (= dims[0]). `plan_tensor_type` only picks a
/// block type when `block_axis` is a multiple of its block size and every caller passes the
/// fastest dim of `raw`, so the length is always block-aligned and encoding cannot fail.
fn add_tensor(
    b: &mut GgufBuilder,
    name: &str,
    raw: &[f32],
    block_axis: usize,
    dims: &[u64],
    is_2d: bool,
    quant: GgufQuant,
) {
    let (ggml_type, fell_back) = plan_tensor_type(is_2d, block_axis, quant);
    if fell_back {
        vprintln!(
            "note: tensor '{name}' (axis {block_axis}) is not block-aligned for the chosen \
             quant; storing it as F16"
        );
    }
    // encode_tensor only fails on a ragged block length, which the plan above rules out.
    let bytes =
        encode_tensor(raw, ggml_type).expect("plan_tensor_type guarantees a block-aligned length");
    b.tensor(name, dims, ggml_type, bytes);
}

#[cfg(test)]
mod tests {
    use super::llama_gguf_bytes;
    use crate::gguf::{Gguf, MetaValue};
    use crate::gguf_write::quant::GgufQuant;
    use crate::gguf_write::test_support::{
        build_tiny_llama, det_fill, tiny_llama_gguf_bytes, tiny_llama_gguf_bytes_ex,
    };

    #[test]
    fn write_llama_gguf_f32_roundtrip_is_exact() {
        use crate::gguf::Gguf;

        let g0 = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = g0.load_llama_prec(None).unwrap();

        let out = llama_gguf_bytes(&model, &g0, GgufQuant::F32).unwrap();
        let g1 = Gguf::parse(out).unwrap();
        let model2 = g1.load_llama_prec(None).unwrap();

        // Same logits for a fixed token sequence (F32 → bit-exact weights).
        let toks = [1usize, 5, 2, 7];
        let l0 = model.forward_tokens(&toks).unwrap();
        let l1 = model2.forward_tokens(&toks).unwrap();
        assert_eq!(l0.data, l1.data);
    }

    #[test]
    fn write_llama_gguf_preserves_tokenizer_metadata() {
        use crate::gguf::Gguf;
        let g0 = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = g0.load_llama_prec(None).unwrap();
        let out = llama_gguf_bytes(&model, &g0, GgufQuant::Q8_0).unwrap();
        let g1 = Gguf::parse(out).unwrap();
        assert_eq!(
            g1.meta("tokenizer.ggml.model").unwrap().as_str(),
            Some("gpt2")
        );
        assert_eq!(
            g0.meta("tokenizer.ggml.tokens"),
            g1.meta("tokenizer.ggml.tokens")
        );
    }

    // Positive-path counterpart to the gating in `add_linear` (Task 11 review,
    // Fix 1/2): `tiny_llama_gguf_bytes` never carries a `*.bias` tensor, so
    // without this test the bias-emission branch is only ever exercised as
    // `false`. This fixture adds deterministic q/k/v biases and checks both
    // that they round-trip and that no other stem ever emits one.
    #[test]
    fn write_llama_gguf_roundtrips_qkv_biases() {
        use crate::gguf::Gguf;

        // Mirrors the (dim, n_layers, n_heads, head_dim) fixed inside
        // `tiny_llama_gguf_bytes_ex`: dim=8, n_heads=2 → head_dim=4, so the q/k/v
        // output dim (n_heads*head_dim) is 8.
        let n_layers = 2usize;
        let qkv_dim = 8usize;

        let mut g0 = Gguf::parse(tiny_llama_gguf_bytes_ex(true)).unwrap();
        let model = g0.load_llama_prec(None).unwrap();

        // Re-arm the writer-gate regression guard: plant a fake directory entry
        // for a non-qkv bias in the *source* (the reader rejects real ones at
        // load time, so the model above is clean — but the writer's
        // `can_have_bias` gate must refuse the emission even when
        // `source.tensor(&bname)` is `Some`). The offset is never read: the
        // writer takes bias values from the model, not the source bytes.
        g0.tensors.push(crate::gguf::TensorInfo {
            name: "blk.0.attn_output.bias".into(),
            dims: vec![8],
            ggml_type: crate::gguf::GGML_F32,
            offset: 0,
        });

        let out = llama_gguf_bytes(&model, &g0, GgufQuant::F32).unwrap();
        let g1 = Gguf::parse(out).unwrap();

        // (a) q/k/v biases are present and match the source values exactly
        // (F32 in, F32 out — lossless).
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            for (stem, seed) in [
                ("attn_q", 300 + i),
                ("attn_k", 310 + i),
                ("attn_v", 320 + i),
            ] {
                let name = format!("{p}.{stem}.bias");
                let expected = det_fill(seed, qkv_dim);
                let t = g1.tensor(&name).expect(&name);
                let got = g1.dequantize(t).unwrap();
                assert_eq!(got.len(), expected.len(), "{name}: length mismatch");
                for (g, e) in got.iter().zip(&expected) {
                    assert!((g - e).abs() < 1e-6, "{name}: got {g}, want {e}");
                }
            }
        }

        // (b) non-qkv stems never emit a bias, even though every Linear (incl.
        // attn_output/ffn_*) carries a zero-filled in-memory `bias` vector.
        assert!(g1.tensor("blk.0.attn_output.bias").is_none());
        assert!(g1.tensor("blk.0.ffn_gate.bias").is_none());
        assert!(g1.tensor("blk.0.ffn_up.bias").is_none());
        assert!(g1.tensor("blk.0.ffn_down.bias").is_none());
        assert!(g1.tensor("blk.1.attn_output.bias").is_none());
    }

    /// D1: fine-tune 2 epochs, then `export-gguf --resume` (apply the checkpoint,
    /// write, reload) must be the exact model `run-gguf --resume` runs, with and
    /// without real q/k/v biases in the source (the trainer must not train a bias
    /// the source file cannot carry).
    fn finetune_export_equals_resume(with_bias: bool, tied: bool) {
        use crate::llm_train::{apply_checkpoint_weights, LlamaTrainer};
        use crate::rng::Rng;

        let mut g0 = Gguf::parse(tiny_llama_gguf_bytes_ex(with_bias)).unwrap();
        if tied {
            // A tied-embedding source (like SmolLM2): no `output.weight`.
            g0.tensors.retain(|t| t.name != "output.weight");
        }
        let mut tr = LlamaTrainer::new(g0.load_llama_prec(None).unwrap()).unwrap();
        tr.set_lr(0.02);
        let tokens: Vec<usize> = (0..100).map(|i| (i * 7) % 32).collect();
        let mut rng = Rng::new(5);
        for _ in 0..2 {
            tr.finetune_epoch(&tokens, 8, 8, &mut rng).unwrap();
        }
        let ckpt = tr.save_checkpoint(&rng);

        let base = g0.load_llama_prec(None).unwrap();
        let mut resumed = g0.load_llama_prec(None).unwrap();
        apply_checkpoint_weights(&mut resumed, ckpt).unwrap();
        let out = llama_gguf_bytes(&resumed, &g0, GgufQuant::F32).unwrap();
        let exported = Gguf::parse(out).unwrap().load_llama_prec(None).unwrap();

        let toks = [1usize, 5, 2, 7, 3];
        let want = resumed.forward_tokens(&toks).unwrap().data;
        assert!(
            want == exported.forward_tokens(&toks).unwrap().data,
            "exported logits differ from base + --resume"
        );
        assert!(want != base.forward_tokens(&toks).unwrap().data, "trained");
        // A tied export stays tied: no `output.weight` is invented.
        let re = Gguf::parse(llama_gguf_bytes(&resumed, &g0, GgufQuant::F32).unwrap()).unwrap();
        assert_eq!(re.tensor("output.weight").is_none(), tied);
    }

    #[test]
    fn export_of_a_bias_free_finetune_equals_the_resumed_model() {
        finetune_export_equals_resume(false, false);
    }

    #[test]
    fn export_of_a_finetune_with_real_qkv_biases_equals_the_resumed_model() {
        finetune_export_equals_resume(true, false);
    }

    /// A tied-embedding source (no `output.weight`) keeps its head tied while
    /// training, so the export (which cannot write a separate head) is lossless.
    #[test]
    fn export_of_a_tied_embedding_finetune_equals_the_resumed_model() {
        finetune_export_equals_resume(false, true);
        finetune_export_equals_resume(true, true);
    }

    // Regression guard (Task 11 final review, Fix 1): a source GGUF that
    // declares a non-default `general.alignment` must not have that value
    // carried into the export. The builder always packs tensor data at
    // `DEFAULT_ALIGNMENT` (32) regardless of what the source claimed, so a
    // stale alignment in the output metadata would make the reader compute
    // every tensor offset from the wrong base — silent corruption.
    #[test]
    fn write_llama_gguf_strips_source_alignment() {
        let mut g0 = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = g0.load_llama_prec(None).unwrap();
        // Simulate a source that declared a non-default alignment (metadata is a pub BTreeMap).
        g0.metadata
            .insert("general.alignment".into(), MetaValue::U32(64));
        let out = llama_gguf_bytes(&model, &g0, GgufQuant::F32).unwrap();
        let g1 = Gguf::parse(out).unwrap();
        // The stale alignment must NOT be propagated (this is the reliable regression guard).
        assert!(g1.meta("general.alignment").is_none());
        // And the model still round-trips exactly.
        let m2 = g1.load_llama_prec(None).unwrap();
        assert_eq!(
            model.forward_tokens(&[1usize, 2, 3]).unwrap().data,
            m2.forward_tokens(&[1usize, 2, 3]).unwrap().data
        );
    }

    // The reader loads biases only for q/k/v; a file carrying one on any other
    // projection (or the LM head) must be rejected loudly at load time, never
    // run silently without it. The planted directory entry is enough — the
    // check fires on presence, before any tensor bytes are read.
    #[test]
    fn load_llama_rejects_non_qkv_biases() {
        for name in ["blk.0.ffn_up.bias", "blk.1.attn_output.bias", "output.bias"] {
            let mut g = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
            g.tensors.push(crate::gguf::TensorInfo {
                name: name.into(),
                dims: vec![8],
                ggml_type: crate::gguf::GGML_F32,
                offset: 0,
            });
            let e = g
                .load_llama_prec(None)
                .err()
                .expect("load must reject a stray bias");
            let msg = e.to_string();
            assert!(msg.contains(name), "error should name '{name}': {msg}");
        }
    }

    // End-to-end k-quant export: at dim 256 every projection's block axis is a
    // QK_K multiple, so no F16 fallback occurs — the output must actually
    // contain Q4_K tensors, carry the k-quant file-type hint, and reload as a
    // runnable model. (The tiny dim-8 fixture always falls back, so without
    // this test k-quants were only covered at the encoder level.)
    #[test]
    fn write_llama_gguf_emits_kquant_at_dim_256() {
        use crate::gguf::GGML_Q4_K;
        let g0 = Gguf::parse(build_tiny_llama(256, 1, false)).unwrap();
        let model = g0.load_llama_prec(None).unwrap();
        let out = llama_gguf_bytes(&model, &g0, GgufQuant::Q4K).unwrap();
        let g1 = Gguf::parse(out).unwrap();
        let t = g1.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(t.ggml_type, GGML_Q4_K, "attn_q should be Q4_K, no fallback");
        assert_eq!(
            g1.meta("general.file_type").and_then(|v| v.as_usize()),
            Some(15), // MOSTLY_Q4_K_M — the requested type was actually used
        );
        let m2 = g1.load_llama_prec(None).unwrap();
        let logits = m2.forward_tokens(&[1usize, 3, 5]).unwrap();
        assert!(logits.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn export_rejects_other_architectures_and_a_missing_architecture() {
        let mut src = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = src.load_llama_prec(None).unwrap();
        src.metadata.insert(
            "general.architecture".into(),
            MetaValue::String("mamba".into()),
        );
        let e = llama_gguf_bytes(&model, &src, GgufQuant::F32).unwrap_err();
        assert!(e.to_string().contains("llama/qwen2 only"), "{e}");
        src.metadata.remove("general.architecture");
        let e = llama_gguf_bytes(&model, &src, GgufQuant::F32).unwrap_err();
        assert!(
            e.to_string().contains("missing general.architecture"),
            "{e}"
        );
    }

    #[test]
    fn a_tied_head_source_exports_without_an_output_tensor() {
        let mut src = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = src.load_llama_prec(None).unwrap();
        src.tensors.retain(|t| t.name != "output.weight");
        let out = Gguf::parse(llama_gguf_bytes(&model, &src, GgufQuant::F32).unwrap()).unwrap();
        assert!(out.tensor("output.weight").is_none());
        assert!(out.tensor("token_embd.weight").is_some());
    }

    #[test]
    fn write_llama_gguf_surfaces_io_errors_and_leaves_no_temp_file() {
        let src = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = src.load_llama_prec(None).unwrap();
        // The destination is an existing directory, so the final rename fails.
        let dir = std::env::temp_dir().join(format!("ferrum_atomic_dir_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.to_str().unwrap();
        let r = super::write_llama_gguf(&model, &src, GgufQuant::F32, p);
        let tmp_left = std::path::Path::new(&format!("{p}.tmp")).exists();
        let _ = std::fs::remove_dir(&dir);
        assert!(r.is_err());
        assert!(!tmp_left);
        // An unwritable directory fails at the first write.
        assert!(
            super::write_llama_gguf(&model, &src, GgufQuant::F32, "/nonexistent-dir/x.gguf")
                .is_err()
        );
    }

    #[test]
    fn model_write_gguf_is_the_atomic_file_form_of_llama_gguf_bytes() {
        let src = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = src.load_llama_prec(None).unwrap();
        let path =
            std::env::temp_dir().join(format!("ferrum_model_write_{}.gguf", std::process::id()));
        model
            .write_gguf(&src, GgufQuant::F32, path.to_str().unwrap())
            .unwrap();
        let on_disk = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            on_disk,
            llama_gguf_bytes(&model, &src, GgufQuant::F32).unwrap()
        );
    }
}
