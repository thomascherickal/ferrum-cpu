//! Parameter enumeration: f32 requirement, shapes and data views.
use crate::error::{InferError, Result};
use crate::layer::Linear;
use crate::llm::LlamaModel;

/// A bias is a trainable parameter only if the layer really has one. A
/// placeholder (the source GGUF had no such tensor) is a zero-length slot: the
/// slot keeps its index, so the parameter layout is stable, but nothing is
/// trained or stored, and the trained model stays exactly what the original
/// architecture (and `export-gguf`) can represent.
fn bias_ref(l: &Linear) -> &[f32] {
    if l.has_bias() {
        &l.bias.data
    } else {
        &[]
    }
}

/// The weight and (if real) bias slots of `l`, as disjoint mutable borrows.
fn wb_mut(l: &mut Linear) -> [&mut [f32]; 2] {
    let real = l.has_bias();
    let Linear { weight, bias, .. } = l;
    [
        &mut weight.data,
        if real { &mut bias.data } else { &mut [] },
    ]
}

/// Training (and applying a checkpoint's f32 masters) needs f32 projections:
/// errors naming the first quantized one.
pub(super) fn ensure_f32_weights(model: &LlamaModel) -> Result<()> {
    for (i, b) in model.blocks.iter().enumerate() {
        for (name, lin) in [
            ("attn_q", &b.attn.wq),
            ("attn_k", &b.attn.wk),
            ("attn_v", &b.attn.wv),
            ("attn_output", &b.attn.wo),
            ("ffn_gate", &b.ffn.gate),
            ("ffn_up", &b.ffn.up),
            ("ffn_down", &b.ffn.down),
        ] {
            if lin.qweight().is_some() {
                return Err(InferError::DimMismatch(format!(
                    "block {i} {name} is quantized; training needs f32 weights \
                     (load with load_llama_prec(None))"
                )));
            }
        }
    }
    if model.lm_head.qweight().is_some() {
        return Err(InferError::DimMismatch(
            "lm_head is quantized; training needs f32 weights".into(),
        ));
    }
    Ok(())
}

/// `(rows, is_matrix)` per parameter in canonical order. `rows` is the number of
/// per-channel quantization groups (used by QAT); `is_matrix` flags the 2-D
/// weight matrices that decoupled weight decay applies to.
pub(super) fn param_meta(model: &LlamaModel) -> Vec<(usize, bool)> {
    let dim = model.cfg.model_dim;
    let mut v: Vec<(usize, bool)> = vec![(model.tok_emb.len() / dim, true)]; // tok_emb [vocab, dim]
    for b in &model.blocks {
        v.push((1, false)); // attn_norm
        v.push((b.attn.wq.in_features(), true));
        v.push((1, false));
        v.push((b.attn.wk.in_features(), true));
        v.push((1, false));
        v.push((b.attn.wv.in_features(), true));
        v.push((1, false));
        v.push((b.attn.wo.in_features(), true));
        v.push((1, false));
        v.push((1, false)); // ffn_norm
        v.push((b.ffn.gate.in_features(), true));
        v.push((1, false));
        v.push((b.ffn.up.in_features(), true));
        v.push((1, false));
        v.push((b.ffn.down.in_features(), true));
        v.push((1, false));
    }
    v.push((1, false)); // final_norm
    v.push((model.lm_head.in_features(), true));
    v.push((1, false));
    v
}

pub(super) fn param_data_ref(model: &LlamaModel) -> Vec<&[f32]> {
    let mut v: Vec<&[f32]> = vec![&model.tok_emb];
    for b in &model.blocks {
        v.push(&b.attn_norm.weight);
        v.push(&b.attn.wq.weight.data);
        v.push(bias_ref(&b.attn.wq));
        v.push(&b.attn.wk.weight.data);
        v.push(bias_ref(&b.attn.wk));
        v.push(&b.attn.wv.weight.data);
        v.push(bias_ref(&b.attn.wv));
        v.push(&b.attn.wo.weight.data);
        v.push(bias_ref(&b.attn.wo));
        v.push(&b.ffn_norm.weight);
        v.push(&b.ffn.gate.weight.data);
        v.push(bias_ref(&b.ffn.gate));
        v.push(&b.ffn.up.weight.data);
        v.push(bias_ref(&b.ffn.up));
        v.push(&b.ffn.down.weight.data);
        v.push(bias_ref(&b.ffn.down));
    }
    v.push(&model.final_norm.weight);
    v.push(&model.lm_head.weight.data);
    v.push(bias_ref(&model.lm_head));
    v
}

pub(super) fn param_data_mut(model: &mut LlamaModel) -> Vec<&mut [f32]> {
    let mut v: Vec<&mut [f32]> = vec![&mut model.tok_emb];
    for b in &mut model.blocks {
        v.push(&mut b.attn_norm.weight);
        v.extend(wb_mut(&mut b.attn.wq));
        v.extend(wb_mut(&mut b.attn.wk));
        v.extend(wb_mut(&mut b.attn.wv));
        v.extend(wb_mut(&mut b.attn.wo));
        v.push(&mut b.ffn_norm.weight);
        v.extend(wb_mut(&mut b.ffn.gate));
        v.extend(wb_mut(&mut b.ffn.up));
        v.extend(wb_mut(&mut b.ffn.down));
    }
    v.push(&mut model.final_norm.weight);
    v.extend(wb_mut(&mut model.lm_head));
    v
}

#[cfg(test)]
mod tests {
    use crate::llm::LlamaModel;
    use crate::llm_train::checkpoint::apply_checkpoint_weights;
    use crate::llm_train::test_support::{lin, tiny, trained_trainer};
    use crate::llm_train::LlamaTrainer;
    use crate::rng::Rng;

    /// `tiny` with every bias a placeholder, as imported from a bias-free GGUF.
    fn bias_free_tiny() -> LlamaModel {
        let mut m = tiny(6, 8, 2, 2, 16, 2);
        for b in &mut m.blocks {
            for l in [
                &mut b.attn.wq,
                &mut b.attn.wk,
                &mut b.attn.wv,
                &mut b.attn.wo,
                &mut b.ffn.gate,
                &mut b.ffn.up,
                &mut b.ffn.down,
            ] {
                *l = std::mem::replace(l, lin(1, 1, 0)).into_bias_free();
            }
        }
        m.lm_head = std::mem::replace(&mut m.lm_head, lin(1, 1, 0)).into_bias_free();
        m
    }

    /// D1: a bias the source model lacks is not a parameter: it has no slot data,
    /// no gradient, no moments, and stays exactly zero through training, so the
    /// trained model is exactly representable by the original architecture.
    #[test]
    fn placeholder_biases_are_not_trained_or_counted() {
        let real = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 2)).unwrap();
        let mut t = LlamaTrainer::new(bias_free_tiny()).unwrap();
        // 2 layers x (3x8 + 8 + 2x16 + 8) biases + 6 for the head.
        assert_eq!(
            real.num_params() - t.num_params(),
            2 * (24 + 8 + 32 + 8) + 6
        );
        t.set_lr(0.02);
        let tokens: Vec<usize> = (0..100).map(|i| (i * 7) % 6).collect();
        let mut rng = Rng::new(3);
        let before = t.model_flat();
        t.finetune_epoch(&tokens, 6, 16, &mut rng).unwrap();
        assert_ne!(before, t.model_flat(), "weights trained");
        for b in &t.model.blocks {
            for l in [&b.attn.wq, &b.attn.wk, &b.attn.wv, &b.attn.wo, &b.ffn.gate] {
                assert!(l.bias.data.iter().all(|&x| x == 0.0), "placeholder moved");
            }
        }
        assert!(t.model.lm_head.bias.data.iter().all(|&x| x == 0.0));
    }

    /// A bias-free fine-tune checkpoint round-trips; one written for a model
    /// that trained placeholder biases (an older Ferrum) is refused by name.
    #[test]
    fn checkpoints_carry_no_placeholder_biases_and_old_ones_are_refused_clearly() {
        let mut t = LlamaTrainer::new(bias_free_tiny()).unwrap();
        t.set_lr(0.02);
        let tokens: Vec<usize> = (0..100).map(|i| (i * 7) % 6).collect();
        let mut rng = Rng::new(3);
        t.finetune_epoch(&tokens, 6, 16, &mut rng).unwrap();
        let bytes = t.save_checkpoint(&rng);
        let mut u = LlamaTrainer::new(bias_free_tiny()).unwrap();
        u.load_checkpoint_into(&bytes).unwrap();
        assert_eq!(t.model_flat(), u.model_flat());
        let mut m = bias_free_tiny();
        apply_checkpoint_weights(&mut m, bytes).unwrap();

        // Older layout: every bias was a trained slot.
        let (old, _) = trained_trainer(false);
        let old_bytes = old.save_checkpoint(&Rng::new(1));
        let mut fresh = LlamaTrainer::new(bias_free_tiny()).unwrap();
        for e in [
            fresh.load_checkpoint_into(&old_bytes).err().unwrap(),
            apply_checkpoint_weights(&mut bias_free_tiny(), old_bytes)
                .err()
                .unwrap(),
        ] {
            let e = e.to_string();
            assert!(
                e.contains("older Ferrum") && e.contains("placeholder biases"),
                "{e}"
            );
        }
    }

    /// Both entry points refuse a quantized model, naming the tensor.
    #[test]
    fn f32_weights_are_required_to_train_or_apply_a_checkpoint() {
        use crate::quant::QKind;
        let (t, rng) = trained_trainer(false);
        let bytes = t.save_checkpoint(&rng);
        let mut block = tiny(6, 8, 2, 2, 16, 2);
        let wq = std::mem::replace(&mut block.blocks[1].attn.wq, lin(8, 8, 1));
        block.blocks[1].attn.wq = wq.quantize(QKind::Int8);
        let e = apply_checkpoint_weights(&mut block, bytes.clone())
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("block 1 attn_q is quantized"), "{e}");
        assert!(LlamaTrainer::new(block).is_err());
        let mut head = tiny(6, 8, 2, 2, 16, 2);
        let h = std::mem::replace(&mut head.lm_head, lin(8, 6, 1));
        head.lm_head = h.quantize(QKind::Int4);
        let e = apply_checkpoint_weights(&mut head, bytes)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("lm_head is quantized"), "{e}");
        assert!(LlamaTrainer::new(head).is_err());
    }
}
