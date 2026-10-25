//! Parameter enumeration: f32 requirement, shapes and data views.
use crate::error::{InferError, Result};
use crate::llm::LlamaModel;

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
        v.push(&b.attn.wq.bias.data);
        v.push(&b.attn.wk.weight.data);
        v.push(&b.attn.wk.bias.data);
        v.push(&b.attn.wv.weight.data);
        v.push(&b.attn.wv.bias.data);
        v.push(&b.attn.wo.weight.data);
        v.push(&b.attn.wo.bias.data);
        v.push(&b.ffn_norm.weight);
        v.push(&b.ffn.gate.weight.data);
        v.push(&b.ffn.gate.bias.data);
        v.push(&b.ffn.up.weight.data);
        v.push(&b.ffn.up.bias.data);
        v.push(&b.ffn.down.weight.data);
        v.push(&b.ffn.down.bias.data);
    }
    v.push(&model.final_norm.weight);
    v.push(&model.lm_head.weight.data);
    v.push(&model.lm_head.bias.data);
    v
}

pub(super) fn param_data_mut(model: &mut LlamaModel) -> Vec<&mut [f32]> {
    let mut v: Vec<&mut [f32]> = vec![&mut model.tok_emb];
    for b in &mut model.blocks {
        v.push(&mut b.attn_norm.weight);
        v.push(&mut b.attn.wq.weight.data);
        v.push(&mut b.attn.wq.bias.data);
        v.push(&mut b.attn.wk.weight.data);
        v.push(&mut b.attn.wk.bias.data);
        v.push(&mut b.attn.wv.weight.data);
        v.push(&mut b.attn.wv.bias.data);
        v.push(&mut b.attn.wo.weight.data);
        v.push(&mut b.attn.wo.bias.data);
        v.push(&mut b.ffn_norm.weight);
        v.push(&mut b.ffn.gate.weight.data);
        v.push(&mut b.ffn.gate.bias.data);
        v.push(&mut b.ffn.up.weight.data);
        v.push(&mut b.ffn.up.bias.data);
        v.push(&mut b.ffn.down.weight.data);
        v.push(&mut b.ffn.down.bias.data);
    }
    v.push(&mut model.final_norm.weight);
    v.push(&mut model.lm_head.weight.data);
    v.push(&mut model.lm_head.bias.data);
    v
}

#[cfg(test)]
mod tests {
    use crate::llm_train::checkpoint::apply_checkpoint_weights;
    use crate::llm_train::test_support::{lin, tiny, trained_trainer};
    use crate::llm_train::LlamaTrainer;

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
