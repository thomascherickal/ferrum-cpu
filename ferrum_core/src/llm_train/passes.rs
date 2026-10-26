//! Whole-model training forward and backward passes.
use crate::error::{InferError, Result};
use crate::llm::LlamaModel;
use crate::llm_train::blocks::{
    attn_forward, attn_output, block_backward, ffn_forward, ffn_output, BlockCache, FwdCache,
};
use crate::llm_train::buffers::{Buffers, LinBuf};
use crate::llm_train::kernels::{
    check_loss_finite, cross_entropy, lin_backward, lin_forward, rmsnorm_backward_row,
    rmsnorm_forward_row,
};

/// Forward pass over `tokens`, returning the activation cache. With `dropout > 0`
/// each block's SwiGLU hidden is inverted-dropout masked from a `seed`-derived
/// generator (per-block offset) so it is reproducible in backward.
pub(super) fn forward_train(
    model: &LlamaModel,
    tokens: &[usize],
    dropout: f32,
    seed: u64,
) -> Result<FwdCache> {
    let cfg = &model.cfg;
    let (seq, dim) = (tokens.len(), cfg.model_dim);
    let mut x = vec![0.0f32; seq * dim];
    for (t, &tok) in tokens.iter().enumerate() {
        if tok >= cfg.vocab_size {
            return Err(InferError::DimMismatch(format!(
                "token {tok} ≥ vocab {}",
                cfg.vocab_size
            )));
        }
        x[t * dim..(t + 1) * dim].copy_from_slice(&model.tok_emb[tok * dim..(tok + 1) * dim]);
    }

    let mut blocks = Vec::with_capacity(model.blocks.len());
    for (bi, block) in model.blocks.iter().enumerate() {
        let mut n1 = vec![0.0f32; seq * dim];
        let mut n1_inv = vec![0.0f32; seq];
        // `t` derives the row offset (for x/n1 slices) and indexes n1_inv.
        #[allow(clippy::needless_range_loop)]
        for t in 0..seq {
            let s = t * dim;
            let (y, inv) =
                rmsnorm_forward_row(&x[s..s + dim], &block.attn_norm.weight, cfg.norm_eps);
            n1[s..s + dim].copy_from_slice(&y);
            n1_inv[t] = inv;
        }
        let attn = attn_forward(cfg, &block.attn, &n1, seq);
        let o = attn_output(&block.attn, &attn, seq);
        let h: Vec<f32> = (0..seq * dim).map(|i| x[i] + o[i]).collect();

        let mut n2 = vec![0.0f32; seq * dim];
        let mut n2_inv = vec![0.0f32; seq];
        #[allow(clippy::needless_range_loop)] // see the n1 loop above
        for t in 0..seq {
            let s = t * dim;
            let (y, inv) =
                rmsnorm_forward_row(&h[s..s + dim], &block.ffn_norm.weight, cfg.norm_eps);
            n2[s..s + dim].copy_from_slice(&y);
            n2_inv[t] = inv;
        }
        let blk_seed = seed.wrapping_add(bi as u64 + 1);
        let ffn = ffn_forward(cfg, &block.ffn, &n2, seq, dropout, blk_seed);
        let f = ffn_output(&block.ffn, &ffn, seq);
        let out: Vec<f32> = (0..seq * dim).map(|i| h[i] + f[i]).collect();

        blocks.push(BlockCache {
            x: x.clone(),
            n1_inv,
            attn,
            h,
            n2_inv,
            ffn,
        });
        x = out;
    }

    let xfinal = x.clone();
    let mut xn = vec![0.0f32; seq * dim];
    let mut nf_inv = vec![0.0f32; seq];
    #[allow(clippy::needless_range_loop)] // `t` derives the row offset and indexes nf_inv
    for t in 0..seq {
        let s = t * dim;
        let (y, inv) =
            rmsnorm_forward_row(&xfinal[s..s + dim], &model.final_norm.weight, cfg.norm_eps);
        xn[s..s + dim].copy_from_slice(&y);
        nf_inv[t] = inv;
    }
    let logits = lin_forward(&model.lm_head, &xn, seq);
    Ok(FwdCache {
        blocks,
        xfinal,
        nf_inv,
        xn,
        logits,
    })
}

/// Backprop `d_logits` through the whole model, **accumulating** into `g` (the
/// embedding scatter and every dW/db/dnorm are `+=`), so summing across a
/// minibatch is just repeated calls into the same buffer.
pub(super) fn backward_into(
    model: &LlamaModel,
    cache: &FwdCache,
    d_logits: &[f32],
    tokens: &[usize],
    g: &mut Buffers,
) -> Result<()> {
    let cfg = &model.cfg;
    let (seq, dim) = (tokens.len(), cfg.model_dim);

    let d_xn = if model.tied_head {
        // ponytail: a vocab*dim scratch per call; fold in place if profiling asks.
        let mut head = LinBuf::zeros(&model.lm_head);
        let d_xn = lin_backward(&model.lm_head, &cache.xn, d_logits, seq, &mut head);
        // Head gradient [dim, vocab] accumulates into the shared embedding [vocab, dim].
        for d in 0..dim {
            for v in 0..cfg.vocab_size {
                g.tok_emb[v * dim + d] += head.w[d * cfg.vocab_size + v];
            }
        }
        d_xn
    } else {
        lin_backward(&model.lm_head, &cache.xn, d_logits, seq, &mut g.lm_head)
    };

    let mut d_x = vec![0.0f32; seq * dim];
    for t in 0..seq {
        let s = t * dim;
        let dxr = rmsnorm_backward_row(
            &d_xn[s..s + dim],
            &cache.xfinal[s..s + dim],
            &model.final_norm.weight,
            cache.nf_inv[t],
            &mut g.final_norm,
        );
        d_x[s..s + dim].copy_from_slice(&dxr);
    }

    for bi in (0..model.blocks.len()).rev() {
        d_x = block_backward(
            cfg,
            &model.blocks[bi],
            &cache.blocks[bi],
            &d_x,
            seq,
            &mut g.blocks[bi],
        );
    }

    for (t, &tok) in tokens.iter().enumerate() {
        let s = t * dim;
        for d in 0..dim {
            g.tok_emb[tok * dim + d] += d_x[s + d];
        }
    }
    Ok(())
}

/// Pure forward+backward over one sequence against the (read-only) model,
/// returning `(loss, flat-gradients)` in canonical order. Used by both the
/// serial accumulator and the data-parallel workers.
pub(super) fn seq_grad_flat(
    model: &LlamaModel,
    tokens: &[usize],
    dropout: f32,
    seed: u64,
) -> Result<(f32, Vec<f32>)> {
    let cache = forward_train(model, tokens, dropout, seed)?;
    let (loss, d_logits) = cross_entropy(&cache.logits, tokens, model.cfg.vocab_size);
    check_loss_finite(loss)?;
    let mut g = Buffers::zeros(model);
    backward_into(model, &cache, &d_logits, tokens, &mut g)?;
    Ok((loss, g.to_flat()))
}

#[cfg(test)]
mod tests {
    use crate::llm_train::test_support::tiny;
    use crate::llm_train::LlamaTrainer;

    #[test]
    fn end_to_end_gradient_matches_finite_difference() {
        let model = tiny(7, 8, 2, 1, 16, 2); // GQA: 2 q-heads share 1 kv-head
        let mut tr = LlamaTrainer::new(model).unwrap();
        let tokens = [1usize, 3, 0, 5, 2];
        tr.loss_and_backward(&tokens).unwrap();

        let eps = 1e-3;
        let check = |g: f32,
                     set: &mut dyn FnMut(&mut LlamaTrainer, f32),
                     tr: &mut LlamaTrainer,
                     label: &str| {
            set(tr, eps);
            let lp = tr.loss(&tokens).unwrap();
            set(tr, -2.0 * eps);
            let lm = tr.loss(&tokens).unwrap();
            set(tr, eps);
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 3e-2, "{label} fd={fd} got={g}");
        };
        let g = tr.grads.tok_emb[3 * 8 + 2];
        check(
            g,
            &mut |t, d| t.model.tok_emb[3 * 8 + 2] += d,
            &mut tr,
            "tok_emb",
        );
        let g = tr.grads.blocks[0].wq.w[5];
        check(
            g,
            &mut |t, d| t.model.blocks[0].attn.wq.weight.data[5] += d,
            &mut tr,
            "wq",
        );
        let g = tr.grads.blocks[1].down.w[9];
        check(
            g,
            &mut |t, d| t.model.blocks[1].ffn.down.weight.data[9] += d,
            &mut tr,
            "down",
        );
        let g = tr.grads.blocks[0].attn_norm[3];
        check(
            g,
            &mut |t, d| t.model.blocks[0].attn_norm.weight[3] += d,
            &mut tr,
            "attn_norm",
        );
        let g = tr.grads.lm_head.w[4];
        check(
            g,
            &mut |t, d| t.model.lm_head.weight.data[4] += d,
            &mut tr,
            "lm_head",
        );
    }
}
