//! Backward passes: in-place and owned-gradient (worker) variants.
use crate::error::Result;
use crate::ops;
use crate::tensor::Tensor;
use crate::train_transformer::block::BlockGrads;
use crate::train_transformer::forward::FwdCache;
use crate::train_transformer::kernels::{attn_bwd, linear_bwd, ln_bwd, matmul_bwd};
use crate::train_transformer::TransformerNet;

impl TransformerNet {
    /// Overwrite every gradient tensor from a flat vector in canonical
    /// parameter order (as produced by [`TransformerNet::backward_grads`], e.g.
    /// a summed cross-shard gradient), so a subsequent [`TransformerNet::step`]
    /// applies it. Panics if `flat` has the wrong length.
    pub(crate) fn load_grads(&mut self, flat: &[f32]) {
        let mut offset = 0usize;
        for g in self.grad_tensors_mut() {
            let n = g.data.len();
            g.data.copy_from_slice(&flat[offset..offset + n]);
            offset += n;
        }
        debug_assert_eq!(offset, flat.len(), "load_grads: length mismatch");
    }

    /// Backprop `dlogits` (e.g. from `softmax_cross_entropy`) through the
    /// whole network, storing the result in every parameter's `.grad`.
    ///
    /// (Each training step calls [`TransformerNet::zero_grad`] then a single
    /// `backward`, so storing is equivalent to the previous accumulate.)
    pub fn backward(&mut self, dlogits: &Tensor, cache: &FwdCache) -> Result<()> {
        let flat = self.backward_grads(dlogits, cache)?;
        self.load_grads(&flat);
        Ok(())
    }

    /// Backprop `dlogits` against the (read-only) weights and return the flat
    /// gradient vector in canonical parameter order — the same layout
    /// [`TransformerNet::load_grads`] consumes.
    ///
    /// Takes `&self` and allocates only the gradient buffer (≈4 B/param), so the
    /// data-parallel workers in [`train_transformer_epoch_threaded`] can share
    /// the master weights instead of deep-cloning the whole network
    /// (weights+grad+Adam moments ≈16 B/param per worker). `backward` is the
    /// in-place wrapper that stores the result into the owned parameters.
    pub(crate) fn backward_grads(&self, dlogits: &Tensor, cache: &FwdCache) -> Result<Vec<f32>> {
        let t = self.context_len;
        let batch = cache.batch;
        let c = self.embed_dim;

        // LM head + final LayerNorm (each produced exactly once).
        let (g_head_w, d_normf) = matmul_bwd(&cache.norm_f, &self.head_w.data, dlogits)?;
        let (mut dx, g_lnf_g, g_lnf_b) =
            ln_bwd(&d_normf, &cache.xhat_f, &cache.inv_std_f, &self.lnf_g.data)?;

        // Per-block grads, computed in reverse but stored by block index so they
        // can be emitted in forward (canonical) order.
        let mut block_grads: Vec<Option<BlockGrads>> =
            (0..self.blocks.len()).map(|_| None).collect();
        for (bi, (blk, bc)) in self.blocks.iter().zip(&cache.blocks).enumerate().rev() {
            // out = x_attn + ffn2(gelu(ffn1(norm2)))
            let d_out = dx;
            let (g_f2_w, g_f2_b, dh) = linear_bwd(&bc.h_act, &blk.f2_w.data, &d_out)?;
            // Backward through (optional) dropout then GELU. The dropout
            // mask×scale carries the inverted-dropout scaling into the gradient.
            let dh_pre = Tensor::new(
                dh.shape.clone(),
                match &bc.ffn_dropout {
                    Some(mask) => dh
                        .data
                        .iter()
                        .zip(&bc.h_pre.data)
                        .zip(mask)
                        .map(|((&g, &h), &m)| g * m * crate::activation::gelu_grad(h))
                        .collect(),
                    None => dh
                        .data
                        .iter()
                        .zip(&bc.h_pre.data)
                        .map(|(&g, &h)| g * crate::activation::gelu_grad(h))
                        .collect(),
                },
            )
            .expect("same element count as dh");
            let (g_f1_w, g_f1_b, d_norm2) = linear_bwd(&bc.norm2, &blk.f1_w.data, &dh_pre)?;
            let (d_xattn_ln, g_ln2_g, g_ln2_b) =
                ln_bwd(&d_norm2, &bc.xhat2, &bc.inv_std2, &blk.ln2_g.data)?;
            // residual 2: gradient flows both into the FFN branch and straight through
            let d_xattn = ops::add(&d_out, &d_xattn_ln)?;

            // x_attn = x + out_proj(concat)
            let (g_o_w, g_o_b, d_concat) = linear_bwd(&bc.concat, &blk.o_w.data, &d_xattn)?;
            let (dq, dk, dv) = attn_bwd(&d_concat, &bc.q, &bc.k, &bc.v, &bc.probs, batch, t, self.num_heads)?;
            let (g_q_w, g_q_b, d_n1a) = linear_bwd(&bc.norm1, &blk.q_w.data, &dq)?;
            let (g_k_w, g_k_b, d_n1b) = linear_bwd(&bc.norm1, &blk.k_w.data, &dk)?;
            let (g_v_w, g_v_b, d_n1c) = linear_bwd(&bc.norm1, &blk.v_w.data, &dv)?;
            let d_norm1 = ops::add(&ops::add(&d_n1a, &d_n1b)?, &d_n1c)?;
            let (d_x_ln, g_ln1_g, g_ln1_b) =
                ln_bwd(&d_norm1, &bc.xhat1, &bc.inv_std1, &blk.ln1_g.data)?;
            // residual 1
            dx = ops::add(&d_xattn, &d_x_ln)?;

            block_grads[bi] = Some(BlockGrads {
                ln1_g: g_ln1_g,
                ln1_b: g_ln1_b,
                q_w: g_q_w,
                q_b: g_q_b,
                k_w: g_k_w,
                k_b: g_k_b,
                v_w: g_v_w,
                v_b: g_v_b,
                o_w: g_o_w,
                o_b: g_o_b,
                ln2_g: g_ln2_g,
                ln2_b: g_ln2_b,
                f1_w: g_f1_w,
                f1_b: g_f1_b,
                f2_w: g_f2_w,
                f2_b: g_f2_b,
            });
        }

        // Embedding: scatter-add row gradients into local tables.
        let mut g_tok = vec![0.0f32; self.tok_emb.data.data.len()];
        let mut g_pos = vec![0.0f32; self.pos_emb.data.data.len()];
        for (r, &tok) in cache.tokens.iter().enumerate() {
            let xb = r * c;
            let tb = tok * c;
            let pb = (r % t) * c;
            for d in 0..c {
                g_tok[tb + d] += dx.data[xb + d];
                g_pos[pb + d] += dx.data[xb + d];
            }
        }

        // Assemble the flat gradient in canonical parameter order.
        let mut out = Vec::with_capacity(self.flat_len());
        out.extend_from_slice(&g_tok);
        out.extend_from_slice(&g_pos);
        for bg in block_grads {
            let bg = bg.expect("every block's gradient was computed");
            for g in [
                &bg.ln1_g, &bg.ln1_b, &bg.q_w, &bg.q_b, &bg.k_w, &bg.k_b, &bg.v_w, &bg.v_b,
                &bg.o_w, &bg.o_b, &bg.ln2_g, &bg.ln2_b, &bg.f1_w, &bg.f1_b, &bg.f2_w, &bg.f2_b,
            ] {
                out.extend_from_slice(&g.data);
            }
        }
        out.extend_from_slice(&g_lnf_g.data);
        out.extend_from_slice(&g_lnf_b.data);
        out.extend_from_slice(&g_head_w.data);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::loss::softmax_cross_entropy;
    use crate::rng::Rng;
    use crate::train_transformer::test_support::{check_param, tiny_net};
    use crate::train_transformer::TransformerNet;

    #[test]
    fn gradient_check_all_param_groups() {
        let mut rng = Rng::new(3);
        let mut net = tiny_net(&mut rng);
        let tokens = [1usize, 2, 3];
        let targets = [2usize, 3, 4];

        macro_rules! check {
            ($field:ident, $label:expr, $idx:expr) => {
                check_param(
                    &mut net,
                    &tokens,
                    &targets,
                    |n, i| n.$field.grad.data[i],
                    |n, i, d| n.$field.data.data[i] += d,
                    $idx,
                    $label,
                );
            };
        }
        macro_rules! check_blk {
            ($field:ident, $label:expr, $idx:expr) => {
                check_param(
                    &mut net,
                    &tokens,
                    &targets,
                    |n, i| n.blocks[0].$field.grad.data[i],
                    |n, i, d| n.blocks[0].$field.data.data[i] += d,
                    $idx,
                    $label,
                );
            };
        }

        check!(tok_emb, "tok_emb", &[4, 9, 13]); // rows of tokens 1,2,3
        check!(pos_emb, "pos_emb", &[0, 5, 11]);
        check!(lnf_g, "lnf_g", &[0, 3]);
        check!(lnf_b, "lnf_b", &[1]);
        check!(head_w, "head_w", &[0, 7, 19]);
        check_blk!(ln1_g, "ln1_g", &[0, 2]);
        check_blk!(ln1_b, "ln1_b", &[1]);
        check_blk!(q_w, "q_w", &[0, 5, 15]);
        check_blk!(k_w, "k_w", &[3, 10]);
        check_blk!(v_w, "v_w", &[2, 12]);
        check_blk!(v_b, "v_b", &[1]);
        check_blk!(o_w, "o_w", &[4, 9]);
        check_blk!(ln2_g, "ln2_g", &[0, 3]);
        check_blk!(f1_w, "f1_w", &[0, 11, 20]);
        check_blk!(f1_b, "f1_b", &[2]);
        check_blk!(f2_w, "f2_w", &[5, 17]);
        check_blk!(f2_b, "f2_b", &[0]);
    }

    /// The dropout backward is correct: against a fixed mask (fixed seed), the
    /// analytic gradient matches central differences both for the FFN-2 weight
    /// (directly downstream of dropout) and the FFN-1 weight (upstream, so the
    /// mask must propagate back through GELU).
    #[test]
    fn dropout_gradient_is_correct_for_fixed_mask() {
        let mut rng = Rng::new(3);
        let mut net = TransformerNet::new(5, 3, 4, 2, 8, 1, &mut rng).unwrap();
        let tokens = [1usize, 2, 3];
        let targets = [2usize, 3, 4];
        let (seed, p) = (77u64, 0.5f32);

        let (logits, cache) = net.forward_train(&tokens, p, seed).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();
        net.zero_grad();
        net.backward(&dl, &cache).unwrap();

        let eps = 1e-2f32;
        macro_rules! check {
            ($field:ident, $idx:expr) => {
                for &i in $idx {
                    let analytic = net.blocks[0].$field.grad.data[i];
                    net.blocks[0].$field.data.data[i] += eps;
                    let lp = softmax_cross_entropy(
                        &net.forward_train(&tokens, p, seed).unwrap().0,
                        &targets,
                    )
                    .unwrap()
                    .0;
                    net.blocks[0].$field.data.data[i] -= 2.0 * eps;
                    let lm = softmax_cross_entropy(
                        &net.forward_train(&tokens, p, seed).unwrap().0,
                        &targets,
                    )
                    .unwrap()
                    .0;
                    net.blocks[0].$field.data.data[i] += eps;
                    let numeric = (lp - lm) / (2.0 * eps);
                    assert!(
                        (numeric - analytic).abs() < 6e-3 + 0.03 * analytic.abs(),
                        "{}[{i}]: analytic={analytic:.6} numeric={numeric:.6}",
                        stringify!($field)
                    );
                }
            };
        }
        check!(f2_w, &[0usize, 5, 11]);
        check!(f1_w, &[0usize, 7, 15]);
    }

    /// `backward_grads(&self)` returns exactly the gradients the in-place
    /// `backward` stores, and leaves the network unmodified. This purity is what
    /// lets the data-parallel workers borrow the master weights read-only
    /// instead of deep-cloning the whole network per shard.
    #[test]
    fn backward_grads_matches_inplace_backward_and_leaves_weights_intact() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut rng).unwrap();
        let tokens = [1usize, 2, 3, 4, 0, 5, 2, 1]; // two windows (T=4)
        let targets = [2usize, 3, 4, 5, 1, 0, 3, 2];
        let (logits, cache) = net.forward(&tokens).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();

        // Pure: no weight or optimizer-moment mutation.
        let weights_before = net.snapshot_weights();
        let flat = net.backward_grads(&dl, &cache).unwrap();
        assert_eq!(
            net.snapshot_weights(),
            weights_before,
            "backward_grads mutated weights"
        );
        assert_eq!(flat.len(), net.num_params());

        // The in-place wrapper stores exactly those gradients (canonical order).
        net.zero_grad();
        net.backward(&dl, &cache).unwrap();
        let mut stored = Vec::new();
        for g in net.grad_tensors_mut() {
            stored.extend_from_slice(&g.data);
        }
        assert_eq!(
            stored, flat,
            "in-place backward differs from backward_grads"
        );
    }
}
