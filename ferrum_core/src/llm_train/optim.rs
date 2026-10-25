//! Optimizer step, gradient clipping and QAT weight snapshots.
use crate::llm_train::params::{param_data_mut, param_data_ref, param_meta};
use crate::llm_train::LlamaTrainer;

impl LlamaTrainer {
    pub(super) fn snapshot_weights(&self) -> Vec<Vec<f32>> {
        param_data_ref(&self.model)
            .iter()
            .map(|s| s.to_vec())
            .collect()
    }

    pub(super) fn restore_weights(&mut self, snap: &[Vec<f32>]) {
        for (s, src) in param_data_mut(&mut self.model).into_iter().zip(snap) {
            s.copy_from_slice(src);
        }
    }

    /// Snap every weight **matrix** onto the int8 grid in place, per output-row.
    /// The GGUF writers use Q8_0 (32-element blocks, f16 scale); per-row scales
    /// are a close QAT-noise approximation of that grid, not an exact match.
    pub(super) fn fake_quantize_weights(&mut self) {
        let meta = param_meta(&self.model);
        for (s, (rows, is_matrix)) in param_data_mut(&mut self.model).into_iter().zip(meta) {
            if is_matrix {
                crate::quant::fake_quantize_int8_per_channel(s, rows.max(1));
            }
        }
    }

    /// Rescale `self.grads` so their global L2 norm is at most `max_norm`,
    /// returning the pre-clip norm.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let mut sumsq = 0.0f64;
        for s in self.grads.slices() {
            for &x in s {
                sumsq += (x as f64) * (x as f64);
            }
        }
        let norm = sumsq.sqrt() as f32;
        if max_norm > 0.0 && norm.is_finite() && norm > max_norm {
            let scale = max_norm / norm;
            for s in self.grads.slices_mut() {
                for x in s.iter_mut() {
                    *x *= scale;
                }
            }
        }
        norm
    }

    fn optimizer_step(&mut self) {
        self.step_t += 1;
        let t = self.step_t;
        let lr = match self.lr_schedule {
            Some(s) => s.lr_at(t),
            None => self.adam.lr,
        };
        let (b1, b2, eps, wd) = (
            self.adam.beta1,
            self.adam.beta2,
            self.adam.eps,
            self.weight_decay,
        );
        let bc1 = 1.0 - b1.powi(t.min(i32::MAX as u64) as i32);
        let bc2 = 1.0 - b2.powi(t.min(i32::MAX as u64) as i32);

        let meta = param_meta(&self.model);
        let mut params = param_data_mut(&mut self.model);
        let grads = self.grads.slices();
        let mut ms = self.m.slices_mut();
        let mut vs = self.v.slices_mut();

        for idx in 0..params.len() {
            let (_, is_matrix) = meta[idx];
            let p = &mut params[idx];
            let g = grads[idx];
            let mb = &mut ms[idx];
            let vb = &mut vs[idx];
            for i in 0..p.len() {
                let gi = g[i];
                mb[i] = b1 * mb[i] + (1.0 - b1) * gi;
                vb[i] = b2 * vb[i] + (1.0 - b2) * gi * gi;
                let m_hat = mb[i] / bc1;
                let v_hat = vb[i] / bc2;
                let mut update = lr * m_hat / (v_hat.sqrt() + eps);
                if wd != 0.0 && is_matrix {
                    update += lr * wd * p[i];
                }
                p[i] -= update;
            }
        }
    }

    /// Run one AdamW step from the gradients currently in `self.grads`
    /// (clipping first if configured).
    pub(super) fn apply_step(&mut self) {
        if let Some(max) = self.grad_clip {
            self.clip_grad_norm(max);
        }
        self.optimizer_step();
    }
}

#[cfg(test)]
mod tests {
    use crate::llm_train::test_support::tiny;
    use crate::llm_train::LlamaTrainer;

    #[test]
    fn adamw_weight_decay_shrinks_matrices_only() {
        let mut tr = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        tr.set_lr(0.5);
        tr.set_weight_decay(0.1);
        // Zero gradients: only decoupled decay acts.
        for v in tr.model.lm_head.weight.data.iter_mut() {
            *v = 2.0;
        } // matrix
        for v in tr.model.lm_head.bias.data.iter_mut() {
            *v = 2.0;
        } // bias (1-D)
        for v in tr.model.final_norm.weight.iter_mut() {
            *v = 2.0;
        } // norm (1-D)
        tr.grads.clear();
        tr.apply_step();
        // matrix: 2 − 0.5·0.1·2 = 1.9
        assert!(
            (tr.model.lm_head.weight.data[0] - 1.9).abs() < 1e-5,
            "matrix not decayed"
        );
        assert!(
            (tr.model.lm_head.bias.data[0] - 2.0).abs() < 1e-6,
            "bias decayed"
        );
        assert!(
            (tr.model.final_norm.weight[0] - 2.0).abs() < 1e-6,
            "norm decayed"
        );
    }

    #[test]
    fn grad_clip_caps_norm_and_stays_finite() {
        let mut tr = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        let tokens = [1usize, 2, 3, 4];
        tr.loss_and_backward(&tokens).unwrap();
        let pre = tr.clip_grad_norm(0.05);
        let mut sumsq = 0.0f64;
        for s in tr.grads.slices() {
            for &x in s {
                sumsq += (x as f64) * (x as f64);
            }
        }
        let post = sumsq.sqrt() as f32;
        assert!(pre > 0.05, "test needs a larger pre-clip norm: {pre}");
        assert!(post <= 0.05 + 1e-4, "post-clip {post} exceeds budget");
    }

    #[test]
    fn qat_training_reduces_loss_and_survives_quant() {
        let mut tr = LlamaTrainer::new(tiny(6, 16, 4, 2, 32, 2)).unwrap();
        tr.set_qat(true);
        assert!(tr.qat_enabled());
        tr.set_lr(0.01);
        let tokens = [1usize, 2, 3, 4, 5, 0, 1, 2];
        let l0 = tr.loss(&tokens).unwrap();
        let mut last = l0;
        for _ in 0..200 {
            last = tr.train_step(&tokens).unwrap();
        }
        assert!(last < l0 * 0.6, "QAT did not reduce loss: {l0} → {last}");
        assert!(last.is_finite());
    }
}
