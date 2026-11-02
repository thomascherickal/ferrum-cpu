//! Adam step, gradient clipping and parameter / gradient views.
use crate::error::Result;
use crate::optim::Adam;
use crate::tensor::Tensor;
use crate::train_transformer::param::Param;
use crate::train_transformer::TransformerNet;

impl TransformerNet {
    /// Rescale all gradients so their global L2 norm is at most `max_norm`,
    /// returning the pre-clip norm. Operates on the gradients currently held in
    /// every parameter (i.e. after [`TransformerNet::backward`]). Exposed for
    /// callers driving their own training loop; the built-in epoch helpers call
    /// it automatically when [`TransformerNet::set_grad_clip`] is set.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let mut grads = self.grad_tensors_mut();
        crate::optim::clip_grad_norm(&mut grads, max_norm)
    }

    /// Every `Param` (weights + Adam moments), in canonical order. Used by the
    /// checkpoint codec (T6).
    pub(super) fn params_all(&self) -> Vec<&Param> {
        let mut v: Vec<&Param> = vec![&self.tok_emb, &self.pos_emb];
        for b in &self.blocks {
            v.extend(b.params());
        }
        v.push(&self.lnf_g);
        v.push(&self.lnf_b);
        v.push(&self.head_w);
        v
    }

    pub(super) fn params_mut_all(&mut self) -> Vec<&mut Param> {
        let mut v: Vec<&mut Param> = vec![&mut self.tok_emb, &mut self.pos_emb];
        for b in &mut self.blocks {
            v.extend(b.params_mut());
        }
        v.push(&mut self.lnf_g);
        v.push(&mut self.lnf_b);
        v.push(&mut self.head_w);
        v
    }

    /// Every parameter tensor, in a fixed order (used by the QAT snapshot /
    /// fake-quantize / restore cycle).
    pub(super) fn param_tensors_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v: Vec<&mut Tensor> = vec![&mut self.tok_emb.data, &mut self.pos_emb.data];
        for b in &mut self.blocks {
            for p in b.params_mut() {
                v.push(&mut p.data);
            }
        }
        v.push(&mut self.lnf_g.data);
        v.push(&mut self.lnf_b.data);
        v.push(&mut self.head_w.data);
        v
    }

    /// Every gradient tensor, in the same fixed order as
    /// [`TransformerNet::param_tensors_mut`]. Used to reduce per-shard
    /// gradients in [`train_transformer_epoch_threaded`].
    pub(super) fn grad_tensors_mut(&mut self) -> Vec<&mut Tensor> {
        let mut v: Vec<&mut Tensor> = vec![&mut self.tok_emb.grad, &mut self.pos_emb.grad];
        for b in &mut self.blocks {
            v.push(&mut b.ln1_g.grad);
            v.push(&mut b.ln1_b.grad);
            v.push(&mut b.q_w.grad);
            v.push(&mut b.q_b.grad);
            v.push(&mut b.k_w.grad);
            v.push(&mut b.k_b.grad);
            v.push(&mut b.v_w.grad);
            v.push(&mut b.v_b.grad);
            v.push(&mut b.o_w.grad);
            v.push(&mut b.o_b.grad);
            v.push(&mut b.ln2_g.grad);
            v.push(&mut b.ln2_b.grad);
            v.push(&mut b.f1_w.grad);
            v.push(&mut b.f1_b.grad);
            v.push(&mut b.f2_w.grad);
            v.push(&mut b.f2_b.grad);
        }
        v.push(&mut self.lnf_g.grad);
        v.push(&mut self.lnf_b.grad);
        v.push(&mut self.head_w.grad);
        v
    }

    pub fn zero_grad(&mut self) {
        self.tok_emb.zero_grad();
        self.pos_emb.zero_grad();
        for b in &mut self.blocks {
            for p in b.params_mut() {
                p.zero_grad();
            }
        }
        self.lnf_g.zero_grad();
        self.lnf_b.zero_grad();
        self.head_w.zero_grad();
    }

    /// Apply one Adam update to every parameter. When a learning-rate schedule
    /// is set ([`TransformerNet::set_lr_schedule`]), the Adam learning rate for
    /// this step is taken from `schedule.lr_at(step_t)` instead of the fixed
    /// `adam.lr`.
    pub fn step(&mut self, adam: &Adam) -> Result<()> {
        self.step_t += 1;
        let t = self.step_t;
        // Adam is `Copy`; override the learning rate when scheduled and fold in
        // the net's decoupled weight decay (AdamW).
        let mut adam = *adam;
        if let Some(sched) = self.lr_schedule {
            adam.lr = sched.lr_at(t);
            vprintln!(
                "[train_transformer::step] step {t}: scheduled lr={:.6e}",
                adam.lr
            );
        }
        if self.weight_decay != 0.0 {
            adam.weight_decay = self.weight_decay;
        }
        let adam = &adam;
        self.tok_emb.step(adam, t)?;
        self.pos_emb.step(adam, t)?;
        for b in &mut self.blocks {
            for p in b.params_mut() {
                p.step(adam, t)?;
            }
        }
        self.lnf_g.step(adam, t)?;
        self.lnf_b.step(adam, t)?;
        self.head_w.step(adam, t)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::loss::softmax_cross_entropy;
    use crate::optim::Adam;
    use crate::rng::Rng;
    use crate::train_transformer::epoch::train_transformer_epoch;
    use crate::train_transformer::TransformerNet;

    /// Decoupled weight decay shrinks weight matrices but never biases or
    /// LayerNorm parameters (rank-1).
    #[test]
    fn weight_decay_decays_matrices_only() {
        let mut rng = Rng::new(1);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_weight_decay(0.1);
        assert!((net.weight_decay() - 0.1).abs() < 1e-9);
        // Known values; zero gradients so only weight decay acts.
        for v in net.tok_emb.data.data.iter_mut() {
            *v = 2.0;
        } // 2-D weight
        for v in net.blocks[0].f1_b.data.data.iter_mut() {
            *v = 2.0;
        } // 1-D bias
        for v in net.lnf_g.data.data.iter_mut() {
            *v = 2.0;
        } // 1-D LN gain
        net.zero_grad();
        let adam = Adam::new(0.5); // lr; decay coefficient comes from the net
        net.step(&adam).unwrap();
        // Weight matrix: 2 − 0.5·0.1·2 = 1.9
        assert!(
            (net.tok_emb.data.data[0] - 1.9).abs() < 1e-5,
            "tok_emb not decayed"
        );
        // Bias and LN gain untouched.
        assert!(
            (net.blocks[0].f1_b.data.data[0] - 2.0).abs() < 1e-6,
            "bias was decayed"
        );
        assert!(
            (net.lnf_g.data.data[0] - 2.0).abs() < 1e-6,
            "LN gain was decayed"
        );
    }

    #[test]
    fn weight_decay_trains_and_shrinks_weight_matrices() {
        let base = {
            let mut r = Rng::new(7);
            TransformerNet::new(6, 4, 8, 2, 16, 1, &mut r).unwrap()
        };
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.01);
        // L2 norm over the actual weight matrices (where decay applies), not the
        // un-decayed LayerNorm gains/biases.
        let matrix_norm = |net: &TransformerNet| -> f32 {
            let b = &net.blocks[0];
            [
                &b.q_w,
                &b.k_w,
                &b.v_w,
                &b.o_w,
                &b.f1_w,
                &b.f2_w,
                &net.head_w,
                &net.tok_emb,
            ]
            .iter()
            .flat_map(|p| p.data.data.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
        };

        let mut plain = base.clone();
        let mut rp = Rng::new(3);
        let mut decayed = base.clone();
        decayed.set_weight_decay(0.3);
        let mut rd = Rng::new(3);

        let mut first = f32::NAN;
        let mut last = 0.0;
        for e in 0..30 {
            train_transformer_epoch(&mut plain, &tokens, 8, &adam, &mut rp).unwrap();
            let l = train_transformer_epoch(&mut decayed, &tokens, 8, &adam, &mut rd).unwrap();
            if e == 0 {
                first = l;
            }
            last = l;
        }
        assert!(
            last < first,
            "decayed run should still reduce loss: {first:.4} → {last:.4}"
        );
        assert!(
            matrix_norm(&decayed) < matrix_norm(&plain),
            "weight decay should shrink the weight matrices"
        );
    }

    /// A warmup+decay schedule is actually applied during training: it drives a
    /// different trajectory than the same base LR held fixed, advances the step
    /// counter, and still reduces loss.
    #[test]
    fn lr_schedule_is_applied_and_reduces_loss() {
        use crate::optim::LrSchedule;
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.02);

        // Fixed-LR reference.
        let mut fixed = base.clone();
        let mut rf = Rng::new(5);
        for _ in 0..10 {
            train_transformer_epoch(&mut fixed, &tokens, 8, &adam, &mut rf).unwrap();
        }

        // Scheduled run: warmup then cosine decay over the same total steps.
        let steps_per_epoch = (tokens.len() - 4).div_ceil(8) as u64;
        let total = steps_per_epoch * 10;
        let mut sched_net = base.clone();
        sched_net.set_lr_schedule(Some(LrSchedule::warmup_cosine(
            0.02,
            steps_per_epoch * 2,
            total,
        )));
        assert!(sched_net.lr_schedule().is_some());
        let mut rs = Rng::new(5);
        let first = train_transformer_epoch(&mut sched_net, &tokens, 8, &adam, &mut rs).unwrap();
        let mut last = first;
        for _ in 0..9 {
            last = train_transformer_epoch(&mut sched_net, &tokens, 8, &adam, &mut rs).unwrap();
        }

        assert_eq!(
            sched_net.step_count(),
            total,
            "step counter must drive the schedule"
        );
        assert!(
            last < first,
            "scheduled run should reduce loss: {first:.4} → {last:.4}"
        );
        // Warmup means the scheduled run takes a genuinely different path.
        assert_ne!(
            fixed.snapshot_weights(),
            sched_net.snapshot_weights(),
            "schedule had no effect on training"
        );
    }

    /// `clip_grad_norm` caps the post-backward global gradient norm at the
    /// requested budget and returns the (larger) pre-clip norm.
    #[test]
    fn clip_grad_norm_caps_global_norm() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng).unwrap();
        let tokens = [1usize, 2, 3, 4];
        let targets = [2usize, 3, 4, 5];
        let (logits, cache) = net.forward(&tokens).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();
        net.zero_grad();
        net.backward(&dl, &cache).unwrap();

        let max_norm = 0.1f32;
        let pre = net.clip_grad_norm(max_norm);
        // Recompute the global norm from the (now clipped) gradients.
        let mut sumsq = 0.0f64;
        for g in net.grad_tensors_mut() {
            for &x in &g.data {
                sumsq += (x as f64) * (x as f64);
            }
        }
        let post = sumsq.sqrt() as f32;
        assert!(
            pre > max_norm,
            "test needs an exploding-enough grad: pre={pre}"
        );
        assert!(
            post <= max_norm + 1e-4,
            "post-clip norm {post} exceeds budget {max_norm}"
        );
    }

    /// Gradient clipping is actually wired into the epoch: with the same seed,
    /// a clipped run takes a different trajectory than an unclipped one (because
    /// the early gradients exceed the budget), and stays finite while still
    /// reducing loss.
    #[test]
    fn grad_clip_changes_trajectory_and_stays_finite() {
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(6, 6, 12, 2, 24, 2, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..300).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.05);

        let finite = |net: &mut TransformerNet| {
            net.snapshot_weights()
                .iter()
                .all(|t| t.iter().all(|v| v.is_finite()))
        };

        let mut unclipped = base.clone();
        let mut r1 = Rng::new(1);
        for _ in 0..5 {
            train_transformer_epoch(&mut unclipped, &tokens, 16, &adam, &mut r1).unwrap();
        }

        let mut clipped = base.clone();
        clipped.set_grad_clip(Some(0.01)); // tiny budget → clips most steps
        assert_eq!(clipped.grad_clip(), Some(0.01));
        let mut r2 = Rng::new(1);
        let first = train_transformer_epoch(&mut clipped, &tokens, 16, &adam, &mut r2).unwrap();
        let mut last = first;
        for _ in 0..4 {
            last = train_transformer_epoch(&mut clipped, &tokens, 16, &adam, &mut r2).unwrap();
        }

        assert!(finite(&mut clipped), "clipped run must stay finite");
        assert!(
            last < first,
            "clipped run should still reduce loss: {first:.4} → {last:.4}"
        );
        // The clip threshold changed the optimization path.
        assert_ne!(
            unclipped.snapshot_weights(),
            clipped.snapshot_weights(),
            "grad clipping had no effect on the weights"
        );
    }
}
