//! Training (the backward pass) for the Llama/Qwen decoder in [`crate::llm`].
//!
//! `llm.rs` is forward-only; without gradients the imported architecture cannot
//! be trained or fine-tuned. This module adds hand-derived backprop for every
//! primitive that block uses — **RMSNorm, RoPE, grouped-query attention (with the
//! softmax), the SwiGLU FFN, the token embedding, and the LM head** — plus a
//! next-token cross-entropy loss. Each primitive's gradient is checked against
//! finite differences in the tests, and an end-to-end test shows the loss falling
//! on a memorisable sequence.
//!
//! On top of that bare gradient it provides a full fine-tuning stack that mirrors
//! [`crate::train_transformer`]:
//!
//! - **AdamW** optimizer with bias correction and decoupled weight decay
//!   (matrices only), replacing the original plain SGD.
//! - **Minibatching** over sequence windows ([`LlamaTrainer::train_batch`]) and
//!   true-epoch drivers that shuffle windows without replacement
//!   ([`LlamaTrainer::finetune_epoch`]).
//! - **Data-parallel** epochs on `std::thread::scope`
//!   ([`LlamaTrainer::finetune_epoch_threaded`]) with deterministic shard
//!   reduction; single-shard is bit-identical to the serial path.
//! - **Gradient clipping** (global L2 norm), a **warmup+decay learning-rate
//!   schedule**, and **FFN-hidden dropout** for regularised fine-tuning.
//! - **Quantization-aware training** (straight-through int8) so a fine-tuned
//!   model survives int8 export, and **checkpoint save / resume**
//!   ([`LlamaTrainer::save_checkpoint`] / [`LlamaTrainer::load_checkpoint_into`])
//!   that preserves the optimizer moments, step counter, and RNG state.
//!
//! Training runs in **f32**: the model must hold full-precision weights
//! (`Gguf::load_llama_prec(None)`, or built f32 directly). Quantized (`QWeight`)
//! Linears have no f32 master to update, so [`LlamaTrainer::new`] rejects them.
//!
//! Scope/reality check: this makes the architecture *trainable* and is exercised
//! on small models. It does **not** make training a 1B model on a CPU feasible —
//! that stays bounded by compute and RAM (see Manual.md §13.5). It adds
//! the missing capability (gradients + a real optimizer), not a claim about scale.
#![warn(missing_docs)]

mod blocks;
mod buffers;
mod checkpoint;
mod finetune;
mod kernels;
mod optim;
mod params;
mod passes;
mod run;
#[cfg(test)]
mod test_support;

pub use self::checkpoint::apply_checkpoint_weights;
pub use self::run::FinetuneRun;

use crate::error::Result;
use crate::llm::LlamaModel;
use crate::llm_train::buffers::Buffers;
use crate::llm_train::kernels::{check_loss_finite, cross_entropy};
use crate::llm_train::params::{ensure_f32_weights, param_data_ref};
use crate::llm_train::passes::{backward_into, forward_train};
use crate::optim::{Adam, LrSchedule};

/// Wraps an f32 [`LlamaModel`] and fine-tunes it by next-token cross-entropy
/// with AdamW. See the module docs for the full feature set.
pub struct LlamaTrainer {
    /// The model being fine-tuned; its weights are updated in place.
    pub model: LlamaModel,
    grads: Buffers,
    m: Buffers,
    v: Buffers,
    adam: Adam,
    weight_decay: f32,
    grad_clip: Option<f32>,
    lr_schedule: Option<LrSchedule>,
    dropout: f32,
    qat: bool,
    step_t: u64,
    /// The run recorded in (and checked against) checkpoints; see [`LlamaTrainer::begin_run`].
    run: run::RunState,
}

impl LlamaTrainer {
    /// Build a trainer over `model`. Errors if any projection is still quantized
    /// (training needs f32 masters — load with `Gguf::load_llama_prec(None)`).
    /// Defaults to AdamW with `lr = 1e-4` and no weight decay; tune with the
    /// setters before training.
    pub fn new(model: LlamaModel) -> Result<Self> {
        ensure_f32_weights(&model)?;
        let grads = Buffers::zeros(&model);
        let m = Buffers::zeros(&model);
        let v = Buffers::zeros(&model);
        Ok(Self {
            model,
            grads,
            m,
            v,
            adam: Adam::new(1e-4),
            weight_decay: 0.0,
            grad_clip: None,
            lr_schedule: None,
            dropout: 0.0,
            qat: false,
            step_t: 0,
            run: Default::default(),
        })
    }

    /// Replace the optimizer settings (learning rate, betas, eps). The decoupled
    /// weight decay is taken from [`LlamaTrainer::set_weight_decay`] /
    /// `Adam::weight_decay`, whichever is non-zero.
    pub fn set_optimizer(&mut self, adam: Adam) {
        self.adam = adam;
        if adam.weight_decay != 0.0 {
            self.weight_decay = adam.weight_decay;
        }
    }

    /// Set the base learning rate (used when no schedule is active).
    pub fn set_lr(&mut self, lr: f32) {
        self.adam.lr = lr;
    }

    /// The base learning rate currently set on the optimizer.
    pub fn lr(&self) -> f32 {
        self.adam.lr
    }

    /// Set the decoupled (AdamW) weight-decay coefficient applied to weight
    /// matrices each step. `0.0` disables it.
    pub fn set_weight_decay(&mut self, wd: f32) {
        self.weight_decay = wd;
    }

    /// The decoupled weight-decay coefficient.
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Set the global-norm gradient-clipping threshold. `Some(max)` clips before
    /// each step; `None` disables.
    pub fn set_grad_clip(&mut self, max_norm: Option<f32>) {
        self.grad_clip = max_norm;
    }

    /// The global-norm gradient-clipping threshold, or `None` when clipping is off.
    pub fn grad_clip(&self) -> Option<f32> {
        self.grad_clip
    }

    /// Set the learning-rate schedule (warmup + decay). When `Some`, it overrides
    /// the Adam learning rate at every step using the internal step counter.
    pub fn set_lr_schedule(&mut self, schedule: Option<LrSchedule>) {
        self.lr_schedule = schedule;
    }

    /// The active learning-rate schedule, or `None` when the fixed rate is used.
    pub fn lr_schedule(&self) -> Option<LrSchedule> {
        self.lr_schedule
    }

    /// Set the FFN-hidden dropout probability used in training forward passes,
    /// in `[0, 1)`. `0.0` disables it; inference is always dropout-free.
    pub fn set_dropout(&mut self, p: f32) {
        self.dropout = p.clamp(0.0, 0.95);
    }

    /// The FFN-hidden dropout probability used in training forward passes.
    pub fn dropout(&self) -> f32 {
        self.dropout
    }

    /// Enable/disable int8 quantization-aware training (straight-through). When
    /// on, each step computes gradients at int8-snapped weights but updates the
    /// f32 masters, so the model is robust to int8 export.
    pub fn set_qat(&mut self, enabled: bool) {
        self.qat = enabled;
    }

    /// Whether int8 quantization-aware training is enabled.
    pub fn qat_enabled(&self) -> bool {
        self.qat
    }

    /// Number of optimizer steps applied so far (the schedule's timestep).
    pub fn step_count(&self) -> u64 {
        self.step_t
    }

    /// Total number of trainable scalar parameters.
    pub fn num_params(&self) -> usize {
        param_data_ref(&self.model).iter().map(|s| s.len()).sum()
    }

    /// A per-parameter clone of the current model weights, in canonical order.
    /// Handy for comparing training runs (e.g. verifying a resumed checkpoint
    /// matches the model it was saved from).
    pub fn model_snapshot(&self) -> Vec<Vec<f32>> {
        param_data_ref(&self.model)
            .iter()
            .map(|s| s.to_vec())
            .collect()
    }

    /// Mean next-token cross-entropy over `tokens` (position `t` predicts
    /// `tokens[t+1]`). Pure forward — no gradients, no dropout.
    pub fn loss(&self, tokens: &[usize]) -> Result<f32> {
        let cache = forward_train(&self.model, tokens, 0.0, 0)?;
        Ok(cross_entropy(&cache.logits, tokens, self.model.cfg.vocab_size).0)
    }

    /// Forward + backward over a single sequence; fills `self.grads`
    /// (dropout-free, deterministic). Returns the loss. Exposed for callers
    /// driving their own loop and used by the gradient-check tests.
    pub fn loss_and_backward(&mut self, tokens: &[usize]) -> Result<f32> {
        let cache = forward_train(&self.model, tokens, 0.0, 0)?;
        let (loss, d_logits) = cross_entropy(&cache.logits, tokens, self.model.cfg.vocab_size);
        check_loss_finite(loss)?;
        self.grads.clear();
        backward_into(&self.model, &cache, &d_logits, tokens, &mut self.grads)?;
        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaTrainer;
    use crate::llm::RopeType;
    use crate::llm_train::kernels::cross_entropy;
    use crate::llm_train::passes::{backward_into, forward_train};
    use crate::llm_train::test_support::{base_clone, tiny};
    use crate::optim::Adam;
    use crate::optim::LrSchedule;
    use crate::rng::Rng;

    #[test]
    fn training_reduces_loss_on_a_memorizable_sequence() {
        let model = tiny(6, 16, 4, 2, 32, 2);
        let mut tr = LlamaTrainer::new(model).unwrap();
        tr.set_lr(0.01);
        let tokens = [1usize, 2, 3, 4, 5, 0, 1, 2];
        let l0 = tr.loss(&tokens).unwrap();
        let mut last = l0;
        for _ in 0..200 {
            last = tr.train_step(&tokens).unwrap();
        }
        let lf = tr.loss(&tokens).unwrap();
        assert!(
            lf < l0 * 0.5,
            "loss did not fall enough: {l0} → {lf} (last step {last})"
        );
        assert!(lf.is_finite());
    }

    #[test]
    fn rejects_quantized_model() {
        let mut model = tiny(6, 8, 2, 2, 16, 1);
        model.lm_head = model.lm_head.quantize(crate::quant::QKind::Int8);
        assert!(LlamaTrainer::new(model).is_err());
    }

    #[test]
    fn lr_schedule_changes_trajectory() {
        let base = tiny(5, 16, 4, 2, 32, 1);
        let tokens: Vec<usize> = (0..100).map(|i| i % 5).collect();

        let mut fixed = LlamaTrainer::new(base_clone(&base)).unwrap();
        fixed.set_lr(0.02);
        let mut rf = Rng::new(5);
        for _ in 0..5 {
            fixed.finetune_epoch(&tokens, 8, 8, &mut rf).unwrap();
        }

        let mut sched = LlamaTrainer::new(base_clone(&base)).unwrap();
        sched.set_lr(0.02);
        let steps_per_epoch = (tokens.len() - 8 + 1).div_ceil(8) as u64;
        sched.set_lr_schedule(Some(LrSchedule::warmup_cosine(
            0.02,
            steps_per_epoch,
            steps_per_epoch * 5,
        )));
        let mut rsc = Rng::new(5);
        for _ in 0..5 {
            sched.finetune_epoch(&tokens, 8, 8, &mut rsc).unwrap();
        }
        assert_eq!(sched.step_count(), steps_per_epoch * 5);
        let differ = fixed
            .model_flat()
            .iter()
            .zip(&sched.model_flat())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differ, "schedule had no effect");
    }

    #[test]
    fn dropout_off_matches_plain_and_gradient_is_correct() {
        // dropout=0 forward equals the plain path.
        let model = tiny(5, 8, 2, 2, 16, 1);
        let tokens = [1usize, 2, 3, 4];
        let a = forward_train(&model, &tokens, 0.0, 123).unwrap();
        let b = forward_train(&model, &tokens, 0.0, 999).unwrap();
        assert_eq!(a.logits, b.logits);

        // Same seed reproduces the masked forward; different seed changes it.
        let c = forward_train(&model, &tokens, 0.5, 7).unwrap();
        let d = forward_train(&model, &tokens, 0.5, 7).unwrap();
        let e = forward_train(&model, &tokens, 0.5, 8).unwrap();
        assert_eq!(c.logits, d.logits);
        assert_ne!(c.logits, e.logits);

        // Gradient through a fixed dropout mask matches finite differences.
        let mut tr = LlamaTrainer::new(base_clone(&model)).unwrap();
        let (seed, p) = (77u64, 0.5f32);
        let cache = forward_train(&tr.model, &tokens, p, seed).unwrap();
        let (_, dl) = cross_entropy(&cache.logits, &tokens, 5);
        tr.grads.clear();
        backward_into(&tr.model, &cache, &dl, &tokens, &mut tr.grads).unwrap();
        let eps = 1e-2f32;
        for &i in &[0usize, 7, 15] {
            let g = tr.grads.blocks[0].down.w[i];
            tr.model.blocks[0].ffn.down.weight.data[i] += eps;
            let lp = cross_entropy(
                &forward_train(&tr.model, &tokens, p, seed).unwrap().logits,
                &tokens,
                5,
            )
            .0;
            tr.model.blocks[0].ffn.down.weight.data[i] -= 2.0 * eps;
            let lm = cross_entropy(
                &forward_train(&tr.model, &tokens, p, seed).unwrap().logits,
                &tokens,
                5,
            )
            .0;
            tr.model.blocks[0].ffn.down.weight.data[i] += eps;
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (fd - g).abs() < 6e-3 + 0.03 * g.abs(),
                "down.w[{i}] fd={fd} got={g}"
            );
        }
    }

    #[test]
    fn hyperparameter_setters_and_getters_roundtrip() {
        let mut t = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        t.set_lr(0.25);
        assert_eq!(t.lr(), 0.25);
        t.set_weight_decay(0.125);
        assert_eq!(t.weight_decay(), 0.125);
        assert_eq!(t.grad_clip(), None);
        t.set_grad_clip(Some(1.5));
        assert_eq!(t.grad_clip(), Some(1.5));
        t.set_grad_clip(None);
        assert_eq!(t.grad_clip(), None);
        assert!(t.lr_schedule().is_none());
        t.set_lr_schedule(Some(LrSchedule::warmup_cosine(0.1, 2, 10)));
        assert!(t.lr_schedule().is_some());
        t.set_dropout(0.3);
        assert_eq!(t.dropout(), 0.3);
        t.set_dropout(5.0); // clamped below 1
        assert!(t.dropout() < 1.0);
        // set_optimizer adopts a non-zero AdamW decay, and keeps the old one otherwise.
        t.set_optimizer(Adam::with_weight_decay(0.5, 0.2));
        assert_eq!((t.lr(), t.weight_decay()), (0.5, 0.2));
        t.set_optimizer(Adam::new(0.75));
        assert_eq!((t.lr(), t.weight_decay()), (0.75, 0.2));
    }

    #[test]
    fn out_of_vocab_tokens_are_an_error() {
        let mut t = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        let e = t.loss(&[0, 1, 99]).unwrap_err();
        assert!(e.to_string().contains("vocab"), "{e}");
        assert!(t.loss_and_backward(&[0, 99]).is_err());
        assert!(t.loss_and_backward(&[0, 1, 2, 3]).unwrap().is_finite());
    }

    #[test]
    fn neox_rope_models_train_with_finite_loss() {
        let mut m = tiny(6, 8, 2, 2, 16, 1);
        m.cfg.rope_type = RopeType::Neox;
        let mut t = LlamaTrainer::new(m).unwrap();
        t.set_lr(0.02);
        let seq: Vec<usize> = (0..12).map(|i| i % 6).collect();
        let first = t.train_step(&seq).unwrap();
        let mut last = first;
        for _ in 0..20 {
            last = t.train_step(&seq).unwrap();
        }
        assert!(first.is_finite() && last.is_finite());
        assert!(last < first, "{first} -> {last}");
    }
}
