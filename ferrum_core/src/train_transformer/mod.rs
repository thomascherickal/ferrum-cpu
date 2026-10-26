//! Trainable decoder-only causal Transformer: full backprop through token +
//! positional embeddings, LayerNorm, multi-head self-attention, and the FFN,
//! optimised with Adam. Mirrors the `train.rs` pattern: a dedicated trainable
//! struct (`TransformerNet`) that exports to an inference `Sequential` whose
//! layers (`Embedding`, `TransformerBlock`, `LayerNorm`, `Linear`) serialize
//! to GGUF and run in WASM.

mod backward;
mod block;
mod checkpoint;
mod epoch;
mod export;
mod forward;
mod kernels;
mod limits;
mod optimizer;
mod param;
mod qat;
mod source;
mod step_metrics;
#[cfg(test)]
mod test_support;
mod threaded;
mod tying;

pub use self::checkpoint::CheckpointInfo;
pub(crate) use self::epoch::steps_over;
pub use self::epoch::{train_transformer_epoch, train_transformer_steps, StepHook};
pub use self::forward::FwdCache;
pub(crate) use self::source::{Prefix, TokenSource};
pub use self::step_metrics::StepMetrics;
pub use self::threaded::train_transformer_epoch_threaded;

use crate::error::{InferError, Result};
use crate::rng::Rng;
use crate::train_transformer::block::BlockT;
use crate::train_transformer::param::Param;

/// A trainable decoder-only causal Transformer:
/// `Embedding → [TransformerBlock × N] → LayerNorm → Linear(vocab)`.
///
/// Train with `forward` / `backward` / `step` (Adam), then `to_inference()`
/// to obtain a `Sequential` that serializes to GGUF and runs anywhere
/// (including WASM with KV-cached generation).
///
/// `Clone` produces a deep copy of every weight, gradient, and Adam moment
/// buffer; [`train_transformer_epoch_threaded`] uses it to give each worker
/// thread an independent network to accumulate gradients into.
#[derive(Clone)]
pub struct TransformerNet {
    vocab_size: usize,
    context_len: usize,
    embed_dim: usize,
    num_heads: usize,
    tok_emb: Param, // [vocab, C]
    pos_emb: Param, // [T, C]
    blocks: Vec<BlockT>,
    lnf_g: Param,
    lnf_b: Param,
    head_w: Param, // [C, vocab]
    step_t: u64,
    /// Quantization Aware Training (int8 QAT simulating GGUF Q8_0): when
    /// enabled, `train_transformer_epoch` runs forward/backward against
    /// Q8_0-snapped weights while Adam updates
    /// full-precision master weights (straight-through estimator).
    qat: bool,
    /// Global-norm gradient clipping threshold. When `Some(max_norm)`, every
    /// training step rescales the gradients so their global L2 norm is at most
    /// `max_norm` before the Adam update, preventing divergence at higher
    /// learning rates / larger models. `None` (default) leaves gradients
    /// untouched, so the optimizer step is bit-identical to before.
    grad_clip: Option<f32>,
    /// Optional learning-rate schedule (warmup + decay). When `Some`, each
    /// [`TransformerNet::step`] overrides the Adam learning rate with
    /// `schedule.lr_at(step_t)`; `None` (default) uses the optimizer's fixed
    /// `lr`, so the step is unchanged.
    lr_schedule: Option<crate::optim::LrSchedule>,
    /// Weight tying (T9): when `true`, the LM head weight is kept equal to the
    /// transpose of the token-embedding table (`head_w = tok_embᵀ`) and their
    /// gradients are summed into the shared parameter, saving `vocab × embed`
    /// independent parameters. `false` (default) trains them independently.
    tie_weights: bool,
    /// Decoupled (AdamW) weight-decay coefficient applied to weight matrices
    /// each step (T7). `0.0` (default) disables it. Folded into the Adam used by
    /// [`TransformerNet::step`].
    weight_decay: f32,
    /// FFN-hidden dropout probability used during training forward passes (T7).
    /// `0.0` (default) disables it; inference is always dropout-free.
    dropout: f32,
    /// Distance in tokens between consecutive training-window starts. `1`
    /// (default) is the legacy fully-overlapping sliding window, where one epoch
    /// trains on every token about `context_len` times; `context_len` gives
    /// non-overlapping windows, so one epoch sees each token once.
    window_stride: usize,
    /// Opt-in per-step metrics (see [`StepMetrics`]); no effect on training.
    metrics: step_metrics::MetricsState,
}

impl TransformerNet {
    pub fn new(
        vocab_size: usize,
        context_len: usize,
        embed_dim: usize,
        num_heads: usize,
        hidden_dim: usize,
        num_blocks: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        if num_heads == 0 || !embed_dim.is_multiple_of(num_heads) {
            return Err(InferError::DimMismatch(format!(
                "embedding_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )));
        }
        if vocab_size == 0
            || context_len == 0
            || num_blocks == 0
            || embed_dim == 0
            || hidden_dim == 0
        {
            // (A zero embed/hidden width otherwise trained fine, then failed to save.)
            return Err(InferError::DimMismatch(
                "vocab_size, context_len, embed_dim, hidden_dim and num_blocks must be > 0".into(),
            ));
        }
        Self::check_size(vocab_size, context_len, embed_dim, hidden_dim, num_blocks)?;
        let scale = 0.02f32; // GPT-style init
        vprintln!("[train_transformer::TransformerNet::new] vocab={}, ctx={}, dim={}, heads={}, hidden={}, blocks={}",
            vocab_size, context_len, embed_dim, num_heads, hidden_dim, num_blocks);
        Ok(Self {
            vocab_size,
            context_len,
            embed_dim,
            num_heads,
            tok_emb: Param::randn(vec![vocab_size, embed_dim], scale, rng),
            pos_emb: Param::randn(vec![context_len, embed_dim], scale, rng),
            blocks: (0..num_blocks)
                .map(|_| BlockT::new(embed_dim, hidden_dim, scale, rng))
                .collect(),
            lnf_g: Param::constant(vec![embed_dim], 1.0),
            lnf_b: Param::constant(vec![embed_dim], 0.0),
            head_w: Param::randn(vec![embed_dim, vocab_size], scale, rng),
            step_t: 0,
            qat: false,
            grad_clip: None,
            lr_schedule: None,
            tie_weights: false,
            weight_decay: 0.0,
            dropout: 0.0,
            window_stride: 1,
            metrics: Default::default(),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn context_len(&self) -> usize {
        self.context_len
    }

    /// Enable or disable int8 QAT simulating GGUF Q8_0 (see the `qat`
    /// field docs). Off by default; `GenerativeSLM` turns it on.
    pub fn set_qat(&mut self, enabled: bool) {
        self.qat = enabled;
    }

    pub fn qat_enabled(&self) -> bool {
        self.qat
    }

    /// Set the global-norm gradient-clipping threshold (see the `grad_clip`
    /// field docs). `Some(max_norm)` clips before each step; `None` disables.
    pub fn set_grad_clip(&mut self, max_norm: Option<f32>) {
        self.grad_clip = max_norm;
    }

    pub fn grad_clip(&self) -> Option<f32> {
        self.grad_clip
    }

    /// Set the learning-rate schedule (warmup + decay). When `Some`, it
    /// overrides the Adam learning rate at every [`TransformerNet::step`] using
    /// the network's internal 1-based step counter; `None` keeps the fixed `lr`.
    pub fn set_lr_schedule(&mut self, schedule: Option<crate::optim::LrSchedule>) {
        self.lr_schedule = schedule;
    }

    pub fn lr_schedule(&self) -> Option<crate::optim::LrSchedule> {
        self.lr_schedule
    }

    /// The number of optimizer steps applied so far (the schedule's timestep).
    pub fn step_count(&self) -> u64 {
        self.step_t
    }

    /// Set the decoupled (AdamW) weight-decay coefficient applied to weight
    /// matrices each step (T7). `0.0` disables it.
    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.weight_decay = weight_decay;
    }

    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Set the FFN-hidden dropout probability used in training forward passes
    /// (T7), in `[0, 1)`. `0.0` disables it; values are clamped to `< 1`.
    pub fn set_dropout(&mut self, p: f32) {
        self.dropout = p.clamp(0.0, 0.95);
    }

    pub fn dropout(&self) -> f32 {
        self.dropout
    }

    /// Set the training-window stride (clamped to `1..=context_len`); see the
    /// field docs. Larger strides make an epoch proportionally cheaper.
    pub fn set_window_stride(&mut self, stride: usize) {
        self.window_stride = stride.clamp(1, self.context_len);
    }

    pub fn window_stride(&self) -> usize {
        self.window_stride
    }

    /// FFN hidden width (read from the first block's first FFN layer).
    fn hidden_dim(&self) -> usize {
        self.blocks[0].f1_b.data.numel()
    }

    /// Trainable parameters. A tied LM head shares the token embedding, so it
    /// is counted once.
    pub fn num_params(&self) -> usize {
        let head = if self.tie_weights {
            self.head_w.data.numel()
        } else {
            0
        };
        self.flat_len() - head
    }

    /// Length of the flat parameter/gradient vector: every tensor, a tied head
    /// included (its gradient is folded into the embedding's but still stored).
    fn flat_len(&self) -> usize {
        let block = |b: &BlockT| -> usize {
            [
                &b.ln1_g, &b.ln1_b, &b.q_w, &b.q_b, &b.k_w, &b.k_b, &b.v_w, &b.v_b, &b.o_w, &b.o_b,
                &b.ln2_g, &b.ln2_b, &b.f1_w, &b.f1_b, &b.f2_w, &b.f2_b,
            ]
            .iter()
            .map(|p| p.data.numel())
            .sum()
        };
        self.tok_emb.data.numel()
            + self.pos_emb.data.numel()
            + self.blocks.iter().map(block).sum::<usize>()
            + self.lnf_g.data.numel()
            + self.lnf_b.data.numel()
            + self.head_w.data.numel()
    }
}

#[cfg(test)]
mod tests {
    use super::TransformerNet;
    use crate::rng::Rng;

    #[test]
    fn invalid_configs_rejected() {
        let mut rng = Rng::new(1);
        assert!(TransformerNet::new(5, 3, 7, 2, 6, 1, &mut rng).is_err()); // 7 % 2 ≠ 0
        assert!(TransformerNet::new(5, 3, 4, 0, 6, 1, &mut rng).is_err());
        assert!(TransformerNet::new(0, 3, 4, 2, 6, 1, &mut rng).is_err());
        assert!(TransformerNet::new(5, 3, 4, 2, 6, 0, &mut rng).is_err());
    }
}
