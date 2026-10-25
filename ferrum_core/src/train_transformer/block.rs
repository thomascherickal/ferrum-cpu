//! One trainable transformer block and its gradient / activation records.
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::train_transformer::param::Param;

#[derive(Clone)]
pub(super) struct BlockT {
    pub(super) ln1_g: Param,
    pub(super) ln1_b: Param,
    pub(super) q_w: Param,
    pub(super) q_b: Param,
    pub(super) k_w: Param,
    pub(super) k_b: Param,
    pub(super) v_w: Param,
    pub(super) v_b: Param,
    pub(super) o_w: Param,
    pub(super) o_b: Param,
    pub(super) ln2_g: Param,
    pub(super) ln2_b: Param,
    pub(super) f1_w: Param,
    pub(super) f1_b: Param,
    pub(super) f2_w: Param,
    pub(super) f2_b: Param,
}

impl BlockT {
    pub(super) fn new(c: usize, hidden: usize, scale: f32, rng: &mut Rng) -> Self {
        Self {
            ln1_g: Param::constant(vec![c], 1.0),
            ln1_b: Param::constant(vec![c], 0.0),
            q_w: Param::randn(vec![c, c], scale, rng),
            q_b: Param::constant(vec![c], 0.0),
            k_w: Param::randn(vec![c, c], scale, rng),
            k_b: Param::constant(vec![c], 0.0),
            v_w: Param::randn(vec![c, c], scale, rng),
            v_b: Param::constant(vec![c], 0.0),
            o_w: Param::randn(vec![c, c], scale, rng),
            o_b: Param::constant(vec![c], 0.0),
            ln2_g: Param::constant(vec![c], 1.0),
            ln2_b: Param::constant(vec![c], 0.0),
            f1_w: Param::randn(vec![c, hidden], scale, rng),
            f1_b: Param::constant(vec![hidden], 0.0),
            f2_w: Param::randn(vec![hidden, c], scale, rng),
            f2_b: Param::constant(vec![c], 0.0),
        }
    }

    pub(super) fn weights_only(&self) -> Self {
        Self {
            ln1_g: self.ln1_g.weights_only(),
            ln1_b: self.ln1_b.weights_only(),
            q_w: self.q_w.weights_only(),
            q_b: self.q_b.weights_only(),
            k_w: self.k_w.weights_only(),
            k_b: self.k_b.weights_only(),
            v_w: self.v_w.weights_only(),
            v_b: self.v_b.weights_only(),
            o_w: self.o_w.weights_only(),
            o_b: self.o_b.weights_only(),
            ln2_g: self.ln2_g.weights_only(),
            ln2_b: self.ln2_b.weights_only(),
            f1_w: self.f1_w.weights_only(),
            f1_b: self.f1_b.weights_only(),
            f2_w: self.f2_w.weights_only(),
            f2_b: self.f2_b.weights_only(),
        }
    }

    pub(super) fn params_mut(&mut self) -> [&mut Param; 16] {
        [
            &mut self.ln1_g,
            &mut self.ln1_b,
            &mut self.q_w,
            &mut self.q_b,
            &mut self.k_w,
            &mut self.k_b,
            &mut self.v_w,
            &mut self.v_b,
            &mut self.o_w,
            &mut self.o_b,
            &mut self.ln2_g,
            &mut self.ln2_b,
            &mut self.f1_w,
            &mut self.f1_b,
            &mut self.f2_w,
            &mut self.f2_b,
        ]
    }

    pub(super) fn params(&self) -> [&Param; 16] {
        [
            &self.ln1_g,
            &self.ln1_b,
            &self.q_w,
            &self.q_b,
            &self.k_w,
            &self.k_b,
            &self.v_w,
            &self.v_b,
            &self.o_w,
            &self.o_b,
            &self.ln2_g,
            &self.ln2_b,
            &self.f1_w,
            &self.f1_b,
            &self.f2_w,
            &self.f2_b,
        ]
    }
}

/// Owned gradient tensors for one `BlockT`, produced by
/// [`TransformerNet::backward_grads`] so a worker can compute gradients without
/// mutating (or cloning) the shared parameters. Fields mirror `BlockT`.
pub(super) struct BlockGrads {
    pub(super) ln1_g: Tensor,
    pub(super) ln1_b: Tensor,
    pub(super) q_w: Tensor,
    pub(super) q_b: Tensor,
    pub(super) k_w: Tensor,
    pub(super) k_b: Tensor,
    pub(super) v_w: Tensor,
    pub(super) v_b: Tensor,
    pub(super) o_w: Tensor,
    pub(super) o_b: Tensor,
    pub(super) ln2_g: Tensor,
    pub(super) ln2_b: Tensor,
    pub(super) f1_w: Tensor,
    pub(super) f1_b: Tensor,
    pub(super) f2_w: Tensor,
    pub(super) f2_b: Tensor,
}

pub(super) struct BlockCache {
    pub(super) xhat1: Tensor,
    pub(super) inv_std1: Vec<f32>,
    pub(super) norm1: Tensor,
    pub(super) q: Tensor,
    pub(super) k: Tensor,
    pub(super) v: Tensor,
    pub(super) probs: Vec<f32>, // [B·H·T·T]
    pub(super) concat: Tensor,  // attention head outputs before out_proj [M, C]
    pub(super) xhat2: Tensor,
    pub(super) inv_std2: Vec<f32>,
    pub(super) norm2: Tensor,
    /// FFN-1 pre-activation [M, hidden].
    pub(super) h_pre: Tensor,
    /// FFN-2 input [M, hidden]: GELU(h) — or, when dropout is active, the
    /// dropped-and-scaled hidden `GELU(h) ⊙ mask`.
    pub(super) h_act: Tensor,
    /// Inverted-dropout mask×scale applied to the GELU output (T7), present only when
    /// dropout was active for this forward; `None` means no dropout.
    pub(super) ffn_dropout: Option<Vec<f32>>,
}
