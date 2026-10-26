//! A trainable tensor with its gradient and Adam moment buffers.
use crate::error::Result;
use crate::optim::Adam;
use crate::rng::Rng;
use crate::tensor::Tensor;

#[derive(Clone)]
pub(super) struct Param {
    pub(super) data: Tensor,
    pub(super) grad: Tensor,
    pub(super) m: Tensor,
    pub(super) v: Tensor,
}

impl Param {
    pub(super) fn new(shape: Vec<usize>, init: Vec<f32>) -> Result<Self> {
        Ok(Self {
            grad: Tensor::zeros(shape.clone()),
            m: Tensor::zeros(shape.clone()),
            v: Tensor::zeros(shape.clone()),
            data: Tensor::new(shape, init)?,
        })
    }
    pub(super) fn randn(shape: Vec<usize>, scale: f32, rng: &mut Rng) -> Self {
        let n: usize = shape.iter().product();
        let init: Vec<f32> = (0..n).map(|_| rng.next_normal() * scale).collect();
        Self::new(shape, init).expect("init has exactly shape.product() elements")
    }
    pub(super) fn constant(shape: Vec<usize>, value: f32) -> Self {
        let n: usize = shape.iter().product();
        Self::new(shape, vec![value; n])
            .expect("vec![value; n] has exactly shape.product() elements")
    }
    /// A copy of the weights alone: gradient and Adam moments are empty, so it
    /// costs 4 B/param instead of 16.
    pub(super) fn weights_only(&self) -> Self {
        let empty = || Tensor::zeros(vec![0]);
        Self {
            data: self.data.clone(),
            grad: empty(),
            m: empty(),
            v: empty(),
        }
    }
    pub(super) fn zero_grad(&mut self) {
        for g in &mut self.grad.data {
            *g = 0.0;
        }
    }
    pub(super) fn step(&mut self, adam: &Adam, t: u64) -> Result<()> {
        // Decoupled weight decay (AdamW) applies to weight matrices only — never
        // to biases or LayerNorm gains/biases (rank-1 params), per convention.
        if adam.weight_decay != 0.0 && self.data.shape.len() < 2 {
            let mut a = *adam;
            a.weight_decay = 0.0;
            return a.step(t, &mut self.data, &self.grad, &mut self.m, &mut self.v);
        }
        adam.step(t, &mut self.data, &self.grad, &mut self.m, &mut self.v)
    }
}
