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
        self.grad.data.fill(0.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_shape_the_buffers() {
        let p = Param::constant(vec![2, 3], 0.5);
        assert_eq!(p.data.data, vec![0.5; 6]);
        assert_eq!(
            (p.grad.shape.clone(), p.m.shape.clone(), p.v.shape.clone()),
            (vec![2, 3], vec![2, 3], vec![2, 3])
        );
        assert!(p
            .grad
            .data
            .iter()
            .chain(&p.m.data)
            .chain(&p.v.data)
            .all(|&x| x == 0.0));
        let r = Param::randn(vec![4], 0.1, &mut Rng::new(7));
        assert_eq!(r.data.data.len(), 4);
        assert!(Param::new(vec![2, 2], vec![1.0; 3]).is_err());
    }

    #[test]
    fn weights_only_keeps_data_and_drops_the_rest() {
        let mut p = Param::constant(vec![2, 2], 1.5);
        p.grad.data.fill(3.0);
        let w = p.weights_only();
        assert_eq!(w.data.data, p.data.data);
        assert!(w.grad.data.is_empty() && w.m.data.is_empty() && w.v.data.is_empty());
        p.zero_grad();
        assert!(p.grad.data.iter().all(|&g| g == 0.0));
    }

    /// AdamW decay applies to matrices only: with a zero gradient, a rank-2 param shrinks
    /// and a rank-1 param (bias, LayerNorm gain) does not move.
    #[test]
    fn weight_decay_skips_rank_one_params() {
        let adam = Adam::with_weight_decay(0.1, 0.5);
        let mut matrix = Param::constant(vec![2, 2], 1.0);
        let mut vector = Param::constant(vec![4], 1.0);
        matrix.step(&adam, 1).unwrap();
        vector.step(&adam, 1).unwrap();
        assert!(
            matrix.data.data.iter().all(|&x| x < 1.0),
            "{:?}",
            matrix.data.data
        );
        assert_eq!(vector.data.data, vec![1.0; 4]);
    }
}
