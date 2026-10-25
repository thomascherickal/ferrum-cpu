//! Shared fixtures for the unit tests of this module tree.
use crate::layer::linear::Linear;
use crate::layer::transformer::TransformerBlock;

pub(super) fn identity_linear(n: usize) -> Linear {
    let mut w = vec![0.0f32; n * n];
    for i in 0..n {
        w[i * n + i] = 1.0;
    }
    Linear::new(n, n, w, vec![0.0; n]).unwrap()
}

pub(super) fn minimal_block(
    ctx: usize,
    heads: usize,
    dim: usize,
    hidden: usize,
) -> TransformerBlock {
    let scale = 0.01f32;
    let c = dim;
    let h = hidden;
    TransformerBlock::new(
        ctx,
        heads,
        dim,
        vec![1.0; c],
        vec![0.0; c], // ln1
        vec![scale; c * c],
        vec![0.0; c], // q
        vec![scale; c * c],
        vec![0.0; c], // k
        vec![scale; c * c],
        vec![0.0; c], // v
        vec![scale; c * c],
        vec![0.0; c], // out
        vec![1.0; c],
        vec![0.0; c], // ln2
        vec![scale; c * h],
        vec![0.0; h], // ffn1
        vec![scale; h * c],
        vec![0.0; c], // ffn2
    )
    .unwrap()
}
