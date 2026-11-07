//! The `Layer` trait and standard architectures: Linear, Activation, LayerNorm, Embedding, and TransformerBlock.

mod activation;
mod embedding;
mod linear;
mod matmul;
mod norm;
#[cfg(test)]
mod test_support;
mod transformer;

pub use self::activation::{ActivationLayer, Flatten};
pub use self::embedding::Embedding;
pub use self::linear::Linear;
pub(crate) use self::matmul::{matmul_naive_helper, matmul_transpose_b_helper};
pub use self::norm::LayerNorm;
pub use self::transformer::{KvCache, TransformerBlock};

use crate::error::Result;
use crate::tensor::Tensor;
use std::any::Any;

/// Everything a layer must provide. `as_any` allows the loader to downcast
/// a trait object back to its concrete type for serialisation.
pub trait Layer {
    /// Run the layer on `input` and return its output tensor.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    /// Human-readable layer name, for logs and summaries.
    fn name(&self) -> String;
    /// The layer as `&dyn Any`, so callers can downcast to the concrete type.
    fn as_any(&self) -> &dyn Any;
    /// Trainable parameters held by this layer (weights + biases).
    fn num_params(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::Activation;
    use crate::layer::activation::{ActivationLayer, Flatten};
    use crate::layer::embedding::Embedding;
    use crate::layer::linear::Linear;
    use crate::layer::norm::LayerNorm;
    use crate::layer::test_support::minimal_block;
    use crate::layer::Layer;
    use crate::quant::{QKind, QWeight};
    use crate::tensor::Tensor;

    #[test]
    fn num_params_counts_weights_and_biases_of_each_layer() {
        // Linear 3->4: 12 weights + 4 biases, f32 and quantized alike.
        let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let lin = Linear::new(3, 4, w, vec![0.0; 4]).unwrap();
        assert_eq!(lin.num_params(), 16);
        assert_eq!(lin.quantize(QKind::Int8).num_params(), 16);
        // A bias-free head holds a zero bias for shape only; it is not a parameter.
        let head = Linear::without_bias(3, 4, vec![0.5; 12]).unwrap();
        assert_eq!(head.num_params(), 12);
        assert_eq!(head.bias.data, vec![0.0; 4]);
        let x = Tensor::matrix(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(head.forward(&x).unwrap().data, vec![3.0; 4]);
        assert_eq!(head.quantize(QKind::Int8).num_params(), 12);
        // A tied head shares the embedding's values: no parameters of its own,
        // and it still computes x · W (quantized or not).
        let tied = Linear::tied_head(3, 4, vec![0.5; 12]).unwrap();
        assert_eq!(tied.num_params(), 0);
        assert_eq!(tied.forward(&x).unwrap().data, vec![3.0; 4]);
        assert_eq!(tied.quantize(QKind::Int8).num_params(), 0);
        assert!(Linear::tied_head(3, 4, vec![0.0; 5]).is_err());
        assert!(Linear::without_bias(3, 4, vec![0.0; 5]).is_err());
        let q = QWeight::from_f32(&[0.5; 12], 3, 4, QKind::Int8);
        assert_eq!(
            Linear::quantized(3, 4, q, vec![0.0; 4])
                .unwrap()
                .num_params(),
            16
        );
        // Embedding: 5x4 token table + 6x4 position table.
        let emb = Embedding::new(5, 6, 4, vec![0.0; 20], vec![0.0; 24]).unwrap();
        assert_eq!(emb.num_params(), 44);
        // LayerNorm: gamma + beta.
        assert_eq!(
            LayerNorm::new(7, vec![1.0; 7], vec![0.0; 7])
                .unwrap()
                .num_params(),
            14
        );
        // Block c=4, hidden 8: 4 projections (16+4) + 2 norms (8) + ffn (32+8, 32+4).
        let b = minimal_block(2, 1, 4, 8);
        assert_eq!(b.num_params(), 4 * 20 + 2 * 8 + 40 + 36);
        // Parameter-free layers report zero (the trait default).
        assert_eq!(ActivationLayer::new(Activation::ReLU).num_params(), 0);
        assert_eq!(Flatten::new().num_params(), 0);
    }
}
