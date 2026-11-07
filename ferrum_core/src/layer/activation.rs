//! Element-wise activation layer and the flatten pass-through.
use crate::activation::Activation;
use crate::error::Result;
use crate::layer::Layer;
use crate::tensor::Tensor;
use std::any::Any;

/// Parameter-free activation wrapper so it lives in the same layer list.
pub struct ActivationLayer {
    /// The element-wise activation this layer applies.
    pub kind: Activation,
}

impl ActivationLayer {
    /// Wrap `kind` in a layer.
    pub fn new(kind: Activation) -> Self {
        Self { kind }
    }
}

impl Layer for ActivationLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.kind.apply(input)
    }
    fn name(&self) -> String {
        format!("Activation({:?})", self.kind)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Flattens any input into a single row: [r, c] → [1, r·c].
///
/// Used between `Embedding` and `Linear` in embedded-MLP language models so
/// the per-position embeddings become one flat feature vector. Serialised in
/// Ferrum GGUF as a `flatten` layer with no tensors.
#[derive(Default)]
pub struct Flatten;

impl Flatten {
    /// Create the (stateless) flatten layer.
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        vprintln!(
            "[layer::Flatten::forward] {:?} → [1, {}]",
            input.shape,
            input.data.len()
        );
        Tensor::matrix(1, input.data.len(), input.data.clone())
    }
    fn name(&self) -> String {
        "Flatten".to_string()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::ActivationLayer;
    use crate::activation::Activation;
    use crate::layer::Layer;
    use crate::tensor::Tensor;

    #[test]
    fn activation_layer_relu_name() {
        let a = ActivationLayer::new(Activation::ReLU);
        assert!(a.name().contains("ReLU"));
    }

    #[test]
    fn activation_layer_forward_delegates() {
        let a = ActivationLayer::new(Activation::ReLU);
        let x = Tensor::vector(vec![-1.0, 2.0]);
        let y = a.forward(&x).unwrap();
        assert_eq!(y.data, vec![0.0, 2.0]);
    }
}
