//! Fully-connected layer `y = xW + b` with optional int8 / int4 quantized weights.
use crate::error::{InferError, Result};
use crate::layer::Layer;
use crate::quant::{QKind, QWeight};
use crate::tensor::Tensor;
use crate::{ops, verbose};
use std::any::Any;
use std::sync::Arc;

/// Fully-connected affine layer: y = x · W + b.
/// Weight shape: [in_features, out_features] — no transpose needed in forward.
///
/// The weights are held either as full f32 (`weight`) or, for inference of large
/// models, as an in-memory quantized matrix (`qweight`, int8/int4). When
/// `qweight` is `Some` the f32 `weight` is an empty placeholder — the whole
/// point of quantizing is to *not* keep the f32 copy resident — and the forward
/// pass runs the packed kernel ([`ops::qlinear`]) directly on the int8/int4
/// data. The two paths produce numerically equivalent results (within the
/// quantization error already baked in at save time).
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    qweight: Option<Arc<QWeight>>,
    in_f: usize,
    out_f: usize,
    /// The zero `bias` only fills the shape: the layer has no bias parameters
    /// (the GPT-2 LM head), so it is not counted by `num_params`.
    bias_free: bool,
    /// The weight is the transposed token embedding (a tied GPT-2 head): its
    /// values are not separate parameters, so `num_params` counts none of it.
    tied: bool,
}

impl Linear {
    pub fn new(in_f: usize, out_f: usize, weight: Vec<f32>, bias: Vec<f32>) -> Result<Self> {
        if bias.len() != out_f {
            return Err(InferError::DimMismatch(format!(
                "bias length {} ≠ out_features {out_f}",
                bias.len()
            )));
        }
        vprintln!(
            "[layer::Linear::new] in={}, out={}, weight_len={}, bias_len={}",
            in_f,
            out_f,
            weight.len(),
            bias.len()
        );
        Ok(Self {
            weight: Tensor::matrix(in_f, out_f, weight)?,
            bias: Tensor::vector(bias),
            qweight: None,
            in_f,
            out_f,
            bias_free: false,
            tied: false,
        })
    }

    /// A Linear with no bias parameters (the output is `x · W`), as the GPT-2
    /// LM head. Stores a zero bias to keep the forward path uniform.
    pub fn without_bias(in_f: usize, out_f: usize, weight: Vec<f32>) -> Result<Self> {
        let mut l = Self::new(in_f, out_f, weight, vec![0.0; out_f])?;
        l.bias_free = true;
        Ok(l)
    }

    /// Mark this layer as having no bias parameters: the zero `bias` only fills the
    /// shape (an imported GGUF projection whose file has no bias tensor). The
    /// fine-tune trainer then leaves it out of the trainable set, so a trained
    /// model stays exactly representable by the file it came from.
    pub fn into_bias_free(mut self) -> Self {
        self.bias_free = true;
        self
    }

    /// Whether the layer has real bias parameters (see [`Linear::into_bias_free`]).
    pub fn has_bias(&self) -> bool {
        !self.bias_free
    }

    /// A bias-free head whose weight is shared with the token embedding
    /// (weight tying). Behaves like [`Linear::without_bias`] but contributes no
    /// parameters, so `Sequential::num_params` counts the shared matrix once.
    pub fn tied_head(in_f: usize, out_f: usize, weight: Vec<f32>) -> Result<Self> {
        let mut l = Self::without_bias(in_f, out_f, weight)?;
        l.tied = true;
        Ok(l)
    }

    /// Build a Linear whose weights are kept **quantized in memory** (int8/int4).
    /// `qw` must be the `[in_f, out_f]` quantization of the weight matrix. No f32
    /// weight copy is retained.
    pub fn quantized(in_f: usize, out_f: usize, qw: QWeight, bias: Vec<f32>) -> Result<Self> {
        if bias.len() != out_f {
            return Err(InferError::DimMismatch(format!(
                "bias length {} ≠ out_features {out_f}",
                bias.len()
            )));
        }
        if qw.rows != in_f || qw.cols != out_f {
            return Err(InferError::DimMismatch(format!(
                "quantized weight [{},{}] ≠ Linear [{in_f},{out_f}]",
                qw.rows, qw.cols
            )));
        }
        vprintln!(
            "[layer::Linear::quantized] in={}, out={}, kind={:?}, resident={} bytes",
            in_f,
            out_f,
            qw.kind,
            qw.resident_bytes()
        );
        Ok(Self {
            weight: Tensor {
                shape: vec![0],
                data: vec![],
            },
            bias: Tensor::vector(bias),
            qweight: Some(Arc::new(qw)),
            in_f,
            out_f,
            bias_free: false,
            tied: false,
        })
    }

    /// Convert an f32 Linear into a quantized one in place, dropping the f32
    /// weight. Returns `self` for chaining. No-op if already quantized.
    pub fn quantize(mut self, kind: QKind) -> Self {
        if self.qweight.is_none() {
            let qw = QWeight::from_f32(&self.weight.data, self.in_f, self.out_f, kind);
            self.weight = Tensor {
                shape: vec![0],
                data: vec![],
            };
            self.qweight = Some(Arc::new(qw));
        }
        self
    }

    /// The in-memory quantized weight, if this Linear holds one.
    pub fn qweight(&self) -> Option<&QWeight> {
        self.qweight.as_deref()
    }

    /// The weight matrix as row-major f32, dequantizing from `qweight` if the
    /// f32 copy was dropped. Used by serialization (not the hot path).
    pub fn weight_f32(&self) -> Vec<f32> {
        match &self.qweight {
            Some(qw) => qw.to_f32(),
            None => self.weight.data.clone(),
        }
    }
    pub fn in_features(&self) -> usize {
        self.in_f
    }
    pub fn out_features(&self) -> usize {
        self.out_f
    }
}

impl Layer for Linear {
    fn num_params(&self) -> usize {
        if self.tied {
            return 0;
        }
        self.in_f * self.out_f + if self.bias_free { 0 } else { self.bias.numel() }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, cols) = input.matrix_dims()?;
        if cols != self.in_f {
            return Err(InferError::DimMismatch(format!(
                "Linear expects width {}, got {cols}",
                self.in_f
            )));
        }
        vprintln!(
            "[layer::Linear::forward] input={:?}, weight=[{},{}]",
            input.shape,
            self.in_f,
            self.out_f
        );
        let result = match &self.qweight {
            Some(qw) => ops::qlinear(input, qw, &self.bias.data)?,
            None => ops::linear_forward(input, &self.weight, &self.bias.data)?,
        };
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&result.data);
            vprintln!(
                "[layer::Linear::forward]   output={:?}, stats: min={:.6}, max={:.6}, mean={:.6}",
                result.shape,
                vmin,
                vmax,
                vmean
            );
            verbose::check_nan_inf(
                &result.data,
                &format!("Linear({}→{}) output", self.in_f, self.out_f),
            );
        }
        Ok(result)
    }
    fn name(&self) -> String {
        format!("Linear({}→{})", self.in_f, self.out_f)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate::error::InferError;
    use crate::layer::test_support::identity_linear;
    use crate::layer::Layer;
    use crate::tensor::Tensor;

    #[test]
    fn linear_identity_transform() {
        let l = identity_linear(3);
        let x = Tensor::row(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(l.forward(&x).unwrap().data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn linear_applies_bias() {
        let l = Linear::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![10.0, 20.0]).unwrap();
        let x = Tensor::row(vec![0.0, 0.0]).unwrap();
        assert_eq!(l.forward(&x).unwrap().data, vec![10.0, 20.0]);
    }

    #[test]
    fn linear_new_wrong_bias_len_errors() {
        let result = Linear::new(2, 3, vec![0.0; 6], vec![0.0; 2]); // bias should be 3
        assert!(matches!(result, Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn linear_forward_wrong_input_width_errors() {
        let l = identity_linear(3);
        let x = Tensor::row(vec![1.0, 2.0]).unwrap(); // width 2, expects 3
        assert!(matches!(l.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn linear_accessors() {
        let l = Linear::new(4, 8, vec![0.0; 32], vec![0.0; 8]).unwrap();
        assert_eq!(l.in_features(), 4);
        assert_eq!(l.out_features(), 8);
    }

    #[test]
    fn linear_name_contains_dims() {
        let l = Linear::new(4, 8, vec![0.0; 32], vec![0.0; 8]).unwrap();
        let name = l.name();
        assert!(name.contains("4") && name.contains("8"));
    }

    #[test]
    fn linear_batch_forward_shape() {
        let l = Linear::new(3, 2, vec![1.0; 6], vec![0.0; 2]).unwrap();
        let x = Tensor::matrix(5, 3, vec![1.0; 15]).unwrap();
        let y = l.forward(&x).unwrap();
        assert_eq!(y.shape, vec![5, 2]);
    }

    #[test]
    fn quantized_linear_validates_shapes_and_exposes_dequantized_weights() {
        use crate::quant::{QKind, QWeight};
        let w = vec![0.5, -0.25, 1.0, 0.75, -0.5, 0.125];
        let q = || QWeight::from_f32(&w, 3, 2, QKind::Int8);
        assert!(Linear::quantized(3, 2, q(), vec![0.0]).is_err()); // bias length
        assert!(Linear::quantized(2, 3, q(), vec![0.0; 3]).is_err()); // weight shape
        let lin = Linear::quantized(3, 2, q(), vec![0.0; 2]).unwrap();
        let back = lin.weight_f32();
        assert_eq!(back.len(), 6);
        for (a, b) in back.iter().zip(&w) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }
}
