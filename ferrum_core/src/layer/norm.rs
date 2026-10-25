//! Per-row layer normalisation.
use crate::error::{InferError, Result};
use crate::layer::Layer;
use crate::tensor::Tensor;
use crate::verbose;
use std::any::Any;

/// Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta.
pub struct LayerNorm {
    pub gamma: Tensor, // scale, shape [dim]
    pub beta: Tensor,  // shift, shape [dim]
    dim: usize,
}

impl LayerNorm {
    pub fn new(dim: usize, gamma: Vec<f32>, beta: Vec<f32>) -> Result<Self> {
        if gamma.len() != dim || beta.len() != dim {
            return Err(InferError::DimMismatch(format!(
                "LayerNorm weights len {}/{} != dim {}",
                gamma.len(),
                beta.len(),
                dim
            )));
        }
        vprintln!("[layer::LayerNorm::new] dim={}", dim);
        Ok(Self {
            gamma: Tensor::vector(gamma),
            beta: Tensor::vector(beta),
            dim,
        })
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Layer for LayerNorm {
    fn num_params(&self) -> usize {
        self.gamma.numel() + self.beta.numel()
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (rows, cols) = input.matrix_dims()?;
        if cols != self.dim {
            return Err(InferError::DimMismatch(format!(
                "LayerNorm expects width {}, got {}",
                self.dim, cols
            )));
        }
        vprintln!("[layer::LayerNorm::forward] [{},{}]", rows, cols);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            let row_slice = &input.data[base..base + cols];
            let mean = row_slice.iter().sum::<f32>() / cols as f32;
            let var = row_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / cols as f32;
            let std = (var + 1e-5).sqrt();

            if verbose::is_verbose() && rows <= 8 {
                vprintln!(
                    "[layer::LayerNorm::forward]   row[{}]: mean={:.6}, var={:.6}, std={:.6}",
                    r,
                    mean,
                    var,
                    std
                );
            }

            for c in 0..cols {
                out[base + c] =
                    ((row_slice[c] - mean) / std) * self.gamma.data[c] + self.beta.data[c];
            }
        }
        if verbose::is_verbose() {
            verbose::check_nan_inf(&out, "LayerNorm output");
        }
        Tensor::matrix(rows, cols, out)
    }
    fn name(&self) -> String {
        format!("LayerNorm({})", self.dim)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::LayerNorm;
    use crate::error::InferError;
    use crate::layer::Layer;
    use crate::tensor::Tensor;

    #[test]
    fn layernorm_preserves_mean_and_var() {
        let x = Tensor::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 10.0, -10.0, 20.0, -20.0]).unwrap();
        let ln = LayerNorm::new(4, vec![1.0; 4], vec![0.0; 4]).unwrap();
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape, vec![2, 4]);
        for r in 0..2 {
            let row = &y.data[r * 4..(r + 1) * 4];
            let mean = row.iter().sum::<f32>() / 4.0;
            let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "row {r} mean={mean}");
            assert!((var - 1.0).abs() < 1e-3, "row {r} var={var}");
        }
    }

    #[test]
    fn layernorm_gamma_beta_applied() {
        // gamma=2, beta=1 on a single-row [0, 0, 0, 0] → all outputs should be beta=1
        // Actually zero-variance input: normalised = 0, output = 0*gamma + beta = beta
        let x = Tensor::matrix(1, 4, vec![5.0; 4]).unwrap(); // constant row → std≈0, normalised≈0
        let ln = LayerNorm::new(4, vec![2.0; 4], vec![1.0; 4]).unwrap();
        let y = ln.forward(&x).unwrap();
        // Each output should be near beta=1 (normalised×gamma=0)
        for &v in &y.data {
            assert!((v - 1.0).abs() < 1e-3, "got {v}");
        }
    }

    #[test]
    fn layernorm_new_wrong_gamma_len_errors() {
        assert!(LayerNorm::new(4, vec![1.0; 3], vec![0.0; 4]).is_err());
    }

    #[test]
    fn layernorm_new_wrong_beta_len_errors() {
        assert!(LayerNorm::new(4, vec![1.0; 4], vec![0.0; 3]).is_err());
    }

    #[test]
    fn layernorm_forward_wrong_width_errors() {
        let ln = LayerNorm::new(4, vec![1.0; 4], vec![0.0; 4]).unwrap();
        let x = Tensor::matrix(1, 3, vec![1.0; 3]).unwrap();
        assert!(matches!(ln.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn layernorm_dim_accessor() {
        let ln = LayerNorm::new(8, vec![1.0; 8], vec![0.0; 8]).unwrap();
        assert_eq!(ln.dim(), 8);
    }

    #[test]
    fn layernorm_name() {
        let ln = LayerNorm::new(6, vec![1.0; 6], vec![0.0; 6]).unwrap();
        assert!(ln.name().contains("6"));
    }
}
