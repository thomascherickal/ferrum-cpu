//! Token + learned positional embedding lookup.
use crate::error::{InferError, Result};
use crate::layer::Layer;
use crate::tensor::Tensor;
use crate::verbose;
use std::any::Any;

/// Embedding table mapping token IDs and positional indices to continuous vectors.
pub struct Embedding {
    pub token_weight: Tensor, // Shape [vocab_size, embedding_dim]
    pub pos_weight: Tensor,   // Shape [max_seq_len, embedding_dim]
    vocab_size: usize,
    max_seq_len: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        embedding_dim: usize,
        token_weight: Vec<f32>,
        pos_weight: Vec<f32>,
    ) -> Result<Self> {
        if token_weight.len() != vocab_size * embedding_dim {
            return Err(InferError::ShapeMismatch {
                expected: vocab_size * embedding_dim,
                got: token_weight.len(),
            });
        }
        if pos_weight.len() != max_seq_len * embedding_dim {
            return Err(InferError::ShapeMismatch {
                expected: max_seq_len * embedding_dim,
                got: pos_weight.len(),
            });
        }
        vprintln!(
            "[layer::Embedding::new] vocab={}, max_seq={}, dim={}",
            vocab_size,
            max_seq_len,
            embedding_dim
        );
        Ok(Self {
            token_weight: Tensor::matrix(vocab_size, embedding_dim, token_weight)?,
            pos_weight: Tensor::matrix(max_seq_len, embedding_dim, pos_weight)?,
            vocab_size,
            max_seq_len,
            embedding_dim,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Embed a single token at a given sequence position → [1, embedding_dim].
    /// Used by the incremental (KV-cached) generation path.
    pub fn embed_one(&self, token: usize, pos: usize) -> Result<Tensor> {
        if token >= self.vocab_size {
            return Err(InferError::DimMismatch(format!(
                "Token index {} out of bounds for vocab_size {}",
                token, self.vocab_size
            )));
        }
        if pos >= self.max_seq_len {
            return Err(InferError::DimMismatch(format!(
                "Position {} exceeds max_seq_len {}",
                pos, self.max_seq_len
            )));
        }
        let d = self.embedding_dim;
        let tok_base = token * d;
        let pos_base = pos * d;
        let data: Vec<f32> = (0..d)
            .map(|i| self.token_weight.data[tok_base + i] + self.pos_weight.data[pos_base + i])
            .collect();
        Tensor::matrix(1, d, data)
    }
}

impl Layer for Embedding {
    fn num_params(&self) -> usize {
        self.token_weight.numel() + self.pos_weight.numel()
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input has shape [B, T] (where T is sequence length)
        let (batch, seq_len) = match input.shape.as_slice() {
            [s] => (1, *s),
            [b, s] => (*b, *s),
            _ => {
                return Err(InferError::DimMismatch(
                    "Embedding expects rank-1 or rank-2 inputs".into(),
                ))
            }
        };
        if seq_len > self.max_seq_len {
            return Err(InferError::DimMismatch(format!(
                "Sequence length {} exceeds max_seq_len {}",
                seq_len, self.max_seq_len
            )));
        }
        vprintln!(
            "[layer::Embedding::forward] batch={}, seq_len={}, vocab={}, dim={}",
            batch,
            seq_len,
            self.vocab_size,
            self.embedding_dim
        );

        let out_cols = self.embedding_dim;
        let mut out_data = vec![0.0f32; batch * seq_len * out_cols];
        for b in 0..batch {
            for t in 0..seq_len {
                let raw = input.data[b * seq_len + t].round();
                // `as usize` saturates: a negative or NaN id would silently become token 0.
                if raw.is_nan() || raw < 0.0 || raw as usize >= self.vocab_size {
                    return Err(InferError::DimMismatch(format!(
                        "Token index {} out of bounds for vocab_size {}",
                        input.data[b * seq_len + t],
                        self.vocab_size
                    )));
                }
                let tok_idx = raw as usize;
                let tok_base = tok_idx * self.embedding_dim;
                let pos_base = t * self.embedding_dim;
                let out_base = (b * seq_len + t) * self.embedding_dim;
                for d in 0..self.embedding_dim {
                    out_data[out_base + d] =
                        self.token_weight.data[tok_base + d] + self.pos_weight.data[pos_base + d];
                }
            }
        }
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&out_data);
            vprintln!("[layer::Embedding::forward]   output=[{},{}], stats: min={:.6}, max={:.6}, mean={:.6}",
                batch * seq_len, out_cols, vmin, vmax, vmean);
        }
        Tensor::matrix(batch * seq_len, out_cols, out_data)
    }
    fn name(&self) -> String {
        format!(
            "Embedding(vocab={}, dim={})",
            self.vocab_size, self.embedding_dim
        )
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::Embedding;
    use crate::error::InferError;
    use crate::layer::Layer;
    use crate::tensor::Tensor;

    #[test]
    fn embedding_lookup_and_positional_addition() {
        let emb = Embedding::new(
            3,
            4,
            2,
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        )
        .unwrap();
        let x = Tensor::matrix(2, 3, vec![0.0, 2.0, 1.0, 1.0, 0.0, 2.0]).unwrap();
        let y = emb.forward(&x).unwrap();
        assert_eq!(y.shape, vec![6, 2]);
        // seq 0, pos 0, tok 0: token[0]=[1,10] + pos[0]=[0.1,0.2] = [1.1, 10.2]
        assert!((y.data[0] - 1.1).abs() < 1e-5);
        assert!((y.data[1] - 10.2).abs() < 1e-5);
    }

    #[test]
    fn embedding_rank1_input_accepted() {
        let emb = Embedding::new(4, 8, 2, vec![0.0; 8], vec![0.0; 16]).unwrap();
        let x = Tensor::vector(vec![0.0, 1.0, 2.0]);
        // rank-1 = single sequence of 3 tokens
        assert!(emb.forward(&x).is_ok());
    }

    #[test]
    fn embedding_new_wrong_token_weight_len_errors() {
        assert!(Embedding::new(3, 4, 2, vec![0.0; 5], vec![0.0; 8]).is_err());
    }

    #[test]
    fn embedding_new_wrong_pos_weight_len_errors() {
        assert!(Embedding::new(3, 4, 2, vec![0.0; 6], vec![0.0; 7]).is_err());
    }

    #[test]
    fn embedding_seq_exceeds_max_seq_len_errors() {
        let emb = Embedding::new(5, 4, 2, vec![0.0; 10], vec![0.0; 8]).unwrap();
        // seq_len=6 > max_seq_len=4
        let x = Tensor::matrix(1, 6, vec![0.0; 6]).unwrap();
        assert!(matches!(emb.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn embedding_oob_token_index_errors() {
        let emb = Embedding::new(3, 4, 2, vec![0.0; 6], vec![0.0; 8]).unwrap();
        let x = Tensor::matrix(1, 2, vec![0.0, 9.0]).unwrap(); // token 9 ≥ vocab_size=3
        assert!(matches!(emb.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn embedding_rank3_input_errors() {
        let emb = Embedding::new(5, 4, 2, vec![0.0; 10], vec![0.0; 8]).unwrap();
        let x = Tensor::new(vec![1, 2, 2], vec![0.0; 4]).unwrap();
        assert!(emb.forward(&x).is_err());
    }

    #[test]
    fn embedding_accessors() {
        let emb = Embedding::new(10, 32, 64, vec![0.0; 640], vec![0.0; 2048]).unwrap();
        assert_eq!(emb.vocab_size(), 10);
        assert_eq!(emb.max_seq_len(), 32);
        assert_eq!(emb.embedding_dim(), 64);
    }

    #[test]
    fn embedding_name() {
        let emb = Embedding::new(5, 8, 16, vec![0.0; 80], vec![0.0; 128]).unwrap();
        let name = emb.name();
        assert!(name.contains("vocab=5") && name.contains("dim=16"));
    }

    #[test]
    fn embed_one_matches_batch_forward() {
        let emb = Embedding::new(
            3,
            4,
            2,
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        )
        .unwrap();
        let x = Tensor::matrix(1, 3, vec![0.0, 2.0, 1.0]).unwrap();
        let batch = emb.forward(&x).unwrap();
        for pos in 0..3 {
            let tok = x.data[pos] as usize;
            let one = emb.embed_one(tok, pos).unwrap();
            for d in 0..2 {
                assert!((one.data[d] - batch.data[pos * 2 + d]).abs() < 1e-6);
            }
        }
        assert!(emb.embed_one(9, 0).is_err()); // token OOB
        assert!(emb.embed_one(0, 9).is_err()); // position OOB
    }

    /// Regression (audit 2026-11-05, A1): a negative or NaN id saturated to token 0.
    #[test]
    fn forward_rejects_negative_and_nan_ids() {
        let emb = Embedding::new(3, 4, 2, vec![0.0; 6], vec![0.0; 8]).unwrap();
        for bad in [-1.0, -0.6, f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 3.0] {
            let x = Tensor::matrix(1, 2, vec![1.0, bad]).unwrap();
            assert!(emb.forward(&x).is_err(), "id {bad} was accepted");
        }
        // -0.4 rounds to -0.0, which is token 0.
        let x = Tensor::matrix(1, 2, vec![-0.4, 2.0]).unwrap();
        assert!(emb.forward(&x).is_ok());
    }
}
