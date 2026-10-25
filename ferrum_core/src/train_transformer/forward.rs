//! Forward passes: plain inference-style and the caching training forward.
use crate::error::{InferError, Result};
use crate::ops;
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::train_transformer::block::BlockCache;
use crate::train_transformer::kernels::{attn_fwd, linear_fwd, ln_fwd};
use crate::train_transformer::TransformerNet;

/// Opaque record of one forward pass, consumed by `backward`.
pub struct FwdCache {
    pub(super) tokens: Vec<usize>,
    pub(super) batch: usize,
    pub(super) blocks: Vec<BlockCache>,
    pub(super) xhat_f: Tensor,
    pub(super) inv_std_f: Vec<f32>,
    pub(super) norm_f: Tensor, // input to the LM head
}

impl TransformerNet {
    /// Forward pass over `tokens` (length must be a multiple of `context_len`;
    /// each `context_len` chunk is one sequence in the batch). Returns logits
    /// of shape [B·T, vocab] plus the activation cache for `backward`.
    ///
    /// Dropout-free — equivalent to [`TransformerNet::forward_train`] with
    /// `dropout = 0`. Used by inference, evaluation, and gradient checks.
    pub fn forward(&self, tokens: &[usize]) -> Result<(Tensor, FwdCache)> {
        self.forward_train(tokens, 0.0, 0)
    }

    /// Forward pass with optional FFN-hidden dropout (T7). When `dropout > 0`,
    /// each block's post-GELU hidden activations are inverted-dropout masked
    /// using a local generator seeded by `seed` (so the masks are deterministic
    /// for a given seed and a worker can reproduce them in `backward` from the
    /// cache). `dropout = 0` is the plain, deterministic forward.
    pub fn forward_train(
        &self,
        tokens: &[usize],
        dropout: f32,
        seed: u64,
    ) -> Result<(Tensor, FwdCache)> {
        let t = self.context_len;
        if tokens.is_empty() || !tokens.len().is_multiple_of(t) {
            return Err(InferError::DimMismatch(format!(
                "token count {} must be a non-zero multiple of context_len {t}",
                tokens.len()
            )));
        }
        let batch = tokens.len() / t;
        let m = tokens.len();
        let c = self.embed_dim;
        let dropout = dropout.clamp(0.0, 0.95);
        let mut drng = Rng::from_state(seed | 1); // local mask generator

        // Embedding: token + positional lookup.
        let mut x = vec![0.0f32; m * c];
        for (r, &tok) in tokens.iter().enumerate() {
            if tok >= self.vocab_size {
                return Err(InferError::DimMismatch(format!(
                    "token {tok} out of range for vocab {}",
                    self.vocab_size
                )));
            }
            let tb = tok * c;
            let pb = (r % t) * c;
            let xb = r * c;
            for d in 0..c {
                x[xb + d] = self.tok_emb.data.data[tb + d] + self.pos_emb.data.data[pb + d];
            }
        }
        let mut cur = Tensor::matrix(m, c, x)?;

        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let x_in = cur;
            let (norm1, xhat1, inv_std1) = ln_fwd(&x_in, &blk.ln1_g.data, &blk.ln1_b.data)?;
            let q = linear_fwd(&norm1, &blk.q_w.data, &blk.q_b.data)?;
            let k = linear_fwd(&norm1, &blk.k_w.data, &blk.k_b.data)?;
            let v = linear_fwd(&norm1, &blk.v_w.data, &blk.v_b.data)?;
            let (concat, probs) = attn_fwd(&q, &k, &v, batch, t, self.num_heads)?;
            let proj = linear_fwd(&concat, &blk.o_w.data, &blk.o_b.data)?;
            let x_attn = ops::add(&x_in, &proj)?;
            let (norm2, xhat2, inv_std2) = ln_fwd(&x_attn, &blk.ln2_g.data, &blk.ln2_b.data)?;
            let h_pre = linear_fwd(&norm2, &blk.f1_w.data, &blk.f1_b.data)?;
            let h_act = h_pre.map(crate::activation::gelu);
            // Inverted FFN dropout: zero each hidden unit with prob `dropout`,
            // scale survivors by 1/(1-dropout) so the expectation is unchanged
            // and inference needs no rescaling.
            let (h_ffn, ffn_dropout) = if dropout > 0.0 {
                let scale = 1.0 / (1.0 - dropout);
                let mask: Vec<f32> = (0..h_act.data.len())
                    .map(|_| {
                        if drng.next_f32() < dropout {
                            0.0
                        } else {
                            scale
                        }
                    })
                    .collect();
                let dropped: Vec<f32> =
                    h_act.data.iter().zip(&mask).map(|(&h, &m)| h * m).collect();
                (Tensor::new(h_act.shape.clone(), dropped)?, Some(mask))
            } else {
                (h_act, None)
            };
            let ff2 = linear_fwd(&h_ffn, &blk.f2_w.data, &blk.f2_b.data)?;
            cur = ops::add(&x_attn, &ff2)?;
            block_caches.push(BlockCache {
                xhat1,
                inv_std1,
                norm1,
                q,
                k,
                v,
                probs,
                concat,
                xhat2,
                inv_std2,
                norm2,
                h_pre,
                h_act: h_ffn,
                ffn_dropout,
            });
        }

        let (norm_f, xhat_f, inv_std_f) = ln_fwd(&cur, &self.lnf_g.data, &self.lnf_b.data)?;
        // LM head has no bias (GPT-2 / llama.cpp graph).
        let logits = ops::matmul(&norm_f, &self.head_w.data)?;

        Ok((
            logits,
            FwdCache {
                tokens: tokens.to_vec(),
                batch,
                blocks: block_caches,
                xhat_f,
                inv_std_f,
                norm_f,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::rng::Rng;
    use crate::train_transformer::test_support::tiny_net;
    use crate::train_transformer::TransformerNet;

    #[test]
    fn forward_rejects_bad_token_counts() {
        let mut rng = Rng::new(1);
        let net = tiny_net(&mut rng);
        assert!(net.forward(&[]).is_err());
        assert!(net.forward(&[1, 2]).is_err()); // not a multiple of T=3
        assert!(net.forward(&[1, 2, 99]).is_err()); // token out of vocab
    }

    #[test]
    fn dropout_forward_is_deterministic_per_seed_and_off_matches_plain() {
        let mut rng = Rng::new(1);
        let net = TransformerNet::new(6, 4, 8, 2, 32, 1, &mut rng).unwrap();
        let tokens = [1usize, 2, 3, 4];
        let (a, _) = net.forward_train(&tokens, 0.5, 123).unwrap();
        let (b, _) = net.forward_train(&tokens, 0.5, 123).unwrap();
        assert_eq!(a.data, b.data, "same seed must reproduce the dropout mask");
        let (c, _) = net.forward_train(&tokens, 0.5, 999).unwrap();
        assert_ne!(
            a.data, c.data,
            "a different seed must change the dropout pattern"
        );
        // dropout = 0 is exactly the plain forward.
        let (d, _) = net.forward_train(&tokens, 0.0, 123).unwrap();
        let (e, _) = net.forward(&tokens).unwrap();
        assert_eq!(d.data, e.data);
    }
}
