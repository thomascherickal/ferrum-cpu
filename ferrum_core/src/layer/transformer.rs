//! Causal multi-head self-attention transformer block and its KV cache.
use crate::error::{InferError, Result};
use crate::layer::linear::Linear;
use crate::layer::matmul::{matmul_naive_helper, matmul_transpose_b_helper};
use crate::layer::norm::LayerNorm;
use crate::layer::Layer;
use crate::tensor::Tensor;
use crate::verbose;
use std::any::Any;
use std::cell::RefCell;

/// Per-block key/value cache for token-at-a-time generation. Holding the K/V
/// rows of already-processed positions turns each new-token forward pass from
/// O(T²) into O(T).
pub struct KvCache {
    k: Vec<f32>, // [len, dim] row-major
    v: Vec<f32>,
    len: usize,
    capacity: usize,
    dim: usize,
}

impl KvCache {
    /// `capacity` should be the block's `context_len`; `dim` its embedding dim.
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            k: Vec::with_capacity(capacity * dim),
            v: Vec::with_capacity(capacity * dim),
            len: 0,
            capacity,
            dim,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Drop all cached positions (start a fresh sequence).
    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.len = 0;
    }
}

/// A Decoder-Only Causal Transformer block containing:
///   LayerNorm -> Causal Multi-Head Self-Attention -> Residual
///   -> LayerNorm -> Feed-Forward Network -> Residual
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub ln2: LayerNorm,
    pub ffn1: Linear,
    pub ffn2: Linear,
    context_len: usize,
    num_heads: usize,
    embedding_dim: usize,
    pub last_attention: RefCell<Vec<f32>>,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context_len: usize,
        num_heads: usize,
        embedding_dim: usize,
        ln1_g: Vec<f32>,
        ln1_b: Vec<f32>,
        q_w: Vec<f32>,
        q_b: Vec<f32>,
        k_w: Vec<f32>,
        k_b: Vec<f32>,
        v_w: Vec<f32>,
        v_b: Vec<f32>,
        out_w: Vec<f32>,
        out_b: Vec<f32>,
        ln2_g: Vec<f32>,
        ln2_b: Vec<f32>,
        ffn1_w: Vec<f32>,
        ffn1_b: Vec<f32>,
        ffn2_w: Vec<f32>,
        ffn2_b: Vec<f32>,
    ) -> Result<Self> {
        if context_len == 0 {
            return Err(InferError::DimMismatch("context_len must be > 0".into()));
        }
        if num_heads == 0 || !embedding_dim.is_multiple_of(num_heads) {
            return Err(InferError::DimMismatch(format!(
                "embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}"
            )));
        }
        vprintln!(
            "[layer::TransformerBlock::new] ctx={}, heads={}, dim={}, hidden={}",
            context_len,
            num_heads,
            embedding_dim,
            ffn1_b.len()
        );

        let ln1 = LayerNorm::new(embedding_dim, ln1_g, ln1_b)?;
        let q_proj = Linear::new(embedding_dim, embedding_dim, q_w, q_b)?;
        let k_proj = Linear::new(embedding_dim, embedding_dim, k_w, k_b)?;
        let v_proj = Linear::new(embedding_dim, embedding_dim, v_w, v_b)?;
        let out_proj = Linear::new(embedding_dim, embedding_dim, out_w, out_b)?;
        let ln2 = LayerNorm::new(embedding_dim, ln2_g, ln2_b)?;
        let hidden_dim = ffn1_b.len();
        let ffn1 = Linear::new(embedding_dim, hidden_dim, ffn1_w, ffn1_b)?;
        let ffn2 = Linear::new(hidden_dim, embedding_dim, ffn2_w, ffn2_b)?;

        Ok(Self {
            ln1,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            ln2,
            ffn1,
            ffn2,
            context_len,
            num_heads,
            embedding_dim,
            last_attention: RefCell::new(Vec::new()),
        })
    }

    /// Assemble a block from already-constructed sublayers. Lets the loader feed
    /// in **quantized** projection Linears (int8/int4) so a transformer block —
    /// which holds the bulk of a large model's weights (≈12·d² per block) — stays
    /// quantized in memory instead of being expanded to f32.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        context_len: usize,
        num_heads: usize,
        embedding_dim: usize,
        ln1: LayerNorm,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
        ln2: LayerNorm,
        ffn1: Linear,
        ffn2: Linear,
    ) -> Result<Self> {
        if context_len == 0 {
            return Err(InferError::DimMismatch("context_len must be > 0".into()));
        }
        if num_heads == 0 || !embedding_dim.is_multiple_of(num_heads) {
            return Err(InferError::DimMismatch(format!(
                "embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}"
            )));
        }
        Ok(Self {
            ln1,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            ln2,
            ffn1,
            ffn2,
            context_len,
            num_heads,
            embedding_dim,
            last_attention: RefCell::new(Vec::new()),
        })
    }

    pub fn context_len(&self) -> usize {
        self.context_len
    }
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    pub fn hidden_dim(&self) -> usize {
        self.ffn1.out_features()
    }

    /// Incremental forward pass for one new token using a KV cache.
    ///
    /// `x` is the embedding of the newest position, shape [1, embedding_dim].
    /// The new K/V rows are appended to `cache` and the query attends over the
    /// whole cached prefix (causality holds automatically — only past
    /// positions are cached). Returns the block output for this position,
    /// shape [1, embedding_dim]. Produces the same values as a full
    /// `forward` over the sequence, but in O(T) per token instead of O(T²).
    ///
    /// Errors if the cache is full (`context_len` positions reached) — call
    /// `cache.clear()` and re-prime with a fresh context to continue.
    pub fn forward_with_cache(&self, x: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let (rows, c) = x.matrix_dims()?;
        if rows != 1 || c != self.embedding_dim {
            return Err(InferError::DimMismatch(format!(
                "forward_with_cache expects [1,{}], got {:?}",
                self.embedding_dim, x.shape
            )));
        }
        if cache.dim != c {
            return Err(InferError::DimMismatch(format!(
                "KvCache dim {} ≠ block embedding_dim {}",
                cache.dim, c
            )));
        }
        if cache.is_full() {
            return Err(InferError::DimMismatch(format!(
                "KvCache full ({} positions): clear and re-prime to continue",
                cache.capacity
            )));
        }

        let norm1 = self.ln1.forward(x)?;
        let q = self.q_proj.forward(&norm1)?;
        let k = self.k_proj.forward(&norm1)?;
        let v = self.v_proj.forward(&norm1)?;
        cache.k.extend_from_slice(&k.data);
        cache.v.extend_from_slice(&v.data);
        cache.len += 1;

        let l = cache.len; // positions visible to the query (incl. itself)
        let heads = self.num_heads;
        let head_dim = c / heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut attn_out = vec![0.0f32; c];
        let mut scores = vec![0.0f32; l];
        for h in 0..heads {
            let hs = h * head_dim;
            // scores[j] = scale · q_h · k_h(j)
            for (j, s) in scores.iter_mut().enumerate() {
                let kb = j * c + hs;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q.data[hs + d] * cache.k[kb + d];
                }
                *s = dot * scale;
            }
            // softmax over the cached prefix
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            // out_h = Σ_j p_j · v_h(j)
            for (j, &s) in scores.iter().enumerate() {
                let p = s / sum;
                let vb = j * c + hs;
                for d in 0..head_dim {
                    attn_out[hs + d] += p * cache.v[vb + d];
                }
            }
        }

        let projected = self.out_proj.forward(&Tensor::matrix(1, c, attn_out)?)?;
        let x_attn: Vec<f32> = (0..c).map(|i| x.data[i] + projected.data[i]).collect();
        let x_attn = Tensor::matrix(1, c, x_attn)?;

        let norm2 = self.ln2.forward(&x_attn)?;
        let hidden = self.ffn1.forward(&norm2)?.map(crate::activation::gelu);
        let ff2 = self.ffn2.forward(&hidden)?;
        let out: Vec<f32> = (0..c).map(|i| x_attn.data[i] + ff2.data[i]).collect();
        Tensor::matrix(1, c, out)
    }
}

impl Layer for TransformerBlock {
    fn num_params(&self) -> usize {
        [
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.out_proj,
            &self.ffn1,
            &self.ffn2,
        ]
        .iter()
        .map(|l| l.num_params())
        .sum::<usize>()
            + self.ln1.num_params()
            + self.ln2.num_params()
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (m, c) = input.matrix_dims()?;
        if c != self.embedding_dim {
            return Err(InferError::DimMismatch(format!(
                "TransformerBlock expects dim {}, got {}",
                self.embedding_dim, c
            )));
        }
        let t = self.context_len;
        if m % t != 0 {
            return Err(InferError::DimMismatch(format!(
                "Input rows {} must be divisible by context_len {}",
                m, t
            )));
        }
        let b = m / t;

        vprintln!(
            "[layer::TransformerBlock::forward] input=[{},{}], batch={}, ctx={}, heads={}, dim={}",
            m,
            c,
            b,
            t,
            self.num_heads,
            self.embedding_dim
        );

        // ── 1. LayerNorm 1 ───────────────────────────────────────────────────────
        vprintln!("[layer::TransformerBlock::forward]   ┌─ LayerNorm1");
        let norm1 = self.ln1.forward(input)?;
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&norm1.data);
            vprintln!("[layer::TransformerBlock::forward]   │  LN1 output: min={:.6}, max={:.6}, mean={:.6}", vmin, vmax, vmean);
            verbose::check_nan_inf(&norm1.data, "TransformerBlock LN1");
        }

        // ── 2. Q, K, V Projections ────────────────────────────────────────────────
        vprintln!("[layer::TransformerBlock::forward]   ├─ Q/K/V projections");
        let q = self.q_proj.forward(&norm1)?;
        let k = self.k_proj.forward(&norm1)?;
        let v = self.v_proj.forward(&norm1)?;
        if verbose::is_verbose() {
            let (qmin, qmax, qmean) = verbose::stats(&q.data);
            let (kmin, kmax, kmean) = verbose::stats(&k.data);
            let (vmin, vmax, vmean) = verbose::stats(&v.data);
            vprintln!(
                "[layer::TransformerBlock::forward]   │  Q: min={:.6}, max={:.6}, mean={:.6}",
                qmin,
                qmax,
                qmean
            );
            vprintln!(
                "[layer::TransformerBlock::forward]   │  K: min={:.6}, max={:.6}, mean={:.6}",
                kmin,
                kmax,
                kmean
            );
            vprintln!(
                "[layer::TransformerBlock::forward]   │  V: min={:.6}, max={:.6}, mean={:.6}",
                vmin,
                vmax,
                vmean
            );
            verbose::check_nan_inf(&q.data, "TransformerBlock Q");
            verbose::check_nan_inf(&k.data, "TransformerBlock K");
            verbose::check_nan_inf(&v.data, "TransformerBlock V");
        }

        // ── 3. Multi-Head Attention ──────────────────────────────────────────────
        vprintln!(
            "[layer::TransformerBlock::forward]   ├─ Multi-Head Attention ({} heads, head_dim={})",
            self.num_heads,
            self.embedding_dim / self.num_heads
        );
        let num_heads = self.num_heads;
        let head_dim = self.embedding_dim / num_heads;
        let head_scale = 1.0 / (head_dim as f32).sqrt();

        let mut attn_out = vec![0.0f32; m * c];
        let mut all_attns = vec![0.0f32; b * num_heads * t * t];

        for batch_idx in 0..b {
            for head_idx in 0..num_heads {
                // Extract Q, K, V for this head and batch item
                let mut q_head = vec![0.0f32; t * head_dim];
                let mut k_head = vec![0.0f32; t * head_dim];
                let mut v_head = vec![0.0f32; t * head_dim];

                for r in 0..t {
                    let src_row = batch_idx * t + r;
                    let head_start = head_idx * head_dim;
                    let src_idx = src_row * self.embedding_dim + head_start;
                    let dst_idx = r * head_dim;
                    q_head[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&q.data[src_idx..src_idx + head_dim]);
                    k_head[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&k.data[src_idx..src_idx + head_dim]);
                    v_head[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&v.data[src_idx..src_idx + head_dim]);
                }

                // Compute causal self-attention scores: S = Q * K^T
                let mut s = matmul_transpose_b_helper(&q_head, &k_head, t, t, head_dim);

                // Scale and apply causal mask
                for i in 0..t {
                    for j in 0..t {
                        let idx = i * t + j;
                        s[idx] *= head_scale;
                        if j > i {
                            s[idx] = -1e9; // Causal mask
                        }
                    }
                }

                // Softmax row-wise
                for i in 0..t {
                    let base = i * t;
                    let row = &mut s[base..base + t];
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for val in row.iter_mut() {
                        let e = (*val - max).exp();
                        *val = e;
                        sum += e;
                    }
                    for val in row.iter_mut() {
                        *val /= sum;
                    }
                }

                if verbose::is_verbose() {
                    let (amin, amax, amean) = verbose::stats(&s);
                    vprintln!("[layer::TransformerBlock::forward]   │  batch[{}] head[{}]: attn_weights: min={:.6}, max={:.6}, mean={:.6}",
                        batch_idx, head_idx, amin, amax, amean);
                    verbose::check_nan_inf(
                        &s,
                        &format!(
                            "TransformerBlock attn_weights batch={} head={}",
                            batch_idx, head_idx
                        ),
                    );
                }

                // Store attention weights for visualization
                let attn_store_base = (batch_idx * num_heads + head_idx) * t * t;
                all_attns[attn_store_base..attn_store_base + t * t].copy_from_slice(&s);

                // Compute output of head: O = S * V
                let o = matmul_naive_helper(&s, &v_head, t, head_dim, t);

                // Copy output back to flat output matrix
                for r in 0..t {
                    let dst_row = batch_idx * t + r;
                    let head_start = head_idx * head_dim;
                    let dst_idx = dst_row * self.embedding_dim + head_start;
                    let src_idx = r * head_dim;
                    attn_out[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&o[src_idx..src_idx + head_dim]);
                }
            }
        }

        // Store attention maps
        self.last_attention.replace(all_attns);

        let attn_out_tensor = Tensor::matrix(m, c, attn_out)?;
        vprintln!("[layer::TransformerBlock::forward]   ├─ Output projection");
        let projected = self.out_proj.forward(&attn_out_tensor)?;

        // Residual connection
        vprintln!(
            "[layer::TransformerBlock::forward]   ├─ Residual connection 1 (input + attention)"
        );
        let x_attn_data: Vec<f32> = input
            .data
            .iter()
            .zip(&projected.data)
            .map(|(a, b)| a + b)
            .collect();
        let x_attn = Tensor::matrix(m, c, x_attn_data)?;
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&x_attn.data);
            vprintln!("[layer::TransformerBlock::forward]   │  post-residual-1: min={:.6}, max={:.6}, mean={:.6}", vmin, vmax, vmean);
            verbose::check_nan_inf(&x_attn.data, "TransformerBlock post-residual-1");
        }

        // ── 4. LayerNorm 2 & FFN ─────────────────────────────────────────────────
        vprintln!("[layer::TransformerBlock::forward]   ├─ LayerNorm2");
        let norm2 = self.ln2.forward(&x_attn)?;

        vprintln!("[layer::TransformerBlock::forward]   ├─ FFN (in → hidden → out)");
        let ff1 = self.ffn1.forward(&norm2)?;
        // GELU
        let ffn_hidden = ff1.map(crate::activation::gelu);
        if verbose::is_verbose() {
            let zeros = ffn_hidden.data.iter().filter(|&&v| v == 0.0).count();
            vprintln!(
                "[layer::TransformerBlock::forward]   │  FFN GELU: {}/{} exact zeros ({:.1}%)",
                zeros,
                ffn_hidden.data.len(),
                100.0 * zeros as f32 / ffn_hidden.data.len() as f32
            );
        }
        let ff2 = self.ffn2.forward(&ffn_hidden)?;

        // Residual connection
        vprintln!("[layer::TransformerBlock::forward]   └─ Residual connection 2 (attn + ffn)");
        let out_data: Vec<f32> = x_attn
            .data
            .iter()
            .zip(&ff2.data)
            .map(|(a, b)| a + b)
            .collect();
        if verbose::is_verbose() {
            let (vmin, vmax, vmean) = verbose::stats(&out_data);
            vprintln!("[layer::TransformerBlock::forward]   Final output: min={:.6}, max={:.6}, mean={:.6}", vmin, vmax, vmean);
            verbose::check_nan_inf(&out_data, "TransformerBlock final output");
        }
        Tensor::matrix(m, c, out_data)
    }

    fn name(&self) -> String {
        format!(
            "TransformerBlock(heads={}, dim={}, hidden={})",
            self.num_heads,
            self.embedding_dim,
            self.hidden_dim()
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{KvCache, TransformerBlock};
    use crate::error::InferError;
    use crate::layer::test_support::minimal_block;
    use crate::layer::Layer;
    use crate::tensor::Tensor;

    #[test]
    fn transformer_block_forward_output_shape() {
        // ctx=4, heads=2, dim=8, hidden=16
        let block = minimal_block(4, 2, 8, 16);
        // Input: [B*T, dim] = [1*4, 8]
        let x = Tensor::matrix(4, 8, vec![0.1f32; 32]).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape, vec![4, 8]);
    }

    #[test]
    fn transformer_block_forward_batch_shape() {
        // batch=2, ctx=4, dim=8 → input [8, 8]
        let block = minimal_block(4, 2, 8, 16);
        let x = Tensor::matrix(8, 8, vec![0.05f32; 64]).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape, vec![8, 8]);
    }

    #[test]
    fn transformer_block_output_is_finite() {
        let block = minimal_block(4, 2, 8, 16);
        let x = Tensor::matrix(4, 8, (0..32).map(|i| i as f32 * 0.01).collect()).unwrap();
        let y = block.forward(&x).unwrap();
        assert!(y.data.iter().all(|v| v.is_finite()), "NaN/Inf in output");
    }

    #[test]
    fn transformer_block_wrong_embedding_dim_errors() {
        let block = minimal_block(4, 2, 8, 16);
        // width 5 instead of 8
        let x = Tensor::matrix(4, 5, vec![0.0; 20]).unwrap();
        assert!(matches!(block.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn transformer_block_rows_not_divisible_by_context_len_errors() {
        let block = minimal_block(4, 2, 8, 16);
        // 6 rows, context_len=4: 6 % 4 ≠ 0
        let x = Tensor::matrix(6, 8, vec![0.0; 48]).unwrap();
        assert!(matches!(block.forward(&x), Err(InferError::DimMismatch(_))));
    }

    #[test]
    fn transformer_block_attention_stored_after_forward() {
        let block = minimal_block(4, 2, 8, 16);
        let x = Tensor::matrix(4, 8, vec![0.1f32; 32]).unwrap();
        block.forward(&x).unwrap();
        let attn = block.last_attention.borrow();
        // Shape: [batch=1, heads=2, T=4, T=4] = 32 elements
        assert_eq!(attn.len(), 2 * 4 * 4);
    }

    #[test]
    fn transformer_block_causal_mask_enforced() {
        // Causal mask: attn[i, j] should be 0 for j > i.
        let block = minimal_block(4, 1, 8, 16);
        let x = Tensor::matrix(4, 8, (0..32).map(|i| (i as f32) * 0.1).collect()).unwrap();
        block.forward(&x).unwrap();
        let attn = block.last_attention.borrow();
        let t = 4;
        for qi in 0..t {
            for ki in 0..t {
                if ki > qi {
                    // Future tokens must have zero attention weight
                    assert!(
                        attn[qi * t + ki] < 1e-6,
                        "attn[{qi},{ki}]={} should be ~0 (causal mask)",
                        attn[qi * t + ki]
                    );
                }
            }
        }
    }

    #[test]
    fn transformer_block_attention_rows_sum_to_one() {
        let block = minimal_block(4, 2, 8, 16);
        let x = Tensor::matrix(4, 8, vec![0.3f32; 32]).unwrap();
        block.forward(&x).unwrap();
        let attn = block.last_attention.borrow();
        // Each query row of the attention matrix should sum to 1 (softmax)
        let t = 4;
        let heads = 2;
        for h in 0..heads {
            for qi in 0..t {
                let base = (h * t + qi) * t;
                let row_sum: f32 = attn[base..base + t].iter().sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "head {h} row {qi}: attn sum = {row_sum}"
                );
            }
        }
    }

    #[test]
    fn transformer_block_accessors() {
        let block = minimal_block(8, 4, 16, 64);
        assert_eq!(block.context_len(), 8);
        assert_eq!(block.num_heads(), 4);
        assert_eq!(block.embedding_dim(), 16);
        assert_eq!(block.hidden_dim(), 64);
    }

    #[test]
    fn transformer_block_name_contains_config() {
        let block = minimal_block(4, 2, 8, 32);
        let name = block.name();
        assert!(name.contains("heads=2") && name.contains("dim=8"));
    }

    #[test]
    fn kv_cache_matches_full_forward() {
        // Token-at-a-time generation through the cache must reproduce the
        // full-sequence forward pass row for row.
        use crate::rng::Rng;
        let (t, heads, dim, hidden) = (6, 2, 8, 16);
        let mut rng = Rng::new(11);
        let mut randn =
            |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_normal() * 0.2).collect() };
        let block = TransformerBlock::new(
            t,
            heads,
            dim,
            vec![1.0; dim],
            vec![0.0; dim],
            randn(dim * dim),
            randn(dim),
            randn(dim * dim),
            randn(dim),
            randn(dim * dim),
            randn(dim),
            randn(dim * dim),
            randn(dim),
            vec![1.0; dim],
            vec![0.0; dim],
            randn(dim * hidden),
            randn(hidden),
            randn(hidden * dim),
            randn(dim),
        )
        .unwrap();

        let x_data = randn(t * dim);
        let x = Tensor::matrix(t, dim, x_data.clone()).unwrap();
        let full = block.forward(&x).unwrap();

        let mut cache = KvCache::new(t, dim);
        for r in 0..t {
            let row = Tensor::matrix(1, dim, x_data[r * dim..(r + 1) * dim].to_vec()).unwrap();
            let inc = block.forward_with_cache(&row, &mut cache).unwrap();
            for d in 0..dim {
                let a = full.data[r * dim + d];
                let b = inc.data[d];
                assert!((a - b).abs() < 1e-4, "row {r} dim {d}: full={a} cached={b}");
            }
        }
        assert!(cache.is_full());
    }

    #[test]
    fn block_ffn_uses_gelu_in_full_and_cached_paths() {
        // Attention contributes nothing (all zero weights), so out = x + ffn2(gelu(ffn1(ln2(x)))).
        // A negative pre-activation distinguishes GELU (-0.154 at -0.5) from ReLU (0).
        let z = |n: usize| vec![0.0f32; n];
        let block = TransformerBlock::new(
            1,
            1,
            2,
            vec![1.0; 2],
            z(2),
            z(4),
            z(2),
            z(4),
            z(2),
            z(4),
            z(2),
            z(4),
            z(2),
            vec![1.0; 2],
            z(2),
            vec![-0.5, 0.0],
            z(1), // ffn1: hidden = -0.5 * norm2[0]
            vec![1.0, 0.0],
            z(2), // ffn2: out[0] += hidden
        )
        .unwrap();
        let x = Tensor::matrix(1, 2, vec![3.0, 1.0]).unwrap();
        let n2 = block.ln2.forward(&x).unwrap();
        assert!(n2.data[0] > 0.9, "norm2[0] = {}", n2.data[0]);
        let pre = -0.5 * n2.data[0];
        let want = [3.0 + crate::activation::gelu(pre), 1.0];
        assert!(
            crate::activation::gelu(pre) < -0.1,
            "test must hit the negative lobe"
        );
        let full = block.forward(&x).unwrap();
        let mut cache = KvCache::new(1, 2);
        let cached = block.forward_with_cache(&x, &mut cache).unwrap();
        for (d, &w) in want.iter().enumerate() {
            assert!(
                (full.data[d] - w).abs() < 1e-6,
                "full[{d}] {} vs {}",
                full.data[d],
                w
            );
            assert!((cached.data[d] - w).abs() < 1e-6, "cached[{d}]");
        }
    }

    #[test]
    fn kv_cache_overflow_errors() {
        let block = minimal_block(2, 1, 4, 8);
        let mut cache = KvCache::new(2, 4);
        let row = Tensor::matrix(1, 4, vec![0.1; 4]).unwrap();
        block.forward_with_cache(&row, &mut cache).unwrap();
        block.forward_with_cache(&row, &mut cache).unwrap();
        assert!(block.forward_with_cache(&row, &mut cache).is_err());
        cache.clear();
        assert!(block.forward_with_cache(&row, &mut cache).is_ok());
    }

    #[test]
    fn transformer_block_invalid_heads_rejected() {
        // 3 heads do not divide dim=8; 0 heads is invalid; 0 context invalid.
        let c = 8usize;
        let mk = |ctx: usize, heads: usize| {
            TransformerBlock::new(
                ctx,
                heads,
                c,
                vec![1.0; c],
                vec![0.0; c],
                vec![0.0; c * c],
                vec![0.0; c],
                vec![0.0; c * c],
                vec![0.0; c],
                vec![0.0; c * c],
                vec![0.0; c],
                vec![0.0; c * c],
                vec![0.0; c],
                vec![1.0; c],
                vec![0.0; c],
                vec![0.0; c * c],
                vec![0.0; c],
                vec![0.0; c * c],
                vec![0.0; c],
            )
        };
        assert!(mk(4, 3).is_err());
        assert!(mk(4, 0).is_err());
        assert!(mk(0, 2).is_err());
        assert!(mk(4, 2).is_ok());
    }

    #[test]
    fn transformer_block_residual_adds_input() {
        // With near-zero weights the block output ≈ input (residual dominates).
        let block = minimal_block(2, 1, 4, 8);
        let input_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let x = Tensor::matrix(2, 4, input_data.clone()).unwrap();
        let y = block.forward(&x).unwrap();
        // Residual ensures y is not all-zero even with tiny weights
        assert!(y.data.iter().any(|&v| v.abs() > 0.01));
    }
}
