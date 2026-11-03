//! Per-block forward / backward for attention and the SwiGLU feed-forward.
use crate::llm::{Attention, FeedForward, LlamaBlock, LlamaConfig};
use crate::llm_train::buffers::BlockBuf;
use crate::llm_train::kernels::{
    lin_backward, lin_forward, rmsnorm_backward_row, rope_apply, silu, silu_grad,
};
use crate::rng::Rng;

pub(super) struct AttnCache {
    pub(super) xn: Vec<f32>,  // [seq, dim] projection input (= rmsnorm1 output)
    pub(super) q: Vec<f32>,   // [seq, q_dim] RoPE'd queries
    pub(super) k: Vec<f32>,   // [seq, kv_dim] RoPE'd keys
    pub(super) v: Vec<f32>,   // [seq, kv_dim] values
    pub(super) ctx: Vec<f32>, // [seq, q_dim] attention context (wo input)
    pub(super) probs: Vec<Vec<f32>>, // [seq*n_heads] softmax weights, entry (t,h) has len t+1
}

pub(super) struct FfnCache {
    pub(super) xn: Vec<f32>, // [seq, dim] FFN input (= rmsnorm2 output)
    pub(super) g: Vec<f32>,  // [seq, ffn] gate pre-activation
    pub(super) u: Vec<f32>,  // [seq, ffn] up
    pub(super) h: Vec<f32>,  // [seq, ffn] down input: silu(g)·u, post-dropout when active
    /// Inverted-dropout mask×scale applied to the hidden, present only when
    /// dropout was active for this forward; `None` means no dropout.
    pub(super) mask: Option<Vec<f32>>,
}

pub(super) struct BlockCache {
    pub(super) x: Vec<f32>,      // [seq, dim] block input
    pub(super) n1_inv: Vec<f32>, // [seq]
    pub(super) attn: AttnCache,
    pub(super) h: Vec<f32>,      // [seq, dim] x + attn output
    pub(super) n2_inv: Vec<f32>, // [seq]
    pub(super) ffn: FfnCache,
}

pub(super) struct FwdCache {
    pub(super) blocks: Vec<BlockCache>,
    pub(super) xfinal: Vec<f32>, // [seq, dim] last block output (final-norm input)
    pub(super) nf_inv: Vec<f32>, // [seq]
    pub(super) xn: Vec<f32>,     // [seq, dim] final-norm output (lm_head input)
    pub(super) logits: Vec<f32>, // [seq, vocab]
}

pub(super) fn attn_forward(
    cfg: &LlamaConfig,
    attn: &Attention,
    xn: &[f32],
    seq: usize,
) -> AttnCache {
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (q_dim, kv_dim, group) = (nh * hd, nkv * hd, nh / nkv);
    let scale = 1.0 / (hd as f32).sqrt();

    let mut q = lin_forward(&attn.wq, xn, seq);
    let mut k = lin_forward(&attn.wk, xn, seq);
    let v = lin_forward(&attn.wv, xn, seq);
    for t in 0..seq {
        rope_apply(
            &mut q[t * q_dim..t * q_dim + q_dim],
            nh,
            hd,
            cfg.rope_dim,
            t,
            cfg.rope_base,
            cfg.rope_type,
            &cfg.rope_scaling,
            false,
        );
        rope_apply(
            &mut k[t * kv_dim..t * kv_dim + kv_dim],
            nkv,
            hd,
            cfg.rope_dim,
            t,
            cfg.rope_base,
            cfg.rope_type,
            &cfg.rope_scaling,
            false,
        );
    }

    let mut ctx = vec![0.0f32; seq * q_dim];
    let mut probs: Vec<Vec<f32>> = vec![Vec::new(); seq * nh];
    for t in 0..seq {
        for h in 0..nh {
            let kvh = h / group;
            let qh = &q[t * q_dim + h * hd..t * q_dim + h * hd + hd];
            let mut sc = vec![0.0f32; t + 1];
            for (j, scj) in sc.iter_mut().enumerate() {
                let kj = &k[j * kv_dim + kvh * hd..j * kv_dim + kvh * hd + hd];
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += qh[d] * kj[d];
                }
                *scj = dot * scale;
            }
            let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in sc.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            for s in sc.iter_mut() {
                *s /= sum;
            }
            let oh = &mut ctx[t * q_dim + h * hd..t * q_dim + h * hd + hd];
            for (j, &p) in sc.iter().enumerate() {
                let vj = &v[j * kv_dim + kvh * hd..j * kv_dim + kvh * hd + hd];
                for d in 0..hd {
                    oh[d] += p * vj[d];
                }
            }
            probs[t * nh + h] = sc;
        }
    }
    AttnCache {
        xn: xn.to_vec(),
        q,
        k,
        v,
        ctx,
        probs,
    }
}

/// Returns the attention sublayer output `o = wo(ctx)` for the cache.
pub(super) fn attn_output(attn: &Attention, c: &AttnCache, seq: usize) -> Vec<f32> {
    lin_forward(&attn.wo, &c.ctx, seq)
}

fn attn_backward(
    cfg: &LlamaConfig,
    attn: &Attention,
    c: &AttnCache,
    d_o: &[f32],
    seq: usize,
    g: &mut BlockBuf,
) -> Vec<f32> {
    let (nh, nkv, hd, dim) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.model_dim);
    let (q_dim, kv_dim, group) = (nh * hd, nkv * hd, nh / nkv);
    let scale = 1.0 / (hd as f32).sqrt();

    let d_ctx = lin_backward(&attn.wo, &c.ctx, d_o, seq, &mut g.wo);
    let mut dq = vec![0.0f32; seq * q_dim];
    let mut dk = vec![0.0f32; seq * kv_dim];
    let mut dv = vec![0.0f32; seq * kv_dim];

    for t in 0..seq {
        for h in 0..nh {
            let kvh = h / group;
            let p = &c.probs[t * nh + h];
            let dctx = &d_ctx[t * q_dim + h * hd..t * q_dim + h * hd + hd];
            // dp_j = dctx · v_j ; dv_j += p_j · dctx
            let mut dp = vec![0.0f32; t + 1];
            for (j, dpj) in dp.iter_mut().enumerate() {
                let vbase = j * kv_dim + kvh * hd;
                let vj = &c.v[vbase..vbase + hd];
                let mut acc = 0.0f32;
                for d in 0..hd {
                    acc += dctx[d] * vj[d];
                    dv[vbase + d] += p[j] * dctx[d];
                }
                *dpj = acc;
            }
            // softmax backward: dscore_j = p_j (dp_j − Σ_l p_l dp_l)
            let dot: f32 = (0..=t).map(|j| p[j] * dp[j]).sum();
            let qbase = t * q_dim + h * hd;
            for j in 0..=t {
                let ds = p[j] * (dp[j] - dot) * scale;
                let kbase = j * kv_dim + kvh * hd;
                for d in 0..hd {
                    dq[qbase + d] += ds * c.k[kbase + d];
                    dk[kbase + d] += ds * c.q[qbase + d];
                }
            }
        }
    }

    // RoPE backward (transpose rotation) on dq, dk.
    for t in 0..seq {
        rope_apply(
            &mut dq[t * q_dim..t * q_dim + q_dim],
            nh,
            hd,
            cfg.rope_dim,
            t,
            cfg.rope_base,
            cfg.rope_type,
            &cfg.rope_scaling,
            true,
        );
        rope_apply(
            &mut dk[t * kv_dim..t * kv_dim + kv_dim],
            nkv,
            hd,
            cfg.rope_dim,
            t,
            cfg.rope_base,
            cfg.rope_type,
            &cfg.rope_scaling,
            true,
        );
    }

    let dxq = lin_backward(&attn.wq, &c.xn, &dq, seq, &mut g.wq);
    let dxk = lin_backward(&attn.wk, &c.xn, &dk, seq, &mut g.wk);
    let dxv = lin_backward(&attn.wv, &c.xn, &dv, seq, &mut g.wv);
    (0..seq * dim).map(|i| dxq[i] + dxk[i] + dxv[i]).collect()
}

/// SwiGLU FFN forward. When `dropout > 0`, the hidden `silu(g)⊙u` is
/// inverted-dropout masked using a generator seeded by `seed`, so a worker can
/// reproduce the mask in backward from the cache.
pub(super) fn ffn_forward(
    cfg: &LlamaConfig,
    ffn: &FeedForward,
    xn: &[f32],
    seq: usize,
    dropout: f32,
    seed: u64,
) -> FfnCache {
    let f = cfg.ffn_dim;
    let g = lin_forward(&ffn.gate, xn, seq);
    let u = lin_forward(&ffn.up, xn, seq);
    let mut h: Vec<f32> = (0..seq * f).map(|i| silu(g[i]) * u[i]).collect();
    let mask = if dropout > 0.0 {
        let scale = 1.0 / (1.0 - dropout);
        let mut drng = Rng::new(seed);
        let mask: Vec<f32> = (0..h.len())
            .map(|_| {
                if drng.next_f32() < dropout {
                    0.0
                } else {
                    scale
                }
            })
            .collect();
        for (hi, &mi) in h.iter_mut().zip(&mask) {
            *hi *= mi;
        }
        Some(mask)
    } else {
        None
    };
    FfnCache {
        xn: xn.to_vec(),
        g,
        u,
        h,
        mask,
    }
}

pub(super) fn ffn_output(ffn: &FeedForward, c: &FfnCache, seq: usize) -> Vec<f32> {
    lin_forward(&ffn.down, &c.h, seq)
}

fn ffn_backward(
    cfg: &LlamaConfig,
    ffn: &FeedForward,
    c: &FfnCache,
    d_f: &[f32],
    seq: usize,
    g: &mut BlockBuf,
) -> Vec<f32> {
    let (dim, f) = (cfg.model_dim, cfg.ffn_dim);
    let d_h_in = lin_backward(&ffn.down, &c.h, d_f, seq, &mut g.down);
    let mut dg = vec![0.0f32; seq * f];
    let mut du = vec![0.0f32; seq * f];
    for i in 0..seq * f {
        // Propagate back through the (optional) inverted-dropout mask to the
        // pre-dropout hidden, then through the SwiGLU gate.
        let d_pre = match &c.mask {
            Some(mask) => d_h_in[i] * mask[i],
            None => d_h_in[i],
        };
        du[i] = d_pre * silu(c.g[i]);
        dg[i] = d_pre * c.u[i] * silu_grad(c.g[i]);
    }
    let dxg = lin_backward(&ffn.gate, &c.xn, &dg, seq, &mut g.gate);
    let dxu = lin_backward(&ffn.up, &c.xn, &du, seq, &mut g.up);
    (0..seq * dim).map(|i| dxg[i] + dxu[i]).collect()
}

pub(super) fn block_backward(
    cfg: &LlamaConfig,
    block: &LlamaBlock,
    bc: &BlockCache,
    d_out: &[f32],
    seq: usize,
    g: &mut BlockBuf,
) -> Vec<f32> {
    let dim = cfg.model_dim;
    // out = h + ffn(rmsnorm2(h)); d_f = d_out (residual to h handled below).
    let d_n2 = ffn_backward(cfg, &block.ffn, &bc.ffn, d_out, seq, g);
    let mut d_h = d_out.to_vec();
    for t in 0..seq {
        let s = t * dim;
        let dxr = rmsnorm_backward_row(
            &d_n2[s..s + dim],
            &bc.h[s..s + dim],
            &block.ffn_norm.weight,
            bc.n2_inv[t],
            &mut g.ffn_norm,
        );
        for d in 0..dim {
            d_h[s + d] += dxr[d];
        }
    }
    // h = x + attn(rmsnorm1(x)); d_o = d_h (residual to x handled below).
    let d_n1 = attn_backward(cfg, &block.attn, &bc.attn, &d_h, seq, g);
    let mut d_x = d_h.clone();
    for t in 0..seq {
        let s = t * dim;
        let dxr = rmsnorm_backward_row(
            &d_n1[s..s + dim],
            &bc.x[s..s + dim],
            &block.attn_norm.weight,
            bc.n1_inv[t],
            &mut g.attn_norm,
        );
        for d in 0..dim {
            d_x[s + d] += dxr[d];
        }
    }
    d_x
}
