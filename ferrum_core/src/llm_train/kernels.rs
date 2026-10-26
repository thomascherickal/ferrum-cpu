//! Scalar / row kernels: linear, RMSNorm, SiLU, RoPE and cross-entropy.
use crate::layer::Linear;
use crate::llm::RopeType;
use crate::llm_train::buffers::LinBuf;

/// `y[r,o] = Σ_i x[r,i]·W[i,o] + b[o]`.
pub(super) fn lin_forward(lin: &Linear, x: &[f32], rows: usize) -> Vec<f32> {
    let (k, n) = (lin.in_features(), lin.out_features());
    let (w, b) = (&lin.weight.data, &lin.bias.data);
    let mut y = vec![0.0f32; rows * n];
    for r in 0..rows {
        let xr = &x[r * k..r * k + k];
        let yr = &mut y[r * n..r * n + n];
        yr.copy_from_slice(b);
        for (i, &xi) in xr.iter().enumerate() {
            let wr = &w[i * n..i * n + n];
            for (o, &wio) in wr.iter().enumerate() {
                yr[o] += xi * wio;
            }
        }
    }
    y
}

/// Backward of [`lin_forward`]: accumulates `dW`/`db` into `g`, returns `dX`.
pub(super) fn lin_backward(
    lin: &Linear,
    x: &[f32],
    dy: &[f32],
    rows: usize,
    g: &mut LinBuf,
) -> Vec<f32> {
    let (k, n) = (lin.in_features(), lin.out_features());
    let w = &lin.weight.data;
    let mut dx = vec![0.0f32; rows * k];
    for r in 0..rows {
        let xr = &x[r * k..r * k + k];
        let dyr = &dy[r * n..r * n + n];
        let dxr = &mut dx[r * k..r * k + k];
        // An empty `g.b` is a frozen placeholder bias: no gradient.
        for (b, &dyo) in g.b.iter_mut().zip(dyr) {
            *b += dyo;
        }
        for i in 0..k {
            let wr = &w[i * n..i * n + n];
            let dwr = &mut g.w[i * n..i * n + n];
            let xi = xr[i];
            let mut acc = 0.0f32;
            for o in 0..n {
                acc += dyr[o] * wr[o];
                dwr[o] += xi * dyr[o];
            }
            dxr[i] = acc;
        }
    }
    dx
}

/// RMSNorm of one row: `(y, inv)` with `inv = 1/sqrt(mean(x²)+eps)`.
pub(super) fn rmsnorm_forward_row(x: &[f32], w: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let d = x.len();
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / d as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    ((0..d).map(|i| x[i] * inv * w[i]).collect(), inv)
}

/// Backward of RMSNorm for one row; adds the weight gradient into `dw`, returns
/// `dx`. With `g_i = dy_i·w_i`: `dx_j = inv·g_j − (inv³·x_j/d)·Σ_i g_i x_i`.
pub(super) fn rmsnorm_backward_row(
    dy: &[f32],
    x: &[f32],
    w: &[f32],
    inv: f32,
    dw: &mut [f32],
) -> Vec<f32> {
    let d = x.len();
    let mut gx = 0.0f32;
    for i in 0..d {
        gx += dy[i] * w[i] * x[i];
        dw[i] += dy[i] * x[i] * inv;
    }
    let inv3 = inv * inv * inv / d as f32;
    (0..d)
        .map(|j| inv * dy[j] * w[j] - inv3 * x[j] * gx)
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(super) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// d/dx silu(x) = σ(x)·(1 + x·(1−σ(x))).
pub(super) fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// Apply RoPE to one `[n_heads·head_dim]` row; `transpose = true` rotates by −θ
/// (the backward direction, since RoPE is an orthogonal rotation).
#[allow(clippy::too_many_arguments)]
pub(super) fn rope_apply(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    base: f32,
    rope_type: RopeType,
    transpose: bool,
) {
    let half = rope_dim / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let freq = (base as f64).powf(-2.0 * i as f64 / rope_dim as f64) as f32;
            let theta = pos as f32 * freq;
            let (mut s, c) = theta.sin_cos();
            if transpose {
                s = -s;
            }
            let (ia, ib) = match rope_type {
                RopeType::Norm => (off + 2 * i, off + 2 * i + 1),
                RopeType::Neox => (off + i, off + i + half),
            };
            let (a, b) = (x[ia], x[ib]);
            x[ia] = a * c - b * s;
            x[ib] = a * s + b * c;
        }
    }
}

/// Mean next-token cross-entropy and its gradient `dL/d logits` (`[seq, vocab]`).
/// Position `t` predicts `tokens[t+1]`; the last position has no target.
pub(super) fn cross_entropy(logits: &[f32], tokens: &[usize], vocab: usize) -> (f32, Vec<f32>) {
    let seq = tokens.len();
    let n_targets = seq.saturating_sub(1).max(1);
    let mut d = vec![0.0f32; seq * vocab];
    let mut loss = 0.0f32;
    for t in 0..seq.saturating_sub(1) {
        let row = &logits[t * vocab..t * vocab + vocab];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; vocab];
        for j in 0..vocab {
            probs[j] = (row[j] - m).exp();
            sum += probs[j];
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let target = tokens[t + 1];
        loss += -crate::loss::clamp_prob(probs[target]).ln();
        for j in 0..vocab {
            d[t * vocab + j] = (probs[j] - if j == target { 1.0 } else { 0.0 }) / n_targets as f32;
        }
    }
    (loss / n_targets as f32, d)
}

#[cfg(test)]
mod tests {
    use super::{rmsnorm_backward_row, rmsnorm_forward_row, rope_apply, silu, silu_grad};
    use crate::llm::RopeType;
    use crate::llm_train::test_support::det;

    #[test]
    fn cross_entropy_of_nan_logits_is_nan() {
        let logits = [f32::NAN, 0.0, 1.0, 0.5, 0.0, 1.0];
        let (loss, _) = super::cross_entropy(&logits, &[0, 1], 3);
        assert!(loss.is_nan(), "loss {loss}");
    }

    #[test]
    fn rmsnorm_backward_matches_finite_difference() {
        let d = 6;
        let x = det(d, 1);
        let w = det(d, 2).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let dy = det(d, 3);
        let (_, inv) = rmsnorm_forward_row(&x, &w, 1e-5);
        let mut dw = vec![0.0; d];
        let dx = rmsnorm_backward_row(&dy, &x, &w, inv, &mut dw);
        let loss = |x: &[f32], w: &[f32]| {
            let (y, _) = rmsnorm_forward_row(x, w, 1e-5);
            y.iter().zip(&dy).map(|(a, b)| a * b).sum::<f32>()
        };
        let eps = 1e-3;
        for i in 0..d {
            let mut xp = x.clone();
            xp[i] += eps;
            let mut xm = x.clone();
            xm[i] -= eps;
            let fd = (loss(&xp, &w) - loss(&xm, &w)) / (2.0 * eps);
            assert!((fd - dx[i]).abs() < 1e-2, "dx[{i}] fd={fd} got={}", dx[i]);
            let mut wp = w.clone();
            wp[i] += eps;
            let mut wm = w.clone();
            wm[i] -= eps;
            let fdw = (loss(&x, &wp) - loss(&x, &wm)) / (2.0 * eps);
            assert!((fdw - dw[i]).abs() < 1e-2, "dw[{i}] fd={fdw} got={}", dw[i]);
        }
    }

    #[test]
    fn silu_grad_matches_finite_difference() {
        for &g in &[-2.0f32, -0.5, 0.0, 0.7, 3.0] {
            let fd = (silu(g + 1e-3) - silu(g - 1e-3)) / 2e-3;
            assert!((fd - silu_grad(g)).abs() < 1e-3, "silu' at {g}");
        }
    }

    #[test]
    fn rope_transpose_is_the_inverse_rotation() {
        let mut x = det(8, 5);
        let orig = x.clone();
        rope_apply(&mut x, 2, 4, 4, 3, 10000.0, RopeType::Norm, false);
        rope_apply(&mut x, 2, 4, 4, 3, 10000.0, RopeType::Norm, true);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
