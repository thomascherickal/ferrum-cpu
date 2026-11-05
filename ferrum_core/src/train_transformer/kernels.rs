//! Forward / backward numeric kernels: linear, LayerNorm and causal multi-head attention.
use crate::error::Result;
use crate::layer::{matmul_naive_helper, matmul_transpose_b_helper};
use crate::ops;
use crate::tensor::Tensor;

const LN_EPS: f32 = 1e-5; // must match layer::LayerNorm

pub(super) fn linear_fwd(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    ops::add_bias(&ops::matmul(x, w)?, b)
}

/// Returns (dW, dx) for y = xW given dy (the bias-free LM head).
pub(super) fn matmul_bwd(x: &Tensor, w: &Tensor, dy: &Tensor) -> Result<(Tensor, Tensor)> {
    let dw = ops::matmul(&ops::transpose(x)?, dy)?;
    let dx = ops::matmul(dy, &ops::transpose(w)?)?;
    Ok((dw, dx))
}

/// Returns (dW, db, dx) for y = xW + b given dy.
pub(super) fn linear_bwd(x: &Tensor, w: &Tensor, dy: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let (dw, dx) = matmul_bwd(x, w, dy)?;
    let db = ops::sum_axis0(dy)?;
    Ok((dw, db, dx))
}

/// LayerNorm forward. Returns (y, x̂, inv_std) — x̂ and inv_std are needed by
/// the backward pass.
pub(super) fn ln_fwd(x: &Tensor, g: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor, Vec<f32>)> {
    let (rows, cols) = x.matrix_dims()?;
    let mut y = vec![0.0f32; rows * cols];
    let mut xhat = vec![0.0f32; rows * cols];
    let mut inv_std = vec![0.0f32; rows];
    // `r` indexes inv_std and derives the row offset for x/y/xhat — an iterator
    // would obscure the per-row math without removing the offset arithmetic.
    #[allow(clippy::needless_range_loop)]
    for r in 0..rows {
        let base = r * cols;
        let row = &x.data[base..base + cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
        let is = 1.0 / (var + LN_EPS).sqrt();
        inv_std[r] = is;
        for c in 0..cols {
            let xh = (row[c] - mean) * is;
            xhat[base + c] = xh;
            y[base + c] = xh * g.data[c] + b.data[c];
        }
    }
    Ok((
        Tensor::matrix(rows, cols, y)?,
        Tensor::matrix(rows, cols, xhat)?,
        inv_std,
    ))
}

/// LayerNorm backward. Returns (dx, dγ, dβ).
pub(super) fn ln_bwd(
    dy: &Tensor,
    xhat: &Tensor,
    inv_std: &[f32],
    g: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (rows, cols) = dy.matrix_dims()?;
    let n = cols as f32;
    let mut dx = vec![0.0f32; rows * cols];
    let mut dg = vec![0.0f32; cols];
    let mut db = vec![0.0f32; cols];
    // `r` derives the row offset and indexes inv_std; see ln_fwd.
    #[allow(clippy::needless_range_loop)]
    for r in 0..rows {
        let base = r * cols;
        let mut sum_dxhat = 0.0f32;
        let mut sum_dxhat_xhat = 0.0f32;
        for c in 0..cols {
            let dyv = dy.data[base + c];
            let xh = xhat.data[base + c];
            dg[c] += dyv * xh;
            db[c] += dyv;
            let dxh = dyv * g.data[c];
            sum_dxhat += dxh;
            sum_dxhat_xhat += dxh * xh;
        }
        let mean_dxhat = sum_dxhat / n;
        let mean_dxhat_xhat = sum_dxhat_xhat / n;
        for c in 0..cols {
            let dxh = dy.data[base + c] * g.data[c];
            dx[base + c] = inv_std[r] * (dxh - mean_dxhat - xhat.data[base + c] * mean_dxhat_xhat);
        }
    }
    Ok((
        Tensor::matrix(rows, cols, dx)?,
        Tensor::vector(dg),
        Tensor::vector(db),
    ))
}

/// out [m,n] = aᵀ·b where a is [k,m] and b is [k,n].
/// Rows `r0..r1` of `Aᵀ·B`: `out[i,j] = Σ_r a[r,i] · b[r,j]` (A is `[k, m]`,
/// B is `[k, n]`), written into a locally-indexed `out`. Output rows (i) are
/// independent and the sum over r keeps its original order, so the split is
/// exact. Shared by the serial and pooled paths.
#[allow(clippy::too_many_arguments)]
fn transpose_a_block(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    r0: usize,
    r1: usize,
    out: &mut [f32],
) {
    for i in r0..r1 {
        let o_row = (i - r0) * n;
        for r in 0..k {
            let a_ri = a[r * m + i];
            let b_row = r * n;
            for j in 0..n {
                out[o_row + j] += a_ri * b[b_row + j];
            }
        }
    }
}

fn matmul_transpose_a_helper(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let cost = m.saturating_mul(n).saturating_mul(k);
    if crate::parallel::should_parallelize(m, cost) {
        let a_arc = std::sync::Arc::<[f32]>::from(a);
        let b_arc = std::sync::Arc::<[f32]>::from(b);
        crate::parallel::run(m, n, move |r0, r1, block| {
            transpose_a_block(&a_arc, &b_arc, m, n, k, r0, r1, block);
        })
    } else {
        let mut out = vec![0.0f32; m * n];
        transpose_a_block(a, b, m, n, k, 0, m, &mut out);
        out
    }
}

/// Causal multi-head attention forward over [M=B·T, C] projections.
/// Returns (concatenated head outputs [M, C], softmax probs [B·H·T·T]).
pub(super) fn attn_fwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: usize,
    t: usize,
    heads: usize,
) -> Result<(Tensor, Vec<f32>)> {
    let (m, c) = q.matrix_dims()?;
    let head_dim = c / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; m * c];
    let mut probs = vec![0.0f32; batch * heads * t * t];

    let mut q_h = vec![0.0f32; t * head_dim];
    let mut k_h = vec![0.0f32; t * head_dim];
    let mut v_h = vec![0.0f32; t * head_dim];
    for b in 0..batch {
        for h in 0..heads {
            let hs = h * head_dim;
            for r in 0..t {
                let src = (b * t + r) * c + hs;
                let dst = r * head_dim;
                q_h[dst..dst + head_dim].copy_from_slice(&q.data[src..src + head_dim]);
                k_h[dst..dst + head_dim].copy_from_slice(&k.data[src..src + head_dim]);
                v_h[dst..dst + head_dim].copy_from_slice(&v.data[src..src + head_dim]);
            }
            // S = scale·QKᵀ with causal mask, then row-softmax.
            let mut s = matmul_transpose_b_helper(&q_h, &k_h, t, t, head_dim);
            for i in 0..t {
                for j in 0..t {
                    let idx = i * t + j;
                    s[idx] *= scale;
                    if j > i {
                        s[idx] = -1e9;
                    }
                }
            }
            for i in 0..t {
                let row = &mut s[i * t..(i + 1) * t];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for val in row.iter_mut() {
                    *val = (*val - max).exp();
                    sum += *val;
                }
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
            let p_base = (b * heads + h) * t * t;
            probs[p_base..p_base + t * t].copy_from_slice(&s);
            // O = P·V
            let o = matmul_naive_helper(&s, &v_h, t, head_dim, t);
            for r in 0..t {
                let dst = (b * t + r) * c + hs;
                let src = r * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&o[src..src + head_dim]);
            }
        }
    }
    Ok((Tensor::matrix(m, c, out)?, probs))
}

/// Backward through causal multi-head attention. Returns (dq, dk, dv), each [M, C].
#[allow(clippy::too_many_arguments)] // the attention backward genuinely needs q/k/v, probs, and the batch/t/heads shape
pub(super) fn attn_bwd(
    d_out: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    probs: &[f32],
    batch: usize,
    t: usize,
    heads: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (m, c) = q.matrix_dims()?;
    let head_dim = c / heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut dq = vec![0.0f32; m * c];
    let mut dk = vec![0.0f32; m * c];
    let mut dv = vec![0.0f32; m * c];

    let mut q_h = vec![0.0f32; t * head_dim];
    let mut k_h = vec![0.0f32; t * head_dim];
    let mut v_h = vec![0.0f32; t * head_dim];
    let mut do_h = vec![0.0f32; t * head_dim];
    for b in 0..batch {
        for h in 0..heads {
            let hs = h * head_dim;
            for r in 0..t {
                let src = (b * t + r) * c + hs;
                let dst = r * head_dim;
                q_h[dst..dst + head_dim].copy_from_slice(&q.data[src..src + head_dim]);
                k_h[dst..dst + head_dim].copy_from_slice(&k.data[src..src + head_dim]);
                v_h[dst..dst + head_dim].copy_from_slice(&v.data[src..src + head_dim]);
                do_h[dst..dst + head_dim].copy_from_slice(&d_out.data[src..src + head_dim]);
            }
            let p = &probs[(b * heads + h) * t * t..(b * heads + h + 1) * t * t];

            // dV = Pᵀ·dO
            let dv_h = matmul_transpose_a_helper(p, &do_h, t, head_dim, t);
            // dP = dO·Vᵀ
            let dp = matmul_transpose_b_helper(&do_h, &v_h, t, t, head_dim);
            // Softmax backward: dS_ij = P_ij·(dP_ij − Σ_k dP_ik·P_ik).
            // Masked entries have P = 0, so dS = 0 there automatically.
            let mut ds = vec![0.0f32; t * t];
            for i in 0..t {
                let base = i * t;
                let row_dot: f32 = (0..t).map(|j| dp[base + j] * p[base + j]).sum();
                for j in 0..t {
                    ds[base + j] = p[base + j] * (dp[base + j] - row_dot) * scale;
                }
            }
            // dQ = dS·K ;  dK = dSᵀ·Q
            let dq_h = matmul_naive_helper(&ds, &k_h, t, head_dim, t);
            let dk_h = matmul_transpose_a_helper(&ds, &q_h, t, head_dim, t);

            for r in 0..t {
                let dst = (b * t + r) * c + hs;
                let src = r * head_dim;
                dq[dst..dst + head_dim].copy_from_slice(&dq_h[src..src + head_dim]);
                dk[dst..dst + head_dim].copy_from_slice(&dk_h[src..src + head_dim]);
                dv[dst..dst + head_dim].copy_from_slice(&dv_h[src..src + head_dim]);
            }
        }
    }
    Ok((
        Tensor::matrix(m, c, dq)?,
        Tensor::matrix(m, c, dk)?,
        Tensor::matrix(m, c, dv)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
    }

    fn mat(rows: usize, cols: usize, seed: f32) -> Tensor {
        Tensor::matrix(rows, cols, data(rows * cols, seed)).unwrap()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Central finite differences of the scalar `loss` at `x`, compared with `analytic`.
    fn assert_fd(what: &str, x: &[f32], analytic: &[f32], loss: impl Fn(&[f32]) -> f32) {
        assert_eq!(x.len(), analytic.len(), "{what}: gradient length");
        let eps = 1e-2f32;
        for i in 0..x.len() {
            let (mut up, mut down) = (x.to_vec(), x.to_vec());
            up[i] += eps;
            down[i] -= eps;
            let num = (loss(&up) - loss(&down)) / (2.0 * eps);
            let tol = 1e-2 + 1e-2 * num.abs();
            assert!(
                (num - analytic[i]).abs() < tol,
                "{what}[{i}]: numeric {num} vs analytic {}",
                analytic[i]
            );
        }
    }

    #[test]
    fn linear_fwd_adds_the_bias_to_every_row() {
        let x = Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let w = Tensor::matrix(2, 1, vec![10.0, 100.0]).unwrap();
        let y = linear_fwd(&x, &w, &Tensor::vector(vec![0.5])).unwrap();
        assert_eq!(y.data, vec![210.5, 430.5]);
    }

    #[test]
    fn linear_bwd_matches_finite_differences() {
        let (x, w, b) = (mat(3, 4, 0.1), mat(4, 2, 0.5), Tensor::vector(data(2, 0.9)));
        let r = data(6, 1.3); // dL/dy for L = Σ r ⊙ y
        let dy = Tensor::matrix(3, 2, r.clone()).unwrap();
        let (dw, db, dx) = linear_bwd(&x, &w, &dy).unwrap();
        let loss = |x: &Tensor, w: &Tensor, b: &Tensor| dot(&linear_fwd(x, w, b).unwrap().data, &r);
        assert_fd("dx", &x.data, &dx.data, |v| {
            loss(&Tensor::matrix(3, 4, v.to_vec()).unwrap(), &w, &b)
        });
        assert_fd("dw", &w.data, &dw.data, |v| {
            loss(&x, &Tensor::matrix(4, 2, v.to_vec()).unwrap(), &b)
        });
        assert_fd("db", &b.data, &db.data, |v| {
            loss(&x, &w, &Tensor::vector(v.to_vec()))
        });
        // matmul_bwd is the bias-free half of the same thing.
        let (dw2, dx2) = matmul_bwd(&x, &w, &dy).unwrap();
        assert_eq!((dw2.data, dx2.data), (dw.data, dx.data));
    }

    #[test]
    fn ln_fwd_normalises_each_row_then_applies_gain_and_bias() {
        let x = Tensor::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, -5.0, 0.0, 5.0, 10.0]).unwrap();
        let g = Tensor::vector(vec![2.0; 4]);
        let b = Tensor::vector(vec![1.0; 4]);
        let (y, xhat, inv_std) = ln_fwd(&x, &g, &b).unwrap();
        assert_eq!(inv_std.len(), 2);
        for r in 0..2 {
            let row = &xhat.data[r * 4..r * 4 + 4];
            let mean = row.iter().sum::<f32>() / 4.0;
            let var = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5 && (var - 1.0).abs() < 1e-3, "row {r}");
        }
        for (yv, xh) in y.data.iter().zip(&xhat.data) {
            assert!((yv - (2.0 * xh + 1.0)).abs() < 1e-6);
        }
        // var of [1,2,3,4] is 1.25, so inv_std = 1/sqrt(1.25 + eps).
        assert!((inv_std[0] - 1.0 / (1.25f32 + LN_EPS).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn ln_bwd_matches_finite_differences() {
        let (x, g, b) = (
            mat(3, 5, 0.2),
            Tensor::vector(data(5, 2.0)),
            Tensor::vector(data(5, 3.0)),
        );
        let r = data(15, 0.7);
        let (_, xhat, inv_std) = ln_fwd(&x, &g, &b).unwrap();
        let dy = Tensor::matrix(3, 5, r.clone()).unwrap();
        let (dx, dg, db) = ln_bwd(&dy, &xhat, &inv_std, &g).unwrap();
        let loss = |x: &Tensor, g: &Tensor, b: &Tensor| dot(&ln_fwd(x, g, b).unwrap().0.data, &r);
        assert_fd("dx", &x.data, &dx.data, |v| {
            loss(&Tensor::matrix(3, 5, v.to_vec()).unwrap(), &g, &b)
        });
        assert_fd("dg", &g.data, &dg.data, |v| {
            loss(&x, &Tensor::vector(v.to_vec()), &b)
        });
        assert_fd("db", &b.data, &db.data, |v| {
            loss(&x, &g, &Tensor::vector(v.to_vec()))
        });
    }

    #[test]
    fn transpose_a_helper_matches_a_plain_loop_serial_and_pooled() {
        // 3x5x4 runs serially; 64x64x32 crosses the pool threshold.
        for (m, n, k) in [(3, 5, 4), (64, 64, 32)] {
            let a = data(k * m, 0.1); // [k, m]
            let b = data(k * n, 0.7); // [k, n]
            let mut want = vec![0.0f32; m * n];
            for i in 0..m {
                for r in 0..k {
                    for j in 0..n {
                        want[i * n + j] += a[r * m + i] * b[r * n + j];
                    }
                }
            }
            assert_eq!(
                matmul_transpose_a_helper(&a, &b, m, n, k),
                want,
                "{m}x{n}x{k}"
            );
        }
    }

    const BATCH: usize = 2;
    const T: usize = 3;
    const HEADS: usize = 2;
    const C: usize = 4;
    const M: usize = BATCH * T;

    #[test]
    fn attn_fwd_is_causal_and_its_probs_are_row_softmaxes() {
        let (q, k, v) = (mat(M, C, 0.1), mat(M, C, 0.4), mat(M, C, 0.9));
        let (out, probs) = attn_fwd(&q, &k, &v, BATCH, T, HEADS).unwrap();
        assert_eq!(probs.len(), BATCH * HEADS * T * T);
        for (n, row) in probs.chunks(T).enumerate() {
            let i = n % T; // query position within its T x T block
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(
                row[i + 1..].iter().all(|&p| p == 0.0),
                "future key has weight"
            );
        }
        // Position 0 of each sequence sees only itself: its output is its own V row.
        for b in 0..BATCH {
            let r = b * T * C;
            assert_eq!(out.data[r..r + C], v.data[r..r + C]);
        }
        // Changing the last position's K and V leaves every earlier output unchanged.
        let (mut k2, mut v2) = (k.clone(), v.clone());
        for c in 0..C {
            k2.data[(T - 1) * C + c] += 3.0;
            v2.data[(T - 1) * C + c] -= 2.0;
        }
        let (out2, _) = attn_fwd(&q, &k2, &v2, BATCH, T, HEADS).unwrap();
        assert_eq!(out.data[..(T - 1) * C], out2.data[..(T - 1) * C]);
        assert_ne!(out.data[(T - 1) * C..T * C], out2.data[(T - 1) * C..T * C]);
    }

    #[test]
    fn attn_bwd_matches_finite_differences() {
        let (q, k, v) = (mat(M, C, 0.1), mat(M, C, 0.4), mat(M, C, 0.9));
        let r = data(M * C, 1.7);
        let (_, probs) = attn_fwd(&q, &k, &v, BATCH, T, HEADS).unwrap();
        let d_out = Tensor::matrix(M, C, r.clone()).unwrap();
        let (dq, dk, dv) = attn_bwd(&d_out, &q, &k, &v, &probs, BATCH, T, HEADS).unwrap();
        let t = |v: &[f32]| Tensor::matrix(M, C, v.to_vec()).unwrap();
        let loss = |q: &Tensor, k: &Tensor, v: &Tensor| {
            dot(&attn_fwd(q, k, v, BATCH, T, HEADS).unwrap().0.data, &r)
        };
        assert_fd("dq", &q.data, &dq.data, |x| loss(&t(x), &k, &v));
        assert_fd("dk", &k.data, &dk.data, |x| loss(&q, &t(x), &v));
        assert_fd("dv", &v.data, &dv.data, |x| loss(&q, &k, &t(x)));
    }
}
