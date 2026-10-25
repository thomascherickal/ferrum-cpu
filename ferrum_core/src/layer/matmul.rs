//! Blocked matmul helpers shared by the layers and the trainer.

/// Rows `r0..r1` of `A·Bᵀ` (A is `[m, k]`, B is `[n, k]`) into a locally-indexed
/// `out`. Shared by the serial and pooled paths.
fn transpose_b_block(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    r0: usize,
    r1: usize,
    out: &mut [f32],
) {
    for i in r0..r1 {
        let a_row = i * k;
        let o_row = (i - r0) * n;
        for j in 0..n {
            let b_row = j * k;
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[a_row + p] * b[b_row + p];
            }
            out[o_row + j] = sum;
        }
    }
}

pub(crate) fn matmul_transpose_b_helper(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let cost = m.saturating_mul(n).saturating_mul(k);
    if crate::parallel::should_parallelize(m, cost) {
        let a_arc = std::sync::Arc::<[f32]>::from(a);
        let b_arc = std::sync::Arc::<[f32]>::from(b);
        crate::parallel::run(m, n, move |r0, r1, block| {
            transpose_b_block(&a_arc, &b_arc, k, n, r0, r1, block);
        })
    } else {
        let mut out = vec![0.0f32; m * n];
        transpose_b_block(a, b, k, n, 0, m, &mut out);
        out
    }
}

/// Rows `r0..r1` of `A·B` (A is `[m, k]`, B is `[k, n]`) into a locally-indexed
/// `out`. Shared by the serial and pooled paths.
fn naive_block(a: &[f32], b: &[f32], k: usize, n: usize, r0: usize, r1: usize, out: &mut [f32]) {
    for i in r0..r1 {
        let a_row = i * k;
        let o_row = (i - r0) * n;
        for p in 0..k {
            let a_ip = a[a_row + p];
            let b_row = p * n;
            for j in 0..n {
                out[o_row + j] += a_ip * b[b_row + j];
            }
        }
    }
}

pub(crate) fn matmul_naive_helper(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let cost = m.saturating_mul(n).saturating_mul(k);
    if crate::parallel::should_parallelize(m, cost) {
        let a_arc = std::sync::Arc::<[f32]>::from(a);
        let b_arc = std::sync::Arc::<[f32]>::from(b);
        crate::parallel::run(m, n, move |r0, r1, block| {
            naive_block(&a_arc, &b_arc, k, n, r0, r1, block);
        })
    } else {
        let mut out = vec![0.0f32; m * n];
        naive_block(a, b, k, n, 0, m, &mut out);
        out
    }
}
