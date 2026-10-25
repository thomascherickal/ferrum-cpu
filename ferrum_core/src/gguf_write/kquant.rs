//! k-quant super-block encoders (Q4_K / Q5_K / Q6_K).
use crate::gguf::{Q4_K_BLOCK, Q5_K_BLOCK, Q6_K_BLOCK, QK_K};
use crate::gguf_write::blocks::f32_to_f16;

/// Pack 8 six-bit sub-block scales and mins into the 12-byte layout that
/// [`crate::gguf::get_scale_min_k4`] reads back. Exact inverse of that function.
fn put_scale_min_k4(sc: &[u8; 8], m: &[u8; 8]) -> [u8; 12] {
    let mut q = [0u8; 12];
    for j in 0..4 {
        q[j] = sc[j] & 63;
        q[j + 4] = m[j] & 63;
    }
    for j in 4..8 {
        q[j + 4] = (sc[j] & 0x0F) | ((m[j] & 0x0F) << 4);
        q[j - 4] |= (sc[j] >> 4) << 6; // top 2 bits of the 6-bit scale
        q[j] |= (m[j] >> 4) << 6; // top 2 bits of the 6-bit min
    }
    q
}

/// Q4_K: per 256-element super-block, 8 sub-blocks of 32. Each sub-block gets an
/// affine (scale, min) chosen to cover its [lo, hi]; those reals are then
/// quantized to 6-bit via super-block `d`/`dmin`. Decode: `d*sc_s*q - dmin*m_s`.
pub(super) fn enc_q4_k(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK_K * Q4_K_BLOCK);
    for sblk in data.chunks_exact(QK_K) {
        // 1. Per sub-block real scale/min covering [lo, hi], with min >= 0.
        let mut scale_r = [0.0f32; 8];
        let mut min_r = [0.0f32; 8];
        for s in 0..8 {
            let seg = &sblk[s * 32..s * 32 + 32];
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &v in seg {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let min = if lo < 0.0 { -lo } else { 0.0 }; // >= 0
            let base = -min; // <= 0
            let scale = (hi - base) / 15.0;
            scale_r[s] = if scale > 0.0 { scale } else { 0.0 };
            min_r[s] = min;
        }
        // 2. Super-block factors and 6-bit sub-block codes.
        let dmax = scale_r.iter().cloned().fold(0.0f32, f32::max);
        let mmax = min_r.iter().cloned().fold(0.0f32, f32::max);
        let d = dmax / 63.0;
        let dmin = mmax / 63.0;
        let (mut sc, mut m) = ([0u8; 8], [0u8; 8]);
        for s in 0..8 {
            sc[s] = if d > 0.0 {
                ((scale_r[s] / d).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
            m[s] = if dmin > 0.0 {
                ((min_r[s] / dmin).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
        }
        // 3. Quantize each element against the ACTUAL reconstructed scale/min.
        let mut qs = [0u8; QK_K / 2];
        for s in 0..8 {
            let a_scale = d * sc[s] as f32;
            let a_min = dmin * m[s] as f32;
            for l in 0..32 {
                let x = sblk[s * 32 + l];
                let q = if a_scale > 0.0 {
                    (((x + a_min) / a_scale).round() as i32).clamp(0, 15) as u8
                } else {
                    0
                };
                let byte = 32 * (s / 2) + l;
                if s % 2 == 0 {
                    qs[byte] |= q;
                } else {
                    qs[byte] |= q << 4;
                }
            }
        }
        // 4. Emit: d, dmin (f16), 12-byte packed scales, 128-byte qs.
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        o.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        o.extend_from_slice(&put_scale_min_k4(&sc, &m));
        o.extend_from_slice(&qs);
    }
    o
}

/// Q5_K: like Q4_K but 5-bit quants (`qv ∈ 0..31`). The low nibble is packed as
/// in Q4_K; the 5th bit rides in `qh` (bit `s` of `qh[l]`). Decode:
/// `d*sc_s*qv - dmin*m_s`.
pub(super) fn enc_q5_k(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK_K * Q5_K_BLOCK);
    for sblk in data.chunks_exact(QK_K) {
        let mut scale_r = [0.0f32; 8];
        let mut min_r = [0.0f32; 8];
        for s in 0..8 {
            let seg = &sblk[s * 32..s * 32 + 32];
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &v in seg {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let min = if lo < 0.0 { -lo } else { 0.0 };
            let base = -min;
            let scale = (hi - base) / 31.0; // 5-bit → 31 levels
            scale_r[s] = if scale > 0.0 { scale } else { 0.0 };
            min_r[s] = min;
        }
        let d = scale_r.iter().cloned().fold(0.0f32, f32::max) / 63.0;
        let dmin = min_r.iter().cloned().fold(0.0f32, f32::max) / 63.0;
        let (mut sc, mut m) = ([0u8; 8], [0u8; 8]);
        for s in 0..8 {
            sc[s] = if d > 0.0 {
                ((scale_r[s] / d).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
            m[s] = if dmin > 0.0 {
                ((min_r[s] / dmin).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
        }
        let mut qs = [0u8; QK_K / 2];
        let mut qh = [0u8; QK_K / 8]; // 32 bytes, one bit per sub-block
        for s in 0..8 {
            let a_scale = d * sc[s] as f32;
            let a_min = dmin * m[s] as f32;
            for l in 0..32 {
                let x = sblk[s * 32 + l];
                let qv = if a_scale > 0.0 {
                    (((x + a_min) / a_scale).round() as i32).clamp(0, 31) as u8
                } else {
                    0
                };
                let low = qv & 0x0F;
                let high = (qv >> 4) & 1;
                let byte = 32 * (s / 2) + l;
                if s % 2 == 0 {
                    qs[byte] |= low;
                } else {
                    qs[byte] |= low << 4;
                }
                if high != 0 {
                    qh[l] |= 1 << s;
                }
            }
        }
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        o.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        o.extend_from_slice(&put_scale_min_k4(&sc, &m));
        o.extend_from_slice(&qh);
        o.extend_from_slice(&qs);
    }
    o
}

/// Q6_K: per 256-element super-block, 16 groups of 16 with signed 6-bit quants
/// (`q ∈ -32..31`) and an `i8` scale per group, all multiplied by a super-block
/// `d` (f16). Decode: `d * scale_g * q`.
pub(super) fn enc_q6_k(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK_K * Q6_K_BLOCK);
    for sblk in data.chunks_exact(QK_K) {
        // 1. Per 16-element group, a real scale mapping amax → q = 31.
        let mut gscale = [0.0f32; 16];
        for g in 0..16 {
            let seg = &sblk[g * 16..g * 16 + 16];
            let amax = seg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            gscale[g] = amax / 31.0;
        }
        // 2. Super-block d and i8 group scales.
        let dmax = gscale.iter().cloned().fold(0.0f32, f32::max);
        let d = dmax / 127.0;
        let mut scales = [0i8; 16];
        for g in 0..16 {
            scales[g] = if d > 0.0 {
                ((gscale[g] / d).round() as i32).clamp(-127, 127) as i8
            } else {
                0
            };
        }
        // 3. Quantize each element to a signed 6-bit code, stored as u = q + 32.
        let mut u = [0u8; QK_K];
        for g in 0..16 {
            let a = d * scales[g] as f32;
            for l in 0..16 {
                let x = sblk[g * 16 + l];
                let q = if a != 0.0 {
                    ((x / a).round() as i32).clamp(-32, 31)
                } else {
                    0
                };
                u[g * 16 + l] = (q + 32) as u8; // 0..63
            }
        }
        // 4. Pack ql/qh per the reader's two-half layout.
        let mut ql = [0u8; QK_K / 2]; // 128
        let mut qh = [0u8; QK_K / 4]; // 64
        for half in 0..2 {
            let (oo, qlo, qho) = (half * 128, half * 64, half * 32);
            for l in 0..32 {
                let ua = u[oo + l];
                let ub = u[oo + l + 32];
                let uc = u[oo + l + 64];
                let ud = u[oo + l + 96];
                ql[qlo + l] = (ua & 0x0F) | ((uc & 0x0F) << 4);
                ql[qlo + l + 32] = (ub & 0x0F) | ((ud & 0x0F) << 4);
                qh[qho + l] = ((ua >> 4) & 3)
                    | (((ub >> 4) & 3) << 2)
                    | (((uc >> 4) & 3) << 4)
                    | (((ud >> 4) & 3) << 6);
            }
        }
        // 5. Emit: ql, qh, 16 i8 scales, d (f16) — matching the reader's order.
        o.extend_from_slice(&ql);
        o.extend_from_slice(&qh);
        for &s in &scales {
            o.push(s as u8);
        }
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    o
}

#[cfg(test)]
mod tests {
    use super::put_scale_min_k4;
    use crate::gguf::QK_K;
    use crate::gguf_write::test_support::{dequant_via_reader, max_abs_err};

    #[test]
    fn put_scale_min_k4_inverts_get_scale_min_k4() {
        use crate::gguf::get_scale_min_k4;
        // Distinct 6-bit values per sub-block, exercising all bit positions.
        let sc = [1u8, 2, 3, 62, 33, 40, 63, 17];
        let m = [7u8, 8, 63, 4, 21, 60, 1, 34];
        let packed = put_scale_min_k4(&sc, &m);
        for j in 0..8 {
            let (d, mn) = get_scale_min_k4(j, &packed);
            assert_eq!(d, sc[j], "scale mismatch at sub-block {j}");
            assert_eq!(mn, m[j], "min mismatch at sub-block {j}");
        }
    }

    #[test]
    fn enc_q4_k_constant_block_is_near_exact() {
        use crate::gguf::GGML_Q4_K;
        for &c in &[0.5f32, -0.75, 0.0] {
            let x = vec![c; QK_K];
            let back = dequant_via_reader(&x, GGML_Q4_K, &[QK_K as u64]);
            let err = max_abs_err(&x, &back);
            assert!(err <= c.abs() * 0.02 + 1e-3, "c={c} err={err}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn enc_q4_k_within_bound() {
        use crate::gguf::GGML_Q4_K;
        // Deterministic pseudo-random-ish values in [-1, 1].
        let x: Vec<f32> = (0..QK_K)
            .map(|i| ((i as f32 * 12.9898).sin() * 43758.5453).fract() * 2.0 - 1.0)
            .collect();
        let back = dequant_via_reader(&x, GGML_Q4_K, &[QK_K as u64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax * 0.15, "q4_k error too high");
    }

    #[test]
    fn enc_q5_k_constant_block_is_near_exact() {
        use crate::gguf::GGML_Q5_K;
        for &c in &[0.5f32, -0.75] {
            let x = vec![c; QK_K];
            let back = dequant_via_reader(&x, GGML_Q5_K, &[QK_K as u64]);
            assert!(max_abs_err(&x, &back) <= c.abs() * 0.02 + 1e-3, "c={c}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn enc_q5_k_within_bound() {
        use crate::gguf::GGML_Q5_K;
        let x: Vec<f32> = (0..QK_K)
            .map(|i| ((i as f32 * 7.13).sin() * 1234.5).fract() * 2.0 - 1.0)
            .collect();
        let back = dequant_via_reader(&x, GGML_Q5_K, &[QK_K as u64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax * 0.10, "q5_k error too high");
    }

    #[test]
    fn enc_q6_k_constant_block_is_near_exact() {
        use crate::gguf::GGML_Q6_K;
        for &c in &[0.5f32, -0.75] {
            let x = vec![c; QK_K];
            let back = dequant_via_reader(&x, GGML_Q6_K, &[QK_K as u64]);
            assert!(max_abs_err(&x, &back) <= c.abs() * 0.02 + 1e-3, "c={c}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn enc_q6_k_within_bound() {
        use crate::gguf::GGML_Q6_K;
        let x: Vec<f32> = (0..QK_K)
            .map(|i| ((i as f32 * 3.7).cos() * 987.65).fract() * 2.0 - 1.0)
            .collect();
        let back = dequant_via_reader(&x, GGML_Q6_K, &[QK_K as u64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax * 0.05, "q6_k error too high");
    }
}
