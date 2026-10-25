//! Per-format dequantizers (f16, legacy Q4/Q5/Q8, k-quants).
use crate::error::Result;
use crate::gguf::cursor::{fmt, Cur};
use crate::gguf::format::{Q2_K_BLOCK, Q3_K_BLOCK, QK, QK_K};

/// Convert an IEEE-754 half (`u16`) to `f32` (normals, subnormals, inf/NaN).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as i32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal f16: value = mant · 2⁻²⁴. Normalize so bit 10 (the hidden
            // 1) is set, tracking the exponent from the subnormal base −14.
            let mut m = mant;
            let mut e = -14i32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = (e + 127) as u32;
            (sign << 31) | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        // Inf / NaN.
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        let f32_exp = (exp - 15 + 127) as u32;
        (sign << 31) | (f32_exp << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub(super) fn dequant_q8_0(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let qs = c.take(QK)?;
        for (j, &q) in qs.iter().enumerate() {
            out[blk * QK + j] = d * (q as i8) as f32;
        }
    }
    Ok(out)
}

pub(super) fn dequant_q8_1(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let _s = f16_to_f32(c.u16()?); // block sum (for dot products); unused here
        let qs = c.take(QK)?;
        for (j, &q) in qs.iter().enumerate() {
            out[blk * QK + j] = d * (q as i8) as f32;
        }
    }
    Ok(out)
}

pub(super) fn dequant_q4_0(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let qs = c.take(QK / 2)?;
        for (j, &byte) in qs.iter().enumerate() {
            // Low nibble → element j, high nibble → element j+16; centered at 8.
            let x0 = (byte & 0x0F) as i32 - 8;
            let x1 = (byte >> 4) as i32 - 8;
            out[blk * QK + j] = d * x0 as f32;
            out[blk * QK + j + QK / 2] = d * x1 as f32;
        }
    }
    Ok(out)
}

pub(super) fn dequant_q4_1(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let m = f16_to_f32(c.u16()?);
        let qs = c.take(QK / 2)?;
        for (j, &byte) in qs.iter().enumerate() {
            let x0 = (byte & 0x0F) as f32;
            let x1 = (byte >> 4) as f32;
            out[blk * QK + j] = d * x0 + m;
            out[blk * QK + j + QK / 2] = d * x1 + m;
        }
    }
    Ok(out)
}

/// Q5_0 (`f16 d`, 4-byte `qh`, 16 bytes `qs`; value `(q-16)*d`) and Q5_1 (adds
/// `f16 m`; value `q*d+m`). `q` is the 4-bit nibble plus bit `j` (first half) /
/// bit `j+16` (second half) of `qh` as the fifth bit, as in ggml.
pub(super) fn dequant_q5(raw: &[u8], n: usize, with_min: bool) -> Result<Vec<f32>> {
    let nb = n / QK;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let m = if with_min { f16_to_f32(c.u16()?) } else { 0.0 };
        let qh = c.take(4)?;
        let qh = u32::from_le_bytes([qh[0], qh[1], qh[2], qh[3]]);
        let qs = c.take(QK / 2)?;
        for (j, &byte) in qs.iter().enumerate() {
            let x0 = ((byte & 0x0F) as u32 | (((qh >> j) & 1) << 4)) as i32;
            let x1 = ((byte >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32;
            let (v0, v1) = if with_min {
                (d * x0 as f32 + m, d * x1 as f32 + m)
            } else {
                (d * (x0 - 16) as f32, d * (x1 - 16) as f32)
            };
            out[blk * QK + j] = v0;
            out[blk * QK + j + QK / 2] = v1;
        }
    }
    Ok(out)
}

/// Unpack the 6-bit scale `d` and min `m` for sub-block `j` (0..8) from the
/// 12-byte packed `scales` of a Q4_K/Q5_K super-block (ggml `get_scale_min_k4`).
#[inline]
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

pub(super) fn dequant_q4_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK_K;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let dmin = f16_to_f32(c.u16()?);
        let scales = c.take(12)?;
        let qs = c.take(QK_K / 2)?; // 128 bytes
        let base = blk * QK_K;
        let (mut y, mut q_off, mut is) = (0usize, 0usize, 0usize);
        while y < QK_K {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                out[base + y + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - mn1;
                out[base + y + 32 + l] = d2 * (qs[q_off + l] >> 4) as f32 - mn2;
            }
            y += 64;
            q_off += 32;
            is += 2;
        }
    }
    Ok(out)
}

pub(super) fn dequant_q5_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK_K;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let d = f16_to_f32(c.u16()?);
        let dmin = f16_to_f32(c.u16()?);
        let scales = c.take(12)?;
        let qh = c.take(QK_K / 8)?; // 32 bytes of high bits
        let qs = c.take(QK_K / 2)?; // 128 bytes of low nibbles
        let base = blk * QK_K;
        let (mut y, mut q_off, mut is) = (0usize, 0usize, 0usize);
        let (mut u1, mut u2) = (1u8, 2u8);
        while y < QK_K {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                out[base + y + l] = d1 * ((qs[q_off + l] & 0x0F) as i32 + hi1) as f32 - mn1;
                out[base + y + 32 + l] = d2 * ((qs[q_off + l] >> 4) as i32 + hi2) as f32 - mn2;
            }
            y += 64;
            q_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

pub(super) fn dequant_q6_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let nb = n / QK_K;
    let mut out = vec![0.0f32; n];
    let mut c = Cur::new(raw);
    for blk in 0..nb {
        let ql = c.take(QK_K / 2)?; // 128 bytes, lower 4 bits
        let qh = c.take(QK_K / 4)?; // 64 bytes, upper 2 bits
        let scales = c.take(QK_K / 16)?; // 16 signed scales
        let d = f16_to_f32(c.u16()?);
        let base = blk * QK_K;
        // Two 128-wide halves; ql/qh/scales advance by 64/32/8 per half.
        for half in 0..2 {
            let (qlo, qho, sco, oo) = (half * 64, half * 32, half * 8, base + half * 128);
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[qlo + l] & 0x0F) as i32 | ((qh[qho + l] & 3) as i32) << 4) - 32;
                let q2 = ((ql[qlo + l + 32] & 0x0F) as i32
                    | (((qh[qho + l] >> 2) & 3) as i32) << 4)
                    - 32;
                let q3 = ((ql[qlo + l] >> 4) as i32 | (((qh[qho + l] >> 4) & 3) as i32) << 4) - 32;
                let q4 =
                    ((ql[qlo + l + 32] >> 4) as i32 | (((qh[qho + l] >> 6) & 3) as i32) << 4) - 32;
                let sc = |k: usize| (scales[sco + k] as i8) as f32;
                out[oo + l] = d * sc(is) * q1 as f32;
                out[oo + l + 32] = d * sc(is + 2) * q2 as f32;
                out[oo + l + 64] = d * sc(is + 4) * q3 as f32;
                out[oo + l + 96] = d * sc(is + 6) * q4 as f32;
            }
        }
    }
    Ok(out)
}

/// Q2_K super-block: 16 packed 4-bit (scale, min) pairs, 64 bytes of 2-bit
/// quants, f16 d + f16 dmin. y = d·sc·q − dmin·m, in ggml's 128-element-half
/// scan order (shift 0,2,4,6 over two 16-element runs per shift).
pub(super) fn dequant_q2_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in raw.chunks_exact(Q2_K_BLOCK) {
        let scales = &block[..16];
        let qs = &block[16..16 + 64];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let mut is = 0usize;
        for half in 0..2 {
            let q = &qs[32 * half..32 * half + 32];
            for shift in [0u8, 2, 4, 6] {
                for run in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for &ql in &q[16 * run..16 * run + 16] {
                        out.push(dl * ((ql >> shift) & 3) as f32 - ml);
                    }
                }
            }
        }
    }
    out.truncate(n);
    if out.len() != n {
        return Err(fmt("Q2_K data shorter than expected"));
    }
    Ok(out)
}

/// Q3_K super-block: 32-byte high-bit mask, 64 bytes of 2-bit low quants,
/// 12 bytes of packed 6-bit scales, f16 d. q = low2 | (¬hbit → −4 offset),
/// y = d·(scale−32)·q, same scan order as Q2_K; the high-bit selector `m`
/// doubles after each shift group.
pub(super) fn dequant_q3_k(raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in raw.chunks_exact(Q3_K_BLOCK) {
        let hmask = &block[..32];
        let qs = &block[32..32 + 64];
        let sc = &block[96..96 + 12];
        let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
        // Unpack 16 six-bit scales from ggml's split encoding.
        let mut scales = [0i8; 16];
        for j in 0..8 {
            scales[j] = (sc[j] & 0xF) as i8;
            scales[j + 8] = (sc[j] >> 4) as i8;
        }
        for j in 0..16 {
            let hi = (sc[8 + j / 4] >> (2 * (j % 4))) & 3;
            scales[j] |= (hi << 4) as i8;
        }
        let mut is = 0usize;
        let mut m: u8 = 1;
        for half in 0..2 {
            let q = &qs[32 * half..32 * half + 32];
            for shift in [0u8, 2, 4, 6] {
                for run in 0..2 {
                    let dl = d * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    for l in 16 * run..16 * run + 16 {
                        let low = ((q[l] >> shift) & 3) as i32;
                        let qv = low - if hmask[l] & m != 0 { 0 } else { 4 };
                        out.push(dl * qv as f32);
                    }
                }
                m <<= 1;
            }
        }
    }
    out.truncate(n);
    if out.len() != n {
        return Err(fmt("Q3_K data shorter than expected"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        dequant_q2_k, dequant_q3_k, dequant_q4_1, dequant_q4_k, dequant_q5, dequant_q5_k,
        dequant_q6_k, dequant_q8_1, f16_to_f32, get_scale_min_k4,
    };
    use crate::gguf::format::{
        type_nbytes, GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_0, GGML_Q5_1, GGML_Q5_K, GGML_Q6_K,
        Q2_K_BLOCK, Q3_K_BLOCK, Q4_K_BLOCK, Q5_K_BLOCK, Q6_K_BLOCK, QK, QK_K,
    };
    use crate::gguf::test_support::f32_to_f16;

    #[test]
    fn f16_roundtrip_of_simple_values() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, -2.5, 16.0] {
            let h = f32_to_f16(v);
            assert!((f16_to_f32(h) - v).abs() < 1e-3, "f16 {v}");
        }
    }

    #[test]
    fn dequant_q8_1_block() {
        let d = 0.5f32;
        let mut raw = Vec::new();
        raw.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        raw.extend_from_slice(&f32_to_f16(0.0).to_le_bytes()); // s (unused)
        for i in 0..QK as i32 {
            raw.push(((i - 16) as i8) as u8);
        }
        let out = dequant_q8_1(&raw, QK).unwrap();
        for (i, &o) in out.iter().enumerate() {
            assert!((o - d * (i as i32 - 16) as f32).abs() < 1e-3);
        }
    }

    #[test]
    fn dequant_q4_1_block() {
        let (d, m) = (0.25f32, 1.0f32);
        let mut raw = Vec::new();
        raw.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        raw.extend_from_slice(&f32_to_f16(m).to_le_bytes());
        // nibble j → lo, nibble j+16 → hi.
        let los: Vec<u8> = (0..16).map(|j| (j % 15) as u8).collect();
        let his: Vec<u8> = (0..16).map(|j| ((j + 3) % 15) as u8).collect();
        for j in 0..16 {
            raw.push((his[j] << 4) | los[j]);
        }
        let out = dequant_q4_1(&raw, QK).unwrap();
        for j in 0..16 {
            assert!((out[j] - (d * los[j] as f32 + m)).abs() < 1e-3);
            assert!((out[j + 16] - (d * his[j] as f32 + m)).abs() < 1e-3);
        }
    }

    #[test]
    fn dequant_q5_high_bit_and_sizing() {
        // qh bit j sets the 5th bit of element j; bit j+16 of element j+16.
        let qh: u32 = (1 << 0) | (1 << 17);
        let mut raw = Vec::new();
        raw.extend_from_slice(&f32_to_f16(2.0).to_le_bytes());
        raw.extend_from_slice(&qh.to_le_bytes());
        raw.extend_from_slice(&[0x21u8; 16]); // lo 1, hi 2
        let q5_0 = dequant_q5(&raw, QK, false).unwrap();
        assert_eq!(q5_0[0], 2.0 * (17 - 16) as f32); // 1 + 16
        assert_eq!(q5_0[1], 2.0 * (1 - 16) as f32);
        assert_eq!(q5_0[17], 2.0 * (18 - 16) as f32); // 2 + 16
        assert_eq!(q5_0[16], 2.0 * (2 - 16) as f32);
        raw.splice(2..2, f32_to_f16(0.5).to_le_bytes()); // d, m, qh, qs
        let q5_1 = dequant_q5(&raw, QK, true).unwrap();
        assert_eq!(q5_1[0], 2.0 * 17.0 + 0.5);
        assert_eq!(q5_1[17], 2.0 * 18.0 + 0.5);
        assert_eq!(type_nbytes(GGML_Q5_0, 64).unwrap(), 2 * 22);
        assert_eq!(type_nbytes(GGML_Q5_1, 64).unwrap(), 2 * 24);
        assert!(type_nbytes(GGML_Q5_0, 30).is_err());
        // Truncated data errors instead of panicking.
        assert!(dequant_q5(&raw[..10], QK, true).is_err());
    }

    #[test]
    fn k_quant_block_sizes() {
        assert_eq!(type_nbytes(GGML_Q4_K, QK_K).unwrap(), Q4_K_BLOCK); // 144
        assert_eq!(type_nbytes(GGML_Q5_K, QK_K).unwrap(), Q5_K_BLOCK); // 176
        assert_eq!(type_nbytes(GGML_Q6_K, QK_K).unwrap(), Q6_K_BLOCK); // 210
        assert!(type_nbytes(GGML_Q4_K, 200).is_err()); // not a multiple of 256
    }

    #[test]
    fn q2k_q3k_block_sizes() {
        assert_eq!(type_nbytes(GGML_Q2_K, QK_K).unwrap(), Q2_K_BLOCK); // 84
        assert_eq!(type_nbytes(GGML_Q3_K, QK_K).unwrap(), Q3_K_BLOCK); // 110
        assert!(type_nbytes(GGML_Q2_K, 200).is_err());
    }

    #[test]
    fn dequant_q2_k_scales_and_mins() {
        // qs = 0x55 → every 2-bit q = 1. d = 1.0, dmin = 0.5 (f16-exact).
        // scales[i] = i | ((15 - i) << 4) → y over group i = 1·i − 0.5·(15 − i).
        let mut b = vec![0u8; Q2_K_BLOCK];
        for (i, s) in b[..16].iter_mut().enumerate() {
            *s = (i as u8) | (((15 - i) as u8) << 4);
        }
        for q in &mut b[16..80] {
            *q = 0x55;
        }
        b[80..82].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        b[82..84].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        let y = dequant_q2_k(&b, 256).unwrap();
        for (g, chunk) in y.chunks_exact(16).enumerate() {
            let expect = 1.0 * g as f32 - 0.5 * (15 - g) as f32;
            for v in chunk {
                assert!((v - expect).abs() < 1e-3, "group {g}: {v} vs {expect}");
            }
        }
    }

    #[test]
    fn dequant_q3_k_high_bit_offsets() {
        // qs = 0x55 (low bits = 1). All 16 scales = 34 → dl = d·2. d = 1.0.
        // hmask all 0xFF → q = 1 → y = 2. hmask all 0 → q = 1−4 = −3 → y = −6.
        let mut b = vec![0u8; Q3_K_BLOCK];
        for q in &mut b[32..96] {
            *q = 0x55;
        }
        // scales all = 34 = 0b100010: low nibble 2 in bytes 0–7 (both nibbles),
        // top bits 0b10 for every j in bytes 8–11 (0b10101010 = 0xAA).
        for j in 0..8 {
            b[96 + j] = 0x22;
        }
        for j in 8..12 {
            b[96 + j] = 0xAA;
        }
        b[108..110].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        for h in &mut b[..32] {
            *h = 0xFF;
        }
        let y = dequant_q3_k(&b, 256).unwrap();
        for v in &y {
            assert!((v - 2.0).abs() < 1e-3, "set-bit case: {v}");
        }
        for h in &mut b[..32] {
            *h = 0;
        }
        let y = dequant_q3_k(&b, 256).unwrap();
        for v in &y {
            assert!((v + 6.0).abs() < 1e-3, "clear-bit case: {v}");
        }
    }

    #[test]
    fn dequant_q6_k_constant_block() {
        // ql nibbles = 8, qh 2-bit fields = 2 ⇒ q = (8|2<<4) − 32 = 8.
        // scales = 2, d = 0.5 ⇒ every value = 0.5·2·8 = 8.0.
        let mut raw = vec![0x88u8; QK_K / 2];
        raw.extend_from_slice(&[0xAAu8; QK_K / 4]);
        raw.extend(std::iter::repeat_n(2i8 as u8, QK_K / 16));
        raw.extend_from_slice(&f32_to_f16(0.5).to_le_bytes());
        let out = dequant_q6_k(&raw, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        for &v in &out {
            assert!((v - 8.0).abs() < 1e-3, "q6_k got {v}");
        }
    }

    /// 12-byte packed scales that decode (via get_scale_min_k4) to d = 2, m = 0
    /// for all 8 sub-blocks.
    const K_SCALES_D2_M0: [u8; 12] = [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2];

    #[test]
    fn get_scale_min_k4_unpacks_all_subblocks() {
        for j in 0..8 {
            let (d, m) = get_scale_min_k4(j, &K_SCALES_D2_M0);
            assert_eq!((d, m), (2, 0), "sub-block {j}");
        }
    }

    #[test]
    fn dequant_q4_k_constant_block() {
        // d·sc = 0.5·2 = 1.0, min = 0, nibble = 3 ⇒ every value = 3.0.
        let mut raw = Vec::new();
        raw.extend_from_slice(&f32_to_f16(0.5).to_le_bytes()); // d
        raw.extend_from_slice(&f32_to_f16(0.0).to_le_bytes()); // dmin
        raw.extend_from_slice(&K_SCALES_D2_M0);
        raw.extend_from_slice(&[0x33u8; QK_K / 2]); // qs: both nibbles = 3
        let out = dequant_q4_k(&raw, QK_K).unwrap();
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-3, "q4_k got {v}");
        }
    }

    #[test]
    fn dequant_q5_k_low_and_high_bits() {
        let build = |qh: u8| {
            let mut raw = Vec::new();
            raw.extend_from_slice(&f32_to_f16(0.5).to_le_bytes());
            raw.extend_from_slice(&f32_to_f16(0.0).to_le_bytes());
            raw.extend_from_slice(&K_SCALES_D2_M0);
            raw.extend_from_slice(&[qh; QK_K / 8]); // 32 bytes qh
            raw.extend_from_slice(&[0x33u8; QK_K / 2]); // qs nibble = 3
            raw
        };
        // No high bit: value = 1.0·3 = 3.0.
        for &v in &dequant_q5_k(&build(0x00), QK_K).unwrap() {
            assert!((v - 3.0).abs() < 1e-3, "q5_k low {v}");
        }
        // Every high bit set: value = 1.0·(3 + 16) = 19.0.
        for &v in &dequant_q5_k(&build(0xFF), QK_K).unwrap() {
            assert!((v - 19.0).abs() < 1e-3, "q5_k high {v}");
        }
    }

    #[test]
    fn f16_special_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), 0.0); // negative zero
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        assert!(f16_to_f32(0x7E00).is_nan());
        // Smallest positive subnormal = 2^-24.
        let sub = f16_to_f32(0x0001);
        assert!(sub > 0.0 && (sub - 2f32.powi(-24)).abs() < 1e-12);
    }

    #[test]
    fn k_quant_dequantizers_reject_data_shorter_than_the_element_count() {
        // One whole block holds 256 weights; asking for 257 must fail loudly.
        assert!(dequant_q2_k(&[0u8; 84], 257).is_err());
        assert!(dequant_q3_k(&[0u8; 110], 257).is_err());
        assert_eq!(dequant_q2_k(&[0u8; 84], 256).unwrap().len(), 256);
        assert_eq!(dequant_q3_k(&[0u8; 110], 256).unwrap().len(), 256);
    }
}
