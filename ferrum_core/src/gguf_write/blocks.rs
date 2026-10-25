//! f16 conversion and the f32 / f16 / legacy Q4 / Q8 block encoders.
use crate::error::{InferError, Result};
use crate::gguf::{
    GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
    GGML_Q8_1, QK, QK_K,
};
use crate::gguf_write::kquant::{enc_q4_k, enc_q5_k, enc_q6_k};

/// Encode an `f32` as an IEEE-754 half (`u16`), round-to-nearest-ties-to-even.
/// Inf/NaN, subnormals, and overflow-to-Inf are all handled. Exact inverse of
/// [`crate::gguf::f16_to_f32`] on values representable as f16.
pub fn f32_to_f16(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp32 = ((x >> 23) & 0xFF) as i32;
    let mant32 = x & 0x007F_FFFF;

    if exp32 == 0xFF {
        // Inf (mant 0) or NaN (mant != 0 → a canonical quiet NaN).
        return if mant32 != 0 {
            sign | 0x7E00
        } else {
            sign | 0x7C00
        };
    }

    let mut exp16 = exp32 - 127 + 15;
    if exp16 >= 0x1F {
        return sign | 0x7C00; // Overflow → Inf.
    }
    if exp16 <= 0 {
        // Subnormal f16, or underflow to signed zero.
        if exp16 < -10 {
            return sign;
        }
        let mant = mant32 | 0x0080_0000; // restore implicit leading 1
        let shift = (14 - exp16) as u32; // 14..=24
        let mut half = mant >> shift;
        let rem = mant & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half as u16;
    }

    // Normalized f16.
    let mut mant16 = (mant32 >> 13) as u16;
    let rem = mant32 & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (mant16 & 1) == 1) {
        mant16 += 1;
        if mant16 == 0x0400 {
            mant16 = 0;
            exp16 += 1;
            if exp16 >= 0x1F {
                return sign | 0x7C00;
            }
        }
    }
    sign | ((exp16 as u16) << 10) | mant16
}

/// Encode a whole tensor's f32 values into GGUF on-disk bytes for `ggml_type`.
/// Quantized types require the length to be a multiple of their block size.
pub(crate) fn encode_tensor(data: &[f32], ggml_type: u32) -> Result<Vec<u8>> {
    let n = data.len();
    let need_mult = |m: usize| -> Result<()> {
        if !n.is_multiple_of(m) {
            return Err(InferError::Format(format!(
                "tensor length {n} is not a multiple of block size {m} for ggml type {ggml_type}"
            )));
        }
        Ok(())
    };
    Ok(match ggml_type {
        GGML_F32 => enc_f32(data),
        GGML_F16 => enc_f16(data),
        GGML_Q8_0 => {
            need_mult(QK)?;
            enc_q8_0(data)
        }
        GGML_Q8_1 => {
            need_mult(QK)?;
            enc_q8_1(data)
        }
        GGML_Q4_0 => {
            need_mult(QK)?;
            enc_q4_0(data)
        }
        GGML_Q4_1 => {
            need_mult(QK)?;
            enc_q4_1(data)
        }
        GGML_Q4_K => {
            need_mult(QK_K)?;
            enc_q4_k(data)
        }
        GGML_Q5_K => {
            need_mult(QK_K)?;
            enc_q5_k(data)
        }
        GGML_Q6_K => {
            need_mult(QK_K)?;
            enc_q6_k(data)
        }
        other => {
            return Err(InferError::Format(format!(
                "encode_tensor: unsupported ggml type {other}"
            )))
        }
    })
}

fn enc_f32(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() * 4);
    for &x in data {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

fn enc_f16(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() * 2);
    for &x in data {
        o.extend_from_slice(&f32_to_f16(x).to_le_bytes());
    }
    o
}

/// Q8_0: per 32-element block, via the shared [`crate::quant::q8_0_block`]
/// (same encoder QAT snaps to). Decode is `d * q`.
fn enc_q8_0(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK * (2 + QK));
    for blk in data.chunks_exact(QK) {
        let (d16, q) = crate::quant::q8_0_block(blk);
        o.extend_from_slice(&d16.to_le_bytes());
        o.extend(q.iter().map(|&v| v as u8));
    }
    o
}

/// Q8_1: like Q8_0 plus a per-block sum `s = d * Σq` (f16). The reader ignores
/// `s` on decode, but ggml stores it, so we compute it faithfully.
fn enc_q8_1(data: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(data.len() / QK * (2 + 2 + QK));
    for blk in data.chunks_exact(QK) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; QK];
        let mut sum = 0i32;
        for (j, &x) in blk.iter().enumerate() {
            let q = (x * id).round().clamp(-127.0, 127.0) as i8;
            qs[j] = q;
            sum += q as i32;
        }
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        o.extend_from_slice(&f32_to_f16(d * sum as f32).to_le_bytes());
        for &q in &qs {
            o.push(q as u8);
        }
    }
    o
}

/// Q4_0: per 32-element block, `d = max/-8` where `max` is the value of largest
/// magnitude; `q = round(x/d)+8` clamped to 0..15. Elements `j` and `j+16` share
/// byte `j` (low/high nibble). Decode is `d * (q - 8)`.
fn enc_q4_0(data: &[f32]) -> Vec<u8> {
    let half = QK / 2;
    let mut o = Vec::with_capacity(data.len() / QK * (2 + half));
    for blk in data.chunks_exact(QK) {
        // Pick the extreme value (largest |x|) to anchor the scale, as ggml does.
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &x in blk {
            if x.abs() > amax {
                amax = x.abs();
                max = x;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..half {
            let q0 = ((blk[j] * id + 8.5) as i32).clamp(0, 15) as u8;
            let q1 = ((blk[j + half] * id + 8.5) as i32).clamp(0, 15) as u8;
            o.push(q0 | (q1 << 4));
        }
    }
    o
}

/// Q4_1: per 32-element block, `d = (max-min)/15`, `q = round((x-min)/d)` in
/// 0..15, storing `d` and `min` (both f16). Decode is `d * q + min`.
fn enc_q4_1(data: &[f32]) -> Vec<u8> {
    let half = QK / 2;
    let mut o = Vec::with_capacity(data.len() / QK * (2 + 2 + half));
    for blk in data.chunks_exact(QK) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &x in blk {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        let d = (hi - lo) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        o.extend_from_slice(&f32_to_f16(lo).to_le_bytes());
        for j in 0..half {
            let q0 = (((blk[j] - lo) * id + 0.5) as i32).clamp(0, 15) as u8;
            let q1 = (((blk[j + half] - lo) * id + 0.5) as i32).clamp(0, 15) as u8;
            o.push(q0 | (q1 << 4));
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::{encode_tensor, f32_to_f16};
    use crate::gguf::f16_to_f32;
    use crate::gguf_write::test_support::{dequant_via_reader, max_abs_err};

    #[test]
    fn f32_to_f16_is_exact_inverse_of_f16_to_f32() {
        // For every representable f16 bit pattern, encoding its f32 value must
        // return the same bits. NaNs are canonicalized; ±0 compared by bits.
        for h in 0u32..=0xFFFF {
            let h = h as u16;
            let exp = (h >> 10) & 0x1F;
            let mant = h & 0x3FF;
            if exp == 0x1F && mant != 0 {
                continue; // skip NaN encodings (many bit patterns, one canonical out)
            }
            let f = f16_to_f32(h);
            assert_eq!(f32_to_f16(f), h, "roundtrip failed for f16 bits {h:#06x}");
        }
    }

    #[test]
    fn f32_to_f16_spot_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF); // max normal f16
        assert_eq!(f32_to_f16(1e5), 0x7C00); // overflow → +Inf
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
    }

    #[test]
    fn enc_f32_is_exact() {
        use crate::gguf::GGML_F32;
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let back = dequant_via_reader(&x, GGML_F32, &[64]);
        assert_eq!(back, x);
    }

    #[test]
    fn enc_f16_within_half_ulp() {
        use crate::gguf::GGML_F16;
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let back = dequant_via_reader(&x, GGML_F16, &[64]);
        // Each value within f16 relative resolution near this magnitude.
        assert!(max_abs_err(&x, &back) < 0.001);
    }

    #[test]
    fn enc_q8_0_within_block_bound() {
        use crate::gguf::GGML_Q8_0;
        let x: Vec<f32> = (0..64).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let back = dequant_via_reader(&x, GGML_Q8_0, &[64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax / 127.0 + 1e-4);
    }

    #[test]
    fn enc_q8_0_layout_and_reader_decode_equal_qat_snap_exactly() {
        use crate::gguf::GGML_Q8_0;
        // positive, negative, zero and outlier blocks
        let mut x: Vec<f32> = (0..32 * 4)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        x[32..64].fill(0.0);
        x[64..96].iter_mut().for_each(|v| *v = -v.abs());
        x[100] = 900.0;
        let bytes = encode_tensor(&x, GGML_Q8_0).unwrap();
        assert_eq!(
            bytes.len(),
            4 * 34,
            "34 bytes per block: f16 scale + 32 x i8"
        );
        for (b, blk) in bytes.chunks_exact(34).enumerate() {
            let (d16, q) = crate::quant::q8_0_block(&x[b * 32..(b + 1) * 32]);
            assert_eq!(
                u16::from_le_bytes([blk[0], blk[1]]),
                d16,
                "scale of block {b}"
            );
            let got: Vec<i8> = blk[2..].iter().map(|&v| v as i8).collect();
            assert_eq!(got, q.to_vec(), "quants of block {b}");
        }
        assert_eq!(&bytes[34..36], &[0, 0], "all-zero block has a zero scale");
        // what llama.cpp reads back is bit-identical to what QAT trains against
        let back = dequant_via_reader(&x, GGML_Q8_0, &[128]);
        let mut snapped = x.clone();
        crate::quant::fake_quantize_q8_0_rows(&mut snapped, 128);
        assert_eq!(back, snapped);
    }

    #[test]
    fn enc_q8_1_within_block_bound() {
        use crate::gguf::GGML_Q8_1;
        let x: Vec<f32> = (0..64).map(|i| ((i * 5) % 11) as f32 - 5.0).collect();
        let back = dequant_via_reader(&x, GGML_Q8_1, &[64]);
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax / 127.0 + 1e-4);
    }

    #[test]
    fn encode_tensor_rejects_ragged_quantized() {
        use crate::gguf::GGML_Q8_0;
        // 20 is not a multiple of QK=32.
        let x = vec![0.0f32; 20];
        assert!(encode_tensor(&x, GGML_Q8_0).is_err());
    }

    #[test]
    fn enc_q4_0_within_block_bound() {
        use crate::gguf::GGML_Q4_0;
        let x: Vec<f32> = (0..64).map(|i| ((i * 3) % 17) as f32 - 8.0).collect();
        let back = dequant_via_reader(&x, GGML_Q4_0, &[64]);
        // Symmetric 4-bit: 15 levels across [-amax, amax] → step ≈ amax/7.5.
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs_err(&x, &back) <= amax / 7.0 + 1e-4);
    }

    #[test]
    fn enc_q4_1_within_block_bound() {
        use crate::gguf::GGML_Q4_1;
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25 - 3.0).collect();
        let back = dequant_via_reader(&x, GGML_Q4_1, &[64]);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &x {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(max_abs_err(&x, &back) <= (hi - lo) / 15.0 + 1e-4);
    }

    #[test]
    fn f32_to_f16_handles_nan_ties_and_rounding_overflow() {
        // NaN keeps its sign and becomes a canonical quiet NaN; infinities map to f16 infinities.
        assert_eq!(f32_to_f16(f32::NAN), 0x7E00);
        assert_eq!(f32_to_f16(-f32::NAN), 0xFE00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
        // Largest finite half is 65504; 65520 is the tie that rounds (to even) up into infinity.
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16(65520.0), 0x7C00);
        assert_eq!(f32_to_f16(1.0e9), 0x7C00);
        // 1.5 * 2^-24 is a tie between the subnormals 1 and 2: ties-to-even picks 2.
        assert_eq!(f32_to_f16(1.5 * 2f32.powi(-24)), 2);
        // Below half the smallest subnormal flushes to signed zero.
        assert_eq!(f32_to_f16(2f32.powi(-26)), 0);
        assert_eq!(f32_to_f16(-(2f32.powi(-26))), 0x8000);
        // A normal value just above a tie rounds up.
        assert_eq!(f32_to_f16(f32::from_bits(0x3F80_1001)), 0x3C01);
    }

    #[test]
    fn encode_tensor_rejects_unsupported_ggml_types() {
        let e = encode_tensor(&[0.0; 32], 99).unwrap_err();
        assert!(e.to_string().contains("unsupported ggml type 99"), "{e}");
    }
}
