//! GGUF / GGML format constants and on-disk size arithmetic.
use crate::error::Result;
use crate::gguf::cursor::fmt;

pub(crate) const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

pub(crate) const DEFAULT_ALIGNMENT: u64 = 32;

// GGML tensor element types (subset we can dequantize).
pub(crate) const GGML_F32: u32 = 0;

pub(crate) const GGML_F16: u32 = 1;

pub(crate) const GGML_Q4_0: u32 = 2;

pub(crate) const GGML_Q4_1: u32 = 3;

pub(crate) const GGML_Q5_0: u32 = 6;

pub(crate) const GGML_Q5_1: u32 = 7;

pub(crate) const GGML_Q8_0: u32 = 8;

pub(crate) const GGML_Q8_1: u32 = 9;

// k-quant super-block formats (G-K). These dominate modern GGUF downloads:
// `Q4_K_M` mixes Q4_K / Q5_K / Q6_K across tensors, so all three are needed to
// load one real checkpoint.
pub(crate) const GGML_Q2_K: u32 = 10;

pub(crate) const GGML_Q3_K: u32 = 11;

pub(crate) const GGML_Q4_K: u32 = 12;

pub(crate) const GGML_Q5_K: u32 = 13;

pub(crate) const GGML_Q6_K: u32 = 14;

pub(crate) const QK: usize = 32; // GGML block length for the legacy quant formats

pub(crate) const QK_K: usize = 256; // super-block length for the k-quant formats

// On-disk bytes per k-quant super-block of QK_K weights (must match ggml).
pub(crate) const Q2_K_BLOCK: usize = 16 + QK_K / 4 + 2 + 2; // scales, 2-bit qs, d, dmin = 84

pub(crate) const Q3_K_BLOCK: usize = QK_K / 8 + QK_K / 4 + 12 + 2; // hmask, qs, scales, d = 110

pub(crate) const Q4_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 2; // d, dmin, 6-bit scales, 4-bit qs = 144

pub(crate) const Q5_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2; // + qh high bits = 176

pub(crate) const Q6_K_BLOCK: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // ql, qh, scales, d = 210

// GGUF metadata value type tags.
pub(crate) const VT_U8: u32 = 0;

pub(crate) const VT_I8: u32 = 1;

pub(crate) const VT_U16: u32 = 2;

pub(crate) const VT_I16: u32 = 3;

pub(crate) const VT_U32: u32 = 4;

pub(crate) const VT_I32: u32 = 5;

pub(crate) const VT_F32: u32 = 6;

pub(crate) const VT_BOOL: u32 = 7;

pub(crate) const VT_STRING: u32 = 8;

pub(crate) const VT_ARRAY: u32 = 9;

pub(crate) const VT_U64: u32 = 10;

pub(crate) const VT_I64: u32 = 11;

pub(crate) const VT_F64: u32 = 12;

/// Bytes one tensor of `n` elements of `ggml_type` occupies on disk.
pub(super) fn type_nbytes(ggml_type: u32, n: usize) -> Result<usize> {
    let blocks = || -> Result<usize> {
        if !n.is_multiple_of(QK) {
            return Err(fmt(&format!(
                "quantized tensor element count {n} is not a multiple of {QK}"
            )));
        }
        Ok(n / QK)
    };
    let kblocks = || -> Result<usize> {
        if !n.is_multiple_of(QK_K) {
            return Err(fmt(&format!(
                "k-quant tensor element count {n} is not a multiple of {QK_K}"
            )));
        }
        Ok(n / QK_K)
    };
    // Checked multiplications: a hostile dims table (whose element count
    // saturates in `num_elements`) must produce a clean error, never an
    // overflow panic (debug) or a wrapped size (release).
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| fmt("tensor byte size overflows"))
    };
    match ggml_type {
        GGML_F32 => mul(n, 4),
        GGML_F16 => mul(n, 2),
        GGML_Q8_0 => mul(blocks()?, 2 + QK),     // f16 d + 32×i8
        GGML_Q8_1 => mul(blocks()?, 2 + 2 + QK), // f16 d + f16 s + 32×i8
        GGML_Q4_0 => mul(blocks()?, 2 + QK / 2), // f16 d + 16 bytes
        GGML_Q4_1 => mul(blocks()?, 2 + 2 + QK / 2), // f16 d + f16 m + 16 bytes
        GGML_Q5_0 => mul(blocks()?, 2 + 4 + QK / 2), // f16 d + 4-byte qh + 16 bytes
        GGML_Q5_1 => mul(blocks()?, 2 + 2 + 4 + QK / 2), // f16 d + f16 m + qh + 16 bytes
        GGML_Q2_K => mul(kblocks()?, Q2_K_BLOCK),
        GGML_Q3_K => mul(kblocks()?, Q3_K_BLOCK),
        GGML_Q4_K => mul(kblocks()?, Q4_K_BLOCK),
        GGML_Q5_K => mul(kblocks()?, Q5_K_BLOCK),
        GGML_Q6_K => mul(kblocks()?, Q6_K_BLOCK),
        other => Err(fmt(&format!(
            "GGUF tensor type {other} is unsupported for sizing"
        ))),
    }
}

pub(super) fn align_up(x: usize, a: usize) -> usize {
    if a == 0 {
        x
    } else {
        x.div_ceil(a) * a
    }
}

#[cfg(test)]
mod tests {
    use super::align_up;

    #[test]
    fn align_up_rounds_up_and_treats_zero_alignment_as_identity() {
        assert_eq!(align_up(5, 0), 5);
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }
}
