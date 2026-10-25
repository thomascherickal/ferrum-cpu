//! Tensor access: raw bytes, f32 dequantization and quantized-weight import.
use crate::error::Result;
use crate::gguf::cursor::{fmt, to_usize};
use crate::gguf::dequant::{
    dequant_q2_k, dequant_q3_k, dequant_q4_0, dequant_q4_1, dequant_q4_k, dequant_q5, dequant_q5_k,
    dequant_q6_k, dequant_q8_0, dequant_q8_1, f16_to_f32,
};
use crate::gguf::format::{
    type_nbytes, GGML_F16, GGML_F32, GGML_Q2_K, GGML_Q3_K, GGML_Q4_0, GGML_Q4_1, GGML_Q4_K,
    GGML_Q5_0, GGML_Q5_1, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, GGML_Q8_1,
};
use crate::gguf::{Gguf, Source, TensorInfo};
use crate::quant::{QKind, QWeight};
use std::borrow::Cow;
use std::io::SeekFrom;
use std::io::{Read, Seek};

impl Gguf {
    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The raw on-disk bytes of one tensor — borrowed from the in-memory buffer,
    /// or read from disk into an owned buffer when streamed.
    pub(crate) fn tensor_bytes(&self, t: &TensorInfo) -> Result<Cow<'_, [u8]>> {
        let n = t.num_elements();
        let len = type_nbytes(t.ggml_type, n)?;
        let start = self
            .data_offset
            .checked_add(to_usize(t.offset))
            .ok_or_else(|| fmt("tensor offset overflow"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| fmt("tensor length overflow"))?;
        match &self.source {
            Source::Memory(bytes) => {
                if end > bytes.len() {
                    return Err(fmt(&format!("tensor '{}' data runs past EOF", t.name)));
                }
                Ok(Cow::Borrowed(&bytes[start..end]))
            }
            Source::Streamed { file, len: flen } => {
                if end > *flen {
                    return Err(fmt(&format!("tensor '{}' data runs past EOF", t.name)));
                }
                let mut f = file.lock().map_err(|_| fmt("GGUF file lock poisoned"))?;
                f.seek(SeekFrom::Start(start as u64))?;
                let mut buf = vec![0u8; len];
                f.read_exact(&mut buf)?;
                Ok(Cow::Owned(buf))
            }
        }
    }

    /// Dequantize a tensor to row-major f32 (in GGML storage order). Handles the
    /// legacy block formats and the common k-quant super-blocks (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K).
    pub fn dequantize(&self, t: &TensorInfo) -> Result<Vec<f32>> {
        let raw = self.tensor_bytes(t)?;
        let raw = raw.as_ref();
        let n = t.num_elements();
        match t.ggml_type {
            GGML_F32 => Ok(raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            GGML_F16 => Ok(raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()),
            GGML_Q8_0 => dequant_q8_0(raw, n),
            GGML_Q8_1 => dequant_q8_1(raw, n),
            GGML_Q4_0 => dequant_q4_0(raw, n),
            GGML_Q4_1 => dequant_q4_1(raw, n),
            GGML_Q5_0 => dequant_q5(raw, n, false),
            GGML_Q5_1 => dequant_q5(raw, n, true),
            GGML_Q2_K => dequant_q2_k(raw, n),
            GGML_Q3_K => dequant_q3_k(raw, n),
            GGML_Q4_K => dequant_q4_k(raw, n),
            GGML_Q5_K => dequant_q5_k(raw, n),
            GGML_Q6_K => dequant_q6_k(raw, n),
            other => Err(fmt(&format!(
                "GGUF tensor type {other} not yet supported for dequant \
                 (the IQ* formats need their own decoders)"
            ))),
        }
    }

    /// Dequantize a tensor by name, then pack it to a Ferrum [`QWeight`] of the
    /// requested in-memory precision (int4/int8). `rows` is the weight's input
    /// dimension (the per-row scale axis); `rows × cols` must equal the tensor's
    /// element count.
    pub fn quantized_weight(&self, name: &str, rows: usize, kind: QKind) -> Result<QWeight> {
        let t = self
            .tensor(name)
            .ok_or_else(|| fmt(&format!("GGUF tensor '{name}' not found")))?;
        let f = self.dequantize(t)?;
        if rows == 0 || f.len() % rows != 0 {
            return Err(fmt(&format!(
                "tensor '{name}' length {} not divisible by rows {rows}",
                f.len()
            )));
        }
        let cols = f.len() / rows;
        Ok(QWeight::from_f32(&f, rows, cols, kind))
    }
}

#[cfg(test)]
mod tests {
    use crate::gguf::format::{type_nbytes, GGML_F32, GGML_Q8_0};
    use crate::gguf::test_support::synth_gguf;
    use crate::gguf::Gguf;
    use crate::quant::QKind;

    #[test]
    fn dequantizes_f32_q8_and_q4() {
        let (bytes, f32_vals, q8_vals, q4_vals) = synth_gguf();
        let g = Gguf::parse(bytes).unwrap();

        let f = g.dequantize(g.tensor("t_f32").unwrap()).unwrap();
        assert_eq!(f, f32_vals);

        let q8 = g.dequantize(g.tensor("t_q8").unwrap()).unwrap();
        for (a, b) in q8.iter().zip(&q8_vals) {
            assert!((a - b).abs() < 1e-4, "q8 {a} vs {b}");
        }

        let q4 = g.dequantize(g.tensor("t_q4").unwrap()).unwrap();
        for (a, b) in q4.iter().zip(&q4_vals) {
            assert!((a - b).abs() < 1e-4, "q4 {a} vs {b}");
        }
    }

    #[test]
    fn imports_tensor_as_int4_qweight() {
        let (bytes, _, _, _) = synth_gguf();
        let g = Gguf::parse(bytes).unwrap();
        // 32 elements as a [4, 8] matrix → int4 QWeight.
        let qw = g.quantized_weight("t_q8", 4, QKind::Int4).unwrap();
        assert_eq!((qw.rows, qw.cols), (4, 8));
        assert_eq!(qw.kind, QKind::Int4);
        // Round-trips within the int4 step of each row.
        let back = qw.to_f32();
        let direct = g.dequantize(g.tensor("t_q8").unwrap()).unwrap();
        let mae: f32 = back
            .iter()
            .zip(&direct)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 32.0;
        assert!(mae < 0.3, "int4 import mae {mae}");
    }

    #[test]
    fn quantized_tensor_count_must_be_block_multiple() {
        // 30 is not a multiple of 32 → sizing rejects it.
        assert!(type_nbytes(GGML_Q8_0, 30).is_err());
        assert!(type_nbytes(GGML_F32, 30).is_ok());
        assert!(type_nbytes(999, 32).is_err());
    }
}
