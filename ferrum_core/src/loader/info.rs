//! Inspection: quantization and format labels of a GGUF file.
use crate::error::Result;
use crate::gguf::{Gguf, MetaValue};
use crate::loader::decode::parse_gguf;

/// The precision a GGUF model was saved at: `ferrum.quant` ("f32" | "int8" |
/// "int4"), else derived from `general.file_type` (0 / 1 / 7 / 2 -> f32 / f16 /
/// int8 / int4). Any other file type is reported as `other(N)`, and a file with
/// neither key as `unknown`, so `info` never refuses a loadable model.
pub fn gguf_quant(bytes: &[u8]) -> Result<String> {
    Ok(quant_of(&parse_gguf(bytes)?))
}

fn quant_of(g: &Gguf) -> String {
    if let Some(q) = g.meta("ferrum.quant").and_then(MetaValue::as_str) {
        return q.to_string();
    }
    match g.meta("general.file_type").and_then(MetaValue::as_usize) {
        Some(0) => "f32".into(),
        Some(7) => "int8".into(),
        Some(1) => "f16".into(),
        Some(2) => "int4".into(),
        Some(ft) => format!("other({ft})"),
        None => "unknown".into(),
    }
}

/// One-line description for `info`: `GGUF v3 (gpt2, int8)`.
pub fn gguf_format(bytes: &[u8]) -> Result<String> {
    let g = parse_gguf(bytes)?;
    Ok(format!(
        "GGUF v{} ({}, {})",
        g.version,
        g.architecture().unwrap_or("unknown"),
        quant_of(&g)
    ))
}

#[cfg(test)]
mod tests {
    use super::{gguf_format, gguf_quant};
    use crate::gguf::MetaValue;
    use crate::gguf_write::GgufBuilder;
    use crate::loader::encode::to_gguf;
    use crate::loader::test_support::{load_err_of, make_bundle, trained_slm, FINF_HINT};
    use crate::loader::{to_bytes_quantized, QPrec};

    #[test]
    fn gguf_quant_falls_back_to_file_type() {
        let mk = |ft: Option<u32>| {
            let mut b = GgufBuilder::new();
            if let Some(ft) = ft {
                b.meta("general.file_type", MetaValue::U32(ft));
            }
            b.into_bytes()
        };
        assert_eq!(gguf_quant(&mk(Some(0))).unwrap(), "f32");
        assert_eq!(gguf_quant(&mk(Some(7))).unwrap(), "int8");
        assert_eq!(gguf_quant(&mk(Some(2))).unwrap(), "int4");
        assert_eq!(gguf_quant(&mk(Some(1))).unwrap(), "f16");
        assert_eq!(gguf_quant(&mk(Some(15))).unwrap(), "other(15)");
        assert_eq!(gguf_quant(&mk(None)).unwrap(), "unknown");
        assert!(load_err_of(gguf_quant(b"FINF\x05\x00\x00\x00")).contains(FINF_HINT));
    }

    #[test]
    fn gguf_format_describes_arch_and_quant() {
        let (m, n, meta) = make_bundle();
        let b = to_bytes_quantized(&m, &n, &meta).unwrap();
        assert_eq!(gguf_format(&b).unwrap(), "GGUF v3 (ferrum, int8)");
        let slm = trained_slm(false);
        let b = to_gguf(&slm.model, &slm.norm, &slm.meta, QPrec::F32).unwrap();
        assert_eq!(gguf_format(&b).unwrap(), "GGUF v3 (gpt2, f32)");
        // No architecture key and no quant info.
        let b = GgufBuilder::new().into_bytes();
        assert_eq!(gguf_format(&b).unwrap(), "GGUF v3 (unknown, unknown)");
        assert!(load_err_of(gguf_format(b"FINF\x05\x00\x00\x00")).contains(FINF_HINT));
        // gguf_format derives quant from the same parse: every arm agrees with gguf_quant.
        for ft in [Some(0), Some(7), Some(1), Some(2), Some(15), None] {
            let mut b = GgufBuilder::new();
            b.meta("general.architecture", MetaValue::String("gpt2".into()));
            if let Some(ft) = ft {
                b.meta("general.file_type", MetaValue::U32(ft));
            }
            let bytes = b.into_bytes();
            let q = gguf_quant(&bytes).unwrap();
            assert_eq!(gguf_format(&bytes).unwrap(), format!("GGUF v3 (gpt2, {q})"));
        }
    }
}
