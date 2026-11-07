//! Requested output precision (`GgufQuant`), size labels and per-tensor type planning.
use crate::gguf::{
    GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
    GGML_Q8_1, QK, QK_K,
};

/// The on-disk GGUF tensor type the writer should emit for weight matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufQuant {
    /// 32-bit float (unquantized).
    F32,
    /// 16-bit float.
    F16,
    /// 8-bit blocks of 32 with one scale.
    Q8_0,
    /// 8-bit blocks of 32 with scale and sum.
    Q8_1,
    /// 4-bit blocks of 32 with one scale.
    Q4_0,
    /// 4-bit blocks of 32 with scale and minimum.
    Q4_1,
    /// 4-bit K-quant super-blocks.
    Q4K,
    /// 5-bit K-quant super-blocks.
    Q5K,
    /// 6-bit K-quant super-blocks.
    Q6K,
}

impl GgufQuant {
    /// Parse a CLI/API name (case-insensitive). Accepts the reader's spellings.
    /// API is deliberately `GgufQuant::from_str`, not implementing `FromStr` trait.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "none" => Some(Self::F32),
            "f16" | "fp16" => Some(Self::F16),
            "q8_0" | "q8" => Some(Self::Q8_0),
            "q8_1" => Some(Self::Q8_1),
            "q4_0" | "q4" => Some(Self::Q4_0),
            "q4_1" => Some(Self::Q4_1),
            "q4_k" | "q4k" => Some(Self::Q4K),
            "q5_k" | "q5k" => Some(Self::Q5K),
            "q6_k" | "q6k" => Some(Self::Q6K),
            _ => None,
        }
    }

    pub(crate) fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => GGML_F32,
            Self::F16 => GGML_F16,
            Self::Q8_0 => GGML_Q8_0,
            Self::Q8_1 => GGML_Q8_1,
            Self::Q4_0 => GGML_Q4_0,
            Self::Q4_1 => GGML_Q4_1,
            Self::Q4K => GGML_Q4_K,
            Self::Q5K => GGML_Q5_K,
            Self::Q6K => GGML_Q6_K,
        }
    }

    /// The GGUF `general.file_type` enum id closest to this target (metadata hint).
    pub(crate) fn file_type(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q8_0 => 7,
            // There is no standard whole-model Q8_1 ftype (9 is MOSTLY_Q5_1);
            // the nearest honest hint is MOSTLY_Q8_0.
            Self::Q8_1 => 7,
            Self::Q4K => 15, // MOSTLY_Q4_K_M
            Self::Q5K => 17, // MOSTLY_Q5_K_M
            Self::Q6K => 18, // MOSTLY_Q6_K
        }
    }

    pub(crate) fn is_kquant(self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K | Self::Q6K)
    }
}

/// llama.cpp-style `general.size_label` for a parameter count: two significant
/// digits below 10 ("1.5B"), whole numbers otherwise ("124M", "507K"). Counts
/// under 1000 are the plain integer; 0 has no label.
pub fn size_label(params: u64) -> Option<String> {
    if params == 0 {
        return None;
    }
    if params < 1000 {
        return Some(params.to_string());
    }
    let units = [(1e3, 'K'), (1e6, 'M'), (1e9, 'B'), (1e12, 'T')];
    let mut i = units.iter().rposition(|&(u, _)| params as f64 >= u)?;
    loop {
        let scaled = params as f64 / units[i].0;
        let r = if scaled < 10.0 {
            (scaled * 10.0).round() / 10.0
        } else {
            scaled.round()
        };
        // 999.6K rounds to 1000K: carry into the next unit.
        if r >= 1000.0 && i + 1 < units.len() {
            i += 1;
            continue;
        }
        let u = units[i].1;
        return Some(if r.fract() == 0.0 {
            format!("{r:.0}{u}")
        } else {
            format!("{r:.1}{u}")
        });
    }
}

/// Decide the on-disk type for one tensor. 1-D tensors (norms, biases) always
/// stay F32. 2-D matrices take `quant`, unless their block axis is not divisible
/// by the block size, in which case they fall back to F16. Returns
/// `(ggml_type, fell_back)`.
pub(crate) fn plan_tensor_type(is_2d: bool, block_axis: usize, quant: GgufQuant) -> (u32, bool) {
    if !is_2d {
        return (GGML_F32, false);
    }
    match quant {
        GgufQuant::F32 => (GGML_F32, false),
        GgufQuant::F16 => (GGML_F16, false),
        q if q.is_kquant() => {
            if block_axis.is_multiple_of(QK_K) {
                (q.ggml_type(), false)
            } else {
                (GGML_F16, true)
            }
        }
        q => {
            if block_axis.is_multiple_of(QK) {
                (q.ggml_type(), false)
            } else {
                (GGML_F16, true)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{plan_tensor_type, size_label, GgufQuant};
    use crate::gguf::Gguf;
    use crate::gguf_write::llama::llama_gguf_bytes;
    use crate::gguf_write::test_support::tiny_llama_gguf_bytes;

    #[test]
    fn size_label_table() {
        let cases: &[(u64, Option<&str>)] = &[
            (0, None),
            (1, Some("1")),
            (999, Some("999")),
            (1_000, Some("1K")),
            (1_049, Some("1K")),
            (1_050, Some("1.1K")),
            (9_949, Some("9.9K")),
            (9_950, Some("10K")),
            (507_552, Some("508K")),
            (999_499, Some("999K")),
            (999_500, Some("1M")),
            (1_000_000, Some("1M")),
            (1_500_000, Some("1.5M")),
            (124_439_808, Some("124M")),
            (999_999_999, Some("1B")),
            (1_000_000_000, Some("1B")),
            (1_540_000_000, Some("1.5B")),
            (8_030_000_000, Some("8B")),
            (70_550_000_000, Some("71B")),
            (999_500_000_000, Some("1T")),
            (1_000_000_000_000, Some("1T")),
            // No unit above T: the number just grows.
            (2_000_000_000_000_000, Some("2000T")),
        ];
        for &(n, want) in cases {
            assert_eq!(size_label(n).as_deref(), want, "{n}");
        }
    }

    #[test]
    fn gguf_quant_parses_and_maps_types() {
        use crate::gguf::{GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_K, GGML_Q6_K, GGML_Q8_0};
        assert_eq!(GgufQuant::from_str("f32").unwrap().ggml_type(), GGML_F32);
        assert_eq!(GgufQuant::from_str("f16").unwrap().ggml_type(), GGML_F16);
        assert_eq!(GgufQuant::from_str("q8_0").unwrap().ggml_type(), GGML_Q8_0);
        assert_eq!(GgufQuant::from_str("q4_0").unwrap().ggml_type(), GGML_Q4_0);
        assert_eq!(GgufQuant::from_str("q4_k").unwrap().ggml_type(), GGML_Q4_K);
        assert_eq!(GgufQuant::from_str("q6_k").unwrap().ggml_type(), GGML_Q6_K);
        assert!(GgufQuant::from_str("q4_k").unwrap().is_kquant());
        assert!(!GgufQuant::from_str("q8_0").unwrap().is_kquant());
        assert!(GgufQuant::from_str("bogus").is_none());
    }

    #[test]
    fn plan_tensor_type_applies_policy() {
        use crate::gguf::{GGML_F16, GGML_F32, GGML_Q4_K, GGML_Q8_0};
        // 1-D tensors (norms/biases) are always F32.
        assert_eq!(
            plan_tensor_type(false, 8, GgufQuant::Q4K),
            (GGML_F32, false)
        );
        // 2-D, divisible by 256 → k-quant target.
        assert_eq!(
            plan_tensor_type(true, 256, GgufQuant::Q4K),
            (GGML_Q4_K, false)
        );
        // 2-D, NOT divisible by 256 → k-quant falls back to F16.
        assert_eq!(
            plan_tensor_type(true, 100, GgufQuant::Q4K),
            (GGML_F16, true)
        );
        // 2-D, divisible by 32 → legacy target.
        assert_eq!(
            plan_tensor_type(true, 64, GgufQuant::Q8_0),
            (GGML_Q8_0, false)
        );
        // 2-D, NOT divisible by 32 → legacy falls back to F16.
        assert_eq!(
            plan_tensor_type(true, 20, GgufQuant::Q8_0),
            (GGML_F16, true)
        );
        // F16/F32 targets have no divisibility requirement.
        assert_eq!(
            plan_tensor_type(true, 20, GgufQuant::F16),
            (GGML_F16, false)
        );
    }

    // When every matrix falls back to F16 (tiny dims), the file-type hint must
    // say F16, not the k-quant that was requested but never used.
    #[test]
    fn file_type_reflects_total_fallback() {
        let g0 = Gguf::parse(tiny_llama_gguf_bytes()).unwrap();
        let model = g0.load_llama_prec(None).unwrap();
        let out = llama_gguf_bytes(&model, &g0, GgufQuant::Q4K).unwrap();
        let g1 = Gguf::parse(out).unwrap();
        assert_eq!(
            g1.meta("general.file_type").and_then(|v| v.as_usize()),
            Some(1), // MOSTLY_F16
        );
    }

    #[test]
    fn every_quant_maps_to_its_ggml_type_and_whole_model_file_type() {
        use crate::gguf::{
            GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q4_1, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
            GGML_Q8_1,
        };
        let table = [
            (GgufQuant::F32, GGML_F32, 0),
            (GgufQuant::F16, GGML_F16, 1),
            (GgufQuant::Q4_0, GGML_Q4_0, 2),
            (GgufQuant::Q4_1, GGML_Q4_1, 3),
            (GgufQuant::Q8_0, GGML_Q8_0, 7),
            (GgufQuant::Q8_1, GGML_Q8_1, 7),
            (GgufQuant::Q4K, GGML_Q4_K, 15),
            (GgufQuant::Q5K, GGML_Q5_K, 17),
            (GgufQuant::Q6K, GGML_Q6_K, 18),
        ];
        for (q, ty, ft) in table {
            assert_eq!(q.ggml_type(), ty, "{q:?}");
            assert_eq!(q.file_type(), ft, "{q:?}");
        }
    }
}
