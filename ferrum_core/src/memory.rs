//! Pre-load memory checks for imported GGUFs, shared by every front end that loads one (the
//! CLI's `run-gguf`/`eval-gguf`/`export-gguf`/`finetune-gguf`, `ferrum-serve`, the GUI): the
//! resident-size estimate, the import-precision names, available RAM and the 90 % refusal.
#![warn(missing_docs)]

use crate::gguf::Gguf;
use crate::quant::QKind;

/// Estimate the resident bytes of a model loaded from `g` at in-memory precision `prec`
/// (`None` = f32), from the tensor directory alone. The token embedding stays f32 in
/// `LlamaModel` (so it is the single largest array for big-vocab models); every other weight
/// packs to int4/int8/f32. A pre-load guard, not an exact figure; saturates instead of
/// overflowing on a hostile header.
pub fn resident_bytes(g: &Gguf, prec: Option<QKind>) -> usize {
    g.tensors.iter().fold(0usize, |total, t| {
        let n = t.num_elements();
        let bytes = if t.name == "token_embd.weight" {
            n.saturating_mul(4)
        } else {
            match prec {
                Some(QKind::Int4) => n.div_ceil(2),
                Some(QKind::Int8) => n,
                None => n.saturating_mul(4),
            }
        };
        total.saturating_add(bytes)
    })
}

/// The in-memory import precision a name stands for: `int4`/`q4`, `int8`/`q8`, or
/// `f32`/`none` (`Some(None)`, full f32). `None` for any other name; each caller words its
/// own refusal.
pub fn import_precision(name: &str) -> Option<Option<QKind>> {
    match name {
        "int4" | "q4" => Some(Some(QKind::Int4)),
        "int8" | "q8" => Some(Some(QKind::Int8)),
        "f32" | "none" => Some(None),
        _ => None,
    }
}

/// Best-effort available RAM in bytes (Linux `/proc/meminfo` `MemAvailable`); `None` where it
/// cannot be determined (other systems, wasm), in which case callers skip the guard.
pub fn available_memory_bytes() -> Option<usize> {
    available_memory_from(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// `MemAvailable` (bytes) from the text of `/proc/meminfo`.
pub fn available_memory_from(info: &str) -> Option<usize> {
    for line in info.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// The refusal when an estimate of `what` exceeds 90 % of the free RAM `avail` (unless
/// `force`); `advice` is appended after the numbers.
pub fn memory_error(
    what: &str,
    est: usize,
    avail: usize,
    force: bool,
    advice: &str,
) -> Option<String> {
    ((est as f64) > 0.9 * avail as f64 && !force).then(|| {
        format!(
            "estimated {what} ({:.2} GB) exceeds 90% of available ({:.2} GB) — {advice}",
            est as f64 / 1e9,
            avail as f64 / 1e9
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_write::GgufBuilder;

    #[test]
    fn resident_bytes_keeps_the_embedding_f32_and_packs_the_rest() {
        let mut b = GgufBuilder::new();
        b.tensor("token_embd.weight", &[5], 0, vec![0; 20]);
        b.tensor("blk.0.w", &[3], 0, vec![0; 12]);
        let g = Gguf::parse(b.into_bytes()).unwrap();
        assert_eq!(resident_bytes(&g, None), 20 + 12);
        assert_eq!(resident_bytes(&g, Some(QKind::Int8)), 20 + 3);
        assert_eq!(resident_bytes(&g, Some(QKind::Int4)), 20 + 2);
    }

    #[test]
    fn import_precision_names() {
        assert_eq!(import_precision("int4"), Some(Some(QKind::Int4)));
        assert_eq!(import_precision("q4"), Some(Some(QKind::Int4)));
        assert_eq!(import_precision("int8"), Some(Some(QKind::Int8)));
        assert_eq!(import_precision("q8"), Some(Some(QKind::Int8)));
        assert_eq!(import_precision("f32"), Some(None));
        assert_eq!(import_precision("none"), Some(None));
        assert_eq!(import_precision("fp8"), None);
        assert_eq!(import_precision(""), None);
    }

    #[test]
    fn memory_error_formats_the_numbers_and_honours_force() {
        let gb = 1_000_000_000;
        // 9.5 of 10 GB is over the 90% line.
        let e = memory_error("peak memory", 95 * gb / 10, 10 * gb, false, "do X.").unwrap();
        assert_eq!(
            e,
            "estimated peak memory (9.50 GB) exceeds 90% of available (10.00 GB) — do X."
        );
        // Exactly 90% and below passes; force always passes.
        assert_eq!(memory_error("m", 9 * gb, 10 * gb, false, "a"), None);
        assert_eq!(memory_error("m", 100 * gb, 10 * gb, true, "a"), None);
    }

    #[test]
    fn available_memory_is_parsed_from_proc_meminfo_text() {
        let info = "MemTotal:       16000000 kB\nMemFree:  100 kB\nMemAvailable:    2048 kB\n";
        assert_eq!(available_memory_from(info), Some(2048 * 1024));
        assert_eq!(available_memory_from("MemTotal: 1 kB\n"), None);
        assert_eq!(available_memory_from("MemAvailable: lots kB\n"), None);
        assert_eq!(available_memory_from(""), None);
        // The real reader agrees with the parser on this machine's /proc (Linux only).
        #[cfg(target_os = "linux")]
        assert!(available_memory_bytes().is_some_and(|b| b > 0));
    }
}
