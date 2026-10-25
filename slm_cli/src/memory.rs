//! Memory estimates and budgets for `run-gguf`, `finetune-gguf` and `export-gguf`.

/// Estimate the resident memory a loaded model will need, from the GGUF tensor
/// directory and the chosen in-memory precision. The token embedding is kept
/// f32 (so it is the single largest array for big-vocab models); every other
/// weight packs to int4/int8/f32. A pre-load guard, not an exact figure.
pub(crate) fn estimate_resident_bytes(
    g: &ferrum_core::Gguf,
    prec: Option<ferrum_core::QKind>,
) -> usize {
    use ferrum_core::QKind;
    let mut total = 0usize;
    for t in &g.tensors {
        let n = t.num_elements();
        let bytes = if t.name == "token_embd.weight" {
            n * 4 // kept f32 in LlamaModel
        } else {
            match prec {
                Some(QKind::Int4) => n.div_ceil(2),
                Some(QKind::Int8) => n,
                None => n * 4,
            }
        };
        total = total.saturating_add(bytes);
    }
    total
}

/// Best-effort available RAM in bytes (Linux `/proc/meminfo` `MemAvailable`);
/// `None` where it cannot be determined, in which case the guard is skipped.
pub(crate) fn available_memory_bytes() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in info.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Peak resident bytes of applying a fine-tune checkpoint (`--resume`) to an f32 model of size
/// `est`: the trainer wrapper holds the weights, gradients and both Adam moments (4x), and the
/// checkpoint is resident as its raw bytes plus the flat f32 vectors rebuilt from it (about its
/// size again). Measured on SmolLM2-135M: est 0.54 GB, a 1.96 GB .flck peaked at 7.7 GB before
/// the buffer was handed over instead of copied (7.7 - 1.96 = 5.7 vs 4 x 0.54 + 2 x 1.96 = 6.1).
pub(crate) fn resume_peak_bytes(est: usize, ckpt_len: usize) -> usize {
    est.saturating_mul(4)
        .saturating_add(ckpt_len.saturating_mul(2))
}

/// Size of the `--resume` checkpoint in bytes (0 if unreadable; the load reports that).
pub(crate) fn file_len(path: &str) -> usize {
    std::fs::metadata(path).map_or(0, |m| usize::try_from(m.len()).unwrap_or(usize::MAX))
}

/// Peak resident bytes of a fine-tune of f32 weights `est`: weights, gradient
/// and two Adam moments (4x), plus the checkpoint save, which serializes the
/// weights and both moments (3x) into the builder and again into the final
/// buffer (measured 6.4 GB for SmolLM2-135M, est 0.54 GB).
pub(crate) fn finetune_peak_bytes(est: usize) -> usize {
    est.saturating_mul(12)
}

/// Worker threads a fine-tune can afford. Every shard holds its own flat f32
/// gradient (about the size of the weights) on top of the weights, shared
/// gradient and two Adam moments (4x the weights), so an 8-thread run of a
/// 135M model peaks near 9 GB. Cap the shards so the total stays within half
/// of the available RAM; always at least one, never above `requested`.
pub(crate) fn finetune_threads(est: usize, avail: Option<usize>, requested: usize) -> usize {
    let Some(avail) = avail else {
        return requested;
    };
    let budget = (avail / 2).saturating_sub(est.saturating_mul(4));
    let per_thread = est.saturating_mul(3) / 2;
    (budget / per_thread.max(1)).clamp(1, requested.max(1))
}

/// The refusal when the fine-tune peak exceeds 90% of the free RAM (unless forced).
pub(crate) fn finetune_memory_error(peak: usize, avail: usize, force: bool) -> Option<String> {
    ((peak as f64) > 0.9 * avail as f64 && !force).then(|| {
        format!(
            "estimated training memory ({:.2} GB) exceeds 90% of available ({:.2} GB) — \
             fine-tune a smaller model, or pass --force to attempt it anyway.",
            peak as f64 / 1e9,
            avail as f64 / 1e9
        )
    })
}

/// The notice printed when [`finetune_threads`] lowered the request.
pub(crate) fn thread_cap_note(capped: usize, requested: usize) -> Option<String> {
    (capped < requested).then(|| {
        format!(
            "  note: using {capped} of {requested} threads (each holds a full gradient copy; \
             keeps training under half of the available RAM)"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finetune_threads_fit_the_memory_budget() {
        const GB: usize = 1_000_000_000;
        let est = 540_000_000; // SmolLM2-135M at f32
                               // Unknown RAM: trust the request.
        assert_eq!(finetune_threads(est, None, 8), 8);
        // 9.5 GB free: (4.75 - 2.16) / 0.81 = 3 shards.
        assert_eq!(finetune_threads(est, Some(95 * GB / 10), 8), 3);
        // Plenty of RAM: the request wins; never zero threads.
        assert_eq!(finetune_threads(est, Some(64 * GB), 4), 4);
        assert_eq!(finetune_threads(est, Some(64 * GB), 0), 1);
        // Weights alone already use the budget: still one thread.
        assert_eq!(finetune_threads(est, Some(3 * GB), 8), 1);
        assert_eq!(finetune_threads(0, Some(GB), 8), 8);
        assert_eq!(finetune_threads(usize::MAX, Some(GB), 8), 1);
        // Qwen2.5-0.5B (2 GB f32 weights) on this 15 GB machine is refused; --force overrides.
        let e = finetune_memory_error(finetune_peak_bytes(2 * GB), 9 * GB, false).unwrap();
        assert!(
            e.contains("24.00 GB") && e.contains("9.00 GB") && e.contains("--force"),
            "{e}"
        );
        assert_eq!(
            finetune_memory_error(finetune_peak_bytes(2 * GB), 9 * GB, true),
            None
        );
        assert_eq!(
            finetune_memory_error(finetune_peak_bytes(540_000_000), 10 * GB, false),
            None
        );
        assert_eq!(thread_cap_note(8, 8), None);
        let note = thread_cap_note(3, 8).unwrap();
        assert!(note.contains("using 3 of 8 threads"), "{note}");
        // The checkpoint save dominates the peak: 12x the f32 weights.
        assert_eq!(finetune_peak_bytes(est), 6_480_000_000);
        assert_eq!(finetune_peak_bytes(usize::MAX), usize::MAX);
    }

    #[test]
    fn resume_peak_accounts_for_trainer_state_and_checkpoint_buffers() {
        // SmolLM2-135M: est 0.54 GB, 1.96 GB checkpoint -> about 6.1 GB (measured 5.7-7.7).
        let gb = 1_000_000_000usize;
        let peak = resume_peak_bytes(540_000_000, 1_960_000_000);
        assert_eq!(peak, 4 * 540_000_000 + 2 * 1_960_000_000);
        assert!(peak > 5 * gb && peak < 7 * gb, "{peak}");
        assert_eq!(resume_peak_bytes(0, 0), 0);
        assert_eq!(resume_peak_bytes(usize::MAX, usize::MAX), usize::MAX);
    }

    #[test]
    fn file_len_reads_size_and_tolerates_missing_files() {
        let f = std::env::temp_dir().join(format!("ferrum_filelen_{}", std::process::id()));
        std::fs::write(&f, [0u8; 123]).unwrap();
        assert_eq!(file_len(f.to_str().unwrap()), 123);
        let _ = std::fs::remove_file(&f);
        assert_eq!(file_len(f.to_str().unwrap()), 0);
    }
}
