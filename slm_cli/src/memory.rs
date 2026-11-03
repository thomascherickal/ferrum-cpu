//! Memory estimates and budgets for `run-gguf`, `finetune-gguf` and `export-gguf`.

pub(crate) use ferrum_core::memory::{
    available_memory_bytes, memory_error, resident_bytes as estimate_resident_bytes,
};

/// Peak resident bytes of applying a fine-tune checkpoint (`--resume`) to an f32 model of size
/// `est` with `apply_checkpoint_weights`: the model, the checkpoint's raw bytes (parsed in
/// place), and one decoded parameter at a time (the embedding table is the largest, 21-27% of
/// the model for SmolLM2/Qwen2.5; budgeted at half). Measured on SmolLM2-135M (est 0.54 GB,
/// 1.96 GB .flck): 2.70 GB peak, down from 5.77 GB when a LlamaTrainer wrapper (4x the model)
/// and flat f32 copies of the weights and both Adam moments were built (and 7.7 GB before the
/// bytes were handed over instead of copied).
pub(crate) fn resume_peak_bytes(est: usize, ckpt_len: usize) -> usize {
    est.saturating_add(ckpt_len).saturating_add(est / 2)
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
    memory_error(
        "training memory",
        peak,
        avail,
        force,
        "fine-tune a smaller model, or pass --force to attempt it anyway.",
    )
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
    fn resume_peak_is_model_plus_checkpoint_plus_one_parameter() {
        // SmolLM2-135M: est 0.54 GB, 1.96 GB checkpoint -> 2.77 GB estimate (measured 2.70).
        let gb = 1_000_000_000usize;
        let peak = resume_peak_bytes(540_000_000, 1_960_000_000);
        assert_eq!(peak, 540_000_000 + 1_960_000_000 + 270_000_000);
        assert!(peak > 2_700_000_000 && peak < 3 * gb, "{peak}");
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

    #[test]
    fn finetune_memory_error_keeps_its_wording() {
        let gb = 1_000_000_000;
        let e = finetune_memory_error(20 * gb, 10 * gb, false).unwrap();
        assert!(e.starts_with("estimated training memory (20.00 GB)"), "{e}");
        assert!(e.ends_with("fine-tune a smaller model, or pass --force to attempt it anyway."));
        assert_eq!(finetune_memory_error(20 * gb, 10 * gb, true), None);
    }
}
