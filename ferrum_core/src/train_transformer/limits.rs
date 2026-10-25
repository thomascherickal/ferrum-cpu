//! Size limits and memory estimates that guard allocation from untrusted dimensions.
use crate::error::{InferError, Result};
use crate::train_transformer::TransformerNet;

impl TransformerNet {
    /// Parameter tensors per transformer block (`BlockT::params`).
    pub(super) const BLOCK_PARAMS: usize = 16;

    /// Estimated peak bytes of training with `batch` sequences per step
    /// (clamped to `windows`, the sequences an epoch has) on `shards` threads:
    /// weights + gradients + both Adam moments (16 B/param), 8 B/param for
    /// every extra shard's net clone and gradient, and the forward cache plus
    /// its gradient (about twice the cached floats) for every position of the
    /// batch. Saturating, so absurd inputs give `usize::MAX`, not a wrap.
    pub fn train_peak_bytes(&self, windows: usize, batch: usize, shards: usize) -> usize {
        let (t, e, v) = (self.context_len, self.embed_dim, self.vocab_size);
        let hidden = self.blocks.first().map_or(0, |b| b.f1_w.data.shape[1]);
        // Floats cached per position: embedding output, per block (LN, q/k/v,
        // attention probabilities over the context, context, projections,
        // residuals, FFN in/out), final LN, and logits + softmax + their gradient.
        let per_block = e
            .saturating_mul(10)
            .saturating_add(hidden.saturating_mul(2))
            .saturating_add(self.num_heads.saturating_mul(t));
        let per_pos = per_block
            .saturating_mul(self.blocks.len())
            .saturating_add(v.saturating_mul(3))
            .saturating_add(e.saturating_mul(2));
        let positions = batch.min(windows.max(1)).saturating_mul(t);
        let act = positions.saturating_mul(per_pos).saturating_mul(8); // x2 cache+grad, 4 B each
        let params = self.num_params();
        params
            .saturating_mul(16)
            .saturating_add(
                params
                    .saturating_mul(8)
                    .saturating_mul(shards.saturating_sub(1)),
            )
            .saturating_add(act)
    }

    /// Largest trainable model: 2^29 parameters (2 GiB of f32 weights, ~8.6 GB
    /// with gradients and both Adam moments) and 4096 blocks. This is an SLM
    /// trainer; the cap turns an absurd `--embed` / `--blocks` into an error
    /// instead of an allocation abort.
    pub const MAX_PARAMS: usize = 1 << 29;

    const MAX_BLOCKS: usize = 4096;

    /// Parameter count of a net with these dims, or an error when it is
    /// overflowing, or above [`TransformerNet::MAX_PARAMS`] / the block cap.
    /// Runs before anything is allocated.
    pub fn check_size(
        vocab: usize,
        ctx: usize,
        embed: usize,
        hidden: usize,
        blocks: usize,
    ) -> Result<usize> {
        let too_big = || {
            InferError::DimMismatch(format!(
                "model too large: vocab {vocab}, context {ctx}, embed {embed}, hidden {hidden}, \
                 blocks {blocks} exceed {} parameters / {} blocks",
                Self::MAX_PARAMS,
                Self::MAX_BLOCKS
            ))
        };
        if blocks > Self::MAX_BLOCKS {
            return Err(too_big());
        }
        let lens = Self::param_lens(vocab, ctx, embed, hidden, blocks).ok_or_else(too_big)?;
        lens.iter()
            .try_fold(0usize, |a, &n| a.checked_add(n))
            .filter(|&n| n <= Self::MAX_PARAMS)
            .ok_or_else(too_big)
    }

    /// Element count of every parameter, in `params_all` order; `None` on
    /// overflow. Allocates `5 + 16·blocks` entries, so callers bound `blocks`
    /// first.
    pub(super) fn param_lens(
        vocab: usize,
        ctx: usize,
        embed: usize,
        hidden: usize,
        blocks: usize,
    ) -> Option<Vec<usize>> {
        let (e, h) = (embed, hidden);
        let sq = e.checked_mul(e)?;
        let eh = e.checked_mul(h)?;
        let block = [e, e, sq, e, sq, e, sq, e, sq, e, e, e, eh, h, eh, e];
        debug_assert_eq!(block.len(), Self::BLOCK_PARAMS);
        let ev = vocab.checked_mul(e)?; // token embedding / LM head
        let mut lens = vec![ev, ctx.checked_mul(e)?]; // + positions
        for _ in 0..blocks {
            lens.extend(block);
        }
        lens.extend([e, e, ev]); // final LayerNorm scale + shift, head
        Some(lens)
    }
}

#[cfg(test)]
mod tests {
    use crate::rng::Rng;
    use crate::train_transformer::TransformerNet;

    #[test]
    fn param_lens_match_real_nets_and_detect_overflow() {
        for (v, c, e, h, hid, b) in [
            (6, 4, 8, 2, 16, 2),
            (30, 8, 16, 2, 32, 1),
            (5, 3, 4, 1, 7, 3),
        ] {
            let net = TransformerNet::new(v, c, e, h, hid, b, &mut Rng::new(1)).unwrap();
            let real: Vec<usize> = net.params_all().iter().map(|p| p.data.data.len()).collect();
            assert_eq!(TransformerNet::param_lens(v, c, e, hid, b), Some(real));
        }
        let big = usize::MAX / 2;
        // u64-based literal: `1 << 40` / `1 << 62` do not fit a 32-bit usize.
        let huge = usize::try_from(1u64 << 40).unwrap_or(usize::MAX);
        let huger = usize::try_from(1u64 << 62).unwrap_or(usize::MAX);
        for args in [
            (big, 1, 4, 1, 1),
            (1, big, 4, 1, 1),
            (1, 1, big, 1, 1),
            (1, 1, huge, 1, 1),
            (1, 1, 4, huger, 1),
        ] {
            let (v, c, e, hid, b) = args;
            assert_eq!(
                TransformerNet::param_lens(v, c, e, hid, b),
                None,
                "{args:?}"
            );
        }
    }

    #[test]
    fn train_peak_bytes_scales_with_batch_clamps_to_windows_and_saturates() {
        let net = TransformerNet::new(100, 16, 32, 4, 64, 2, &mut Rng::new(1)).unwrap();
        let fixed = net.num_params() * 16;
        // One shard, one sequence: weights + optimizer + activations.
        let one = net.train_peak_bytes(1000, 1, 1);
        let ten = net.train_peak_bytes(1000, 10, 1);
        assert!(one > fixed && ten > one);
        assert_eq!(
            (ten - fixed) / (one - fixed),
            10,
            "activations are linear in batch"
        );
        // A batch larger than the epoch's windows counts only the windows.
        assert_eq!(
            net.train_peak_bytes(3, 1 << 40, 1),
            net.train_peak_bytes(3, 3, 1)
        );
        // Each extra shard adds a net clone + gradient (8 B/param).
        assert_eq!(
            net.train_peak_bytes(1000, 1, 3) - one,
            2 * net.num_params() * 8
        );
        // 0 windows is treated as 1; absurd inputs saturate instead of wrapping.
        assert_eq!(net.train_peak_bytes(0, 5, 1), net.train_peak_bytes(1, 5, 1));
        assert_eq!(net.train_peak_bytes(usize::MAX, usize::MAX, 1), usize::MAX);
        // Attention scores grow with the square of the context.
        let long = TransformerNet::new(100, 4096, 32, 4, 64, 2, &mut Rng::new(1)).unwrap();
        let short = TransformerNet::new(100, 64, 32, 4, 64, 2, &mut Rng::new(1)).unwrap();
        assert!(
            long.train_peak_bytes(10_000, 1, 1) > 64 * short.train_peak_bytes(10_000, 1, 1) / 4
        );
    }

    #[test]
    fn zero_width_dims_are_rejected_up_front() {
        let mut r = Rng::new(1);
        for (v, c, e, h, hid, b) in [
            (8, 4, 0, 1, 16, 1), // embed 0 (heads 1 divides 0)
            (8, 4, 8, 2, 0, 1),  // hidden 0
            (0, 4, 8, 2, 16, 1),
            (8, 0, 8, 2, 16, 1),
            (8, 4, 8, 2, 16, 0),
        ] {
            let err = TransformerNet::new(v, c, e, h, hid, b, &mut r)
                .err()
                .unwrap();
            assert!(err.to_string().contains("must be > 0"), "{err}");
        }
    }

    #[test]
    fn absurd_dimensions_are_an_error_not_an_allocation_abort() {
        // Each of these used to ask the allocator for 100 GB - 3 TB and abort.
        let mut r = Rng::new(1);
        for (v, c, e, h, hid, b) in [
            (512, 64, 100_000_000, 1, 1, 1),     // embed
            (512, 64, 64, 1, 64, 1_000_000_000), // blocks
            (usize::MAX, 4, 8, 2, 16, 1),        // vocab overflow
            (4_000_000_000, 4, 8, 2, 16, 1),     // vocab
            (512, 4_000_000_000, 8, 2, 16, 1),   // context
            (512, 64, 64, 1, usize::MAX, 1),     // hidden overflow
            (512, 64, 4096, 1, 4096, 4096),      // just over MAX_PARAMS
        ] {
            let err = TransformerNet::new(v, c, e, h, hid, b, &mut r).err();
            assert!(
                err.as_ref()
                    .is_some_and(|x| x.to_string().contains("model too large")),
                "{v},{c},{e}: {err:?}"
            );
        }
        // Boundary: the 507k reference config and a tiny net pass; the cap itself is exact.
        assert_eq!(
            TransformerNet::check_size(1024, 128, 96, 320, 3).unwrap(),
            507_552
        );
        let exact = TransformerNet::MAX_PARAMS;
        // vocab chosen so params == MAX_PARAMS + k: check_size is monotone around the cap.
        let (e, ctx) = (64usize, 1usize);
        let fixed = TransformerNet::check_size(1, ctx, e, 1, 1).unwrap();
        let per_vocab = 2 * e;
        let v_ok = (exact - fixed) / per_vocab + 1;
        assert!(TransformerNet::check_size(v_ok, ctx, e, 1, 1).is_ok());
        assert!(TransformerNet::check_size(v_ok + 1, ctx, e, 1, 1).is_err());
    }
}
