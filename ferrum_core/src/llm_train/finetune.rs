//! Training steps, batches and (threaded) fine-tuning epochs.
use crate::error::{InferError, Result};
#[cfg(not(target_arch = "wasm32"))]
use crate::llm::LlamaModel;
use crate::llm_train::passes::seq_grad_flat;
use crate::llm_train::LlamaTrainer;
use crate::rng::Rng;

impl LlamaTrainer {
    /// One AdamW step on a single sequence. Returns the pre-step loss.
    pub fn train_step(&mut self, tokens: &[usize]) -> Result<f32> {
        self.train_batch(&[tokens])
    }

    /// One AdamW step over a minibatch of sequences. The per-sequence gradients
    /// (each already a per-position mean) are averaged across the batch, so the
    /// update matches the mean loss. Returns the mean pre-step loss.
    ///
    /// With QAT enabled the gradients are computed at int8-snapped weights and
    /// the f32 masters are updated (straight-through estimator).
    pub fn train_batch(&mut self, batch: &[&[usize]]) -> Result<f32> {
        self.train_batch_seeded(batch, 0)
    }

    /// Like [`LlamaTrainer::train_batch`] but with an explicit dropout-mask base
    /// seed (each sequence offset from it). `seed = 0` with dropout off is the
    /// plain deterministic path.
    fn train_batch_seeded(&mut self, batch: &[&[usize]], base_seed: u64) -> Result<f32> {
        if batch.is_empty() {
            return Err(InferError::DimMismatch("empty minibatch".into()));
        }
        let dropout = self.dropout;
        let masters = if self.qat {
            let snap = self.snapshot_weights();
            self.fake_quantize_weights();
            Some(snap)
        } else {
            None
        };

        self.grads.clear();
        let mut total = 0.0f32;
        for (si, seq) in batch.iter().enumerate() {
            let seed = base_seed.wrapping_add(si as u64 + 1);
            let (loss, flat) = seq_grad_flat(&self.model, seq, dropout, seed)?;
            self.grads.add_assign_flat(&flat);
            total += loss;
        }
        let inv_b = 1.0 / batch.len() as f32;
        self.grads.scale(inv_b);

        if let Some(snap) = &masters {
            self.restore_weights(snap);
        }
        self.apply_step();
        Ok(total * inv_b)
    }

    /// One epoch of minibatch AdamW over a token stream. Each example is a window
    /// of `seq_len` tokens (next-token targets at every interior position); the
    /// windows are shuffled and drawn without replacement, so an epoch covers the
    /// corpus once. Returns the mean loss. `seq_len` is capped at the model's
    /// `context_len`.
    pub fn finetune_epoch(
        &mut self,
        tokens: &[usize],
        seq_len: usize,
        batch_size: usize,
        rng: &mut Rng,
    ) -> Result<f32> {
        self.finetune_epoch_threaded(tokens, seq_len, batch_size, rng, 1)
    }

    /// Data-parallel [`LlamaTrainer::finetune_epoch`]: each minibatch is split
    /// into up to `threads` shards run concurrently on `std::thread::scope`,
    /// their gradients summed in a fixed shard order, and one AdamW step applied.
    /// `threads <= 1` is the serial path, bit-for-bit. On `wasm32` it always runs
    /// serially.
    pub fn finetune_epoch_threaded(
        &mut self,
        tokens: &[usize],
        seq_len: usize,
        batch_size: usize,
        rng: &mut Rng,
        threads: usize,
    ) -> Result<f32> {
        let seq_len = seq_len.min(self.model.cfg.context_len).max(2);
        if tokens.len() < seq_len {
            return Err(InferError::DimMismatch(format!(
                "need at least seq_len = {seq_len} tokens, got {}",
                tokens.len()
            )));
        }
        let batch_size = batch_size.max(1);
        let num_windows = tokens.len() - seq_len + 1;
        let steps = num_windows.div_ceil(batch_size);
        let perm = rng.shuffled_indices(num_windows);
        let dropout = self.dropout;
        let mut total = 0.0f32;

        for step in 0..steps {
            let idxs = &perm[step * batch_size..((step + 1) * batch_size).min(num_windows)];
            let windows: Vec<&[usize]> = idxs.iter().map(|&s| &tokens[s..s + seq_len]).collect();
            let base_seed = if dropout > 0.0 { rng.next_u64() } else { 0 };
            let loss = self.step_over_windows(&windows, base_seed, threads)?;
            total += loss;
        }
        Ok(total / steps as f32)
    }

    /// Shared gradient computation for one minibatch of already-sliced windows,
    /// optionally across threads, followed by a single AdamW step. Returns the
    /// mean loss over the batch.
    fn step_over_windows(
        &mut self,
        windows: &[&[usize]],
        base_seed: u64,
        threads: usize,
    ) -> Result<f32> {
        // No threads on wasm: always take the serial accumulator path.
        #[cfg(target_arch = "wasm32")]
        {
            let _ = threads;
            self.train_batch_seeded(windows, base_seed)
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let nshards = threads.max(1).min(windows.len().max(1));
            // One shard (or a single-window batch) is exactly the serial path.
            if nshards <= 1 {
                return self.train_batch_seeded(windows, base_seed);
            }

            let dropout = self.dropout;
            let masters = if self.qat {
                let snap = self.snapshot_weights();
                self.fake_quantize_weights();
                Some(snap)
            } else {
                None
            };

            let model_ref: &LlamaModel = &self.model;
            let n = windows.len();
            let per = n.div_ceil(nshards);
            // Each shard returns (loss_sum, summed flat grads) over its windows.
            let results: Vec<Result<(f32, Vec<f32>)>> = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                let mut w0 = 0usize;
                while w0 < n {
                    let w1 = (w0 + per).min(n);
                    let shard = &windows[w0..w1];
                    let off = w0;
                    handles.push(scope.spawn(move || -> Result<(f32, Vec<f32>)> {
                        let mut acc: Option<Vec<f32>> = None;
                        let mut loss_sum = 0.0f32;
                        for (i, seq) in shard.iter().enumerate() {
                            let seed = base_seed.wrapping_add((off + i) as u64 + 1);
                            let (loss, flat) = seq_grad_flat(model_ref, seq, dropout, seed)?;
                            loss_sum += loss;
                            match &mut acc {
                                None => acc = Some(flat),
                                Some(a) => {
                                    for (x, y) in a.iter_mut().zip(&flat) {
                                        *x += y;
                                    }
                                }
                            }
                        }
                        Ok((loss_sum, acc.expect("each shard has ≥1 window")))
                    }));
                    w0 = w1;
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("ferrum: finetune worker thread panicked"))
                    .collect()
            });

            let mut summed: Option<Vec<f32>> = None;
            let mut loss_total = 0.0f32;
            for r in results {
                let (loss, flat) = r?;
                loss_total += loss;
                match &mut summed {
                    None => summed = Some(flat),
                    Some(a) => {
                        for (x, y) in a.iter_mut().zip(&flat) {
                            *x += y;
                        }
                    }
                }
            }
            let mut summed = summed.expect("at least one shard runs");
            let inv_b = 1.0 / n as f32;
            for x in &mut summed {
                *x *= inv_b;
            }

            if let Some(snap) = &masters {
                self.restore_weights(snap);
            }
            self.grads.load_flat(&summed);
            self.apply_step();
            Ok(loss_total * inv_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::llm_train::passes::seq_grad_flat;
    use crate::llm_train::test_support::{base_clone, tiny};
    use crate::llm_train::LlamaTrainer;
    use crate::rng::Rng;

    #[test]
    fn batch_equals_accumulated_singletons() {
        // One step over a 2-sequence batch == manual average of the two
        // per-sequence gradients fed through the same optimizer.
        let base = tiny(6, 8, 2, 2, 16, 1);
        let s0 = [1usize, 2, 3, 4];
        let s1 = [5usize, 0, 1, 2];

        let mut a = LlamaTrainer::new(base_clone(&base)).unwrap();
        a.set_lr(0.05);
        a.train_batch(&[&s0, &s1]).unwrap();

        // Reference: build the averaged grad by hand then step once.
        let mut b = LlamaTrainer::new(base_clone(&base)).unwrap();
        b.set_lr(0.05);
        let (_, g0) = seq_grad_flat(&b.model, &s0, 0.0, 0).unwrap();
        let (_, g1) = seq_grad_flat(&b.model, &s1, 0.0, 0).unwrap();
        let avg: Vec<f32> = g0.iter().zip(&g1).map(|(x, y)| 0.5 * (x + y)).collect();
        b.grads.load_flat(&avg);
        b.apply_step();

        for (x, y) in a
            .model
            .lm_head
            .weight
            .data
            .iter()
            .zip(&b.model.lm_head.weight.data)
        {
            assert!((x - y).abs() < 1e-6, "batch != hand-averaged: {x} vs {y}");
        }
    }

    #[test]
    fn finetune_epoch_reduces_loss() {
        let mut tr = LlamaTrainer::new(tiny(5, 16, 4, 2, 32, 2)).unwrap();
        tr.set_lr(0.01);
        let tokens: Vec<usize> = (0..120).map(|i| i % 5).collect();
        let mut rng = Rng::new(7);
        let first = tr.finetune_epoch(&tokens, 8, 8, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..15 {
            last = tr.finetune_epoch(&tokens, 8, 8, &mut rng).unwrap();
        }
        assert!(
            last < first * 0.7,
            "epoch training did not reduce loss: {first:.4} → {last:.4}"
        );
    }

    #[test]
    fn threaded_one_shard_matches_serial_bitwise() {
        let base = tiny(6, 8, 2, 2, 16, 1);
        let tokens: Vec<usize> = (0..80).map(|i| (i * 7) % 6).collect();

        let mut serial = LlamaTrainer::new(base_clone(&base)).unwrap();
        serial.set_lr(0.02);
        let mut rs = Rng::new(99);
        let ls = serial
            .finetune_epoch_threaded(&tokens, 6, 8, &mut rs, 1)
            .unwrap();

        let mut threaded = LlamaTrainer::new(base_clone(&base)).unwrap();
        threaded.set_lr(0.02);
        let mut rt = Rng::new(99);
        let lt = threaded
            .finetune_epoch_threaded(&tokens, 6, 8, &mut rt, 1)
            .unwrap();

        assert_eq!(ls, lt);
        for (x, y) in serial.model_flat().iter().zip(&threaded.model_flat()) {
            assert_eq!(x, y, "single-shard threaded diverged from serial");
        }
    }

    #[test]
    fn threaded_multishard_matches_serial_closely() {
        let base = tiny(6, 8, 2, 2, 16, 1);
        let tokens: Vec<usize> = (0..80).map(|i| (i * 7) % 6).collect();

        let mut serial = LlamaTrainer::new(base_clone(&base)).unwrap();
        serial.set_lr(0.02);
        let mut rs = Rng::new(1);
        for _ in 0..3 {
            serial
                .finetune_epoch_threaded(&tokens, 6, 16, &mut rs, 1)
                .unwrap();
        }
        let mut par = LlamaTrainer::new(base_clone(&base)).unwrap();
        par.set_lr(0.02);
        let mut rp = Rng::new(1);
        for _ in 0..3 {
            par.finetune_epoch_threaded(&tokens, 6, 16, &mut rp, 4)
                .unwrap();
        }
        for (x, y) in serial.model_flat().iter().zip(&par.model_flat()) {
            assert!((x - y).abs() < 1e-4, "multishard drifted: {x} vs {y}");
        }
    }
}
