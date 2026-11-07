//! Data-parallel (sharded) training epochs with a deterministic gradient reduction.
use crate::error::Result;
use crate::optim::Adam;
use crate::rng::Rng;
use crate::train_transformer::epoch::train_transformer_steps;
#[cfg(not(target_arch = "wasm32"))]
use crate::train_transformer::epoch::{check_loss_finite, StepHook};
#[cfg(not(target_arch = "wasm32"))]
use crate::train_transformer::source::TokenSource;
use crate::train_transformer::TransformerNet;
#[cfg(not(target_arch = "wasm32"))]
use crate::verbose;

/// Data-parallel version of [`train_transformer_epoch`](super::train_transformer_epoch): each minibatch is split
/// into up to `threads` shards processed concurrently on separate OS threads,
/// then their gradients are summed and a single Adam update is applied.
///
/// Parallelism is **across training examples** (sequences), complementing the
/// per-matmul row parallelism in [`crate::ops::matmul`]. It is built only on
/// `std::thread::scope` — no external crates, no `unsafe`. Each shard clones the
/// network (cheap relative to forward/backward for non-trivial models), computes
/// gradients over its sequences against the shared weights, and the partial
/// gradients are reduced in a **fixed shard order**, so a run is reproducible for
/// a given `threads` value. With `threads <= 1` (or a single sequence) this is
/// bit-for-bit identical to [`train_transformer_epoch`](super::train_transformer_epoch); with more threads the
/// only differences are the floating-point regrouping of the gradient sum.
///
/// The RNG is drawn exactly as in the serial path (the whole minibatch is
/// sampled up front, then sharded), so the sequence of training windows does not
/// depend on the thread count. QAT semantics are preserved: weights are
/// int8-snapped on the shared master before the shards fork, and the fp32
/// masters are restored before the optimizer step.
///
/// On `wasm32` (no threads) this transparently runs the serial path.
pub fn train_transformer_epoch_threaded(
    net: &mut TransformerNet,
    tokens: &[usize],
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
    threads: usize,
) -> Result<f32> {
    train_transformer_steps(
        net,
        tokens,
        batch_size,
        adam,
        rng,
        threads,
        usize::MAX,
        &mut |_, _| Ok(true),
    )
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn epoch_sharded<S: TokenSource>(
    net: &mut TransformerNet,
    tokens: &mut S,
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
    threads: usize,
    max_steps: usize,
    replay: Option<(Rng, usize)>,
    on_step: &mut StepHook,
) -> Result<f32> {
    {
        use crate::loss::softmax_cross_entropy;
        let nshards = threads.max(1).min(batch_size.max(1));
        let t = net.context_len();
        // Same per-epoch shuffle as the serial path (T4): identical RNG use, so
        // a fixed seed gives the same minibatches regardless of thread count.
        let (perm, skip) = net.epoch_plan(tokens.len(), rng, replay)?;
        let num_windows = perm.len();
        let steps = num_windows
            .div_ceil(batch_size)
            .min(skip.saturating_add(max_steps));
        let mut total = 0.0f32;
        let mut done = 0usize;

        for step in skip..steps {
            // The step's minibatch is the next slice of the permutation (the
            // final batch may be smaller than `batch_size`).
            let batch = &perm[step * batch_size..((step + 1) * batch_size).min(num_windows)];
            let cur_bs = batch.len();
            let total_rows = (cur_bs * t) as f32;
            let mut input = Vec::with_capacity(cur_bs * t);
            let mut targets = Vec::with_capacity(cur_bs * t);
            for &start in batch {
                tokens.cut(start, t, &mut input, &mut targets)?;
            }

            // Weight tying: mirror tok_embᵀ into the shared head before forking,
            // so every shard's forward sees the tied head (T9).
            net.sync_tied_head();
            // QAT: snap the shared master weights before forking the shards, so
            // every shard's forward/backward runs at the int8 grid.
            let masters = if net.qat_enabled() {
                let snapshot = net.snapshot_weights();
                net.fake_quantize_weights();
                Some(snapshot)
            } else {
                None
            };

            // FFN dropout (T7): one base mask seed per step, offset per shard so
            // each shard's examples get independent masks. RNG is untouched off.
            let dropout = net.dropout;
            let base_seed = if dropout > 0.0 { rng.next_u64() } else { 0 };

            let per = cur_bs.div_ceil(nshards);
            let net_ref: &TransformerNet = net;
            let input_ref = &input;
            let targets_ref = &targets;

            // Each shard returns its weighted (loss, flat-gradients). Weighting by
            // shard_rows / total_rows makes the summed result equal the serial
            // gradient, which normalises by the full batch's row count.
            let results: Vec<Result<(f32, Vec<f32>)>> = std::thread::scope(|scope| {
                let mut handles = Vec::new();
                let mut w0 = 0usize;
                while w0 < cur_bs {
                    let w1 = (w0 + per).min(cur_bs);
                    handles.push(scope.spawn(move || -> Result<(f32, Vec<f32>)> {
                        // Share the master weights read-only (no clone): forward
                        // and backward_grads borrow `net_ref` and allocate only a
                        // flat gradient buffer (≈4 B/param) instead of a full
                        // network copy (weights+grad+Adam moments ≈16 B/param).
                        let sub_in = &input_ref[w0 * t..w1 * t];
                        let sub_tg = &targets_ref[w0 * t..w1 * t];
                        let (logits, cache) = net_ref.forward_train(
                            sub_in,
                            dropout,
                            base_seed.wrapping_add(w0 as u64 + 1),
                        )?;
                        let (loss, dlogits) = softmax_cross_entropy(&logits, sub_tg)?;
                        let shard_rows = ((w1 - w0) * t) as f32;
                        let weight = shard_rows / total_rows;
                        let mut g = net_ref.backward_grads(&dlogits, &cache)?;
                        for x in &mut g {
                            *x *= weight;
                        }
                        Ok((loss * weight, g))
                    }));
                    w0 = w1;
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("ferrum: training worker thread panicked"))
                    .collect()
            });

            // Reduce in deterministic shard order (independent of scheduling).
            let mut summed: Option<Vec<f32>> = None;
            let mut step_loss = 0.0f32;
            for r in results {
                let (loss, g) = r?;
                step_loss += loss;
                match &mut summed {
                    None => summed = Some(g),
                    Some(acc) => {
                        for (a, b) in acc.iter_mut().zip(&g) {
                            *a += b;
                        }
                    }
                }
            }
            let summed = summed.expect("at least one shard always runs");
            check_loss_finite(step_loss, net.step_count() + 1)?;

            if let Some(snapshot) = &masters {
                net.restore_weights(snapshot);
            }
            net.load_grads(&summed);
            // Sum the head gradient back into the shared embedding (T9).
            net.fold_tied_head_grad();
            net.finish_grads(step_loss);
            net.step(adam)?;
            total += step_loss;
            done += 1;
            if !on_step(net, rng)? {
                break;
            }
        }

        let avg = total / done.max(1) as f32;
        if verbose::is_verbose() {
            vprintln!("[train_transformer::epoch_threaded] shards={nshards}, steps={done}, mean loss={avg:.6}");
        }
        Ok(avg)
    }
}

#[cfg(test)]
mod tests {
    use super::train_transformer_epoch_threaded;
    use crate::optim::Adam;
    use crate::rng::Rng;
    use crate::train_transformer::epoch::train_transformer_epoch;
    use crate::train_transformer::TransformerNet;

    /// The sharded path stops on a diverged loss too.
    #[test]
    fn threaded_early_stop_breaks_step_loop() {
        let mut rng0 = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(99);
        let mut steps = 0;
        let mut on_step = |_: &TransformerNet, _: &Rng| -> crate::error::Result<bool> {
            steps += 1;
            Ok(false)
        };
        let mut src = tokens.as_slice();
        let res = super::epoch_sharded(
            &mut net,
            &mut src,
            8,
            &adam,
            &mut rng,
            2,
            usize::MAX,
            None,
            &mut on_step,
        );
        assert!(res.is_ok());
        assert_eq!(steps, 1);
    }

    #[test]
    fn threaded_diverging_training_returns_an_error() {
        let mut rng0 = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(1e30);
        let mut rng = Rng::new(99);
        let e = (0..5)
            .map(|_| train_transformer_epoch_threaded(&mut net, &tokens, 8, &adam, &mut rng, 2))
            .find(Result::is_err)
            .expect("lr 1e30 must diverge")
            .unwrap_err();
        assert!(matches!(e, crate::error::InferError::Diverged(_)), "{e:?}");
    }

    /// A token outside the vocabulary fails the shard that sees it, and the epoch reports it.
    #[test]
    fn threaded_out_of_range_token_is_an_error() {
        let mut rng0 = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200)
            .map(|i| (i * 7) % 6 + (i == 100) as usize * 50)
            .collect();
        let r = train_transformer_epoch_threaded(
            &mut net,
            &tokens,
            8,
            &Adam::new(0.01),
            &mut Rng::new(1),
            2,
        );
        assert!(r.is_err());
    }

    /// `threads <= 1` must be the serial path, bit-for-bit.
    #[test]
    fn threaded_one_shard_matches_serial_bitwise() {
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.01);

        let mut serial = base.clone();
        let mut rs = Rng::new(99);
        let ls = train_transformer_epoch(&mut serial, &tokens, 8, &adam, &mut rs).unwrap();

        let mut threaded = base.clone();
        let mut rt = Rng::new(99);
        let lt =
            train_transformer_epoch_threaded(&mut threaded, &tokens, 8, &adam, &mut rt, 1).unwrap();

        assert_eq!(ls, lt, "loss differs with a single shard");
        assert_eq!(
            serial.snapshot_weights(),
            threaded.snapshot_weights(),
            "weights differ with a single shard"
        );
    }

    /// Multi-shard data-parallel must match the serial gradient/step to within
    /// floating-point regrouping error after one step (batch = all windows).
    #[test]
    fn threaded_multishard_matches_serial_one_step() {
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.05);
        let batch = tokens.len() - 4; // a single optimizer step over every window

        let mut serial = base.clone();
        let mut rs = Rng::new(1);
        let ls = train_transformer_epoch(&mut serial, &tokens, batch, &adam, &mut rs).unwrap();

        let mut par = base.clone();
        let mut rp = Rng::new(1);
        let lp =
            train_transformer_epoch_threaded(&mut par, &tokens, batch, &adam, &mut rp, 4).unwrap();

        assert!(
            (ls - lp).abs() < 1e-4,
            "loss diverged: serial {ls} vs threaded {lp}"
        );
        let sw = serial.snapshot_weights();
        let pw = par.snapshot_weights();
        for (a, b) in sw.iter().zip(&pw) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-4, "weight diverged: {x} vs {y}");
            }
        }
    }

    /// The QAT path must also behave identically for a single shard.
    #[test]
    fn threaded_qat_one_shard_matches_serial() {
        let mut rng0 = Rng::new(13);
        let mut base = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        base.set_qat(true);
        let tokens: Vec<usize> = (0..200).map(|i| (i * 3) % 6).collect();
        let adam = Adam::new(0.01);

        let mut serial = base.clone();
        let mut rs = Rng::new(5);
        for _ in 0..3 {
            train_transformer_epoch(&mut serial, &tokens, 8, &adam, &mut rs).unwrap();
        }
        let mut threaded = base.clone();
        let mut rt = Rng::new(5);
        for _ in 0..3 {
            train_transformer_epoch_threaded(&mut threaded, &tokens, 8, &adam, &mut rt, 1).unwrap();
        }
        assert_eq!(serial.snapshot_weights(), threaded.snapshot_weights());
    }

    /// QAT and gradient clipping must also run (and stay finite, restoring the
    /// f32 masters) on the multi-shard path.
    #[test]
    fn threaded_multishard_qat_and_clip_train_and_keep_masters() {
        let mut rng0 = Rng::new(21);
        let mut net = TransformerNet::new(4, 32, 32, 2, 32, 1, &mut rng0).unwrap();
        net.set_qat(true);
        net.set_grad_clip(Some(0.05));
        let tokens: Vec<usize> = (0..200).map(|i| (i * 3) % 4).collect();
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(9);
        let before = net.snapshot_weights();
        let first =
            train_transformer_epoch_threaded(&mut net, &tokens, 16, &adam, &mut rng, 3).unwrap();
        let mut last = first;
        for _ in 0..5 {
            last = train_transformer_epoch_threaded(&mut net, &tokens, 16, &adam, &mut rng, 3)
                .unwrap();
        }
        assert!(first.is_finite() && last < first, "{first} -> {last}");
        let after = net.snapshot_weights();
        assert!(after.iter().flatten().all(|v| v.is_finite()));
        assert_ne!(before, after);
        // masters are restored after each step: they are not sitting on the Q8_0 grid
        let mut snapped = after.clone();
        net.fake_quantize_weights();
        let grid = net.snapshot_weights();
        assert_ne!(
            snapped.remove(0),
            grid[0],
            "f32 masters must not be left snapped"
        );
    }

    /// Data-parallel training reduces loss like the serial path.
    #[test]
    fn threaded_multishard_reduces_loss() {
        let mut rng0 = Rng::new(7);
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..400).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(123);
        let first =
            train_transformer_epoch_threaded(&mut net, &tokens, 16, &adam, &mut rng, 4).unwrap();
        let mut last = first;
        for _ in 0..30 {
            last = train_transformer_epoch_threaded(&mut net, &tokens, 16, &adam, &mut rng, 4)
                .unwrap();
        }
        assert!(
            last < first * 0.5,
            "threaded loss did not halve: {first:.4} → {last:.4}"
        );
    }

    /// Tied training is still bit-identical across the serial and single-shard
    /// threaded paths (the sync/fold hooks run in both).
    #[test]
    fn tied_threaded_one_shard_matches_serial() {
        let mut rng0 = Rng::new(7);
        let mut base = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        base.set_weight_tying(true);
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.01);
        let mut serial = base.clone();
        let mut rs = Rng::new(9);
        let mut threaded = base.clone();
        let mut rt = Rng::new(9);
        for _ in 0..3 {
            train_transformer_epoch(&mut serial, &tokens, 8, &adam, &mut rs).unwrap();
            train_transformer_epoch_threaded(&mut threaded, &tokens, 8, &adam, &mut rt, 1).unwrap();
        }
        assert_eq!(serial.snapshot_weights(), threaded.snapshot_weights());
    }

    /// The shared-weight (no-clone) data-parallel path stays numerically in
    /// step with the serial path across several epochs — the memory fix changes
    /// only allocation, not the math.
    #[test]
    fn threaded_shared_weights_track_serial_over_epochs() {
        let mut rng0 = Rng::new(7);
        let base = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.02);
        let batch = tokens.len() - 4; // one optimizer step per epoch (all windows)

        let mut serial = base.clone();
        let mut rs = Rng::new(3);
        let mut par = base.clone();
        let mut rp = Rng::new(3);
        for _ in 0..5 {
            train_transformer_epoch(&mut serial, &tokens, batch, &adam, &mut rs).unwrap();
            train_transformer_epoch_threaded(&mut par, &tokens, batch, &adam, &mut rp, 4).unwrap();
        }
        let sw = serial.snapshot_weights();
        let pw = par.snapshot_weights();
        for (a, b) in sw.iter().zip(&pw) {
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x - y).abs() < 1e-3,
                    "shared-weight path drifted from serial: {x} vs {y}"
                );
            }
        }
    }

    /// For a fixed `threads` value, training is reproducible regardless of how
    /// the OS schedules the worker threads (deterministic shard reduction).
    #[test]
    fn threaded_training_is_reproducible() {
        let mut rng0 = Rng::new(5);
        let base = TransformerNet::new(5, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..300).map(|i| (i * 11) % 5).collect();
        let adam = Adam::new(0.01);
        let run = || {
            let mut n = base.clone();
            let mut r = Rng::new(321);
            for _ in 0..10 {
                train_transformer_epoch_threaded(&mut n, &tokens, 12, &adam, &mut r, 4).unwrap();
            }
            n.snapshot_weights()
        };
        assert_eq!(run(), run(), "threaded training is not reproducible");
    }
}
