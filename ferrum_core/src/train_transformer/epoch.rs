//! Serial training epochs and step-budget training.
use crate::error::{InferError, Result};
use crate::optim::Adam;
use crate::rng::Rng;
#[cfg(not(target_arch = "wasm32"))]
use crate::train_transformer::threaded::epoch_sharded;
use crate::train_transformer::TransformerNet;
use crate::verbose;

impl TransformerNet {
    /// Training-window start positions for one epoch over `n_tokens`, in a
    /// fresh random order. With stride > 1 the grid gets a random phase each
    /// epoch, so window boundaries differ across epochs. Stride 1 draws exactly
    /// as before (one shuffle), keeping existing runs reproducible.
    pub(super) fn epoch_window_starts(&self, n_tokens: usize, rng: &mut Rng) -> Result<Vec<usize>> {
        let t = self.context_len;
        if n_tokens < t + 1 {
            return Err(InferError::DimMismatch(format!(
                "need at least context_len+1 = {} tokens, got {}",
                t + 1,
                n_tokens
            )));
        }
        let last = n_tokens - t - 1; // largest valid start
        let s = self.window_stride;
        let phase = if s > 1 {
            (rng.next_u64() % s as u64) as usize % (last + 1)
        } else {
            0
        };
        let count = (last - phase) / s + 1;
        Ok(rng
            .shuffled_indices(count)
            .into_iter()
            .map(|i| phase + i * s)
            .collect())
    }
}

/// Aborts a run whose loss went NaN or infinite, before the update that would
/// write garbage into the weights. `step` is 1-based.
pub(super) fn check_loss_finite(loss: f32, step: u64) -> Result<()> {
    if loss.is_finite() {
        Ok(())
    } else {
        Err(InferError::Diverged(format!(
            "the loss became {loss} at step {step}; lower --lr or set --clip"
        )))
    }
}

/// One epoch of minibatch Adam over a token stream. Each example is a window
/// of `context_len` tokens with next-token targets at every position.
/// Returns the mean train loss.
///
/// If the net has QAT enabled ([`TransformerNet::set_qat`]), each step runs
/// forward and backward against int8-snapped weights, then applies the Adam
/// update to the full-precision master weights (straight-through estimator),
/// so the trained model is robust to int8 export.
pub fn train_transformer_epoch(
    net: &mut TransformerNet,
    tokens: &[usize],
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
) -> Result<f32> {
    train_transformer_steps(
        net,
        tokens,
        batch_size,
        adam,
        rng,
        1,
        usize::MAX,
        &mut |_, _| Ok(true),
    )
}

/// Callback run after every optimizer step of [`train_transformer_steps`],
/// with the updated net and the RNG (e.g. to write a checkpoint). It returns
/// `Ok(true)` to keep going or `Ok(false)` to end the epoch after this step.
pub type StepHook<'a> = dyn FnMut(&TransformerNet, &Rng) -> Result<bool> + 'a;

/// One epoch (see [`train_transformer_epoch_threaded`](super::train_transformer_epoch_threaded)) that stops after at
/// most `max_steps` optimizer steps and calls `on_step` after each one, so a
/// driver can enforce a token/step budget mid-epoch and checkpoint as it goes.
/// Returns the mean loss over the steps actually run (fewer if `on_step` ends
/// the epoch early).
#[allow(clippy::too_many_arguments)]
pub fn train_transformer_steps(
    net: &mut TransformerNet,
    tokens: &[usize],
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
    threads: usize,
    max_steps: usize,
    on_step: &mut StepHook,
) -> Result<f32> {
    // One shard (or a single-sequence batch, or wasm32) is the serial path.
    #[cfg(not(target_arch = "wasm32"))]
    if threads.max(1).min(batch_size.max(1)) > 1 {
        return epoch_sharded(
            net, tokens, batch_size, adam, rng, threads, max_steps, on_step,
        );
    }
    let _ = threads;
    epoch_serial(net, tokens, batch_size, adam, rng, max_steps, on_step)
}

fn epoch_serial(
    net: &mut TransformerNet,
    tokens: &[usize],
    batch_size: usize,
    adam: &Adam,
    rng: &mut Rng,
    max_steps: usize,
    on_step: &mut StepHook,
) -> Result<f32> {
    use crate::loss::softmax_cross_entropy;
    let t = net.context_len();
    // Shuffle the epoch's windows once and draw minibatches without
    // replacement (T4), so an epoch visits every window exactly once.
    let perm = net.epoch_window_starts(tokens.len(), rng)?;
    let num_windows = perm.len();
    let steps = num_windows.div_ceil(batch_size).min(max_steps);
    let mut total = 0.0f32;
    let mut done = 0usize;
    for step in 0..steps {
        let batch = &perm[step * batch_size..((step + 1) * batch_size).min(num_windows)];
        let mut input = Vec::with_capacity(batch.len() * t);
        let mut targets = Vec::with_capacity(batch.len() * t);
        for &start in batch {
            input.extend_from_slice(&tokens[start..start + t]);
            targets.extend_from_slice(&tokens[start + 1..start + t + 1]);
        }
        // Weight tying: mirror tok_embᵀ into the head before the forward (T9).
        net.sync_tied_head();
        // QAT: gradients are computed at the int8-snapped weights, but the
        // optimizer updates the fp32 masters (straight-through estimator).
        let masters = if net.qat_enabled() {
            let snapshot = net.snapshot_weights();
            net.fake_quantize_weights();
            Some(snapshot)
        } else {
            None
        };
        // FFN dropout (T7): draw a per-step mask seed only when enabled, so the
        // RNG stream (and thus reproducibility) is untouched when dropout is off.
        let dropout = net.dropout;
        let drop_seed = if dropout > 0.0 { rng.next_u64() } else { 0 };
        let (logits, cache) = net.forward_train(&input, dropout, drop_seed)?;
        let (loss, dlogits) = softmax_cross_entropy(&logits, &targets)?;
        check_loss_finite(loss, net.step_count() + 1)?;
        net.zero_grad();
        net.backward(&dlogits, &cache)?;
        if let Some(snapshot) = &masters {
            net.restore_weights(snapshot);
        }
        // Sum the head gradient back into the shared embedding (T9).
        net.fold_tied_head_grad();
        if let Some(max_norm) = net.grad_clip {
            net.clip_grad_norm(max_norm);
        }
        net.step(adam)?;
        total += loss;
        done += 1;
        if !on_step(net, rng)? {
            break;
        }
    }
    let avg = total / done.max(1) as f32;
    if verbose::is_verbose() {
        vprintln!(
            "[train_transformer::epoch] steps={}, mean loss={:.6}",
            done,
            avg
        );
    }
    Ok(avg)
}

#[cfg(test)]
mod tests {
    use super::train_transformer_epoch;
    use crate::optim::Adam;
    use crate::rng::Rng;
    use crate::train_transformer::TransformerNet;

    #[test]
    fn check_loss_finite_rejects_nan_and_infinity() {
        assert!(super::check_loss_finite(1.5, 1).is_ok());
        for bad in [f32::NAN, f32::INFINITY] {
            let e = super::check_loss_finite(bad, 7).unwrap_err();
            assert!(matches!(e, crate::error::InferError::Diverged(_)), "{e:?}");
            assert!(e.to_string().contains("step 7"), "{e}");
        }
    }

    /// An absurd learning rate blows the weights up; training must stop with an
    /// error instead of carrying on with NaN weights.
    #[test]
    fn diverging_training_returns_an_error() {
        let mut rng0 = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..200).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(1e30);
        let mut rng = Rng::new(99);
        let mut result = Ok(0.0);
        for _ in 0..5 {
            result = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng);
            if result.is_err() {
                break;
            }
        }
        let e = result.expect_err("lr 1e30 must diverge");
        assert!(matches!(e, crate::error::InferError::Diverged(_)), "{e:?}");
    }

    #[test]
    fn training_reduces_loss() {
        let mut rng = Rng::new(7);
        // vocab=4, T=4, C=8, heads=2, hidden=16, 1 block
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng).unwrap();
        // A perfectly periodic token stream — trivially learnable.
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        let first = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..30 {
            last = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        assert!(
            last < first * 0.5,
            "loss did not halve: {first:.4} → {last:.4}"
        );
    }

    #[test]
    fn training_with_dropout_reduces_loss() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_dropout(0.2);
        assert!((net.dropout() - 0.2).abs() < 1e-6);
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        let first = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..40 {
            last = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        assert!(
            last < first * 0.7,
            "dropout training did not reduce loss: {first:.4} → {last:.4}"
        );
    }

    /// A true epoch covers every window exactly once: with `batch_size = 1` it
    /// takes exactly `num_windows` optimizer steps, and with a batch larger than
    /// the corpus exactly one step — never sampling with replacement.
    #[test]
    fn epoch_makes_one_pass_over_all_windows() {
        let mut rng0 = Rng::new(7);
        let mut net = TransformerNet::new(5, 4, 8, 2, 16, 1, &mut rng0).unwrap();
        let tokens: Vec<usize> = (0..50).map(|i| i % 5).collect();
        let num_windows = (tokens.len() - 4) as u64;
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(1);

        let before = net.step_count();
        train_transformer_epoch(&mut net, &tokens, 1, &adam, &mut rng).unwrap();
        assert_eq!(
            net.step_count() - before,
            num_windows,
            "batch_size=1 must take exactly num_windows steps"
        );

        let before = net.step_count();
        train_transformer_epoch(&mut net, &tokens, 10_000, &adam, &mut rng).unwrap();
        assert_eq!(
            net.step_count() - before,
            1,
            "an over-sized batch must take exactly one full-corpus step"
        );
    }

    #[test]
    fn epochs_over_too_few_tokens_are_an_error() {
        use crate::optim::Adam;
        use crate::rng::Rng;
        use crate::train_transformer::{train_transformer_epoch, TransformerNet};
        let mut net = TransformerNet::new(10, 4, 8, 2, 16, 1, &mut Rng::new(1)).unwrap();
        let tokens: Vec<usize> = (0..4).collect(); // needs context_len + 1 = 5
        let e = train_transformer_epoch(&mut net, &tokens, 2, &Adam::new(0.01), &mut Rng::new(2))
            .unwrap_err();
        assert!(e.to_string().contains("context_len+1"), "{e}");
    }
}
