//! The epoch/step training loop with resume, validation and checkpointing.
use crate::error::{InferError, Result};
use crate::optim::Adam;
use crate::rng::Rng;
use crate::slm::config::TransformerConfig;
use crate::slm::memory::{available_memory_bytes, memory_gate};
use crate::slm::metrics_log::{format_eta, CsvLog, Row};
use crate::train_transformer::{steps_over, TokenSource, TransformerNet};

/// A fresh (or checkpoint-resumed) training net configured from `cfg`.
pub(super) fn prepare_net(
    cfg: &TransformerConfig,
    vocab: usize,
    rng: &mut Rng,
) -> Result<TransformerNet> {
    let ckpt = std::path::Path::new(&cfg.checkpoint_path);
    let mut net = if cfg.resume && !cfg.checkpoint_path.is_empty() && ckpt.exists() {
        let (net, ckpt_rng) = TransformerNet::load_checkpoint_from_vec(std::fs::read(ckpt)?)?;
        if net.vocab_size() != vocab || net.context_len() != cfg.context_len {
            return Err(InferError::DimMismatch(format!(
                "checkpoint has vocab {} / context {}, this run needs vocab {vocab} / context {}",
                net.vocab_size(),
                net.context_len(),
                cfg.context_len
            )));
        }
        *rng = ckpt_rng;
        vprintln!(
            "[slm::train] resumed {} at step {}",
            ckpt.display(),
            net.step_count()
        );
        net
    } else {
        TransformerNet::new(
            vocab,
            cfg.context_len,
            cfg.embed_dim,
            cfg.num_heads,
            cfg.hidden_dim,
            cfg.num_blocks,
            rng,
        )?
    };
    net.set_qat(cfg.qat);
    net.set_weight_decay(cfg.weight_decay);
    net.set_dropout(cfg.dropout);
    net.set_weight_tying(cfg.weight_tying);
    net.set_grad_clip((cfg.grad_clip > 0.0).then_some(cfg.grad_clip));
    net.set_window_stride(if cfg.window_stride == 0 {
        cfg.context_len
    } else {
        cfg.window_stride
    });
    vprintln!(
        "[slm::train] {} params, lr={}, wd={}, dropout={}, qat={}, stride={}, tied={}",
        net.num_params(),
        cfg.lr,
        cfg.weight_decay,
        cfg.dropout,
        cfg.qat,
        net.window_stride(),
        cfg.weight_tying
    );
    Ok(net)
}

/// The throttled progress line printed every `log_every_steps` steps. The ETA assumes the
/// steps so far are typical of the remaining ones.
fn progress_line(
    step: u64,
    total_steps: u64,
    steps_done: u64,
    tokens_done: f64,
    secs: f64,
    grad_norm: Option<f32>,
) -> String {
    let rate = tokens_done / secs.max(1e-9);
    let eta = secs / steps_done.max(1) as f64 * total_steps.saturating_sub(step) as f64;
    let norm = grad_norm.map_or(String::new(), |n| format!("  grad_norm {n:.3}"));
    format!(
        "[train] step {step}/{total_steps}  tokens/s {rate:.0}{norm}  elapsed {secs:.0}s  eta {}",
        format_eta(eta)
    )
}

/// Write a training checkpoint atomically (temp file + rename), so a crash
/// mid-write never leaves a truncated checkpoint behind.
fn write_checkpoint(path: &str, net: &TransformerNet, rng: &Rng) -> Result<()> {
    crate::fs_atomic::atomic_write(path, &net.save_checkpoint(rng))
}

/// The training loop shared by every config-driven trainer: runs `cfg.epochs`
/// epochs, or until `cfg.max_tokens` is spent, checkpointing as configured.
/// `on_epoch(net, epoch, mean_loss)` runs after each (possibly partial) epoch
/// and returns `false` to stop early.
pub(super) fn run_training<S, F>(
    net: &mut TransformerNet,
    tokens: &mut S,
    cfg: &TransformerConfig,
    threads: usize,
    rng: &mut Rng,
    mut on_epoch: F,
) -> Result<()>
where
    S: TokenSource,
    F: FnMut(&TransformerNet, usize, f32) -> Result<bool>,
{
    let t = net.context_len();
    let batch = cfg.batch_size.max(1);
    let windows = tokens.len().saturating_sub(t + 1) / net.window_stride() + 1;
    memory_gate(
        cfg.skip_memory_check,
        net.train_peak_bytes(windows, batch, threads.max(1).min(batch)),
        available_memory_bytes(),
    )?;
    let steps_per_epoch = windows.div_ceil(batch).max(1) as u64;
    let budgeted = cfg.max_tokens > 0;
    let total_steps = if budgeted {
        cfg.max_tokens.div_ceil((batch * t) as u64)
    } else {
        cfg.epochs as u64 * steps_per_epoch
    };
    if cfg.cosine_lr {
        net.set_lr_schedule(Some(crate::optim::LrSchedule::warmup_cosine(
            cfg.lr,
            cfg.warmup_steps,
            total_steps,
        )));
    }
    let adam = Adam::new(cfg.lr);
    cfg.stop.clear_record();
    let ckpt = &cfg.checkpoint_path;
    let every = cfg.checkpoint_every;
    let t0 = std::time::Instant::now();
    let start_step = net.step_count();
    let log_every = cfg.log_every_steps;
    let tokens_per_step = (batch * t) as f64;
    let halted = std::cell::Cell::new(false);
    let epoch = std::cell::Cell::new(0usize);
    let mut csv = (!cfg.metrics_csv.is_empty())
        .then(|| CsvLog::open(&cfg.metrics_csv, cfg.resume))
        .transpose()?;
    // The gradient norm costs a pass over the gradients: only measure it when it is shown.
    net.set_track_grad_norm(log_every > 0 || csv.is_some());
    let mut on_step = |n: &TransformerNet, r: &Rng| -> Result<bool> {
        let m = n.last_step_metrics();
        if let (Some(log), Some(m)) = (csv.as_mut(), m) {
            let step = n.step_count();
            log.write(&Row {
                step,
                epoch: epoch.get(),
                loss: m.loss,
                lr: n.lr_schedule().map_or(cfg.lr, |s| s.lr_at(step)),
                grad_norm: m.grad_norm,
                tokens: tokens_per_step,
                elapsed: t0.elapsed().as_secs_f64(),
            })?;
        }
        if !ckpt.is_empty() && every > 0 && n.step_count().is_multiple_of(every) {
            write_checkpoint(ckpt, n, r)?;
        }
        if log_every > 0 && n.step_count().is_multiple_of(log_every) {
            let done = (n.step_count() - start_step) as f64 * tokens_per_step;
            println!(
                "{}",
                progress_line(
                    n.step_count(),
                    total_steps,
                    n.step_count() - start_step,
                    done,
                    t0.elapsed().as_secs_f64(),
                    m.and_then(|m| m.grad_norm),
                )
            );
        }
        // A stop request ends the run after this step, unless it was the last one.
        if n.step_count() < total_steps {
            if let Some(cause) = cfg.stop.should_stop() {
                cfg.stop.note_stop(cause, n.step_count());
                halted.set(true);
                return Ok(false);
            }
        }
        Ok(true)
    };
    // ponytail: a resumed run reshuffles its partially-done epoch (so the
    // remaining steps see a fresh permutation); exact mid-epoch replay would
    // need the permutation in the checkpoint. The epoch still ends where an
    // uninterrupted run's would.
    let mut ep = (net.step_count() / steps_per_epoch) as usize;
    let mut partial = net.step_count() % steps_per_epoch; // steps already done in this epoch
    loop {
        let remaining = total_steps.saturating_sub(net.step_count());
        if (budgeted && remaining == 0) || (!budgeted && ep >= cfg.epochs) {
            break;
        }
        ep += 1;
        epoch.set(ep);
        let epoch_left = (steps_per_epoch - partial) as usize;
        partial = 0;
        let max_steps = if budgeted {
            (remaining as usize).min(epoch_left)
        } else {
            epoch_left
        };
        let loss = steps_over(
            net,
            tokens,
            batch,
            &adam,
            rng,
            threads,
            max_steps,
            &mut on_step,
        )?;
        vprintln!(
            "[slm::train] epoch {ep}: step {}/{total_steps}, loss={loss:.6}",
            net.step_count()
        );
        // After a stop the (partial) epoch is still reported, so a validated run
        // keeps its best-so-far model; then the final checkpoint below is written.
        if !on_epoch(net, ep, loss)? || halted.get() {
            break;
        }
    }
    if !ckpt.is_empty() {
        write_checkpoint(ckpt, net, rng)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::progress_line;
    use crate::rng::Rng;
    use crate::slm::config::{TransformerConfig, ValidationConfig};
    use crate::slm::test_support::{epoch_ends, resume_cfg, scratch_ckpt, RESUME_CORPUS};
    use crate::slm::GenerativeSLM;
    use crate::train_transformer::TransformerNet;

    #[test]
    fn resume_mid_epoch_finishes_the_partial_epoch() {
        let ckpt = scratch_ckpt("resume");
        let corpus = RESUME_CORPUS.repeat(40);
        let base = resume_cfg(&ckpt);
        // Uninterrupted run: record the step count at the end of every epoch.
        let full = epoch_ends(&corpus, &base);
        assert_eq!(full.len(), 3);
        let spe = full[0];
        assert!(spe > 5 && full == vec![spe, 2 * spe, 3 * spe], "{full:?}");
        // Interrupted run: train only 5 steps, then resume to 3 epochs.
        let _ = std::fs::remove_file(&ckpt);
        let short = TransformerConfig {
            max_tokens: 5 * 4 * 8,
            ..base.clone()
        };
        assert_eq!(epoch_ends(&corpus, &short), vec![5]);
        let resumed = TransformerConfig {
            resume: true,
            ..base.clone()
        };
        // The partial epoch is finished first: epoch boundaries line up.
        assert_eq!(epoch_ends(&corpus, &resumed), full);

        // Through the public API the epochs are numbered 1..=3 as well.
        let _ = std::fs::remove_file(&ckpt);
        GenerativeSLM::train_transformer_config(
            &corpus,
            &TransformerConfig {
                max_tokens: 5 * 4 * 8,
                ..base.clone()
            },
            &mut Rng::new(1),
            |_, _| {},
        )
        .unwrap();
        let mut epochs_seen = Vec::new();
        GenerativeSLM::train_transformer_config(&corpus, &resumed, &mut Rng::new(1), |ep, _| {
            epochs_seen.push(ep)
        })
        .unwrap();
        assert_eq!(epochs_seen, vec![1, 2, 3]);
        let (net, _) = TransformerNet::load_checkpoint(&std::fs::read(&ckpt).unwrap()).unwrap();
        assert_eq!(
            net.step_count(),
            3 * spe,
            "resumed run must end on the same step"
        );
    }

    #[test]
    fn resume_mid_epoch_with_a_token_budget_keeps_epoch_boundaries() {
        let ckpt = scratch_ckpt("resume_budget");
        let corpus = RESUME_CORPUS.repeat(40);
        let base = resume_cfg(&ckpt);
        let spe = epoch_ends(&corpus, &base)[0];
        let _ = std::fs::remove_file(&ckpt);
        let short = TransformerConfig {
            max_tokens: 5 * 4 * 8,
            ..base.clone()
        };
        epoch_ends(&corpus, &short);
        // Budget of 2 epochs + 3 steps, resumed 5 steps in: finishes epoch 1,
        // runs epoch 2, then stops 3 steps into epoch 3.
        let resumed = TransformerConfig {
            resume: true,
            max_tokens: (2 * spe + 3) * 4 * 8,
            ..base
        };
        assert_eq!(
            epoch_ends(&corpus, &resumed),
            vec![spe, 2 * spe, 2 * spe + 3]
        );
    }

    #[test]
    fn validated_resume_starts_from_the_resumed_models_score() {
        let ckpt = scratch_ckpt("vresume");
        let corpus = RESUME_CORPUS.repeat(40);
        let cfg = TransformerConfig {
            epochs: 4,
            checkpoint_every: 0,
            ..resume_cfg(&ckpt)
        };
        let val = ValidationConfig {
            val_fraction: 0.2,
            patience: 0,
        };
        let mut first = Vec::new();
        GenerativeSLM::train_transformer_config_validated(
            &corpus,
            &cfg,
            1,
            &val,
            &mut Rng::new(1),
            |p| first.push(p.is_best),
        )
        .unwrap();
        assert!(first[0], "a fresh run's first epoch is the best so far");
        // Resume with 4 more epochs at lr 0 (no change): no epoch may claim
        // "best", because none improves on the resumed model's own score.
        let resumed = TransformerConfig {
            resume: true,
            epochs: 8,
            lr: 0.0,
            ..cfg
        };
        let mut bests = Vec::new();
        let slm = GenerativeSLM::train_transformer_config_validated(
            &corpus,
            &resumed,
            1,
            &val,
            &mut Rng::new(1),
            |p| bests.push(p.is_best),
        )
        .unwrap();
        assert_eq!(bests.len(), 4);
        assert!(bests.iter().all(|b| !b), "{bests:?}");
        // The resumed model itself is what comes back.
        assert!(slm.generate("the cat", 5, 0.5, &mut Rng::new(1)).is_ok());
        // Resuming a run that already finished trains nothing and still returns it.
        let done = TransformerConfig {
            epochs: 8,
            ..resumed
        };
        let mut calls = 0;
        GenerativeSLM::train_transformer_config_validated(
            &corpus,
            &done,
            1,
            &val,
            &mut Rng::new(1),
            |_| calls += 1,
        )
        .unwrap();
        assert_eq!(calls, 0);
    }

    #[test]
    fn a_stop_request_checkpoints_and_resume_finishes_the_run_exactly() {
        use crate::stop::{StopCause, StopCondition};
        let corpus = RESUME_CORPUS.repeat(40);
        // One step per epoch, so a stop after step 1 lands on an epoch boundary.
        let one_step = |ckpt: &str| TransformerConfig {
            batch_size: 4096,
            checkpoint_every: 0,
            ..resume_cfg(ckpt)
        };
        let full_ck = scratch_ckpt("stop_full");
        assert_eq!(epoch_ends(&corpus, &one_step(&full_ck)), vec![1, 2, 3]);

        let ck = scratch_ckpt("stop_part");
        let stop_file = std::path::PathBuf::from(&ck).with_extension("stop");
        std::fs::write(&stop_file, b"").unwrap();
        let stop = StopCondition::none().with_stop_file(&stop_file);
        let cfg = TransformerConfig {
            stop: stop.clone(),
            ..one_step(&ck)
        };
        assert_eq!(epoch_ends(&corpus, &cfg), vec![1]);
        assert_eq!(stop.stopped(), Some((StopCause::StopFile, 1)));
        let (net, _) = TransformerNet::load_checkpoint(&std::fs::read(&ck).unwrap()).unwrap();
        assert_eq!(net.step_count(), 1);

        // Resume without the stop request: the remaining epochs run and the
        // final checkpoint equals the uninterrupted run's, byte for byte.
        let resumed = TransformerConfig {
            resume: true,
            ..one_step(&ck)
        };
        assert_eq!(epoch_ends(&corpus, &resumed), vec![2, 3]);
        assert_eq!(
            std::fs::read(&ck).unwrap(),
            std::fs::read(&full_ck).unwrap()
        );
        std::fs::remove_file(&stop_file).unwrap();
    }

    #[test]
    fn a_stop_during_the_last_step_does_not_count_as_a_stop() {
        use crate::stop::StopCondition;
        let corpus = RESUME_CORPUS.repeat(40);
        let stop = StopCondition::none().with_deadline(std::time::Duration::ZERO);
        let cfg = TransformerConfig {
            epochs: 1,
            batch_size: 4096,
            stop: stop.clone(),
            ..resume_cfg("")
        };
        // The only step is also the last: the run completes normally.
        assert_eq!(epoch_ends(&corpus, &cfg), vec![1]);
        assert_eq!(stop.stopped(), None);
    }

    #[test]
    fn metrics_csv_has_a_finite_row_per_step_and_leaves_the_weights_alone() {
        let corpus = RESUME_CORPUS.repeat(40);
        let csv = scratch_ckpt("metrics").replace(".fckp", ".csv");
        let _ = std::fs::remove_file(&csv);
        let (ck_plain, ck_csv) = (scratch_ckpt("metrics_a"), scratch_ckpt("metrics_b"));
        let plain = TransformerConfig {
            cosine_lr: true,
            warmup_steps: 2,
            ..resume_cfg(&ck_plain)
        };
        let with_csv = TransformerConfig {
            metrics_csv: csv.clone(),
            ..TransformerConfig {
                checkpoint_path: ck_csv.clone(),
                ..plain.clone()
            }
        };
        let steps = *epoch_ends(&corpus, &plain).last().unwrap();
        epoch_ends(&corpus, &with_csv);
        assert_eq!(
            std::fs::read(&ck_plain).unwrap(),
            std::fs::read(&ck_csv).unwrap(),
            "the CSV must not change training"
        );
        let text = std::fs::read_to_string(&csv).unwrap();
        let mut lines = text.lines();
        assert_eq!(
            lines.next(),
            Some("step,epoch,loss,lr,grad_norm,tokens_per_sec,elapsed")
        );
        let rows: Vec<Vec<f64>> = lines
            .map(|l| l.split(',').map(|x| x.parse().unwrap()).collect())
            .collect();
        assert_eq!(rows.len() as u64, steps);
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.len(), 7);
            assert!(r.iter().all(|x| x.is_finite()), "{r:?}");
            assert_eq!(r[0], (i + 1) as f64);
            assert!(r[2] > 0.0 && r[4] > 0.0 && r[5] > 0.0, "{r:?}");
        }
        assert_eq!(rows.last().unwrap()[1], 3.0, "three epochs");
        assert!(rows[0][3] < rows[2][3], "warmup raises the lr: {rows:?}");
    }

    #[test]
    fn progress_line_reports_step_rate_and_elapsed() {
        let line = progress_line(30, 120, 30, 3000.0, 60.0, Some(0.5));
        assert_eq!(
            line,
            "[train] step 30/120  tokens/s 50  grad_norm 0.500  elapsed 60s  eta 3m00s"
        );
        // Without a norm the field is absent; a zero elapsed time never divides by zero.
        let line = progress_line(1, 2, 1, 8.0, 0.0, None);
        assert!(
            line.contains("tokens/s") && !line.contains("grad_norm"),
            "{line}"
        );
    }

    #[test]
    fn log_every_steps_defaults_off_and_training_runs_with_it_on() {
        assert_eq!(TransformerConfig::default().log_every_steps, 0);
        let cfg = TransformerConfig {
            log_every_steps: 2,
            epochs: 1,
            checkpoint_every: 0,
            ..resume_cfg("")
        };
        let corpus = RESUME_CORPUS.repeat(40);
        // Epoch mode and budget mode both print progress and still train.
        assert_eq!(epoch_ends(&corpus, &cfg).len(), 1);
        let budget = TransformerConfig {
            max_tokens: 6 * 4 * 8,
            ..cfg
        };
        assert_eq!(epoch_ends(&corpus, &budget), vec![6]);
    }
}
