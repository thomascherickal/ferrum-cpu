//! The epoch/step training loop with resume, validation and checkpointing.
use crate::error::{InferError, Result};
use crate::optim::Adam;
use crate::rng::Rng;
use crate::slm::config::TransformerConfig;
use crate::slm::memory::{available_memory_bytes, memory_gate};
use crate::slm::metrics_log::{format_eta, CsvLog, Row};
use crate::train_transformer::checkpoint::RunSettings;
use crate::train_transformer::{steps_over, EpochStart, TokenSource, TransformerNet};

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
        crate::resume::check_corpus(
            net.corpus_id.as_deref(),
            cfg.corpus_id.as_deref(),
            " (a text corpus and the token file made from it have different ids, so neither \
             resumes the other's checkpoint)",
        )?;
        if let Some(was) = &net.run_settings {
            if let Some(note) = was.compare(&RunSettings::of(cfg))? {
                crate::verbose::log_line(&note); // also reaches a GUI log sink
            }
        } else {
            crate::verbose::log_line(
                "note: the checkpoint predates its recorded training settings; batch, stride \
                 and learning-rate settings could not be compared",
            );
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
    // A run without an id keeps the checkpoint's; one with an id (re)records it.
    net.corpus_id = cfg.corpus_id.clone().or(net.corpus_id.take());
    net.run_settings = Some(RunSettings::of(cfg));
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

/// Where a resumed net picks up: the epochs already finished, the steps an older checkpoint
/// has done in the current epoch (it is reshuffled), and for a checkpoint written inside an
/// epoch, that epoch's start and the steps done in it (its window order is replayed).
type ResumePoint = (usize, u64, Option<(EpochStart, usize)>);

fn resume_point(
    net: &TransformerNet,
    n_tokens: usize,
    batch: usize,
    steps_per_epoch: u64,
) -> Result<ResumePoint> {
    let step = net.step_count();
    let Some(start) = net.epoch_start else {
        let partial = step % steps_per_epoch;
        if partial > 0 {
            crate::verbose::log_line(&reshuffle_note(step)); // also reaches a GUI log sink
        }
        return Ok(((step / steps_per_epoch) as usize, partial, None));
    };
    // The epoch's real length: with a stride above 1 it varies with the drawn grid phase.
    let order = net.epoch_window_starts(n_tokens, &mut Rng::from_state(start.rng))?;
    let done = (step - start.step) as usize;
    Ok(if done < order.len().div_ceil(batch) {
        (start.epoch as usize - 1, 0, Some((start, done)))
    } else {
        // Written on the epoch's last step: it is finished and the next one starts fresh.
        (start.epoch as usize, 0, None)
    })
}

/// What a resume from a checkpoint without `ferrum.ckpt.epoch_rng` (written before Ferrum
/// stored it) prints: the rest of its epoch runs in a fresh order, so the result is close
/// to, not bit-identical with, an uninterrupted run.
fn reshuffle_note(step: u64) -> String {
    format!(
        "note: the checkpoint (step {step}) predates ferrum.ckpt.epoch_rng; the rest of its \
         epoch is reshuffled, so the result will differ slightly from an uninterrupted run"
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
pub(super) fn run_training<S: TokenSource>(
    net: &mut TransformerNet,
    tokens: &mut S,
    cfg: &TransformerConfig,
    threads: usize,
    rng: &mut Rng,
    on_epoch: &mut dyn FnMut(&TransformerNet, usize, f32) -> Result<bool>,
) -> Result<()> {
    run_training_with(net, tokens, cfg, threads, rng, on_epoch, CsvLog::open)
}

/// Tokens one optimizer step consumes (`batch` windows of `t` tokens), or an error when the
/// product does not fit a `usize`.
fn tokens_per_step(batch: usize, t: usize) -> Result<usize> {
    batch.checked_mul(t).ok_or_else(|| {
        InferError::DimMismatch(format!(
            "--batch {batch} x --context {t} is too large to train (the product overflows); \
             lower --batch or --context"
        ))
    })
}

/// [`run_training`] with the `metrics_csv` log opened by `open_csv(path, append)`, so a test
/// can hand it a writer that fails.
fn run_training_with<S: TokenSource>(
    net: &mut TransformerNet,
    tokens: &mut S,
    cfg: &TransformerConfig,
    threads: usize,
    rng: &mut Rng,
    on_epoch: &mut dyn FnMut(&TransformerNet, usize, f32) -> Result<bool>,
    open_csv: impl FnOnce(&str, bool) -> Result<CsvLog>,
) -> Result<()> {
    let t = net.context_len();
    let batch = cfg.batch_size.max(1);
    // Refused whatever the RAM guard knows: where the free RAM is unknown (non-Linux) the guard
    // only enforces its hard cap, and an absurd batch would otherwise overflow the product below.
    let step_tokens = tokens_per_step(batch, t)?;
    let windows = tokens.len().saturating_sub(t + 1) / net.window_stride() + 1;
    memory_gate(
        cfg.skip_memory_check,
        net.train_peak_bytes(windows, batch, threads.max(1).min(batch)),
        available_memory_bytes(),
    )?;
    let steps_per_epoch = windows.div_ceil(batch).max(1) as u64;
    let budgeted = cfg.max_tokens > 0;
    let total_steps = if budgeted {
        cfg.max_tokens.div_ceil(step_tokens as u64)
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
    let tokens_per_step = step_tokens as f64;
    let halted = std::cell::Cell::new(false);
    let epoch = std::cell::Cell::new(0usize);
    let mut csv = (!cfg.metrics_csv.is_empty())
        .then(|| open_csv(&cfg.metrics_csv, cfg.resume))
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
    let (mut ep, mut partial, mut replay) =
        resume_point(net, tokens.len(), batch, steps_per_epoch)?;
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
        // Record where this epoch began, so a checkpoint written inside it can replay it.
        let (start, skip) = replay.take().unwrap_or((
            EpochStart {
                rng: rng.state(),
                step: net.step_count(),
                epoch: ep as u64,
            },
            0,
        ));
        net.epoch_start = Some(start);
        let loss = steps_over(
            net,
            tokens,
            batch,
            &adam,
            rng,
            threads,
            max_steps,
            (skip > 0).then(|| (Rng::from_state(start.rng), skip)),
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
    #[test]
    fn tokens_per_step_is_the_product_and_refuses_overflow() {
        assert_eq!(super::tokens_per_step(16, 256).unwrap(), 4096);
        let e = super::tokens_per_step(usize::MAX, 512)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("too large to train") && e.contains("--batch"),
            "{e}"
        );
        assert_eq!(super::tokens_per_step(1, usize::MAX).unwrap(), usize::MAX);
    }

    use super::progress_line;
    use crate::rng::Rng;
    use crate::slm::config::{TransformerConfig, ValidationConfig};
    use crate::slm::test_support::{
        epoch_ends, epoch_ends_threaded, resume_cfg, scratch_ckpt, RESUME_CORPUS,
    };
    use crate::slm::GenerativeSLM;
    use crate::train_transformer::TransformerNet;

    /// Train 5 steps on `RESUME_CORPUS` recording `id`, return the checkpoint path.
    fn short_run(name: &str, id: Option<&str>) -> String {
        let ckpt = scratch_ckpt(name);
        let cfg = TransformerConfig {
            max_tokens: 5 * 4 * 8,
            corpus_id: id.map(String::from),
            ..resume_cfg(&ckpt)
        };
        GenerativeSLM::train_transformer_config(
            &RESUME_CORPUS.repeat(40),
            &cfg,
            &mut Rng::new(1),
            crate::slm::test_support::ignore_progress,
        )
        .unwrap();
        ckpt
    }

    fn resume_with(ckpt: &str, id: Option<&str>) -> crate::error::Result<GenerativeSLM> {
        let cfg = TransformerConfig {
            resume: true,
            max_tokens: 8 * 4 * 8,
            corpus_id: id.map(String::from),
            ..resume_cfg(ckpt)
        };
        GenerativeSLM::train_transformer_config(
            &RESUME_CORPUS.repeat(40),
            &cfg,
            &mut Rng::new(1),
            crate::slm::test_support::ignore_progress,
        )
    }

    fn saved_id(ckpt: &str) -> Option<String> {
        let (net, _) = TransformerNet::load_checkpoint(&std::fs::read(ckpt).unwrap()).unwrap();
        net.corpus_id
    }

    /// A `metrics_csv` sink whose writes fail, like a full disk.
    struct FullDisk;
    impl std::io::Write for FullDisk {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_failing_metrics_row_write_ends_the_run() {
        let cfg = TransformerConfig {
            metrics_csv: "unused.csv".into(),
            max_tokens: 4 * 8,
            ..resume_cfg("")
        };
        let tc = crate::slm::corpus::tokenize_for_lm_cfg(RESUME_CORPUS, &cfg).unwrap();
        let mut rng = Rng::new(1);
        let mut net = super::prepare_net(&cfg, tc.vocab_size, &mut rng).unwrap();
        let err = super::run_training_with(
            &mut net,
            &mut tc.tokens.as_slice(),
            &cfg,
            1,
            &mut rng,
            &mut |_, _, _| Ok(true),
            |_, _| Ok(super::CsvLog::to(Box::new(FullDisk))),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "i/o error: disk full");
        assert_eq!(net.step_count(), 1, "the step ran, its row did not land");
        std::io::Write::flush(&mut FullDisk).unwrap(); // only writes fail
    }

    /// Resume `ckpt` with `edit` applied to the config the checkpoint was written under.
    fn resume_edited(
        ckpt: &str,
        edit: impl Fn(&mut TransformerConfig),
    ) -> crate::error::Result<GenerativeSLM> {
        let mut cfg = TransformerConfig {
            resume: true,
            max_tokens: 8 * 4 * 8,
            ..resume_cfg(ckpt)
        };
        edit(&mut cfg);
        GenerativeSLM::train_transformer_config(
            &RESUME_CORPUS.repeat(40),
            &cfg,
            &mut Rng::new(1),
            crate::slm::test_support::ignore_progress,
        )
    }

    #[test]
    fn resume_refuses_a_changed_batch_or_stride_and_accepts_the_same() {
        let ckpt = short_run("settings_refuse", None);
        let before = std::fs::read(&ckpt).unwrap();
        let was = resume_cfg(&ckpt);
        let err = resume_edited(&ckpt, |c| c.batch_size = was.batch_size + 1)
            .err()
            .unwrap()
            .to_string();
        let (a, b) = (was.batch_size, was.batch_size + 1);
        assert!(
            err.contains("--batch is ")
                && err.contains(&format!("batch {a}"))
                && err.contains(&b.to_string()),
            "{err}"
        );
        let err = resume_edited(&ckpt, |c| c.window_stride = 3)
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("--stride is 3") && err.contains("stride"),
            "{err}"
        );
        assert_eq!(
            std::fs::read(&ckpt).unwrap(),
            before,
            "a refusal writes nothing"
        );
        resume_edited(&ckpt, |_| {}).unwrap();
    }

    #[test]
    fn resume_with_changed_learning_rate_settings_proceeds() {
        let ckpt = short_run("settings_note", None);
        resume_edited(&ckpt, |c| {
            c.lr *= 0.5;
            c.cosine_lr = true;
            c.warmup_steps = 2;
            c.grad_clip = 1.0;
        })
        .unwrap();
    }

    #[test]
    fn a_checkpoint_without_recorded_settings_resumes_unchanged() {
        let ckpt = short_run("settings_old", None);
        let strip = |b: &[u8]| {
            crate::ckpt::rewrite(
                b,
                |m| {
                    for k in ["batch_size", "window_stride", "lr_settings"] {
                        assert!(m.remove(&format!("ferrum.ckpt.{k}")).is_some());
                    }
                },
                None,
            )
        };
        let old = strip(&std::fs::read(&ckpt).unwrap());
        let (net, _) = TransformerNet::load_checkpoint(&old).unwrap();
        assert!(net.run_settings.is_none());
        std::fs::write(&ckpt, &old).unwrap();
        // Any batch/stride resumes (nothing to compare), and the run records them again.
        resume_edited(&ckpt, |c| c.batch_size += 1).unwrap();
        let (net, _) = TransformerNet::load_checkpoint(&std::fs::read(&ckpt).unwrap()).unwrap();
        assert!(net.run_settings.is_some());
    }

    #[test]
    fn resume_refuses_a_different_corpus() {
        let ckpt = short_run("corpus_refuse", Some("sha256:aaaa"));
        assert_eq!(saved_id(&ckpt).as_deref(), Some("sha256:aaaa"));
        let before = std::fs::read(&ckpt).unwrap();
        let err = resume_with(&ckpt, Some("sha256:bbbb"))
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("corpus") && err.contains("sha256:aaaa") && err.contains("sha256:bbbb"),
            "{err}"
        );
        assert_eq!(
            std::fs::read(&ckpt).unwrap(),
            before,
            "a refusal writes nothing"
        );
        // The same corpus resumes.
        resume_with(&ckpt, Some("sha256:aaaa")).unwrap();
    }

    #[test]
    fn resume_refuses_different_vocab_or_context() {
        let ckpt = short_run("vocab_ctx_refuse", None);
        let mut cfg = resume_cfg(&ckpt);
        cfg.resume = true;
        let mut rng = Rng::new(1);
        let err = super::prepare_net(&cfg, 999, &mut rng)
            .err()
            .expect("err")
            .to_string();
        assert!(err.contains("checkpoint has vocab"), "{err}");

        let mut cfg = resume_cfg(&ckpt);
        cfg.resume = true;
        cfg.context_len += 2;
        let err = super::prepare_net(&cfg, 290, &mut rng)
            .err()
            .expect("err")
            .to_string();
        assert!(err.contains("context"), "{err}");

        // Also test prepare_net on fresh net with invalid config (dim mismatch in TransformerNet::new):
        let bad_cfg = TransformerConfig {
            context_len: 0,
            ..TransformerConfig::default()
        };
        assert!(super::prepare_net(&bad_cfg, 10, &mut rng).is_err());
    }

    #[test]
    fn resume_without_corpus_id_on_either_side_proceeds() {
        // Checkpoint has one, the resuming run gives none: proceeds and keeps the id.
        let ckpt = short_run("corpus_ck_only", Some("sha256:aaaa"));
        resume_with(&ckpt, None).unwrap();
        assert_eq!(saved_id(&ckpt).as_deref(), Some("sha256:aaaa"));
        // Checkpoint has none, the resuming run gives one: proceeds and records it.
        let ckpt = short_run("corpus_run_only", None);
        assert_eq!(saved_id(&ckpt), None);
        resume_with(&ckpt, Some("sha256:bbbb")).unwrap();
        assert_eq!(saved_id(&ckpt).as_deref(), Some("sha256:bbbb"));
        // Neither has one.
        let ckpt = short_run("corpus_none", None);
        resume_with(&ckpt, None).unwrap();
        assert_eq!(saved_id(&ckpt), None);
    }

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
            crate::slm::test_support::ignore_progress,
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

    /// Five copies of the corpus at batch 2: the stride grid's random phase makes the epochs
    /// 5, 5, 6 and 5 steps long against an estimate of 6, so `step / steps_per_epoch` alone
    /// misplaces a checkpoint inside its epoch. Dropout draws from the RNG on every step.
    fn drift_cfg(ckpt: &str) -> TransformerConfig {
        TransformerConfig {
            batch_size: 2,
            epochs: 4,
            dropout: 0.1,
            checkpoint_every: 0,
            ..resume_cfg(ckpt)
        }
    }

    /// Train `stop` steps (a token budget), resume to the full run, and return the epoch ends
    /// of the resumed run and its final checkpoint. `edit` may rewrite the interrupted checkpoint.
    fn interrupted_run(
        tag: &str,
        stop: u64,
        threads: usize,
        edit: impl Fn(Vec<u8>) -> Vec<u8>,
    ) -> (Vec<u64>, Vec<u8>) {
        let corpus = RESUME_CORPUS.repeat(5);
        let ck = scratch_ckpt(tag);
        let short = TransformerConfig {
            max_tokens: stop * 2 * 8,
            ..drift_cfg(&ck)
        };
        assert_eq!(
            epoch_ends_threaded(&corpus, &short, threads).last(),
            Some(&stop)
        );
        std::fs::write(&ck, edit(std::fs::read(&ck).unwrap())).unwrap();
        let resumed = TransformerConfig {
            resume: true,
            ..drift_cfg(&ck)
        };
        let ends = epoch_ends_threaded(&corpus, &resumed, threads);
        (ends, std::fs::read(&ck).unwrap())
    }

    #[test]
    fn resume_mid_epoch_replays_the_interrupted_epoch_exactly() {
        for threads in [1, 2] {
            let full_ck = scratch_ckpt(&format!("exact_full{threads}"));
            let full = epoch_ends_threaded(&RESUME_CORPUS.repeat(5), &drift_cfg(&full_ck), threads);
            assert_eq!(full, vec![5, 10, 16, 21], "the epochs must drift");
            let full_bytes = std::fs::read(&full_ck).unwrap();
            // 21 / 2 + 3 = 13: three steps into the third epoch (13 % 6 would say one), and
            // 10: the last step of the second epoch (10 % 6 would say four into it).
            for stop in [13, 10] {
                let tag = format!("exact_{threads}_{stop}");
                let (ends, bytes) = interrupted_run(&tag, stop, threads, |b| b);
                let rest: Vec<u64> = full.iter().copied().filter(|&e| e > stop).collect();
                assert_eq!(ends, rest, "threads {threads}, stop {stop}");
                assert!(
                    bytes == full_bytes,
                    "threads {threads}, stop {stop}: checkpoints differ"
                );
            }
        }
    }

    #[test]
    fn a_checkpoint_without_epoch_rng_resumes_with_a_reshuffle() {
        let full_ck = scratch_ckpt("reshuffle_full");
        let full = epoch_ends_threaded(&RESUME_CORPUS.repeat(5), &drift_cfg(&full_ck), 1);
        let strip = |b: Vec<u8>| {
            crate::ckpt::rewrite(
                &b,
                |m| {
                    assert!(m.remove("ferrum.ckpt.epoch_rng").is_some());
                },
                None,
            )
        };
        // As before the key existed: 13 % 6 = 1 step counts as done, the epoch is reshuffled.
        let (ends, bytes) = interrupted_run("reshuffle", 13, 1, strip);
        assert_eq!(ends.len(), 2, "{ends:?}");
        assert!(
            bytes != std::fs::read(&full_ck).unwrap(),
            "{full:?} vs {ends:?}"
        );
        // The resumed run itself writes the key again.
        let (net, _) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert_eq!(net.epoch_start.map(|e| e.epoch), Some(4));
        let note = super::reshuffle_note(13);
        assert!(
            note.contains("step 13") && note.contains("ferrum.ckpt.epoch_rng"),
            "{note}"
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
    fn a_cancel_flag_set_before_the_run_stops_mid_epoch_and_resumes_exactly() {
        use crate::stop::{StopCause, StopCondition};
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let corpus = RESUME_CORPUS.repeat(5);
        let full_ck = scratch_ckpt("flag_full");
        assert_eq!(
            epoch_ends(&corpus, &drift_cfg(&full_ck)),
            vec![5, 10, 16, 21]
        );
        // Already set when the loop starts (a Ctrl-C during setup): one step, inside the
        // first 5-step epoch, then the checkpoint.
        let ck = scratch_ckpt("flag_part");
        let stop = StopCondition::none().with_cancel(Arc::new(AtomicBool::new(true)));
        let cfg = TransformerConfig {
            stop: stop.clone(),
            ..drift_cfg(&ck)
        };
        assert_eq!(epoch_ends(&corpus, &cfg), vec![1]);
        assert_eq!(stop.stopped(), Some((StopCause::Cancelled, 1)));
        let (net, _) = TransformerNet::load_checkpoint(&std::fs::read(&ck).unwrap()).unwrap();
        assert_eq!(net.step_count(), 1);
        let resumed = TransformerConfig {
            resume: true,
            ..drift_cfg(&ck)
        };
        assert_eq!(epoch_ends(&corpus, &resumed), vec![5, 10, 16, 21]);
        assert!(std::fs::read(&ck).unwrap() == std::fs::read(&full_ck).unwrap());

        // No checkpoint path: the run still stops after one step and writes nothing.
        let none = StopCondition::none().with_cancel(Arc::new(AtomicBool::new(true)));
        let cfg = TransformerConfig {
            stop: none.clone(),
            ..drift_cfg("")
        };
        assert_eq!(epoch_ends(&corpus, &cfg), vec![1]);
        assert_eq!(none.stopped(), Some((StopCause::Cancelled, 1)));
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

    #[test]
    fn write_checkpoint_fails_on_unwritable_path() {
        let mut rng = Rng::new(1);
        let net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng).unwrap();
        let res = super::write_checkpoint("/dev/null/impossible/ckpt.ferrum", &net, &mut rng);
        assert!(res.is_err());
    }
}
