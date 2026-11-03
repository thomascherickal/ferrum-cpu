//! What a fine-tune run records in its `.flck` and checks on `--resume`: the corpus id, the
//! batch and window it trains with, its learning-rate settings, and where its current epoch
//! began, so a resume from the middle of an epoch replays the rest of its window order.
//! Every key is optional; a checkpoint without them loads and resumes as before.
use crate::ckpt;
use crate::error::Result;
use crate::gguf::Gguf;
use crate::gguf_write::GgufBuilder;
use crate::llm_train::LlamaTrainer;
use crate::resume::{check_corpus, read_one, refuse_changed, EpochStart, RunSettings};
use crate::rng::Rng;

/// The settings of one fine-tune run. [`LlamaTrainer::begin_run`] checks them against a
/// resumed checkpoint and records them in the next one.
#[derive(Debug, Clone, PartialEq)]
pub struct FinetuneRun {
    /// `sha256:<hex>` of the corpus file (`ferrum.ckpt.corpus_id`), if known.
    pub corpus_id: Option<String>,
    pub batch: usize,
    /// Window length (capped at the model's context, at least 2).
    pub seq: usize,
    pub lr: f32,
    pub warmup: u64,
    pub clip: f32,
    pub weight_decay: f32,
    pub dropout: f32,
}

/// What a resume prints when the `.flck` predates the recorded run (an older Ferrum wrote it).
pub(crate) const OLD_CHECKPOINT_NOTE: &str =
    "note: the checkpoint predates its recorded corpus, epoch and training settings; none \
     could be checked, and a resume from the middle of an epoch starts a fresh one, so the \
     result may differ from an uninterrupted run";

/// The run state a trainer carries between checkpoints.
#[derive(Debug, Default)]
pub(crate) struct RunState {
    corpus_id: Option<String>,
    settings: Option<RunSettings>,
    /// The window length (`ferrum.ckpt.window_len`).
    window: Option<u64>,
    pub(crate) epoch_start: Option<EpochStart>,
    /// Set by a checkpoint load: the next epoch replays the interrupted one.
    replay: bool,
    resumed: bool,
}

impl RunState {
    pub(crate) fn write(&self, b: &mut GgufBuilder) {
        if let Some(id) = &self.corpus_id {
            ckpt::put_str(b, "corpus_id", id);
        }
        if let Some(s) = &self.settings {
            s.write(b);
        }
        if let Some(w) = self.window {
            ckpt::put_u64s(b, "window_len", &[w]);
        }
        if let Some(e) = self.epoch_start {
            e.write(b);
        }
    }

    /// The state of a checkpoint taken at `step` (every key optional).
    pub(crate) fn read(g: &Gguf, step: u64) -> Result<Self> {
        Ok(Self {
            corpus_id: ckpt::get_str(g, "corpus_id")?,
            settings: RunSettings::read(g)?,
            window: read_one(g, "window_len")?,
            epoch_start: EpochStart::read(g, step)?,
            replay: true,
            resumed: true,
        })
    }
}

impl LlamaTrainer {
    /// Start a run on `n_tokens` corpus tokens. After a checkpoint load it refuses another
    /// corpus (both ids known), batch or window, and notes changed learning-rate settings
    /// (through [`crate::verbose::log_line`]); then it records `run` for the next checkpoint.
    /// Returns the step the run's first epoch counts from: the interrupted epoch's start when
    /// it will be replayed, else the current step (a learning-rate schedule spans from there).
    pub fn begin_run(&mut self, run: &FinetuneRun, n_tokens: usize) -> Result<u64> {
        let seq = run.seq.min(self.model.cfg.context_len).max(2);
        let batch = run.batch.max(1);
        let now = RunSettings {
            batch: batch as u64,
            stride: 1,
            lr: run.lr,
            cosine: run.warmup > 0, // the warmup schedule decays by cosine
            warmup: run.warmup,
            clip: run.clip,
            weight_decay: run.weight_decay,
            dropout: run.dropout,
        };
        if let Some(note) = self.resume_check(&now, seq as u64, run.corpus_id.as_deref())? {
            crate::verbose::log_line(&note); // also reaches a GUI log sink
        }
        self.run.resumed = false;
        self.run.corpus_id = run.corpus_id.clone().or(self.run.corpus_id.take());
        self.run.settings = Some(now);
        self.run.window = Some(seq as u64);
        let steps = (n_tokens.saturating_sub(seq) + 1).div_ceil(batch);
        Ok(self
            .pending_replay(steps)
            .map_or(self.step_count(), |(s, _)| s.step))
    }

    /// After a checkpoint load: refuse another corpus, batch or window; the note to print for
    /// changed learning-rate settings or for a checkpoint without recorded settings.
    fn resume_check(
        &self,
        now: &RunSettings,
        seq: u64,
        corpus_id: Option<&str>,
    ) -> Result<Option<String>> {
        if !self.run.resumed {
            return Ok(None);
        }
        check_corpus(self.run.corpus_id.as_deref(), corpus_id, "")?;
        let Some(was) = &self.run.settings else {
            return Ok(Some(match self.run.epoch_start {
                None => OLD_CHECKPOINT_NOTE.into(),
                Some(_) => "note: the checkpoint has no recorded corpus or training settings; \
                            they could not be checked"
                    .into(),
            }));
        };
        let note = was.compare(now)?;
        if let Some(w) = self.run.window {
            refuse_changed("seq", w, seq)?;
        }
        Ok(note)
    }

    /// The interrupted epoch a freshly loaded checkpoint replays, and the steps of it done:
    /// `None` without a recorded epoch start, or when that epoch had finished.
    fn pending_replay(&self, steps: usize) -> Option<(EpochStart, usize)> {
        let s = self.run.epoch_start.filter(|_| self.run.replay)?;
        let done = self.step_count().saturating_sub(s.step) as usize;
        (done < steps).then_some((s, done))
    }

    /// The epoch's window order and the steps of it already done. A fresh epoch records its
    /// start (the RNG state its order is drawn from) and shuffles; the first epoch after a
    /// mid-epoch checkpoint rebuilds that epoch's order instead and skips the done steps.
    pub(crate) fn epoch_plan(
        &mut self,
        num_windows: usize,
        steps: usize,
        rng: &mut Rng,
    ) -> (Vec<usize>, usize) {
        let replay = self.pending_replay(steps);
        self.run.replay = false;
        if let Some((s, done)) = replay {
            return (Rng::from_state(s.rng).shuffled_indices(num_windows), done);
        }
        let step = self.step_count();
        let epoch = self
            .run
            .epoch_start
            .map_or(step / steps as u64, |e| e.epoch)
            + 1;
        self.run.epoch_start = Some(EpochStart {
            rng: rng.state(),
            step,
            epoch,
        });
        (rng.shuffled_indices(num_windows), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::{FinetuneRun, OLD_CHECKPOINT_NOTE};
    use crate::llm_train::test_support::tiny;
    use crate::llm_train::LlamaTrainer;
    use crate::optim::{Adam, LrSchedule};
    use crate::rng::Rng;

    /// 40 tokens at seq 6 and batch 4: 35 windows, 9 steps an epoch (the last one short).
    fn corpus() -> Vec<usize> {
        (0..40).map(|i| (i * 7 + i / 5) % 6).collect()
    }

    /// Dropout draws from the RNG every step and the warmup schedule depends on the run's
    /// total, so both have to be resumed exactly for the bytes to match.
    fn run() -> FinetuneRun {
        FinetuneRun {
            corpus_id: Some("sha256:aa".into()),
            batch: 4,
            seq: 6,
            lr: 0.02,
            warmup: 3,
            clip: 1.0,
            weight_decay: 0.01,
            dropout: 0.1,
        }
    }

    fn trainer(r: &FinetuneRun) -> LlamaTrainer {
        let mut t = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        t.set_optimizer(Adam::new(r.lr));
        t.set_weight_decay(r.weight_decay);
        t.set_dropout(r.dropout);
        t.set_grad_clip((r.clip > 0.0).then_some(r.clip));
        t
    }

    /// What `finetune-gguf` and the GUI do: begin the run, span the schedule from the step it
    /// returns, run `epochs` epochs, stopping after step `stop` (never the run's last).
    fn drive(
        t: &mut LlamaTrainer,
        r: &FinetuneRun,
        epochs: u64,
        threads: usize,
        rng: &mut Rng,
        stop: u64,
    ) {
        let tokens = corpus();
        let base = t.begin_run(r, tokens.len()).unwrap();
        let total = base + 9 * epochs;
        if r.warmup > 0 {
            t.set_lr_schedule(Some(LrSchedule::warmup_cosine(r.lr, r.warmup, total)));
        }
        for _ in 0..epochs {
            let (_, halted) = t
                .finetune_epoch_until(&tokens, r.seq, r.batch, rng, threads, &mut |s| {
                    s >= stop && s < total
                })
                .unwrap();
            if halted {
                break;
            }
        }
    }

    /// Run 3 epochs, or stop after `stop` steps, save, reload into a fresh trainer and run the
    /// epochs not finished; return the final checkpoint.
    fn interrupted(threads: usize, stop: u64) -> Vec<u8> {
        let r = run();
        let mut a = trainer(&r);
        let mut rng = Rng::new(5);
        drive(&mut a, &r, 3, threads, &mut rng, stop);
        assert_eq!(a.step_count(), stop);
        let bytes = a.save_checkpoint(&rng);
        let mut b = trainer(&r);
        let mut rng = b.load_checkpoint_into(&bytes).unwrap();
        drive(&mut b, &r, 3 - stop / 9, threads, &mut rng, u64::MAX);
        b.save_checkpoint(&rng)
    }

    #[test]
    fn a_resume_from_inside_an_epoch_is_bit_identical_to_an_uninterrupted_run() {
        for threads in [1, 2] {
            let r = run();
            let mut full = trainer(&r);
            let mut rng = Rng::new(5);
            drive(&mut full, &r, 3, threads, &mut rng, u64::MAX);
            assert_eq!(full.step_count(), 27);
            let full = full.save_checkpoint(&rng);
            // Inside the first and second epoch, and on the first epoch's last step.
            for stop in [4, 13, 9] {
                assert!(
                    interrupted(threads, stop) == full,
                    "threads {threads}, stop {stop}: checkpoints differ"
                );
            }
        }
    }

    #[test]
    fn the_run_keys_roundtrip_and_a_fresh_run_never_notes() {
        let r = run();
        let mut t = trainer(&r);
        let mut rng = Rng::new(5);
        drive(&mut t, &r, 1, 1, &mut rng, 4);
        let bytes = t.save_checkpoint(&rng);
        let g = crate::gguf::Gguf::parse(bytes.clone()).unwrap();
        for k in [
            "corpus_id",
            "batch_size",
            "window_stride",
            "window_len",
            "lr_settings",
            "epoch_rng",
        ] {
            assert!(g.meta(&format!("ferrum.ckpt.{k}")).is_some(), "{k}");
        }
        let mut u = trainer(&r);
        let rng2 = u.load_checkpoint_into(&bytes).unwrap();
        assert_eq!(u.run.corpus_id, t.run.corpus_id);
        assert_eq!(u.run.settings, t.run.settings);
        assert_eq!(
            (u.run.window, u.run.epoch_start),
            (Some(6), t.run.epoch_start)
        );
        assert_eq!(u.save_checkpoint(&rng2), bytes);
        // Same run: no note. Not resumed: nothing is checked.
        let now = t.run.settings.unwrap();
        assert_eq!(u.resume_check(&now, 6, Some("sha256:aa")).unwrap(), None);
        assert_eq!(t.resume_check(&now, 99, Some("sha256:bb")).unwrap(), None);
        // A malformed key is an error, not a silent skip.
        let bad = crate::ckpt::rewrite(
            &bytes,
            |m| {
                m.insert("ferrum.ckpt.window_len".into(), crate::MetaValue::U64(6));
            },
            None,
        );
        let e = u.load_checkpoint_into(&bad).err().unwrap().to_string();
        assert!(e.contains("window_len"), "{e}");
        let mut model = tiny(6, 8, 2, 2, 16, 1);
        let e = crate::llm_train::apply_checkpoint_weights(&mut model, bad).unwrap_err();
        assert!(e.to_string().contains("window_len"), "{e}");
    }

    fn resumed() -> LlamaTrainer {
        let r = run();
        let mut t = trainer(&r);
        let mut rng = Rng::new(5);
        drive(&mut t, &r, 1, 1, &mut rng, 4);
        let mut u = trainer(&r);
        u.load_checkpoint_into(&t.save_checkpoint(&rng)).unwrap();
        u
    }

    #[test]
    fn a_resume_refuses_another_corpus_batch_or_window() {
        let n = corpus().len();
        let e = resumed()
            .begin_run(
                &FinetuneRun {
                    corpus_id: Some("sha256:bb".into()),
                    ..run()
                },
                n,
            )
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("corpus sha256:aa") && e.contains("corpus is sha256:bb"),
            "{e}"
        );
        let e = resumed()
            .begin_run(&FinetuneRun { batch: 2, ..run() }, n)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("--batch is 2 but the checkpoint was trained with batch 4"),
            "{e}"
        );
        let e = resumed()
            .begin_run(&FinetuneRun { seq: 8, ..run() }, n)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("--seq is 8 but the checkpoint was trained with seq 6"),
            "{e}"
        );
        // A run without an id is not checked and keeps the checkpoint's.
        let mut t = resumed();
        t.begin_run(
            &FinetuneRun {
                corpus_id: None,
                ..run()
            },
            n,
        )
        .unwrap();
        assert_eq!(t.run.corpus_id.as_deref(), Some("sha256:aa"));
    }

    #[test]
    fn changed_learning_rate_settings_are_only_noted() {
        let mut t = resumed();
        let mut now = t.run.settings.unwrap();
        now.lr = 0.01;
        now.warmup = 0;
        now.cosine = false;
        now.dropout = 0.0;
        let note = t.resume_check(&now, 6, Some("sha256:aa")).unwrap().unwrap();
        for part in [
            "lr 0.02 -> 0.01",
            "warmup 3 -> 0",
            "dropout 0.1 -> 0",
            "will not match",
        ] {
            assert!(note.contains(part), "{part}: {note}");
        }
        let base = t
            .begin_run(
                &FinetuneRun {
                    lr: 0.01,
                    warmup: 0,
                    dropout: 0.0,
                    ..run()
                },
                corpus().len(),
            )
            .unwrap();
        assert_eq!(
            base, 0,
            "the interrupted first epoch is replayed from its start"
        );
    }

    /// Written by Ferrum before the run keys existed: 3 of 9 steps of the first epoch of
    /// `tiny(6, 8, 2, 2, 16, 1)`, lr 0.02, seq 6, batch 4, seed 5, on `(0..40).map(|i| i * 7 % 6)`.
    const PRE_RUN_KEYS: &[u8] = include_bytes!("testdata/pre_epoch_rng.flck");

    /// A run that never resumes trains exactly as before the run keys: the old build's
    /// checkpoint of this run holds bit-identical weights, Adam moments, step and RNG.
    /// (`ckpt::rewrite` does not re-emit a file byte for byte, so the state is compared.)
    #[test]
    fn a_run_that_never_resumes_is_bit_identical_to_the_old_build() {
        let mut t = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        t.set_lr(0.02);
        let tokens: Vec<usize> = (0..40).map(|i| (i * 7) % 6).collect();
        let mut rng = Rng::new(5);
        t.finetune_epoch_until(&tokens, 6, 4, &mut rng, 1, &mut |s| s >= 3)
            .unwrap();
        let bytes = t.save_checkpoint(&rng);
        let g = crate::gguf::Gguf::parse(bytes.clone()).unwrap();
        assert!(
            g.meta("ferrum.ckpt.epoch_rng").is_some(),
            "the one key it adds"
        );
        let parse = |b: &[u8]| {
            let mut u = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
            let rng = u.load_checkpoint_into(b).unwrap();
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            (
                rng.state(),
                u.step_count(),
                bits(u.model_flat()),
                bits(u.m.to_flat()),
                bits(u.v.to_flat()),
            )
        };
        assert!(
            parse(&bytes) == parse(PRE_RUN_KEYS),
            "weights, moments, step or RNG changed"
        );
    }

    #[test]
    fn an_old_checkpoint_resumes_as_before_with_a_note() {
        let mut t = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        t.set_lr(0.02);
        let mut rng = t.load_checkpoint_into(PRE_RUN_KEYS).unwrap();
        assert_eq!(t.step_count(), 3);
        assert_eq!(
            t.save_checkpoint(&rng),
            PRE_RUN_KEYS,
            "no key appears on a re-save"
        );
        let r = FinetuneRun {
            dropout: 0.0,
            warmup: 0,
            weight_decay: 0.0,
            ..run()
        };
        let note = t
            .resume_check(
                &super::RunSettings {
                    batch: 4,
                    stride: 1,
                    lr: 0.02,
                    cosine: false,
                    warmup: 0,
                    clip: 1.0,
                    weight_decay: 0.0,
                    dropout: 0.0,
                },
                6,
                Some("sha256:aa"),
            )
            .unwrap();
        assert_eq!(note.as_deref(), Some(OLD_CHECKPOINT_NOTE));
        // As before: no replay, the next epoch is a fresh full one (9 more steps).
        let tokens: Vec<usize> = (0..40).map(|i| (i * 7) % 6).collect();
        assert_eq!(t.begin_run(&r, tokens.len()).unwrap(), 3);
        t.finetune_epoch_until(&tokens, 6, 4, &mut rng, 1, &mut |_| false)
            .unwrap();
        assert_eq!(t.step_count(), 12);
        assert_eq!(t.run.epoch_start.map(|e| (e.step, e.epoch)), Some((3, 1)));
    }
}
