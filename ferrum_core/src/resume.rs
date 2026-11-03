//! What a resumable checkpoint records about its run, shared by `train` (`ferrum-checkpoint`)
//! and `finetune-gguf` (`ferrum-llama-checkpoint`): where the current epoch began (so a resume
//! replays its window order), the settings that must not change across a resume, and the
//! corpus identity check. All keys are optional: a checkpoint without them still loads.
use crate::ckpt;
use crate::error::{InferError, Result};

/// An optional one-number key (`[v]`): `None` when absent, an error when malformed.
pub(crate) fn read_one(g: &crate::gguf::Gguf, k: &str) -> Result<Option<u64>> {
    match ckpt::get_u64s(g, k)? {
        None => Ok(None),
        Some(&[v]) => Ok(Some(v)),
        Some(_) => Err(InferError::Format(format!(
            "checkpoint key 'ferrum.ckpt.{k}' is not one number"
        ))),
    }
}

/// Refuse a resume whose `--{name}` differs from the checkpoint's: it would be a different run.
pub(crate) fn refuse_changed(name: &str, was: u64, is: u64) -> Result<()> {
    if was == is {
        return Ok(());
    }
    Err(InferError::DimMismatch(format!(
        "--{name} is {is} but the checkpoint was trained with {name} {was}; a \
         resume with another {name} is a different run (window order and step \
         count change). Resume with --{name} {was}, or start a new run"
    )))
}

/// Refuse a resume on another corpus when both the checkpoint and the run have an id.
/// `hint` ends the message (what the caller's corpora are).
pub(crate) fn check_corpus(was: Option<&str>, now: Option<&str>, hint: &str) -> Result<()> {
    match (was, now) {
        (Some(was), Some(now)) if was != now => Err(InferError::DimMismatch(format!(
            "the checkpoint was trained on corpus {was}, this run's corpus is {now}; \
             resume on the corpus it was started with{hint}"
        ))),
        _ => Ok(()),
    }
}

/// The start of the epoch a checkpoint was written in (`ferrum.ckpt.epoch_rng`, stored as
/// `[rng, step, epoch]`): the RNG state its window order was drawn from, the optimizer step
/// it began after, and its 1-based number. Epochs are not all the same length (a stride
/// above 1 draws a random grid phase per epoch), so the step and the number are stored
/// rather than derived from the step count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EpochStart {
    pub(crate) rng: u64,
    pub(crate) step: u64,
    pub(crate) epoch: u64,
}

impl EpochStart {
    /// Read it from a checkpoint taken at `step`: absent in checkpoints older than the key,
    /// an error when malformed or later than the checkpoint itself.
    pub(crate) fn read(g: &crate::gguf::Gguf, step: u64) -> Result<Option<Self>> {
        match ckpt::get_u64s(g, "epoch_rng")? {
            None => Ok(None),
            Some(&[rng, start, epoch]) if start <= step && epoch >= 1 => Ok(Some(Self {
                rng,
                step: start,
                epoch,
            })),
            Some(_) => Err(InferError::Format(
                "checkpoint key 'ferrum.ckpt.epoch_rng' is not [rng, step, epoch]".into(),
            )),
        }
    }

    pub(crate) fn write(&self, b: &mut crate::gguf_write::GgufBuilder) {
        ckpt::put_u64s(b, "epoch_rng", &[self.rng, self.step, self.epoch]);
    }
}

/// The training settings a checkpoint was written under (`ferrum.ckpt.batch_size`,
/// `ferrum.ckpt.window_stride`, `ferrum.ckpt.lr_settings`), so a `--resume` can tell when they
/// changed. A different batch or stride changes the window order and the step count, so a
/// resume refuses it; the learning-rate settings may change and are only noted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RunSettings {
    pub(crate) batch: u64,
    /// Effective stride (`--stride 0` already resolved to the context length).
    pub(crate) stride: u64,
    pub(crate) lr: f32,
    pub(crate) cosine: bool,
    pub(crate) warmup: u64,
    pub(crate) clip: f32,
    pub(crate) weight_decay: f32,
    pub(crate) dropout: f32,
}

impl RunSettings {
    pub(crate) fn write(&self, b: &mut crate::gguf_write::GgufBuilder) {
        ckpt::put_u64s(b, "batch_size", &[self.batch]);
        ckpt::put_u64s(b, "window_stride", &[self.stride]);
        ckpt::put_u64s(
            b,
            "lr_settings",
            &[
                self.lr.to_bits() as u64,
                self.cosine as u64,
                self.warmup,
                self.clip.to_bits() as u64,
                self.weight_decay.to_bits() as u64,
                self.dropout.to_bits() as u64,
            ],
        );
    }

    /// `None` for a checkpoint older than the keys; an error for a malformed one.
    pub(crate) fn read(g: &crate::gguf::Gguf) -> Result<Option<Self>> {
        let one = |k: &str| read_one(g, k);
        let (Some(batch), Some(stride)) = (one("batch_size")?, one("window_stride")?) else {
            return Ok(None);
        };
        let Some(&[lr, cosine, warmup, clip, wd, dropout]) = ckpt::get_u64s(g, "lr_settings")?
        else {
            return Ok(None);
        };
        let f = |bits: u64| f32::from_bits(bits as u32);
        Ok(Some(Self {
            batch,
            stride,
            lr: f(lr),
            cosine: cosine != 0,
            warmup,
            clip: f(clip),
            weight_decay: f(wd),
            dropout: f(dropout),
        }))
    }

    /// Compare a checkpoint's settings (`self`) with this run's. `Err` names a changed batch
    /// or stride; `Ok` lists the changed learning-rate settings, one `note:` line.
    pub(crate) fn compare(&self, now: &Self) -> Result<Option<String>> {
        refuse_changed("batch", self.batch, now.batch)?;
        refuse_changed("stride", self.stride, now.stride)?;
        let mut changed = Vec::new();
        let mut diff = |name: &str, was: String, is: String, same: bool| {
            if !same {
                changed.push(format!("{name} {was} -> {is}"));
            }
        };
        diff(
            "lr",
            self.lr.to_string(),
            now.lr.to_string(),
            self.lr.to_bits() == now.lr.to_bits(),
        );
        diff(
            "cosine",
            self.cosine.to_string(),
            now.cosine.to_string(),
            self.cosine == now.cosine,
        );
        diff(
            "warmup",
            self.warmup.to_string(),
            now.warmup.to_string(),
            self.warmup == now.warmup,
        );
        diff(
            "clip",
            self.clip.to_string(),
            now.clip.to_string(),
            self.clip.to_bits() == now.clip.to_bits(),
        );
        diff(
            "weight_decay",
            self.weight_decay.to_string(),
            now.weight_decay.to_string(),
            self.weight_decay.to_bits() == now.weight_decay.to_bits(),
        );
        diff(
            "dropout",
            self.dropout.to_string(),
            now.dropout.to_string(),
            self.dropout.to_bits() == now.dropout.to_bits(),
        );
        Ok((!changed.is_empty()).then(|| {
            format!(
                "note: training settings changed since the checkpoint was written ({}); the \
                 resumed run will not match an uninterrupted one",
                changed.join(", ")
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{check_corpus, read_one, refuse_changed, EpochStart};
    use crate::{ckpt, MetaValue};

    #[test]
    fn corpus_check_refuses_only_two_different_ids() {
        assert!(check_corpus(None, Some("sha256:b"), "").is_ok());
        assert!(check_corpus(Some("sha256:a"), None, "").is_ok());
        assert!(check_corpus(Some("sha256:a"), Some("sha256:a"), "").is_ok());
        let e = check_corpus(Some("sha256:a"), Some("sha256:b"), " (hint)")
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("corpus sha256:a")
                && e.contains("corpus is sha256:b")
                && e.ends_with("with (hint)"),
            "{e}"
        );
    }

    #[test]
    fn a_changed_setting_is_refused_by_name() {
        assert!(refuse_changed("seq", 8, 8).is_ok());
        let e = refuse_changed("seq", 8, 16).unwrap_err().to_string();
        assert!(
            e.contains("--seq is 16 but the checkpoint was trained with seq 8")
                && e.contains("Resume with --seq 8"),
            "{e}"
        );
    }

    #[test]
    fn one_number_keys_and_epoch_starts_roundtrip() {
        let mut b = ckpt::begin("ferrum-test");
        ckpt::put_u64s(&mut b, "one", &[5]);
        ckpt::put_u64s(&mut b, "two", &[5, 6]);
        let start = EpochStart {
            rng: 9,
            step: 4,
            epoch: 2,
        };
        start.write(&mut b);
        b.meta("ferrum.ckpt.scalar", MetaValue::U64(1));
        let g = ckpt::open(&b.into_bytes(), b"NONE", "ferrum-test").unwrap();
        assert_eq!(read_one(&g, "one").unwrap(), Some(5));
        assert_eq!(read_one(&g, "absent").unwrap(), None);
        assert!(read_one(&g, "two")
            .unwrap_err()
            .to_string()
            .contains("not one number"));
        assert!(read_one(&g, "scalar").is_err());
        assert_eq!(EpochStart::read(&g, 4).unwrap(), Some(start));
        assert!(
            EpochStart::read(&g, 3).is_err(),
            "a start after the checkpoint"
        );
    }
}
