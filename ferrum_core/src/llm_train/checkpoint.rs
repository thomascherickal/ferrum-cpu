//! GGUF checkpoint save / load and weights-only application.
use crate::ckpt;
use crate::error::{InferError, Result};
use crate::llm::{LlamaConfig, LlamaModel};
use crate::llm_train::params::{ensure_f32_weights, param_data_mut, param_data_ref};
use crate::llm_train::run::RunState;
use crate::llm_train::LlamaTrainer;
use crate::rng::Rng;

/// The model shape recorded in (and checked against) a checkpoint.
fn shape_keys_of(c: &LlamaConfig) -> [(&'static str, usize); 7] {
    [
        ("vocab", c.vocab_size),
        ("model_dim", c.model_dim),
        ("layers", c.n_layers),
        ("heads", c.n_heads),
        ("kv_heads", c.n_kv_heads),
        ("head_dim", c.head_dim),
        ("ffn", c.ffn_dim),
    ]
}

/// A parameter slot with nothing to train (a placeholder bias, see
/// `params::bias_ref`, or the LM head of a tied-embedding model) is stored as an
/// empty tensor. Data in such a slot means an older Ferrum trained it, which the
/// source architecture cannot represent: name the cause instead of a bare shape
/// error.
fn check_no_placeholder_biases(
    g: &crate::gguf::Gguf,
    lens: &[usize],
    tied_head: bool,
) -> Result<()> {
    for (i, _) in lens.iter().enumerate().filter(|(_, &n)| n == 0) {
        let name = format!("p{i}.data");
        if g.tensor(&name).is_some_and(|t| t.dims != [0u64]) {
            if tied_head && i + 2 == lens.len() {
                return Err(InferError::Format(format!(
                    "checkpoint tensor '{name}' holds a separately trained LM head, but this \
                     model has tied embeddings (its GGUF has no output.weight): the checkpoint \
                     was written by an older Ferrum that trained the head independently of the \
                     embedding, which cannot be exported. Re-run finetune-gguf to make a new \
                     checkpoint"
                )));
            }
            return Err(InferError::Format(format!(
                "checkpoint tensor '{name}' holds a bias this model does not have (its GGUF \
                 has no such bias tensor): the checkpoint was written by an older Ferrum \
                 that trained placeholder biases, which cannot be exported. Re-run \
                 finetune-gguf to make a new checkpoint"
            )));
        }
    }
    Ok(())
}

/// Overlay a fine-tune checkpoint's f32 master **weights** onto `model` without
/// a [`LlamaTrainer`]: no gradient buffers or Adam-moment copies are allocated
/// and no flat f32 vectors are built (the optimizer state in the file is
/// skipped), so applying a checkpoint for inference or export peaks at the
/// model plus the checkpoint bytes plus one parameter, instead of 4x the model
/// plus twice the checkpoint. The bytes are parsed in place. The shape, the
/// scalars a trainer would restore, and every weight tensor are validated
/// before the first weight is written, so a bad checkpoint leaves `model`
/// untouched. Like the trainer, it needs f32 projections.
pub fn apply_checkpoint_weights(model: &mut LlamaModel, bytes: Vec<u8>) -> Result<()> {
    ensure_f32_weights(model)?;
    let g = ckpt::open_owned(bytes, b"FLCK", "ferrum-llama-checkpoint")?;
    for (k, want) in shape_keys_of(&model.cfg) {
        if ckpt::get_usize(&g, k)? != want {
            return Err(InferError::Format(
                "checkpoint shape does not match this model".into(),
            ));
        }
    }
    // The same scalars a full restore reads: a malformed file fails alike.
    ckpt::get_bool(&g, "qat")?;
    let step = ckpt::get_u64(&g, "step")?;
    ckpt::get_u64(&g, "rng")?;
    RunState::read(&g, step)?;
    let tied = model.tied_head;
    let mut targets = param_data_mut(model);
    check_no_placeholder_biases(
        &g,
        &targets.iter().map(|s| s.len()).collect::<Vec<_>>(),
        tied,
    )?;
    for (i, s) in targets.iter().enumerate() {
        ckpt::f32_tensor(&g, &format!("p{i}.data"), s.len())?;
    }
    for (i, s) in targets.iter_mut().enumerate() {
        let t = ckpt::f32_tensor(&g, &format!("p{i}.data"), s.len())?;
        s.copy_from_slice(&g.dequantize(t)?);
    }
    drop(targets);
    model.sync_tied_head();
    Ok(())
}

impl LlamaTrainer {
    /// Serialize the full **optimizer state**: a shape header, the QAT flag, the
    /// step counter, the supplied RNG's state, the run's optional keys (corpus id,
    /// batch, window and learning-rate settings when [`LlamaTrainer::begin_run`] set
    /// them, and where the current epoch began), and every parameter's weights and
    /// both Adam moments. Pair with [`LlamaTrainer::load_checkpoint_into`] on a
    /// freshly-loaded base model of the same shape to resume exactly.
    pub fn save_checkpoint(&self, rng: &Rng) -> Vec<u8> {
        let mut b = ckpt::begin("ferrum-llama-checkpoint");
        for (k, v) in self.shape_keys() {
            ckpt::put_u32(&mut b, k, v);
        }
        ckpt::put_bool(&mut b, "qat", self.qat);
        ckpt::put_u64(&mut b, "step", self.step_t);
        ckpt::put_u64(&mut b, "rng", rng.state());
        self.run.write(&mut b);
        let (data, m, v) = (
            param_data_ref(&self.model),
            self.m.slices(),
            self.v.slices(),
        );
        for (i, ((d, m), v)) in data.into_iter().zip(m).zip(v).enumerate() {
            ckpt::put_param(&mut b, i, d, m, v);
        }
        b.into_bytes()
    }

    /// The model shape recorded in (and checked against) a checkpoint.
    fn shape_keys(&self) -> [(&'static str, usize); 7] {
        shape_keys_of(&self.model.cfg)
    }

    #[cfg(test)]
    pub(super) fn model_flat(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for s in param_data_ref(&self.model) {
            out.extend_from_slice(s);
        }
        out
    }

    /// Restore weights, Adam moments, step counter, QAT flag, and RNG state from
    /// [`LlamaTrainer::save_checkpoint`] bytes into `self` (which must already
    /// hold a model of the matching shape). Returns the restored RNG. Nothing is
    /// modified unless the whole checkpoint validates.
    pub fn load_checkpoint_into(&mut self, bytes: &[u8]) -> Result<Rng> {
        self.apply_checkpoint(ckpt::open(bytes, b"FLCK", "ferrum-llama-checkpoint")?)
    }

    /// [`LlamaTrainer::load_checkpoint_into`] taking ownership of the bytes: a
    /// checkpoint just read from disk is parsed in place instead of copied
    /// (a 2 GB `.flck` otherwise peaks one file's size higher).
    pub fn load_checkpoint_from_vec(&mut self, bytes: Vec<u8>) -> Result<Rng> {
        self.apply_checkpoint(ckpt::open_owned(bytes, b"FLCK", "ferrum-llama-checkpoint")?)
    }

    fn apply_checkpoint(&mut self, g: crate::gguf::Gguf) -> Result<Rng> {
        for (k, want) in self.shape_keys() {
            if ckpt::get_usize(&g, k)? != want {
                return Err(InferError::Format(
                    "checkpoint shape does not match this model".into(),
                ));
            }
        }
        let qat = ckpt::get_bool(&g, "qat")?;
        let step_t = ckpt::get_u64(&g, "step")?;
        let rng_state = ckpt::get_u64(&g, "rng")?;
        let run = RunState::read(&g, step_t)?;

        let lens: Vec<usize> = param_data_ref(&self.model)
            .iter()
            .map(|s| s.len())
            .collect();
        check_no_placeholder_biases(&g, &lens, self.model.tied_head)?;
        let read = |suffix: &str| -> Result<Vec<f32>> {
            let mut flat = Vec::with_capacity(self.num_params());
            for (i, &n) in lens.iter().enumerate() {
                flat.extend(ckpt::get_f32(&g, &format!("p{i}.{suffix}"), n)?);
            }
            Ok(flat)
        };
        let (wflat, mflat, vflat) = (read("data")?, read("m")?, read("v")?);

        self.qat = qat;
        self.step_t = step_t;
        self.run = run;
        let mut off = 0usize;
        for s in param_data_mut(&mut self.model) {
            let n = s.len();
            s.copy_from_slice(&wflat[off..off + n]);
            off += n;
        }
        self.m.load_flat(&mflat);
        self.v.load_flat(&vflat);
        self.model.sync_tied_head();
        Ok(Rng::from_state(rng_state))
    }
}

#[cfg(test)]
mod tests {
    use super::apply_checkpoint_weights;
    use crate::ckpt;
    use crate::llm_train::params::param_data_ref;
    use crate::llm_train::test_support::{base_clone, bits_of, tiny, trained_trainer};
    use crate::llm_train::LlamaTrainer;
    use crate::rng::Rng;

    #[test]
    fn checkpoint_resume_matches_uninterrupted() {
        let base = tiny(6, 8, 2, 2, 16, 2);
        let tokens: Vec<usize> = (0..100).map(|i| (i * 7) % 6).collect();

        // Uninterrupted: 6 epochs.
        let mut a = LlamaTrainer::new(base_clone(&base)).unwrap();
        a.set_lr(0.02);
        let mut ra = Rng::new(123);
        for _ in 0..6 {
            a.finetune_epoch(&tokens, 6, 16, &mut ra).unwrap();
        }

        // Interrupted: 3 epochs, checkpoint, reload into a fresh base, 3 more.
        let mut b = LlamaTrainer::new(base_clone(&base)).unwrap();
        b.set_lr(0.02);
        let mut rb = Rng::new(123);
        for _ in 0..3 {
            b.finetune_epoch(&tokens, 6, 16, &mut rb).unwrap();
        }
        let bytes = b.save_checkpoint(&rb);
        let mut c = LlamaTrainer::new(base_clone(&base)).unwrap();
        c.set_lr(0.02);
        let mut rc = c.load_checkpoint_into(&bytes).unwrap();
        for _ in 0..3 {
            c.finetune_epoch(&tokens, 6, 16, &mut rc).unwrap();
        }

        assert_eq!(a.step_count(), c.step_count());
        for (x, y) in a.model_flat().iter().zip(&c.model_flat()) {
            assert!((x - y).abs() < 1e-5, "resumed run diverged: {x} vs {y}");
        }
    }

    #[test]
    fn checkpoint_is_gguf_and_roundtrips_every_field_bit_exactly() {
        for qat in [false, true] {
            let (t, rng) = trained_trainer(qat);
            let bytes = t.save_checkpoint(&rng);
            assert_eq!(&bytes[..4], b"GGUF");
            let g = crate::gguf::Gguf::parse(bytes.clone()).unwrap();
            assert_eq!(g.architecture(), Some("ferrum-llama-checkpoint"));
            let mut u = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 2)).unwrap();
            let rng2 = u.load_checkpoint_into(&bytes).unwrap();
            assert_eq!(rng2.state(), rng.state());
            assert_eq!(u.qat_enabled(), qat);
            assert_eq!(u.step_count(), t.step_count());
            assert!(u.step_count() > 0);
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(bits(u.model_flat()), bits(t.model_flat()));
            assert_eq!(bits(u.m.to_flat()), bits(t.m.to_flat()));
            assert_eq!(bits(u.v.to_flat()), bits(t.v.to_flat()));
            assert!(t.m.to_flat().iter().any(|&x| x != 0.0), "moments exercised");
            assert_eq!(u.save_checkpoint(&rng2), bytes);
            // The owning variant (no copy of the file) restores the same state.
            let mut w = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 2)).unwrap();
            let rng3 = w.load_checkpoint_from_vec(bytes.clone()).unwrap();
            assert_eq!(w.save_checkpoint(&rng3), bytes);
            assert!(w.load_checkpoint_from_vec(Vec::new()).is_err());
        }
    }

    #[test]
    fn old_checkpoint_magic_is_a_clear_error() {
        let mut tr = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        let err = tr
            .load_checkpoint_into(b"FLCK\x01\x00\x00\x00")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("pre-GGUF checkpoint (FLCK)"), "{err}");
        assert!(err.contains("retrain or start a new checkpoint"), "{err}");
    }

    #[test]
    fn checkpoint_rejects_wrong_architecture_truncation_and_tampering() {
        use crate::gguf::MetaValue;
        let (t, rng) = trained_trainer(true);
        let bytes = t.save_checkpoint(&rng);
        let mut fresh = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 2)).unwrap();
        for cut in [0, 4, 100, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                fresh.load_checkpoint_into(&bytes[..cut]).is_err(),
                "cut {cut}"
            );
        }
        // A TransformerNet checkpoint is not a Llama checkpoint.
        let mut r = Rng::new(1);
        let net = crate::train_transformer::TransformerNet::new(6, 4, 8, 2, 16, 1, &mut r).unwrap();
        let e = fresh
            .load_checkpoint_into(&net.save_checkpoint(&r))
            .err()
            .unwrap();
        assert!(
            e.to_string().contains("not a ferrum-llama-checkpoint"),
            "{e}"
        );
        let key = |k: &str| format!("ferrum.ckpt.{k}");
        for k in [
            "vocab",
            "model_dim",
            "layers",
            "heads",
            "kv_heads",
            "head_dim",
            "ffn",
            "step",
            "rng",
            "qat",
        ] {
            let kk = key(k);
            let b = crate::ckpt::rewrite(&bytes, |m| drop(m.remove(&kk)), None);
            let e = fresh.load_checkpoint_into(&b).err().unwrap().to_string();
            assert!(e.contains(&kk), "{k}: {e}");
        }
        // Every shape key is cross-checked against the model it loads into.
        for k in [
            "vocab",
            "model_dim",
            "layers",
            "heads",
            "kv_heads",
            "head_dim",
            "ffn",
        ] {
            for v in [99, 4_000_000_000] {
                let b = crate::ckpt::rewrite(
                    &bytes,
                    |m| {
                        m.insert(key(k), MetaValue::U32(v));
                    },
                    None,
                );
                let e = fresh.load_checkpoint_into(&b).err().unwrap().to_string();
                assert!(e.contains("does not match this model"), "{k}={v}: {e}");
            }
        }
        // Tensors that disagree with the model's parameter sizes.
        let b = crate::ckpt::rewrite(&bytes, |_| {}, Some("p2.m"));
        let e = fresh.load_checkpoint_into(&b).err().unwrap().to_string();
        assert!(e.contains("missing tensor 'p2.m'"), "{e}");
        // A failed load leaves the target trainer untouched: weights and Adam
        // moments are bit-identical, even when the failure is the *last* tensor
        // (so earlier ones were already read successfully).
        let bits = |t: &LlamaTrainer| {
            [t.model_flat(), t.m.to_flat(), t.v.to_flat()]
                .map(|v| v.into_iter().map(f32::to_bits).collect::<Vec<_>>())
        };
        let before = bits(&fresh);
        let last = param_data_ref(&fresh.model).len() - 1;
        let b = crate::ckpt::rewrite(&bytes, |_| {}, Some(&format!("p{last}.v")));
        assert!(fresh.load_checkpoint_into(&b).is_err());
        assert_eq!(bits(&fresh), before);
        assert_eq!(fresh.step_count(), 0);
        assert!(!fresh.qat_enabled());
    }

    /// Nothing allocated or read is sized by the header: shape keys are
    /// compared with the existing model, and tensor sizes come from the model
    /// and the directory (a tensor claiming 4e9 elements in a tiny file is
    /// rejected before it is read).
    #[test]
    fn crafted_checkpoint_sizes_do_not_drive_allocation() {
        let mut fresh = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        let mut b = ckpt::begin("ferrum-llama-checkpoint");
        for (k, v) in fresh.shape_keys() {
            ckpt::put_u32(&mut b, k, v);
        }
        ckpt::put_bool(&mut b, "qat", false);
        ckpt::put_u64(&mut b, "step", 0);
        ckpt::put_u64(&mut b, "rng", 1);
        for suffix in ["data", "m", "v"] {
            b.tensor(&format!("p0.{suffix}"), &[4_000_000_000], 0, vec![0u8; 16]);
        }
        assert!(fresh.load_checkpoint_into(&b.into_bytes()).is_err());
        assert_eq!(fresh.step_count(), 0);
    }

    /// The weights-only apply (no trainer, no flat vectors, no optimizer-state
    /// copies) must give the model exactly the weights a full restore does.
    #[test]
    fn apply_checkpoint_weights_equals_a_full_restore_bit_for_bit() {
        for qat in [false, true] {
            let (t, rng) = trained_trainer(qat);
            let bytes = t.save_checkpoint(&rng);
            let mut via_trainer = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 2)).unwrap();
            via_trainer.load_checkpoint_from_vec(bytes.clone()).unwrap();
            let mut model = tiny(6, 8, 2, 2, 16, 2);
            let fresh = bits_of(&model);
            apply_checkpoint_weights(&mut model, bytes).unwrap();
            assert_eq!(bits_of(&model), bits_of(&via_trainer.model));
            assert_eq!(bits_of(&model), bits_of(&t.model));
            assert_ne!(bits_of(&model), fresh, "training moved the weights");
        }
    }

    /// Regression (deferred minor): `--resume` built a LlamaTrainer (4x the
    /// model) just to copy weights. The weights-only path must reject what the
    /// trainer rejects and leave the model untouched on every failure.
    #[test]
    fn apply_checkpoint_weights_rejects_bad_input_and_leaves_the_model_untouched() {
        use crate::gguf::MetaValue;
        let (t, rng) = trained_trainer(false);
        let bytes = t.save_checkpoint(&rng);
        let mut model = tiny(6, 8, 2, 2, 16, 2);
        let before = bits_of(&model);
        let mut fails = |b: Vec<u8>, want: &str| {
            let e = apply_checkpoint_weights(&mut model, b)
                .err()
                .unwrap()
                .to_string();
            assert!(e.contains(want), "{want}: {e}");
            assert_eq!(bits_of(&model), before, "model modified on failure");
        };
        fails(Vec::new(), "");
        fails(b"XXXXjunk".to_vec(), "");
        fails(bytes[..bytes.len() / 2].to_vec(), "");
        let key = |k: &str| format!("ferrum.ckpt.{k}");
        for k in ["vocab", "ffn", "step", "rng", "qat"] {
            let kk = key(k);
            let b = crate::ckpt::rewrite(&bytes, |m| drop(m.remove(&kk)), None);
            fails(b, &kk);
        }
        let b = crate::ckpt::rewrite(
            &bytes,
            |m| {
                m.insert(key("model_dim"), MetaValue::U32(99));
            },
            None,
        );
        fails(b, "does not match this model");
        // Even a missing *last* weight tensor changes nothing: all are checked first.
        let last = param_data_ref(&tiny(6, 8, 2, 2, 16, 2)).len() - 1;
        let b = crate::ckpt::rewrite(&bytes, |_| {}, Some(&format!("p{last}.data")));
        fails(b, &format!("missing tensor 'p{last}.data'"));
        // A checkpoint from another architecture family.
        let mut r = Rng::new(1);
        let net = crate::train_transformer::TransformerNet::new(6, 4, 8, 2, 16, 1, &mut r).unwrap();
        fails(net.save_checkpoint(&r), "not a ferrum-llama-checkpoint");
    }

    #[test]
    fn checkpoint_rejects_corrupt_and_mismatched() {
        let mut tr = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 1)).unwrap();
        assert!(tr.load_checkpoint_into(b"").is_err());
        assert!(tr.load_checkpoint_into(b"XXXXjunk").is_err());
        // Shape mismatch: checkpoint from a different model.
        let other = LlamaTrainer::new(tiny(7, 8, 2, 2, 16, 1)).unwrap();
        let bytes = other.save_checkpoint(&Rng::new(1));
        assert!(tr.load_checkpoint_into(&bytes).is_err());
    }
}
