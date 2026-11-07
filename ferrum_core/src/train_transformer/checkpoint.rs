//! GGUF checkpoint save / load with strict header validation.
use crate::ckpt;
use crate::error::{InferError, Result};
pub(crate) use crate::resume::{EpochStart, RunSettings};
use crate::rng::Rng;
use crate::train_transformer::TransformerNet;

/// What a training checkpoint records about the net: its architecture and progress.
/// (Not the corpus, learning rate or seed: a checkpoint stores none of those.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointInfo {
    pub vocab: usize,
    pub context: usize,
    pub embed: usize,
    pub heads: usize,
    pub hidden: usize,
    pub blocks: usize,
    pub step: u64,
}

impl RunSettings {
    pub(crate) fn of(cfg: &crate::slm::TransformerConfig) -> Self {
        Self {
            batch: cfg.batch_size.max(1) as u64,
            stride: if cfg.window_stride == 0 {
                cfg.context_len
            } else {
                cfg.window_stride
            } as u64,
            lr: cfg.lr,
            cosine: cfg.cosine_lr,
            warmup: cfg.warmup_steps,
            clip: cfg.grad_clip,
            weight_decay: cfg.weight_decay,
            dropout: cfg.dropout,
        }
    }
}

impl TransformerNet {
    /// Read just the header of [`TransformerNet::save_checkpoint`] bytes (no weights are decoded),
    /// e.g. to tell the user which architecture a `--resume` will really use.
    pub fn checkpoint_info(bytes: Vec<u8>) -> Result<CheckpointInfo> {
        let g = ckpt::open_owned(bytes, b"FCKP", "ferrum-checkpoint")?;
        let n = |k| ckpt::get_usize(&g, k);
        let info = CheckpointInfo {
            vocab: n("vocab")?,
            context: n("context")?,
            embed: n("embed")?,
            heads: n("heads")?,
            hidden: n("hidden")?,
            blocks: n("blocks")?,
            step: ckpt::get_u64(&g, "step")?,
        };
        let dims = [
            info.vocab,
            info.context,
            info.embed,
            info.heads,
            info.hidden,
            info.blocks,
        ];
        Self::check_header(&g, dims)?;
        Ok(info)
    }

    /// Serialize the **full training state** (T6): architecture, QAT/tying
    /// flags, optimizer timestep, the supplied RNG's state, the current epoch's start when a
    /// config-driven run is under way (so a resume replays the rest of its window order), and every
    /// parameter's weights *and* Adam moment buffers (`m`, `v`). Pair with
    /// [`TransformerNet::load_checkpoint`] to resume an interrupted run exactly
    /// where it stopped — unlike a finished-model save, this preserves the
    /// optimizer state so momentum/adaptive scales are not lost.
    ///
    /// (The learning-rate schedule and grad-clip threshold are training-driver
    /// settings, not weights; re-apply them on the resumed net.)
    pub fn save_checkpoint(&self, rng: &Rng) -> Vec<u8> {
        let mut b = ckpt::begin("ferrum-checkpoint");
        for (k, v) in [
            ("vocab", self.vocab_size),
            ("context", self.context_len),
            ("embed", self.embed_dim),
            ("heads", self.num_heads),
            ("hidden", self.hidden_dim()),
            ("blocks", self.blocks.len()),
        ] {
            ckpt::put_u32(&mut b, k, v);
        }
        ckpt::put_bool(&mut b, "qat", self.qat);
        ckpt::put_bool(&mut b, "tie", self.tie_weights);
        ckpt::put_u64(&mut b, "step", self.step_t);
        ckpt::put_u64(&mut b, "rng", rng.state());
        if let Some(id) = &self.corpus_id {
            ckpt::put_str(&mut b, "corpus_id", id);
        }
        if let Some(r) = &self.run_settings {
            r.write(&mut b);
        }
        if let Some(e) = self.epoch_start {
            e.write(&mut b);
        }
        for (i, p) in self.params_all().into_iter().enumerate() {
            ckpt::put_param(&mut b, i, &p.data.data, &p.m.data, &p.v.data);
        }
        b.into_bytes()
    }

    /// Reject a checkpoint whose header dims do not account for exactly the
    /// tensors it carries (`dims` = vocab, context, embed, heads, hidden,
    /// blocks). Runs before any header-sized allocation.
    fn check_header(g: &crate::gguf::Gguf, dims: [usize; 6]) -> Result<()> {
        let bad = || InferError::Format("checkpoint header does not match its tensors".into());
        let [vocab, ctx, embed, heads, hidden, blocks] = dims;
        if dims.contains(&0) || !embed.is_multiple_of(heads) {
            return Err(bad());
        }
        // 5 non-block params (2 embeddings, final LN scale + shift, head).
        let n_params = blocks
            .checked_mul(Self::BLOCK_PARAMS)
            .and_then(|b| b.checked_add(5))
            .ok_or_else(bad)?;
        if n_params.checked_mul(3) != Some(g.tensors.len()) {
            return Err(bad());
        }
        let lens = Self::param_lens(vocab, ctx, embed, hidden, blocks).ok_or_else(bad)?;
        // Exactly `p{i}.{data,m,v}`, F32, of the expected length. With the
        // count check above and distinct names, this accounts for every
        // tensor, so (non-overlapping, in-file) extents bound what `new` may
        // allocate by the file size.
        let by_name: std::collections::HashMap<&str, &crate::gguf::TensorInfo> =
            g.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        for (i, &len) in lens.iter().enumerate() {
            for s in ["data", "m", "v"] {
                match by_name.get(format!("p{i}.{s}").as_str()) {
                    Some(t) if t.ggml_type == crate::gguf::GGML_F32 && t.dims == [len as u64] => {}
                    _ => return Err(bad()),
                }
            }
        }
        Ok(())
    }

    /// Rebuild a net and RNG from [`TransformerNet::save_checkpoint`] bytes,
    /// restoring weights, Adam moments, optimizer timestep, and RNG state so
    /// training resumes deterministically. Pre-GGUF (`FCKP`) checkpoints are
    /// rejected with a pointed error.
    pub fn load_checkpoint(bytes: &[u8]) -> Result<(Self, Rng)> {
        Self::from_checkpoint(ckpt::open(bytes, b"FCKP", "ferrum-checkpoint")?)
    }

    /// [`TransformerNet::load_checkpoint`] taking ownership of the bytes (no copy).
    pub fn load_checkpoint_from_vec(bytes: Vec<u8>) -> Result<(Self, Rng)> {
        Self::from_checkpoint(ckpt::open_owned(bytes, b"FCKP", "ferrum-checkpoint")?)
    }

    fn from_checkpoint(g: crate::gguf::Gguf) -> Result<(Self, Rng)> {
        let n = |k| ckpt::get_usize(&g, k);
        let (vocab, ctx, embed) = (n("vocab")?, n("context")?, n("embed")?);
        let (heads, hidden, blocks) = (n("heads")?, n("hidden")?, n("blocks")?);
        // The header must be consistent with the (file-bounded) tensors before
        // `new` allocates anything sized by it.
        Self::check_header(&g, [vocab, ctx, embed, heads, hidden, blocks])?;
        let mut seed_rng = Rng::new(1); // throwaway: param data is overwritten
        let mut net = TransformerNet::new(vocab, ctx, embed, heads, hidden, blocks, &mut seed_rng)?;
        net.qat = ckpt::get_bool(&g, "qat")?;
        net.tie_weights = ckpt::get_bool(&g, "tie")?;
        net.step_t = ckpt::get_u64(&g, "step")?;
        let rng_state = ckpt::get_u64(&g, "rng")?;
        net.epoch_start = EpochStart::read(&g, net.step_t)?;
        net.corpus_id = ckpt::get_str(&g, "corpus_id")?;
        net.run_settings = RunSettings::read(&g)?;
        for (i, p) in net.params_mut_all().into_iter().enumerate() {
            // m and v share the parameter's shape, so each is the same length.
            let len = p.data.data.len();
            p.data.data = ckpt::get_f32(&g, &format!("p{i}.data"), len)?;
            p.m.data = ckpt::get_f32(&g, &format!("p{i}.m"), len)?;
            p.v.data = ckpt::get_f32(&g, &format!("p{i}.v"), len)?;
        }
        Ok((net, Rng::from_state(rng_state)))
    }
}

#[cfg(test)]
mod tests {
    use crate::optim::Adam;
    use crate::rng::Rng;
    use crate::train_transformer::epoch::train_transformer_epoch;
    use crate::train_transformer::test_support::{mangled, trained_net};
    use crate::train_transformer::{EpochStart, TransformerNet};
    use crate::{MetaValue, NumArray};

    /// Resuming from a mid-training checkpoint reproduces uninterrupted training
    /// bit-for-bit — proving weights, Adam moments, step counter, and RNG state
    /// are all preserved.
    #[test]
    fn checkpoint_resume_matches_uninterrupted_training() {
        let base = {
            let mut r = Rng::new(7);
            let mut n = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
            n.set_qat(true); // exercise the QAT path through the checkpoint
            n
        };
        let tokens: Vec<usize> = (0..300).map(|i| (i * 7) % 6).collect();
        let adam = Adam::new(0.02);

        // Uninterrupted: 6 epochs straight through.
        let mut net_a = base.clone();
        let mut rng_a = Rng::new(123);
        for _ in 0..6 {
            train_transformer_epoch(&mut net_a, &tokens, 16, &adam, &mut rng_a).unwrap();
        }

        // Interrupted: 3 epochs, checkpoint, reload, 3 more.
        let mut net_b = base.clone();
        let mut rng_b = Rng::new(123);
        for _ in 0..3 {
            train_transformer_epoch(&mut net_b, &tokens, 16, &adam, &mut rng_b).unwrap();
        }
        let bytes = net_b.save_checkpoint(&rng_b);
        drop(net_b);
        let (mut net_c, mut rng_c) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert!(net_c.qat_enabled(), "QAT flag must survive the checkpoint");
        for _ in 0..3 {
            train_transformer_epoch(&mut net_c, &tokens, 16, &adam, &mut rng_c).unwrap();
        }

        assert_eq!(net_a.step_count(), net_c.step_count());
        assert_eq!(
            net_a.snapshot_weights(),
            net_c.snapshot_weights(),
            "resumed training diverged from the uninterrupted run"
        );
    }

    #[test]
    fn checkpoint_roundtrips_epoch_rng() {
        let (mut net, rng) = trained_net(false, false);
        // A net outside the config-driven loop writes no key and reads none back.
        let (plain, _) = TransformerNet::load_checkpoint(&net.save_checkpoint(&rng)).unwrap();
        assert_eq!(plain.epoch_start, None);
        let start = EpochStart {
            rng: u64::MAX - 1,
            step: net.step_count(),
            epoch: 3,
        };
        net.epoch_start = Some(start);
        let bytes = net.save_checkpoint(&rng);
        let (back, _) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert_eq!(back.epoch_start, Some(start));
        assert_eq!(back.save_checkpoint(&rng), bytes);
        // Malformed: a wrong type, a wrong length, a start after the checkpoint, epoch 0.
        let bad = |v: MetaValue| {
            let b = crate::ckpt::rewrite(
                &bytes,
                |m| {
                    m.insert("ferrum.ckpt.epoch_rng".into(), v);
                },
                None,
            );
            TransformerNet::load_checkpoint(&b)
                .err()
                .unwrap()
                .to_string()
        };
        let step = net.step_count();
        let u64s = |v: Vec<u64>| NumArray::U64(v).into_meta();
        assert!(bad(MetaValue::U64(1)).contains("epoch_rng"));
        assert!(bad(u64s(vec![1, 2])).contains("[rng, step, epoch]"));
        assert!(bad(u64s(vec![1, step + 1, 1])).contains("[rng, step, epoch]"));
        assert!(bad(u64s(vec![1, step, 0])).contains("[rng, step, epoch]"));
    }

    #[test]
    fn checkpoint_roundtrips_run_settings() {
        use super::RunSettings;
        let (mut net, rng) = trained_net(false, false);
        let (plain, _) = TransformerNet::load_checkpoint(&net.save_checkpoint(&rng)).unwrap();
        assert_eq!(plain.run_settings, None);
        let s = RunSettings {
            batch: 4,
            stride: 16,
            lr: 0.003,
            cosine: true,
            warmup: 20,
            clip: 1.0,
            weight_decay: 0.01,
            dropout: 0.1,
        };
        net.run_settings = Some(s);
        let (back, _) = TransformerNet::load_checkpoint(&net.save_checkpoint(&rng)).unwrap();
        assert_eq!(back.run_settings, Some(s));
        // A key of the wrong shape is an error, not a silent skip.
        let bad = crate::ckpt::rewrite(
            &net.save_checkpoint(&rng),
            |m| {
                m.insert(
                    "ferrum.ckpt.batch_size".into(),
                    MetaValue::Nums(NumArray::U64(vec![1, 2])),
                );
            },
            None,
        );
        let err = TransformerNet::load_checkpoint(&bad)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("batch_size"), "{err}");
    }

    #[test]
    fn run_settings_compare_words_and_orders_the_findings() {
        use super::RunSettings;
        let s = RunSettings {
            batch: 4,
            stride: 16,
            lr: 0.01,
            cosine: false,
            warmup: 0,
            clip: 0.0,
            weight_decay: 0.0,
            dropout: 0.0,
        };
        assert_eq!(s.compare(&s).unwrap(), None);
        let err = |n: RunSettings| s.compare(&n).unwrap_err().to_string();
        let e = err(RunSettings { batch: 8, ..s });
        assert!(e.contains("--batch is 8") && e.contains("batch 4"), "{e}");
        let e = err(RunSettings { stride: 1, ..s });
        assert!(
            e.contains("--stride is 1") && e.contains("stride 16"),
            "{e}"
        );
        let note = s
            .compare(&RunSettings {
                lr: 0.02,
                cosine: true,
                warmup: 5,
                clip: 1.0,
                weight_decay: 0.1,
                dropout: 0.2,
                ..s
            })
            .unwrap()
            .unwrap();
        for part in [
            "lr 0.01 -> 0.02",
            "cosine false -> true",
            "warmup 0 -> 5",
            "clip 0 -> 1",
            "weight_decay 0 -> 0.1",
            "dropout 0 -> 0.2",
        ] {
            assert!(note.contains(part), "{part} in {note}");
        }
        assert!(note.starts_with("note:"), "{note}");
    }

    #[test]
    fn checkpoint_roundtrips_corpus_id() {
        let (mut net, rng) = trained_net(false, false);
        assert_eq!(net.corpus_id, None);
        let plain = net.save_checkpoint(&rng);
        net.corpus_id = Some("sha256:00ff".into());
        let bytes = net.save_checkpoint(&rng);
        assert!(bytes != plain);
        let (back, _) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert_eq!(back.corpus_id.as_deref(), Some("sha256:00ff"));
        assert_eq!(back.save_checkpoint(&rng), bytes);
        // A wrong type is an error, not a silent "no corpus".
        let b = crate::ckpt::rewrite(
            &bytes,
            |m| {
                m.insert("ferrum.ckpt.corpus_id".into(), MetaValue::U64(1));
            },
            None,
        );
        let err = TransformerNet::load_checkpoint(&b)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("corpus_id"), "{err}");
    }

    /// Written by Ferrum before `ferrum.ckpt.epoch_rng` existed (3 steps of a 3-token net).
    const PRE_EPOCH_RNG: &[u8] = include_bytes!("testdata/pre_epoch_rng.ckpt.gguf");

    #[test]
    fn old_checkpoint_without_epoch_rng_still_loads() {
        let (net, rng) = TransformerNet::load_checkpoint(PRE_EPOCH_RNG).unwrap();
        assert_eq!(net.epoch_start, None);
        assert_eq!(net.corpus_id, None);
        assert_eq!(net.step_count(), 3);
        assert_eq!(rng.state(), 3354630805809648812);
        // Without an epoch start the writer emits exactly the old layout.
        assert_eq!(net.save_checkpoint(&rng), PRE_EPOCH_RNG);
        let info = TransformerNet::checkpoint_info(PRE_EPOCH_RNG.to_vec()).unwrap();
        assert_eq!((info.vocab, info.context, info.step), (3, 2, 3));
    }

    #[test]
    fn checkpoint_info_reports_architecture_and_step() {
        let mut r = Rng::new(1);
        let mut net = TransformerNet::new(30, 8, 16, 2, 48, 3, &mut r).unwrap();
        let tokens: Vec<usize> = (0..100).map(|i| i % 30).collect();
        let mut rng = Rng::new(5);
        train_transformer_epoch(&mut net, &tokens, 8, &Adam::new(0.01), &mut rng).unwrap();
        let bytes = net.save_checkpoint(&rng);
        let info = TransformerNet::checkpoint_info(bytes).unwrap();
        assert_eq!(
            (
                info.vocab,
                info.context,
                info.embed,
                info.heads,
                info.hidden,
                info.blocks
            ),
            (30, 8, 16, 2, 48, 3)
        );
        assert_eq!(info.step, net.step_count());
        assert!(TransformerNet::checkpoint_info(b"junk".to_vec()).is_err());
    }

    #[test]
    fn checkpoint_roundtrip_preserves_flags_and_rng() {
        let mut r = Rng::new(1);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
        net.set_qat(true);
        net.set_weight_tying(true);
        let tokens: Vec<usize> = (0..100).map(|i| i % 6).collect();
        let adam = Adam::new(0.01);
        let mut rng = Rng::new(5);
        train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();

        let bytes = net.save_checkpoint(&rng);
        let (mut net2, rng2) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert_eq!(net.snapshot_weights(), net2.snapshot_weights());
        assert_eq!(net.step_count(), net2.step_count());
        assert!(net2.qat_enabled());
        assert!(net2.weight_tying());
        assert_eq!(rng.state(), rng2.state(), "RNG state must round-trip");
    }

    #[test]
    fn load_checkpoint_rejects_corrupt_data() {
        assert!(TransformerNet::load_checkpoint(b"").is_err());
        assert!(TransformerNet::load_checkpoint(b"XXXXjunkdata").is_err());
        let mut r = Rng::new(1);
        let net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut r).unwrap();
        let bytes = net.save_checkpoint(&r);
        // Truncated payload is rejected, not silently zero-filled.
        assert!(TransformerNet::load_checkpoint(&bytes[..bytes.len() / 2]).is_err());
    }

    #[test]
    fn checkpoint_is_gguf_and_roundtrips() {
        let mut rng = Rng::new(5);
        let net = TransformerNet::new(30, 8, 16, 2, 32, 1, &mut rng).unwrap();
        let bytes = net.save_checkpoint(&rng);
        assert_eq!(&bytes[..4], b"GGUF");
        let g = crate::gguf::Gguf::parse(bytes.clone()).unwrap();
        assert_eq!(g.architecture(), Some("ferrum-checkpoint"));
        let (net2, rng2) = TransformerNet::load_checkpoint(&bytes).unwrap();
        assert_eq!(net2.save_checkpoint(&rng2), bytes);
        // The owning variant parses in place and yields the same net; errors are the same.
        let (net3, rng3) = TransformerNet::load_checkpoint_from_vec(bytes.clone()).unwrap();
        assert_eq!(net3.save_checkpoint(&rng3), bytes);
        let e = TransformerNet::load_checkpoint_from_vec(b"FCKP\x01\x00\x00\x00".to_vec())
            .err()
            .unwrap();
        assert!(e.to_string().contains("pre-GGUF checkpoint (FCKP)"), "{e}");
    }

    #[test]
    fn checkpoint_roundtrips_every_field_bit_exactly() {
        for (tie, qat) in [(false, false), (true, false), (false, true), (true, true)] {
            let (net, rng) = trained_net(tie, qat);
            assert!(net.step_count() > 0);
            let (net2, rng2) = TransformerNet::load_checkpoint(&net.save_checkpoint(&rng)).unwrap();
            assert_eq!(net2.qat_enabled(), qat);
            assert_eq!(net2.weight_tying(), tie);
            assert_eq!(net2.step_count(), net.step_count());
            assert_eq!(rng2.state(), rng.state());
            assert_eq!(net2.hidden_dim(), net.hidden_dim());
            let (a, b) = (net.params_all(), net2.params_all());
            assert_eq!(a.len(), b.len());
            let mut moments = 0.0f32;
            for (p, q) in a.iter().zip(&b) {
                for (x, y) in [(&p.data, &q.data), (&p.m, &q.m), (&p.v, &q.v)] {
                    assert_eq!(x.shape, y.shape);
                    assert!(x
                        .data
                        .iter()
                        .zip(&y.data)
                        .all(|(u, w)| u.to_bits() == w.to_bits()));
                }
                moments += p.m.data.iter().map(|v| v.abs()).sum::<f32>();
            }
            assert!(moments > 0.0, "test must exercise non-zero Adam moments");
        }
    }

    #[test]
    fn old_checkpoint_magic_is_a_clear_error() {
        let err = TransformerNet::load_checkpoint(b"FCKP\x01\x00\x00\x00")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("pre-GGUF checkpoint (FCKP)"), "{err}");
        assert!(err.contains("retrain or start a new checkpoint"), "{err}");
    }

    #[test]
    fn checkpoint_rejects_foreign_truncated_and_wrong_architecture() {
        for junk in [&b""[..], b"XXXXjunkdata", b"GGUF"] {
            assert!(TransformerNet::load_checkpoint(junk).is_err());
        }
        let (net, rng) = trained_net(false, false);
        let bytes = net.save_checkpoint(&rng);
        for cut in [bytes.len() / 2, bytes.len() - 1, 100] {
            assert!(
                TransformerNet::load_checkpoint(&bytes[..cut]).is_err(),
                "cut {cut}"
            );
        }
        let other = crate::ckpt::rewrite(
            &bytes,
            |m| {
                m.insert(
                    "general.architecture".into(),
                    crate::gguf::MetaValue::String("gpt2".into()),
                );
            },
            None,
        );
        let err = TransformerNet::load_checkpoint(&other)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("not a ferrum-checkpoint"), "{err}");
    }

    #[test]
    fn checkpoint_rejects_inconsistent_headers_and_tensors() {
        use crate::gguf::MetaValue;
        let (net, rng) = trained_net(true, true);
        let bytes = net.save_checkpoint(&rng);
        let key = |k: &str| format!("ferrum.ckpt.{k}");
        // A missing or mistyped scalar names its key.
        for k in [
            "vocab", "context", "embed", "heads", "hidden", "blocks", "step", "rng",
        ] {
            let kk = key(k);
            let b = crate::ckpt::rewrite(&bytes, |m| drop(m.remove(&kk)), None);
            let e = TransformerNet::load_checkpoint(&b)
                .err()
                .unwrap()
                .to_string();
            assert!(e.contains(&kk), "{k}: {e}");
        }
        for k in ["qat", "tie"] {
            let kk = key(k);
            let b = crate::ckpt::rewrite(&bytes, |m| drop(m.remove(&kk)), None);
            assert!(TransformerNet::load_checkpoint(&b).is_err(), "{k}");
            let b = crate::ckpt::rewrite(
                &bytes,
                |m| {
                    m.insert(kk.clone(), MetaValue::U32(1));
                },
                None,
            );
            assert!(TransformerNet::load_checkpoint(&b).is_err(), "{k} type");
        }
        // Header says a different shape than the tensors hold.
        let b = crate::ckpt::rewrite(
            &bytes,
            |m| {
                m.insert(key("embed"), MetaValue::U32(16));
            },
            None,
        );
        let e = TransformerNet::load_checkpoint(&b)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("does not match its tensors"), "{e}");
        // Invalid architecture numbers are rejected by the constructor.
        for (k, v) in [("vocab", 0), ("heads", 3)] {
            let b = crate::ckpt::rewrite(
                &bytes,
                |m| {
                    m.insert(key(k), MetaValue::U32(v));
                },
                None,
            );
            assert!(TransformerNet::load_checkpoint(&b).is_err(), "{k}");
        }
        // A dropped moment tensor is an error (the directory no longer matches
        // the header), never zero-filled.
        let b = crate::ckpt::rewrite(&bytes, |_| {}, Some("p3.v"));
        let e = TransformerNet::load_checkpoint(&b)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("does not match its tensors"), "{e}");
    }

    /// Crafted headers must fail fast with `Err`; if any of them drove an
    /// allocation (4e9 x 4e9 elements, 4e9 blocks) the test process would abort.
    #[test]
    fn crafted_headers_are_rejected_before_any_allocation() {
        use crate::gguf::MetaValue;
        let mut r = Rng::new(1);
        let net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
        let bytes = net.save_checkpoint(&r);
        let with = |k: &str, v: u32| {
            let k = format!("ferrum.ckpt.{k}");
            crate::ckpt::rewrite(
                &bytes,
                move |m| {
                    m.insert(k, MetaValue::U32(v));
                },
                None,
            )
        };
        let cases: [(&str, u32); 14] = [
            ("vocab", 4_000_000_000),
            ("context", 4_000_000_000),
            ("embed", 4_000_000_000),
            ("hidden", 4_000_000_000),
            ("blocks", 4_000_000_000),
            ("heads", 4_000_000_000),
            ("embed", 0),
            ("context", 0),
            ("hidden", 0),
            ("blocks", 0),
            ("heads", 0),
            ("heads", 3),  // does not divide embed 8
            ("blocks", 1), // fewer blocks than the tensor directory holds
            ("blocks", 3), // more blocks than it holds
        ];
        for (k, v) in cases {
            let t = std::time::Instant::now();
            let e = TransformerNet::load_checkpoint(&with(k, v)).err();
            assert!(e.is_some(), "{k}={v} must be rejected");
            assert!(t.elapsed().as_secs() < 5, "{k}={v} took too long");
        }
        // Tensor extents beyond EOF (header untouched).
        for cut in [4, 40] {
            assert!(TransformerNet::load_checkpoint(&bytes[..bytes.len() - cut]).is_err());
        }
        // A self-consistent claim of a huge net with no tensors at all.
        let mut huge = crate::ckpt::begin("ferrum-checkpoint");
        for k in ["vocab", "context", "embed", "heads", "hidden", "blocks"] {
            crate::ckpt::put_u32(&mut huge, k, 1 << 30);
        }
        crate::ckpt::put_bool(&mut huge, "qat", false);
        crate::ckpt::put_bool(&mut huge, "tie", false);
        crate::ckpt::put_u64(&mut huge, "step", 0);
        crate::ckpt::put_u64(&mut huge, "rng", 1);
        assert!(TransformerNet::load_checkpoint(&huge.into_bytes()).is_err());
    }

    /// `check_header` demands exactly `p{i}.{data,m,v}` F32 tensors of the
    /// header-implied length, even when the extents are all valid.
    #[test]
    fn check_header_requires_exact_tensor_names_dims_and_types() {
        let mut r = Rng::new(1);
        let net = TransformerNet::new(6, 4, 8, 2, 16, 2, &mut r).unwrap();
        let bytes = net.save_checkpoint(&r);
        TransformerNet::load_checkpoint(&bytes).unwrap(); // control
        type Edit = Box<dyn Fn(&mut (String, Vec<u64>, u32))>;
        let cases: Vec<(&str, Edit)> = vec![
            (
                "renamed tensor",
                Box::new(|t| {
                    if t.0 == "p3.m" {
                        t.0 = "q3.m".into();
                    }
                }),
            ),
            (
                "wrong dims",
                Box::new(|t| {
                    if t.0 == "p5.data" {
                        t.1 = vec![t.1[0] + 1];
                    }
                }),
            ),
            (
                "wrong rank",
                Box::new(|t| {
                    if t.0 == "p5.v" {
                        t.1 = vec![1, t.1[0]];
                    }
                }),
            ),
            (
                "wrong type (extent changes)",
                Box::new(|t| {
                    if t.0 == "p5.m" {
                        t.2 = crate::gguf::GGML_F16;
                    }
                }),
            ),
        ];
        for (why, edit) in cases {
            let e = TransformerNet::load_checkpoint(&mangled(&bytes, &edit))
                .err()
                .expect("must be rejected");
            assert!(
                e.to_string().contains("does not match its tensors"),
                "{why}: {e}"
            );
        }
    }

    /// The round-1 bypass: every `*.data` tensor aliases the same bytes, with
    /// a header chosen so the summed element count equals the architecture's
    /// parameter count. Rejected centrally (overlap) before any net exists,
    /// and by `check_header` on its own once the aliasing is removed.
    #[test]
    fn aliased_tensor_checkpoint_is_rejected_quickly() {
        const BLOCKS: usize = 100;
        const L: u64 = 1001; // elements per aliased `*.data` tensor (odd)
        let n_params = 5 + 16 * BLOCKS;
        // embed = heads = hidden = ctx = 1: total params = 2*vocab + 16*blocks + 3
        // must equal n_params * L, so vocab is solved for.
        let vocab = (n_params as u64 * L - (16 * BLOCKS as u64 + 3)) / 2;
        let mut b = crate::ckpt::begin("ferrum-checkpoint");
        for (k, v) in [
            ("vocab", vocab as usize),
            ("context", 1),
            ("embed", 1),
            ("heads", 1),
            ("hidden", 1),
            ("blocks", BLOCKS),
        ] {
            crate::ckpt::put_u32(&mut b, k, v);
        }
        crate::ckpt::put_bool(&mut b, "qat", false);
        crate::ckpt::put_bool(&mut b, "tie", false);
        crate::ckpt::put_u64(&mut b, "step", 0);
        crate::ckpt::put_u64(&mut b, "rng", 1);
        let data = vec![0.0f32; L as usize];
        for i in 0..n_params {
            crate::ckpt::put_param(&mut b, i, &data, &[0.0], &[0.0]);
        }
        let clean = b.into_bytes();
        // The file is ~(3 * n_params) directory entries plus ~L*4 data bytes
        // after aliasing; the net it claims would be ~2*vocab*16 B = ~51 MB.
        let mut bytes = clean.clone();
        for i in 0..3 * n_params {
            crate::gguf::patch_tensor_offset(&mut bytes, i, 0);
        }
        let t = std::time::Instant::now();
        let e = TransformerNet::load_checkpoint(&bytes).err().unwrap();
        assert!(e.to_string().contains("overlap"), "{e}");
        assert!(t.elapsed().as_secs() < 5, "took {:?}", t.elapsed());
        // Without aliasing the tensors are legitimate extents but wrong dims
        // for this header (data tensors are L long, the header wants 1 or
        // vocab): check_header refuses them before `new` allocates.
        let e = TransformerNet::load_checkpoint(&clean).err().unwrap();
        assert!(e.to_string().contains("does not match its tensors"), "{e}");
    }
}
