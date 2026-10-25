//! Quantization-aware training: snapshot, Q8_0 snap, and restore of master weights.
use crate::train_transformer::TransformerNet;

impl TransformerNet {
    /// Copy of every parameter tensor's data (the fp32 master weights).
    pub(crate) fn snapshot_weights(&mut self) -> Vec<Vec<f32>> {
        self.param_tensors_mut()
            .iter()
            .map(|t| t.data.clone())
            .collect()
    }

    /// Snap exactly the tensors the GGUF writer stores as Q8_0 onto their
    /// Q8_0 values (32-blocks, f16 scale, same axis as GGML): the token
    /// embedding along its rows, `[in, out]` weights down each column, the
    /// untied head likewise. 1-D params, the position embedding (GGUF keeps
    /// `position_embd` F32, the llama-quantize convention) and matrices whose
    /// GGML row length is not a multiple of 32 (stored F32) are left alone.
    /// With weight tying the head is re-mirrored from the snapped embedding.
    pub(crate) fn fake_quantize_weights(&mut self) {
        use crate::quant::{fake_quantize_q8_0_cols, fake_quantize_q8_0_rows};
        let e = self.embed_dim;
        fake_quantize_q8_0_rows(&mut self.tok_emb.data.data, e);
        for b in &mut self.blocks {
            for w in [
                &mut b.q_w,
                &mut b.k_w,
                &mut b.v_w,
                &mut b.o_w,
                &mut b.f1_w,
                &mut b.f2_w,
            ] {
                let (rows, cols) = (w.data.shape[0], w.data.shape[1]);
                fake_quantize_q8_0_cols(&mut w.data.data, rows, cols);
            }
        }
        if self.tie_weights {
            self.sync_tied_head();
        } else {
            let v = self.vocab_size;
            fake_quantize_q8_0_cols(&mut self.head_w.data.data, e, v);
        }
    }

    /// Restore master weights captured by [`TransformerNet::snapshot_weights`].
    pub(crate) fn restore_weights(&mut self, snapshot: &[Vec<f32>]) {
        for (t, s) in self.param_tensors_mut().into_iter().zip(snapshot) {
            t.data.copy_from_slice(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::optim::Adam;
    use crate::rng::Rng;
    use crate::tensor::Tensor;
    use crate::train_transformer::epoch::train_transformer_epoch;
    use crate::train_transformer::TransformerNet;

    #[test]
    fn qat_training_reduces_loss() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_qat(true);
        assert!(net.qat_enabled());
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        let first = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..30 {
            last = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        assert!(
            last < first * 0.5,
            "QAT loss did not halve: {first:.4} → {last:.4}"
        );
    }

    #[test]
    fn qat_snaps_only_q8_0_eligible_matrices() {
        let mut rng = Rng::new(2);
        // embed 48: ne0 = 48 is not a multiple of 32, so those tensors stay f32;
        // hidden 64: ffn2 has ne0 = 64, so it is snapped.
        let mut net = TransformerNet::new(40, 8, 48, 2, 64, 1, &mut rng).unwrap();
        let before = net.snapshot_weights();
        net.fake_quantize_weights();
        let after = net.snapshot_weights();
        assert!(
            before.iter().zip(&after).any(|(a, b)| a != b),
            "ffn2 must be snapped"
        );
        let shapes: Vec<Vec<usize>> = net
            .param_tensors_mut()
            .iter()
            .map(|t| t.shape.clone())
            .collect();
        let mut changed_2d = 0;
        for ((sh, a), b) in shapes.iter().zip(&before).zip(&after) {
            if sh.len() == 1 {
                assert_eq!(a, b, "1-D params must never be snapped");
            } else if a != b {
                changed_2d += 1;
                assert_eq!(sh, &vec![64, 48], "only ffn2 [64,48] is eligible");
            }
        }
        assert_eq!(changed_2d, 1);
    }

    #[test]
    fn qat_snap_is_idempotent_keeps_pos_emb_f32_and_tied_head_mirrors_embedding() {
        let mut rng = Rng::new(3);
        let mut net = TransformerNet::new(8, 4, 32, 2, 64, 1, &mut rng).unwrap();
        net.set_weight_tying(true);
        let pos_before = net.pos_emb.data.data.clone();
        net.fake_quantize_weights();
        // position_embd stays F32 in GGUF (llama-quantize convention): never snapped.
        assert_eq!(
            net.pos_emb.data.data, pos_before,
            "pos_emb must not be snapped"
        );
        let once = net.snapshot_weights();
        net.fake_quantize_weights();
        assert_eq!(net.snapshot_weights(), once, "snap must be idempotent");
        // head mirrors the snapped embedding
        let (v, e) = (8, 32);
        for i in 0..e {
            for j in 0..v {
                assert_eq!(
                    net.head_w.data.data[i * v + j],
                    net.tok_emb.data.data[j * e + i]
                );
            }
        }
    }

    #[test]
    fn qat_snapshot_quantize_restore_cycle() {
        let mut rng = Rng::new(11);
        let mut net = TransformerNet::new(8, 4, 32, 2, 64, 1, &mut rng).unwrap();
        let masters = net.snapshot_weights();
        net.fake_quantize_weights();
        let quantized = net.snapshot_weights();
        // Quantization actually moved (large) weights off their fp32 values.
        assert!(
            masters.iter().zip(&quantized).any(|(m, q)| m != q),
            "fake quantization changed no weights"
        );
        // Snapping is idempotent — re-quantizing lands on the same grid (so
        // every value already sits on its row's int8 grid).
        net.fake_quantize_weights();
        assert_eq!(
            net.snapshot_weights(),
            quantized,
            "fake quantization is not idempotent"
        );
        // Restore must bring back the fp32 masters exactly.
        net.restore_weights(&masters);
        assert_eq!(net.snapshot_weights(), masters);
    }

    #[test]
    fn qat_model_survives_int8_export_exactly() {
        use crate::loader::{from_bytes, to_bytes_quantized};
        use crate::meta::{ModelMetadata, Normalizer, TaskType};
        let mut rng = Rng::new(13);
        let mut net = TransformerNet::new(4, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_qat(true);
        let tokens: Vec<usize> = (0..200).map(|i| i % 4).collect();
        let adam = Adam::new(0.01);
        for _ in 0..20 {
            train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        let model = net.to_inference().unwrap();
        let meta = ModelMetadata {
            dataset_name: "qat".into(),
            task: TaskType::TransformerSLM,
            feature_names: vec![],
            feature_ranges: vec![],
            class_names: (0..4).map(|i| i.to_string()).collect(),
            target_name: "next".into(),
            target_range: [0.0, 4.0],
            input_dim: 4,
            output_dim: 4,
            tokenizer_state: String::new(),
        };
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let bytes = to_bytes_quantized(&model, &norm, &meta).unwrap();
        let (m2, _, _) = from_bytes(&bytes).unwrap();
        let x = Tensor::matrix(1, 4, vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let before = model.forward(&x).unwrap();
        let after = m2.forward(&x).unwrap();
        // QAT trained on the exact grid the Q8_0 export uses, so the export is
        // lossless: the reloaded model reproduces the trained one bit for bit.
        assert_eq!(
            before.data, after.data,
            "int8 export of a QAT model drifted"
        );
    }
}
