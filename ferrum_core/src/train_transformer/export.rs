//! Conversion of the trainable net into an inference Sequential.
use crate::error::Result;
use crate::layer::{ActivationLayer, Embedding, LayerNorm, Linear, TransformerBlock};
use crate::model::Sequential;
use crate::train_transformer::block::BlockT;
use crate::train_transformer::TransformerNet;

impl TransformerNet {
    /// Export to an inference `Sequential`:
    /// `Embedding → blocks → LayerNorm → Linear → Softmax`.
    /// Serializes to GGUF v3 and is compatible with `TransformerSLMModel`.
    /// A copy holding only the weights (no gradients or Adam moments): enough
    /// to snap, mirror the tied head and export, never to train.
    pub(super) fn weights_only(&self) -> Self {
        Self {
            vocab_size: self.vocab_size,
            context_len: self.context_len,
            embed_dim: self.embed_dim,
            num_heads: self.num_heads,
            tok_emb: self.tok_emb.weights_only(),
            pos_emb: self.pos_emb.weights_only(),
            blocks: self.blocks.iter().map(BlockT::weights_only).collect(),
            lnf_g: self.lnf_g.weights_only(),
            lnf_b: self.lnf_b.weights_only(),
            head_w: self.head_w.weights_only(),
            step_t: self.step_t,
            qat: self.qat,
            grad_clip: self.grad_clip,
            lr_schedule: self.lr_schedule,
            tie_weights: self.tie_weights,
            weight_decay: self.weight_decay,
            dropout: self.dropout,
            window_stride: self.window_stride,
            metrics: Default::default(),
            epoch_start: self.epoch_start,
            corpus_id: self.corpus_id.clone(),
            run_settings: self.run_settings,
        }
    }

    /// [`TransformerNet::to_inference`] for shipping: a QAT net is exported at
    /// its Q8_0-snapped values (what the training saw), so an int8 export
    /// changes nothing. Snaps a weights-only copy, leaving `self` (the
    /// full-precision masters) untouched, without cloning gradients or Adam
    /// state (4 B/param instead of 16).
    pub(crate) fn to_inference_snapped(&self) -> Result<Sequential> {
        if !self.qat {
            return self.to_inference();
        }
        let mut snapped = self.weights_only();
        snapped.fake_quantize_weights();
        snapped.to_inference()
    }

    /// Export the net as an inference `Sequential` (embedding, blocks, final LayerNorm, head) from the current f32 weights, without int8 snapping.
    pub fn to_inference(&self) -> Result<Sequential> {
        let mut m = Sequential::new();
        m.push(Box::new(Embedding::new(
            self.vocab_size,
            self.context_len,
            self.embed_dim,
            self.tok_emb.data.data.clone(),
            self.pos_emb.data.data.clone(),
        )?));
        for b in &self.blocks {
            m.push(Box::new(TransformerBlock::new(
                self.context_len,
                self.num_heads,
                self.embed_dim,
                b.ln1_g.data.data.clone(),
                b.ln1_b.data.data.clone(),
                b.q_w.data.data.clone(),
                b.q_b.data.data.clone(),
                b.k_w.data.data.clone(),
                b.k_b.data.data.clone(),
                b.v_w.data.data.clone(),
                b.v_b.data.data.clone(),
                b.o_w.data.data.clone(),
                b.o_b.data.data.clone(),
                b.ln2_g.data.data.clone(),
                b.ln2_b.data.data.clone(),
                b.f1_w.data.data.clone(),
                b.f1_b.data.data.clone(),
                b.f2_w.data.data.clone(),
                b.f2_b.data.data.clone(),
            )?));
        }
        m.push(Box::new(LayerNorm::new(
            self.embed_dim,
            self.lnf_g.data.data.clone(),
            self.lnf_b.data.data.clone(),
        )?));
        // With weight tying the shipped head is the (current) transpose of the
        // token embedding; otherwise it is the independently trained head.
        let head_w_data = if self.tie_weights {
            let (vocab, embed) = (self.vocab_size, self.embed_dim);
            let mut w = vec![0.0f32; embed * vocab];
            for e in 0..embed {
                for v in 0..vocab {
                    w[e * vocab + v] = self.tok_emb.data.data[v * embed + e];
                }
            }
            w
        } else {
            self.head_w.data.data.clone()
        };
        let head = if self.tie_weights {
            Linear::tied_head
        } else {
            Linear::without_bias
        };
        m.push(Box::new(head(
            self.embed_dim,
            self.vocab_size,
            head_w_data,
        )?));
        m.push(Box::new(ActivationLayer::new(
            crate::activation::Activation::Softmax,
        )));
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops;
    use crate::optim::Adam;
    use crate::rng::Rng;
    use crate::tensor::Tensor;
    use crate::train_transformer::epoch::train_transformer_epoch;
    use crate::train_transformer::kernels::{linear_bwd, linear_fwd, matmul_bwd};
    use crate::train_transformer::TransformerNet;

    /// The bias-free head path (`matmul_bwd`, plain `matmul` in `forward`) must
    /// be bit-identical to the old zero-bias `linear_*` route it replaced.
    #[test]
    fn bias_free_head_path_matches_linear_with_zero_bias() {
        let x = Tensor::matrix(3, 2, vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5]).unwrap();
        let w = Tensor::matrix(2, 4, vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]).unwrap();
        let dy = Tensor::matrix(3, 4, (0..12).map(|i| (i as f32 - 5.0) * 0.1).collect()).unwrap();
        let zero = Tensor::vector(vec![0.0; 4]);
        assert_eq!(
            ops::matmul(&x, &w).unwrap().data,
            linear_fwd(&x, &w, &zero).unwrap().data
        );
        let (dw, dx) = matmul_bwd(&x, &w, &dy).unwrap();
        let (dw2, db2, dx2) = linear_bwd(&x, &w, &dy).unwrap();
        assert_eq!((dw.data, dx.data), (dw2.data, dx2.data));
        assert_eq!(db2.data.len(), 4, "linear_bwd still returns the bias grad");
        // End to end: the net's logits are exactly xW of the final norm.
        let mut rng = Rng::new(5);
        let net = TransformerNet::new(7, 4, 8, 2, 16, 1, &mut rng).unwrap();
        let (logits, cache) = net.forward(&[1, 2, 3, 4]).unwrap();
        assert_eq!(
            logits.data,
            ops::matmul(&cache.norm_f, &net.head_w.data).unwrap().data
        );
    }

    /// `to_inference_snapped` must equal the old clone-then-snap export bit for
    /// bit, leave the masters alone, and not carry gradients or Adam state.
    #[test]
    fn snapped_export_matches_full_clone_and_keeps_masters() {
        for tie in [false, true] {
            let mut rng = Rng::new(17);
            let mut net = TransformerNet::new(8, 4, 32, 2, 64, 2, &mut rng).unwrap();
            net.set_weight_tying(tie);
            net.set_qat(true);
            let tokens: Vec<usize> = (0..120).map(|i| (i * 7) % 8).collect();
            let adam = Adam::new(0.01);
            for _ in 0..3 {
                train_transformer_epoch(&mut net, &tokens, 4, &adam, &mut rng).unwrap();
            }
            let masters = net.snapshot_weights();
            let mut full = net.clone();
            full.fake_quantize_weights();
            let want = full.to_inference().unwrap();
            let got = net.to_inference_snapped().unwrap();
            let x = Tensor::matrix(1, 4, vec![0.0, 3.0, 5.0, 7.0]).unwrap();
            assert_eq!(
                got.forward(&x).unwrap().data,
                want.forward(&x).unwrap().data
            );
            assert_eq!(
                net.snapshot_weights(),
                masters,
                "masters must not be snapped"
            );
            // Without QAT it is a plain export.
            net.set_qat(false);
            let plain = net.to_inference().unwrap();
            assert_eq!(
                net.to_inference_snapped()
                    .unwrap()
                    .forward(&x)
                    .unwrap()
                    .data,
                plain.forward(&x).unwrap().data
            );
        }
        // The copy really is weights-only.
        let mut rng = Rng::new(1);
        let net = TransformerNet::new(8, 4, 32, 2, 64, 1, &mut rng).unwrap();
        let mut w = net.weights_only();
        for p in w.params_mut_all() {
            assert_eq!((p.grad.numel(), p.m.numel(), p.v.numel()), (0, 0, 0));
            assert!(p.data.numel() > 0);
        }
    }

    /// A net whose tensors disagree with its dimensions cannot be exported: each
    /// layer's constructor rejects it and the error reaches the caller.
    #[test]
    fn to_inference_rejects_inconsistent_weights() {
        let fresh = || TransformerNet::new(6, 4, 8, 2, 12, 1, &mut Rng::new(2)).unwrap();
        assert!(fresh().to_inference().is_ok());
        let mut net = fresh();
        net.tok_emb.data.data.pop();
        assert!(net.to_inference().is_err());
        let mut net = fresh();
        net.blocks[0].f2_b.data.data.pop();
        assert!(net.to_inference().is_err());
        let mut net = fresh();
        net.lnf_g.data.data.pop();
        assert!(net.to_inference().is_err());
        let mut net = fresh();
        net.head_w.data.data.pop();
        assert!(net.to_inference().is_err());
    }

    #[test]
    fn head_has_no_bias() {
        let mut rng = Rng::new(9);
        let net = TransformerNet::new(40, 8, 16, 2, 32, 2, &mut rng).unwrap();
        // tok 640 + pos 128 + 2 blocks * 2224 + lnf 32 + head_w 640; no `+ vocab` bias term.
        assert_eq!(net.num_params(), 640 + 128 + 2 * 2224 + 32 + 640);
    }

    #[test]
    fn to_inference_matches_trainable_forward() {
        let mut rng = Rng::new(21);
        let net = TransformerNet::new(6, 4, 8, 2, 12, 2, &mut rng).unwrap();
        let tokens = [0usize, 3, 1, 5];
        let (logits, _) = net.forward(&tokens).unwrap();

        let model = net.to_inference().unwrap();
        let ids: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
        let x = Tensor::matrix(1, 4, ids).unwrap();
        let out = model.forward(&x).unwrap(); // softmaxed [T, vocab]

        // Softmax the trainable logits and compare row by row.
        let probs = ops::softmax_rows(&logits).unwrap();
        assert_eq!(out.shape, probs.shape);
        for (a, b) in out.data.iter().zip(&probs.data) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn exported_model_roundtrips_gguf() {
        use crate::loader::{from_bytes, to_bytes};
        use crate::meta::{ModelMetadata, Normalizer, TaskType};
        let mut rng = Rng::new(33);
        let net = TransformerNet::new(6, 4, 8, 2, 12, 1, &mut rng).unwrap();
        let model = net.to_inference().unwrap();
        let meta = ModelMetadata {
            dataset_name: "t".into(),
            task: TaskType::TransformerSLM,
            feature_names: vec![],
            feature_ranges: vec![],
            class_names: (0..6).map(|i| i.to_string()).collect(),
            target_name: "next".into(),
            target_range: [0.0, 6.0],
            input_dim: 4,
            output_dim: 6,
            tokenizer_state: String::new(),
        };
        let norm = Normalizer {
            means: vec![],
            stds: vec![],
        };
        let bytes = to_bytes(&model, &norm, &meta).unwrap();
        let (m2, _, _) = from_bytes(&bytes).unwrap();
        let x = Tensor::matrix(1, 4, vec![0.0, 3.0, 1.0, 5.0]).unwrap();
        let a = model.forward(&x).unwrap();
        let b = m2.forward(&x).unwrap();
        for (p, q) in a.data.iter().zip(&b.data) {
            assert!((p - q).abs() < 1e-6);
        }
    }
}
