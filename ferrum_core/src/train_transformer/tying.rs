//! Weight tying: LM head kept equal to the transposed token embedding.
use crate::train_transformer::TransformerNet;

impl TransformerNet {
    /// Enable or disable weight tying between the token embedding and the LM
    /// head (T9). When enabled, [`TransformerNet::sync_tied_head`] mirrors the
    /// embedding into the head before each step and
    /// [`TransformerNet::fold_tied_head_grad`] sums the head gradient back into
    /// the embedding, so the two share parameters.
    pub fn set_weight_tying(&mut self, enabled: bool) {
        self.tie_weights = enabled;
        if enabled {
            self.sync_tied_head();
        }
    }

    /// Whether the LM head is tied to the token embedding.
    pub fn weight_tying(&self) -> bool {
        self.tie_weights
    }

    /// Copy `tok_embᵀ` into the LM-head weight so the head mirrors the embedding
    /// (no-op unless weight tying is enabled). Called before each forward.
    pub fn sync_tied_head(&mut self) {
        if !self.tie_weights {
            return;
        }
        let (vocab, embed) = (self.vocab_size, self.embed_dim);
        // head_w is [embed, vocab] = transpose of tok_emb [vocab, embed].
        for e in 0..embed {
            for v in 0..vocab {
                self.head_w.data.data[e * vocab + v] = self.tok_emb.data.data[v * embed + e];
            }
        }
    }

    /// Fold the LM-head gradient into the token-embedding gradient
    /// (`tok_emb.grad += head_w.gradᵀ`) so the shared parameter receives both
    /// contributions, then zero the head gradient so the redundant head copy is
    /// not stepped independently. No-op unless weight tying is enabled.
    pub fn fold_tied_head_grad(&mut self) {
        if !self.tie_weights {
            return;
        }
        let (vocab, embed) = (self.vocab_size, self.embed_dim);
        for e in 0..embed {
            for v in 0..vocab {
                self.tok_emb.grad.data[v * embed + e] += self.head_w.grad.data[e * vocab + v];
                self.head_w.grad.data[e * vocab + v] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::layer::Linear;
    use crate::loss::softmax_cross_entropy;
    use crate::optim::Adam;
    use crate::rng::Rng;
    use crate::train_transformer::epoch::train_transformer_epoch;
    use crate::train_transformer::TransformerNet;

    #[test]
    fn sync_tied_head_mirrors_embedding_transpose() {
        let mut rng = Rng::new(5);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_weight_tying(true); // enabling syncs immediately
        assert!(net.weight_tying());
        let (vocab, embed) = (6usize, 8usize);
        for v in 0..vocab {
            for e in 0..embed {
                assert_eq!(
                    net.head_w.data.data[e * vocab + v],
                    net.tok_emb.data.data[v * embed + e],
                    "head is not tok_embᵀ at ({v},{e})"
                );
            }
        }
    }

    /// The folded gradient must be the true gradient of the loss w.r.t. the
    /// shared embedding — accounting for the embedding affecting BOTH the input
    /// lookup and the (tied) output head. Verified by central differences that
    /// re-sync the head after each perturbation.
    #[test]
    fn weight_tying_gradient_is_correct() {
        let mut rng = Rng::new(3);
        let mut net = TransformerNet::new(5, 3, 4, 2, 6, 1, &mut rng).unwrap();
        net.set_weight_tying(true);
        let tokens = [1usize, 2, 3];
        let targets = [2usize, 3, 4];

        net.sync_tied_head();
        let (logits, cache) = net.forward(&tokens).unwrap();
        let (_, dl) = softmax_cross_entropy(&logits, &targets).unwrap();
        net.zero_grad();
        net.backward(&dl, &cache).unwrap();
        net.fold_tied_head_grad();

        let loss_at = |net: &mut TransformerNet, idx: usize, delta: f32| -> f32 {
            net.tok_emb.data.data[idx] += delta;
            net.sync_tied_head(); // the head tracks the embedding
            let (logits, _) = net.forward(&tokens).unwrap();
            let l = softmax_cross_entropy(&logits, &targets).unwrap().0;
            net.tok_emb.data.data[idx] -= delta; // restore
            net.sync_tied_head();
            l
        };

        let eps = 1e-2f32;
        for &i in &[4usize, 9, 13] {
            let analytic = net.tok_emb.grad.data[i];
            let lp = loss_at(&mut net, i, eps);
            let lm = loss_at(&mut net, i, -eps);
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (numeric - analytic).abs() < 6e-3 + 0.03 * analytic.abs(),
                "tied grad[{i}]: analytic={analytic:.6} numeric={numeric:.6}"
            );
        }
    }

    #[test]
    fn weight_tying_trains_and_exports_transposed_head() {
        let mut rng = Rng::new(7);
        let mut net = TransformerNet::new(6, 4, 8, 2, 16, 1, &mut rng).unwrap();
        net.set_weight_tying(true);
        let tokens: Vec<usize> = (0..200).map(|i| i % 6).collect();
        let adam = Adam::new(0.01);
        let first = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        let mut last = first;
        for _ in 0..30 {
            last = train_transformer_epoch(&mut net, &tokens, 8, &adam, &mut rng).unwrap();
        }
        assert!(
            last < first * 0.6,
            "tied training did not reduce loss: {first:.4} → {last:.4}"
        );

        // The exported head must equal the final tok_embᵀ.
        let model = net.to_inference().unwrap();
        let head = model
            .layers()
            .iter()
            .rev()
            .find_map(|l| l.as_any().downcast_ref::<Linear>())
            .expect("model has an LM head");
        let (vocab, embed) = (6usize, 8usize);
        for v in 0..vocab {
            for e in 0..embed {
                assert!(
                    (head.weight.data[e * vocab + v] - net.tok_emb.data.data[v * embed + e]).abs()
                        < 1e-6,
                    "exported head ≠ tok_embᵀ at ({v},{e})"
                );
            }
        }
    }
}
