//! Flat gradient / moment buffers mirroring the Llama parameter layout.
use crate::layer::Linear;
use crate::llm::LlamaModel;

/// A weight+bias buffer mirroring one [`Linear`]. Used both for gradients (then
/// `w`/`b` hold dW/db) and for Adam moment buffers (then they hold m or v).
#[derive(Clone)]
pub(super) struct LinBuf {
    pub(super) w: Vec<f32>, // [in·out] row-major, like Linear::weight
    pub(super) b: Vec<f32>, // [out], or empty when the layer has no real bias
}

impl LinBuf {
    pub(super) fn zeros(lin: &Linear) -> Self {
        Self {
            w: vec![0.0; lin.in_features() * lin.out_features()],
            b: vec![
                0.0;
                if lin.has_bias() {
                    lin.out_features()
                } else {
                    0
                }
            ],
        }
    }
    pub(super) fn clear(&mut self) {
        self.w.iter_mut().for_each(|x| *x = 0.0);
        self.b.iter_mut().for_each(|x| *x = 0.0);
    }
}

#[derive(Clone)]
pub(super) struct BlockBuf {
    pub(super) attn_norm: Vec<f32>,
    pub(super) wq: LinBuf,
    pub(super) wk: LinBuf,
    pub(super) wv: LinBuf,
    pub(super) wo: LinBuf,
    pub(super) ffn_norm: Vec<f32>,
    pub(super) gate: LinBuf,
    pub(super) up: LinBuf,
    pub(super) down: LinBuf,
}

/// One full set of buffers shaped like the model — gradients or an Adam moment.
#[derive(Clone)]
pub(super) struct Buffers {
    pub(super) tok_emb: Vec<f32>,
    pub(super) blocks: Vec<BlockBuf>,
    pub(super) final_norm: Vec<f32>,
    pub(super) lm_head: LinBuf,
}

impl Buffers {
    pub(super) fn zeros(model: &LlamaModel) -> Self {
        let blocks = model
            .blocks
            .iter()
            .map(|b| BlockBuf {
                attn_norm: vec![0.0; b.attn_norm.weight.len()],
                wq: LinBuf::zeros(&b.attn.wq),
                wk: LinBuf::zeros(&b.attn.wk),
                wv: LinBuf::zeros(&b.attn.wv),
                wo: LinBuf::zeros(&b.attn.wo),
                ffn_norm: vec![0.0; b.ffn_norm.weight.len()],
                gate: LinBuf::zeros(&b.ffn.gate),
                up: LinBuf::zeros(&b.ffn.up),
                down: LinBuf::zeros(&b.ffn.down),
            })
            .collect();
        Self {
            tok_emb: vec![0.0; model.tok_emb.len()],
            blocks,
            final_norm: vec![0.0; model.final_norm.weight.len()],
            lm_head: {
                let mut h = LinBuf::zeros(&model.lm_head);
                if model.tied_head {
                    h.w = Vec::new(); // its gradient folds into tok_emb
                }
                h
            },
        }
    }
    pub(super) fn clear(&mut self) {
        self.tok_emb.iter_mut().for_each(|x| *x = 0.0);
        self.final_norm.iter_mut().for_each(|x| *x = 0.0);
        self.lm_head.clear();
        for b in &mut self.blocks {
            b.attn_norm.iter_mut().for_each(|x| *x = 0.0);
            b.ffn_norm.iter_mut().for_each(|x| *x = 0.0);
            for g in [
                &mut b.wq,
                &mut b.wk,
                &mut b.wv,
                &mut b.wo,
                &mut b.gate,
                &mut b.up,
                &mut b.down,
            ] {
                g.clear();
            }
        }
    }

    /// Flat view of every buffer in canonical order (matches [`param_data_ref`]).
    pub(super) fn slices(&self) -> Vec<&[f32]> {
        let mut v: Vec<&[f32]> = vec![&self.tok_emb];
        for b in &self.blocks {
            v.push(&b.attn_norm);
            v.push(&b.wq.w);
            v.push(&b.wq.b);
            v.push(&b.wk.w);
            v.push(&b.wk.b);
            v.push(&b.wv.w);
            v.push(&b.wv.b);
            v.push(&b.wo.w);
            v.push(&b.wo.b);
            v.push(&b.ffn_norm);
            v.push(&b.gate.w);
            v.push(&b.gate.b);
            v.push(&b.up.w);
            v.push(&b.up.b);
            v.push(&b.down.w);
            v.push(&b.down.b);
        }
        v.push(&self.final_norm);
        v.push(&self.lm_head.w);
        v.push(&self.lm_head.b);
        v
    }

    pub(super) fn slices_mut(&mut self) -> Vec<&mut [f32]> {
        let mut v: Vec<&mut [f32]> = vec![&mut self.tok_emb];
        for b in &mut self.blocks {
            v.push(&mut b.attn_norm);
            v.push(&mut b.wq.w);
            v.push(&mut b.wq.b);
            v.push(&mut b.wk.w);
            v.push(&mut b.wk.b);
            v.push(&mut b.wv.w);
            v.push(&mut b.wv.b);
            v.push(&mut b.wo.w);
            v.push(&mut b.wo.b);
            v.push(&mut b.ffn_norm);
            v.push(&mut b.gate.w);
            v.push(&mut b.gate.b);
            v.push(&mut b.up.w);
            v.push(&mut b.up.b);
            v.push(&mut b.down.w);
            v.push(&mut b.down.b);
        }
        v.push(&mut self.final_norm);
        v.push(&mut self.lm_head.w);
        v.push(&mut self.lm_head.b);
        v
    }

    pub(super) fn to_flat(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for s in self.slices() {
            out.extend_from_slice(s);
        }
        out
    }

    /// Overwrite from a flat vector in canonical order. Panics on length mismatch.
    pub(super) fn load_flat(&mut self, flat: &[f32]) {
        let mut off = 0usize;
        for s in self.slices_mut() {
            let n = s.len();
            s.copy_from_slice(&flat[off..off + n]);
            off += n;
        }
        debug_assert_eq!(off, flat.len(), "load_flat: length mismatch");
    }

    /// Add another buffer set in place (`self += other`).
    pub(super) fn add_assign_flat(&mut self, flat: &[f32]) {
        let mut off = 0usize;
        for s in self.slices_mut() {
            let n = s.len();
            for (a, b) in s.iter_mut().zip(&flat[off..off + n]) {
                *a += *b;
            }
            off += n;
        }
    }

    /// Scale every element (`self *= k`).
    pub(super) fn scale(&mut self, k: f32) {
        for s in self.slices_mut() {
            for x in s.iter_mut() {
                *x *= k;
            }
        }
    }
}
