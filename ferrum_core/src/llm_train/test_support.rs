//! Shared fixtures for the unit tests of this module tree.
use crate::layer::Linear;
use crate::llm::{Attention, FeedForward, LlamaBlock, LlamaConfig, LlamaModel, RmsNorm, RopeType};
use crate::llm_train::params::param_data_ref;
use crate::llm_train::LlamaTrainer;
use crate::rng::Rng;

pub(super) fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

pub(super) fn lin(in_f: usize, out_f: usize, seed: u64) -> Linear {
    let w: Vec<f32> = det(in_f * out_f, seed).iter().map(|v| v * 0.3).collect();
    Linear::new(in_f, out_f, w, vec![0.0; out_f]).unwrap()
}

pub(super) fn tiny(
    vocab: usize,
    dim: usize,
    n_heads: usize,
    n_kv: usize,
    ffn: usize,
    n_layers: usize,
) -> LlamaModel {
    let hd = dim / n_heads;
    let cfg = LlamaConfig {
        vocab_size: vocab,
        model_dim: dim,
        n_layers,
        n_heads,
        n_kv_heads: n_kv,
        head_dim: hd,
        ffn_dim: ffn,
        rope_dim: hd,
        rope_base: 10000.0,
        rope_type: RopeType::Norm,
        norm_eps: 1e-5,
        context_len: 64,
    };
    let blocks = (0..n_layers)
        .map(|l| {
            let s = 100 + l as u64 * 50;
            LlamaBlock {
                attn_norm: RmsNorm::new(vec![1.0; dim], 1e-5),
                attn: Attention::new(
                    lin(dim, n_heads * hd, s),
                    lin(dim, n_kv * hd, s + 1),
                    lin(dim, n_kv * hd, s + 2),
                    lin(n_heads * hd, dim, s + 3),
                    n_heads,
                    n_kv,
                    hd,
                    hd,
                    10000.0,
                    RopeType::Norm,
                )
                .unwrap(),
                ffn_norm: RmsNorm::new(vec![1.0; dim], 1e-5),
                ffn: FeedForward::new(
                    lin(dim, ffn, s + 4),
                    lin(dim, ffn, s + 5),
                    lin(ffn, dim, s + 6),
                ),
            }
        })
        .collect();
    LlamaModel {
        cfg,
        tok_emb: det(vocab * dim, 7).iter().map(|v| v * 0.3).collect(),
        blocks,
        final_norm: RmsNorm::new(vec![1.0; dim], 1e-5),
        lm_head: lin(dim, vocab, 999),
        tied_head: false,
    }
}

pub(super) fn base_clone(m: &LlamaModel) -> LlamaModel {
    LlamaModel {
        cfg: m.cfg.clone(),
        tok_emb: m.tok_emb.clone(),
        blocks: m.blocks.clone(),
        final_norm: m.final_norm.clone(),
        lm_head: m.lm_head.clone(),
        tied_head: m.tied_head,
    }
}

pub(super) fn trained_trainer(qat: bool) -> (LlamaTrainer, Rng) {
    let mut t = LlamaTrainer::new(tiny(6, 8, 2, 2, 16, 2)).unwrap();
    t.set_lr(0.02);
    t.set_qat(qat);
    let tokens: Vec<usize> = (0..100).map(|i| (i * 7) % 6).collect();
    let mut rng = Rng::new(77);
    t.finetune_epoch(&tokens, 6, 16, &mut rng).unwrap();
    (t, rng)
}

pub(super) fn bits_of(m: &LlamaModel) -> Vec<u32> {
    param_data_ref(m)
        .into_iter()
        .flatten()
        .map(|x| x.to_bits())
        .collect()
}
