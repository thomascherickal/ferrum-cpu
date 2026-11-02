//! Shared fixtures for the unit tests of this module tree.
use crate::meta::{ModelMetadata, Normalizer, TaskType};
use crate::model::Sequential;
use crate::rng::Rng;
use crate::slm::cached::FEEDS;
use crate::slm::config::TransformerConfig;
use crate::slm::corpus::{char_to_hex, tokenize_for_lm_cfg};
use crate::slm::sampling::SamplingParams;
use crate::slm::training_loop::{prepare_net, run_training};
use crate::slm::GenerativeSLM;
use crate::tensor::Tensor;
use crate::tokenizer::ByteBpeTokenizer;
use crate::train_transformer::TransformerNet;

/// A small, genuine decoder-only transformer as an inference `Sequential`
/// (`Embedding → TransformerBlock×2 → LayerNorm → Linear → Softmax`).
pub(super) fn tiny_transformer_model() -> Sequential {
    let mut rng = Rng::new(7);
    // vocab=6, context=8, embed=8, heads=2, hidden=16, 2 blocks
    let net = TransformerNet::new(6, 8, 8, 2, 16, 2, &mut rng).unwrap();
    net.to_inference().unwrap()
}

/// A progress callback that ignores every epoch (a named fn, so it counts as covered when training calls it).
pub(super) fn ignore_progress(_: usize, _: f32) {}

/// [`ignore_progress`] for the validated trainers.
pub(super) fn ignore_validation(_: &crate::slm::ValidationProgress) {}

pub(super) fn t5_cfg(epochs: usize) -> TransformerConfig {
    TransformerConfig {
        context_len: 4,
        embed_dim: 8,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 16,
        epochs,
        lr: 0.02,
        batch_size: 8,
        vocab_size: 0,
        weight_decay: 0.0,
        dropout: 0.0,
        ..TransformerConfig::default()
    }
}

/// A tiny trained BPE transformer SLM for scoring tests (context 8).
pub(super) fn tiny_bpe_slm() -> GenerativeSLM {
    let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(20);
    let cfg = TransformerConfig {
        context_len: 8,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        epochs: 2,
        vocab_size: 270,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(3), ignore_progress)
        .unwrap()
}

pub(super) fn tiny_char_slm() -> GenerativeSLM {
    let corpus = "abcabcabc abc\n".repeat(20);
    let cfg = TransformerConfig {
        context_len: 8,
        embed_dim: 8,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 16,
        epochs: 1,
        vocab_size: 0,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(1), |_, _| {}).unwrap()
}

/// Reference: score each target with its own full forward over the
/// preceding (<= context_len) tokens, right-padded, reading the row of the
/// last real token. Slow but obviously correct.
pub(super) fn naive_logprobs(slm: &GenerativeSLM, ids: &[usize], from: usize) -> Vec<(f64, bool)> {
    let t = slm.meta.input_dim;
    (from..ids.len())
        .map(|i| {
            let start = i.saturating_sub(t);
            let ctx = &ids[start..i];
            let mut row: Vec<f32> = ctx.iter().map(|&x| x as f32).collect();
            row.resize(t, 0.0);
            let out = slm
                .model
                .forward(&Tensor::matrix(1, t, row).unwrap())
                .unwrap();
            let v = out.matrix_dims().unwrap().1;
            let r = &out.data[(ctx.len() - 1) * v..ctx.len() * v];
            let p = r[ids[i]].max(1e-12) as f64;
            let argmax = (0..v).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
            (p.ln(), argmax == ids[i])
        })
        .collect()
}

/// A model with no transformer blocks (just Linear + Softmax) exercises the
/// one-forward-per-target fallback; constant bias gives a closed-form oracle.
pub(super) fn bias_only_slm() -> GenerativeSLM {
    let bias = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
    let lin = crate::layer::Linear::new(4, 5, vec![0.0; 20], bias.clone()).unwrap();
    let model =
        Sequential::new()
            .with(Box::new(lin))
            .with(Box::new(crate::layer::ActivationLayer::new(
                crate::activation::Activation::Softmax,
            )));
    let meta = ModelMetadata {
        task: TaskType::TransformerSLM,
        class_names: "abcde".chars().map(char_to_hex).collect(),
        input_dim: 4,
        output_dim: 5,
        ..ModelMetadata::default()
    };
    GenerativeSLM::new(
        model,
        Normalizer {
            means: vec![],
            stds: vec![],
        },
        meta,
    )
}

pub(super) fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    }
}

/// Trained on short documents so greedy decoding reaches EOS.
pub(super) fn tiny_eos_slm() -> GenerativeSLM {
    let corpus = "the cat sat.<|endoftext|>".repeat(60);
    let cfg = TransformerConfig {
        context_len: 16,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        epochs: 30,
        vocab_size: 270,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(5), |_, _| {}).unwrap()
}

/// Byte-level model (no merges) trained on a two-byte UTF-8 character, so
/// generation emits partial UTF-8 sequences token by token.
pub(super) fn tiny_utf8_slm() -> GenerativeSLM {
    let corpus = "\u{e9}".repeat(200);
    let cfg = TransformerConfig {
        context_len: 8,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        epochs: 30,
        vocab_size: 257,
        ..TransformerConfig::default()
    };
    GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(5), |_, _| {}).unwrap()
}

pub(super) fn feeds() -> usize {
    FEEDS.with(|f| f.get())
}

/// Byte-level BPE model without transformer blocks whose output is a fixed
/// distribution: `Linear` of zero weights plus `bias` over the token ids.
pub(super) fn constant_bpe_slm(bias_for: impl Fn(&ByteBpeTokenizer) -> Vec<f32>) -> GenerativeSLM {
    let tok = ByteBpeTokenizer::train("the cat sat on the mat. ", 280).unwrap();
    let bias = bias_for(&tok);
    let v = tok.vocab_size();
    let lin = crate::layer::Linear::new(4, v, vec![0.0; 4 * v], bias).unwrap();
    let model =
        Sequential::new()
            .with(Box::new(lin))
            .with(Box::new(crate::layer::ActivationLayer::new(
                crate::activation::Activation::Softmax,
            )));
    let meta = ModelMetadata {
        task: TaskType::TransformerSLM,
        input_dim: 4,
        output_dim: v,
        tokenizer_state: tok.encode_state(),
        ..ModelMetadata::default()
    };
    GenerativeSLM::new(
        model,
        Normalizer {
            means: vec![],
            stds: vec![],
        },
        meta,
    )
}

pub(super) const RESUME_CORPUS: &str = "the cat sat on the mat. the dog sat on the log. ";

pub(super) fn resume_cfg(ckpt: &str) -> TransformerConfig {
    TransformerConfig {
        context_len: 8,
        embed_dim: 8,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 16,
        epochs: 3,
        batch_size: 4,
        vocab_size: 270,
        checkpoint_path: ckpt.to_string(),
        checkpoint_every: 5,
        ..TransformerConfig::default()
    }
}

pub(super) fn scratch_ckpt(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("ferrum_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("r.fckp").to_string_lossy().to_string();
    let _ = std::fs::remove_file(&p);
    p
}

/// Optimizer step counts at the end of every epoch `run_training` reports.
pub(super) fn epoch_ends(corpus: &str, cfg: &TransformerConfig) -> Vec<u64> {
    let tc = tokenize_for_lm_cfg(corpus, cfg).unwrap();
    let mut rng = Rng::new(1);
    let mut net = prepare_net(cfg, tc.vocab_size, &mut rng).unwrap();
    let mut ends = Vec::new();
    run_training(
        &mut net,
        &mut tc.tokens.as_slice(),
        cfg,
        1,
        &mut rng,
        &mut |n, _, _| {
            ends.push(n.step_count());
            Ok(true)
        },
    )
    .unwrap();
    ends
}
