//! Unit-test copy of the verbose-tracing walk in `tests/tests/test_verbose_logging.rs`.
//! `ferrum_core` is compiled once for its own tests and once as a dependency of the other
//! crates' tests; the diagnostic branches (behind the process-global switch) must be reached
//! by the build that sees the whole function, so they are walked here too. The trace is
//! captured through the log sink and asserted on.
use crate::activation::Activation;
use crate::layer::{
    ActivationLayer, Embedding, Flatten, Layer, LayerNorm, Linear, TransformerBlock,
};
use crate::model::Sequential;
use crate::optim::{Adam, Sgd};
use crate::quant::{QKind, QWeight};
use crate::slm::ValidationConfig;
use crate::{
    clear_log_sink, mse, set_log_sink, set_verbose, softmax_cross_entropy, GenerativeSLM, Rng,
    Tensor, TransformerConfig,
};
use std::sync::{Arc, Mutex};

fn block(c: usize, h: usize, heads: usize, ctx: usize) -> TransformerBlock {
    TransformerBlock::new(
        ctx,
        heads,
        c,
        vec![1.0; c],
        vec![0.0; c],
        vec![0.1; c * c],
        vec![0.0; c],
        vec![0.1; c * c],
        vec![0.0; c],
        vec![0.1; c * c],
        vec![0.0; c],
        vec![0.1; c * c],
        vec![0.0; c],
        vec![1.0; c],
        vec![0.0; c],
        vec![0.1; c * h],
        vec![0.0; h],
        vec![0.1; h * c],
        vec![0.0; c],
    )
    .unwrap()
}

#[test]
fn verbose_tracing_covers_every_layer_loss_and_optimizer() {
    let _guard = crate::verbose::TEST_GUARD
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let lines: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = Arc::clone(&lines);
    set_log_sink(move |l| sink.lock().unwrap().push(l.to_string()));
    set_verbose(true);
    assert!(crate::is_verbose());
    let count = |needle: &str| {
        lines
            .lock()
            .unwrap()
            .iter()
            .filter(|l| l.contains(needle))
            .count()
    };

    // Every activation, including the softmax and SiLU paths.
    let x = Tensor::matrix(2, 3, vec![-1.0, 0.5, 2.0, 0.0, 1.0, -2.0]).unwrap();
    for tag in 0..=5u8 {
        let a = Activation::from_tag(tag).unwrap();
        assert_eq!(ActivationLayer::new(a).forward(&x).unwrap().shape, x.shape);
    }
    assert!(count("[activation::apply]") >= 6);

    // Linear: float and quantized weights, then a flatten and a layer norm.
    let w = vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6];
    let lin = Linear::new(3, 2, w.clone(), vec![0.0, 0.1]).unwrap();
    assert_eq!(lin.forward(&x).unwrap().shape, vec![2, 2]);
    let qlin = Linear::quantized(
        3,
        2,
        QWeight::from_f32(&w, 3, 2, QKind::Int8),
        vec![0.0, 0.1],
    )
    .unwrap();
    assert_eq!(qlin.forward(&x).unwrap().shape, vec![2, 2]);
    assert_eq!(qlin.weight_f32().len(), 6);
    assert!(Linear::quantized(3, 2, QWeight::from_f32(&w, 3, 2, QKind::Int8), vec![0.0]).is_err());
    assert!(
        Linear::quantized(2, 3, QWeight::from_f32(&w, 3, 2, QKind::Int8), vec![0.0; 3]).is_err()
    );
    assert_eq!(Flatten::new().forward(&x).unwrap().shape, vec![1, 6]);
    let ln = LayerNorm::new(3, vec![1.0; 3], vec![0.0; 3]).unwrap();
    assert_eq!(ln.forward(&x).unwrap().shape, vec![2, 3]);

    // Embedding and a full transformer block (two sequences of four positions).
    let emb = Embedding::new(5, 4, 8, vec![0.1; 40], vec![0.01; 32]).unwrap();
    let tokens = Tensor::matrix(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
    let embedded = emb.forward(&tokens).unwrap();
    assert_eq!(embedded.shape, vec![8, 8]);
    let tb = block(8, 16, 2, 4);
    assert_eq!(tb.forward(&embedded).unwrap().shape, vec![8, 8]);

    // Sequential walks its layers and reports each.
    let seq = Sequential::new()
        .with(Box::new(lin))
        .with(Box::new(ActivationLayer::new(Activation::Tanh)));
    assert_eq!(seq.forward(&x).unwrap().shape, vec![2, 2]);

    // Losses, including the non-finite warnings.
    let logits = Tensor::matrix(2, 3, vec![1.0, 2.0, 3.0, 0.5, 0.1, -0.3]).unwrap();
    let (loss, _) = softmax_cross_entropy(&logits, &[2, 0]).unwrap();
    assert!(loss.is_finite());
    // An empty batch divides 0 by 0: the only way this loss ever goes non-finite.
    let empty = Tensor::matrix(0, 3, vec![]).unwrap();
    let (nan_loss, _) = softmax_cross_entropy(&empty, &[]).unwrap();
    assert!(nan_loss.is_nan());
    assert!(count("returned NaN loss") >= 1);
    let preds = Tensor::matrix(2, 1, vec![0.5, 1.5]).unwrap();
    let (m, _) = mse(&preds, &[1.0, 1.0]).unwrap();
    assert!((m - 0.25).abs() < 1e-6);
    let nan_preds = Tensor::matrix(1, 1, vec![f32::NAN]).unwrap();
    assert!(mse(&nan_preds, &[1.0]).unwrap().0.is_nan());
    assert!(count("MSE returned NaN loss") >= 1);
    assert!(mse(&logits, &[0.0, 0.0]).is_err()); // more than one output column
    assert!(mse(&preds, &[0.0]).is_err()); // wrong number of targets

    // Optimizers.
    let mut p = Tensor::vector(vec![1.0, 2.0]);
    let g = Tensor::vector(vec![0.1, 0.2]);
    let mut vel = Tensor::vector(vec![0.0, 0.0]);
    Sgd::with_momentum(0.1, 0.9)
        .step(&mut p, &g, &mut vel)
        .unwrap();
    assert!(count("[optim::Sgd::step]") >= 3);
    let (mut m1, mut v1) = (Tensor::vector(vec![0.0; 2]), Tensor::vector(vec![0.0; 2]));
    Adam::new(0.01)
        .step(1, &mut p, &g, &mut m1, &mut v1)
        .unwrap();

    // Shape-checked elementwise ops report their error.
    assert!(crate::ops::mul(&x, &g).is_err());
    assert_eq!(x.rank(), 2);

    // Fitting a normalizer narrates per-column statistics, summarising wide inputs.
    let wide = Tensor::matrix(2, 10, (0..20).map(|v| v as f32).collect()).unwrap();
    let norm = crate::Normalizer::fit(&wide).unwrap();
    assert_eq!(norm.means.len(), 10);
    assert!(count("... (2 more columns)") >= 1);

    // End to end: train (with validation and a resumed checkpoint), score, generate,
    // and serialise, with the narration on.
    let dir = std::env::temp_dir().join(format!("ferrum_verbose_{}", std::process::id()));
    let ckpt = dir.join("run.fckp");
    let corpus = "the cat sat on the mat. the dog sat on the log. ".repeat(12);
    let cfg = TransformerConfig {
        context_len: 8,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        epochs: 2,
        batch_size: 4,
        vocab_size: 270,
        checkpoint_path: ckpt.to_string_lossy().into_owned(),
        checkpoint_every: 2,
        tokenizer_path: dir.join("tok.gguf").to_string_lossy().into_owned(),
        ..TransformerConfig::default()
    };
    let slm = GenerativeSLM::train_transformer_config(&corpus, &cfg, &mut Rng::new(3), |_, _| {})
        .unwrap();
    assert!(ckpt.exists());
    let resumed = TransformerConfig {
        resume: true,
        epochs: 3,
        ..cfg.clone()
    };
    GenerativeSLM::train_transformer_config(&corpus, &resumed, &mut Rng::new(3), |_, _| {})
        .unwrap();
    assert!(count("resumed") >= 1);
    let validated = GenerativeSLM::train_transformer_config_validated(
        &corpus,
        &TransformerConfig {
            resume: false,
            checkpoint_path: String::new(),
            ..cfg.clone()
        },
        1,
        &ValidationConfig::default(),
        &mut Rng::new(3),
        |_| {},
    )
    .unwrap();
    assert!(count("train_validated") >= 1);
    assert!(
        slm.evaluate("the cat sat on the mat.")
            .unwrap()
            .num_predictions
            > 0
    );
    assert!(!slm
        .generate("the cat", 8, 0.8, &mut Rng::new(1))
        .unwrap()
        .is_empty());
    let tied = validated.to_bytes().unwrap();
    assert_eq!(&tied[..4], b"GGUF");
    assert_eq!(&slm.to_bytes_quantized_int4().unwrap()[..4], b"GGUF");
    for (name, save) in [("q8.gguf", 0), ("q4.gguf", 1), ("f32.gguf", 2)] {
        let path = dir.join(name);
        let p = path.to_str().unwrap();
        match save {
            0 => slm.save(p).unwrap(),
            1 => slm.save_int4(p).unwrap(),
            _ => slm.save_f32(p).unwrap(),
        }
        assert!(GenerativeSLM::load(p).is_ok(), "{name}");
    }
    let _ = std::fs::remove_dir_all(&dir);

    // Character-level generation, both the KV-cached transformer path and the
    // full-forward loop used by models without transformer blocks.
    let char_cfg = TransformerConfig {
        vocab_size: 0,
        checkpoint_path: String::new(),
        tokenizer_path: String::new(),
        ..cfg.clone()
    };
    let chars = GenerativeSLM::train_transformer_config(
        "abcabcabc abc\n".repeat(20).as_str(),
        &char_cfg,
        &mut Rng::new(4),
        |_, _| {},
    )
    .unwrap();
    assert!(!chars
        .generate("abc", 6, 0.7, &mut Rng::new(2))
        .unwrap()
        .is_empty());
    assert!(count("generate_cached") >= 1);
    let head = Linear::new(4, 5, vec![0.0; 20], vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let flat = GenerativeSLM::new(
        Sequential::new()
            .with(Box::new(head))
            .with(Box::new(ActivationLayer::new(Activation::Softmax))),
        crate::Normalizer {
            means: vec![],
            stds: vec![],
        },
        crate::ModelMetadata {
            task: crate::TaskType::TransformerSLM,
            class_names: "abcde".chars().map(crate::slm::char_to_hex).collect(),
            input_dim: 4,
            output_dim: 5,
            ..crate::ModelMetadata::default()
        },
    );
    assert_eq!(
        flat.generate("ab", 3, 0.0, &mut Rng::new(1)).unwrap(),
        "abeee"
    );
    assert!(count("logits stats") >= 2);

    set_verbose(false);
    clear_log_sink();
    assert!(!crate::is_verbose());
    assert!(count("Verbose tracing ENABLED") >= 1);
}
