//! The library API end to end with a tiny model (about 14k parameters): train, save, load,
//! generate, score and evaluate, plus checkpoint resume. Complements the per-feature tests in
//! `test_qat_cache.rs` and `test_slm_library.rs`, which each look at one piece.

use ferrum_core::{GenerativeSLM, Rng, SamplingParams, TransformerConfig};
use std::path::PathBuf;

/// About 1.8 KB of text over an 18-word vocabulary, from a fixed LCG.
fn corpus() -> String {
    let words: Vec<&str> =
        "the cat dog sat ran on a mat log hill and then it saw big small red old"
            .split(' ')
            .collect();
    let mut state = 99u64;
    let mut next = move |n: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize % n
    };
    (0..40)
        .map(|_| {
            let body: Vec<&str> = (0..8 + next(7)).map(|_| words[next(words.len())]).collect();
            format!("{}. ", body.join(" "))
        })
        .collect()
}

fn cfg(epochs: usize) -> TransformerConfig {
    TransformerConfig {
        context_len: 16,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 2,
        hidden_dim: 32,
        epochs,
        vocab_size: 280,
        batch_size: 8,
        ..TransformerConfig::default()
    }
}

fn train(c: &TransformerConfig, seed: u64) -> GenerativeSLM {
    GenerativeSLM::train_transformer_config(&corpus(), c, &mut Rng::new(seed), |_, _| {}).unwrap()
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    }
}

/// A unique temp dir, removed on drop (also when an assertion panics).
struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("ferrum_tiny_lib_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    fn s(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_string()
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Everything observable about a model on a fixed prompt, for exact comparison.
fn fingerprint(m: &GenerativeSLM) -> (String, String, u64, u64, usize) {
    let g = m
        .generate_tokens("the cat", 25, &[], &greedy(), &mut Rng::new(0))
        .unwrap();
    let sampled = SamplingParams {
        temperature: 0.9,
        top_k: 8,
        ..greedy()
    };
    let s = m
        .generate_tokens("the cat", 25, &[], &sampled, &mut Rng::new(5))
        .unwrap();
    let c = m.score("the cat sat on", " the mat.").unwrap();
    let r = m.score_rolling(&corpus()[..300]).unwrap();
    (
        g.text,
        s.text,
        c.logprob.to_bits(),
        r.logprob.to_bits(),
        r.n_tokens,
    )
}

#[test]
fn training_is_deterministic_and_the_seed_matters() {
    let a = train(&cfg(3), 1);
    let b = train(&cfg(3), 1);
    assert_eq!(a.to_bytes().unwrap(), b.to_bytes().unwrap());
    assert_eq!(
        a.to_bytes_quantized().unwrap(),
        b.to_bytes_quantized().unwrap()
    );
    assert_ne!(a.to_bytes().unwrap(), train(&cfg(3), 2).to_bytes().unwrap());
    // The data-parallel trainer is reproducible for a fixed thread count too.
    let t = |seed| {
        GenerativeSLM::train_transformer_config_threaded(
            &corpus(),
            &cfg(3),
            2,
            &mut Rng::new(seed),
            |_, _| {},
        )
        .unwrap()
        .to_bytes()
        .unwrap()
    };
    assert_eq!(t(1), t(1));
}

#[test]
fn save_and_load_are_bit_identical_for_every_precision() {
    let t = Tmp::new("save");
    let m = train(&cfg(3), 1);
    assert_eq!(m.model.num_params(), 13696);

    // f32 keeps the in-memory weights exactly, so the loaded model behaves identically.
    m.save_f32(&t.s("f32.gguf")).unwrap();
    let f32m = GenerativeSLM::load(&t.s("f32.gguf")).unwrap();
    assert_eq!(fingerprint(&m), fingerprint(&f32m));

    // int8 is lossy against the f32 model but every load of the file is the same model.
    m.save(&t.s("int8.gguf")).unwrap();
    let (a, b) = (
        GenerativeSLM::load(&t.s("int8.gguf")).unwrap(),
        GenerativeSLM::from_vec(std::fs::read(t.s("int8.gguf")).unwrap()).unwrap(),
    );
    assert_eq!(fingerprint(&a), fingerprint(&b));
    // Saving what was loaded writes the same bytes (the quantization is idempotent).
    a.save(&t.s("int8_again.gguf")).unwrap();
    assert_eq!(
        std::fs::read(t.s("int8.gguf")).unwrap(),
        std::fs::read(t.s("int8_again.gguf")).unwrap()
    );

    m.save_int4(&t.s("int4.gguf")).unwrap();
    let i4 = GenerativeSLM::load(&t.s("int4.gguf")).unwrap();
    let (e8, e4, ef) = (
        a.evaluate(&corpus()).unwrap().perplexity,
        i4.evaluate(&corpus()).unwrap().perplexity,
        m.evaluate(&corpus()).unwrap().perplexity,
    );
    assert!(
        e4.is_finite() && (e4 / ef - 1.0).abs() < 0.2,
        "{e4} vs {ef}"
    );
    assert!((e8 / ef - 1.0).abs() < 0.2, "{e8} vs {ef}");
    let size = |n: &str| std::fs::metadata(t.s(n)).unwrap().len();
    assert!(size("int4.gguf") < size("int8.gguf") && size("int8.gguf") < size("f32.gguf"));
}

#[test]
fn evaluate_agrees_with_score_rolling_and_beats_the_uniform_baseline() {
    let m = train(&cfg(4), 1);
    let text = corpus();
    let ev = m.evaluate(&text).unwrap();
    let sc = m.score_rolling(&text).unwrap();
    assert_eq!(ev.num_predictions, sc.n_tokens);
    let ce = -sc.logprob / sc.n_tokens as f64;
    assert!(
        (ev.cross_entropy as f64 - ce).abs() < 1e-4,
        "{ev:?} vs {ce}"
    );
    assert!((ev.perplexity as f64 - ce.exp()).abs() < 1e-3 * ce.exp());
    assert!((ev.bits_per_token as f64 - ce / std::f64::consts::LN_2).abs() < 1e-4);
    assert!(ev.perplexity < m.meta.output_dim as f32, "{ev:?}");
    // More training helps on the training text.
    assert!(ev.cross_entropy < train(&cfg(1), 1).evaluate(&text).unwrap().cross_entropy);
    assert!(sc.logprob.is_finite() && sc.logprob < 0.0);
}

#[test]
fn checkpoint_resume_at_an_epoch_boundary_equals_uninterrupted_training() {
    let t = Tmp::new("resume");
    let whole = train(&cfg(4), 7);

    // One epoch, checkpoint written at the end; then continue to four epochs from it.
    let first = TransformerConfig {
        checkpoint_path: t.s("run.fckp"),
        ..cfg(1)
    };
    train(&first, 7);
    assert!(std::fs::metadata(t.s("run.fckp")).unwrap().len() > 0);
    let resumed_cfg = TransformerConfig {
        resume: true,
        checkpoint_path: t.s("run.fckp"),
        ..cfg(4)
    };
    // The seed argument is ignored on resume: the checkpoint carries the RNG state.
    let resumed = train(&resumed_cfg, 12345);

    assert_eq!(whole.to_bytes().unwrap(), resumed.to_bytes().unwrap());
    assert_eq!(fingerprint(&whole), fingerprint(&resumed));
}
