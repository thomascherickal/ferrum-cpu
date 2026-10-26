//! Regression test: the estimates stay in the right ballpark of real wall
//! times on tiny models. The band is deliberately very generous (0.05x to 20x,
//! on a floor of 10 ms because milliseconds are noise); it exists to catch an
//! order-of-magnitude mistake in a constant or a formula, not to benchmark.
#[path = "../src/test_support.rs"]
mod test_support;

use ferrum_core::{
    Adam, GenerativeSLM, Gguf, LlamaTrainer, Rng, SamplingParams, TransformerConfig,
};
use ferrum_gui_lib::estimate::{Estimate, JobKind, JobParams};
use ferrum_gui_lib::estimate_cmd::estimate_now;
use std::time::Instant;

const TEXT: &str = "the quick brown fox jumps over the lazy dog. a stitch in time saves nine. \
                    all that glitters is not gold, but a rolling stone gathers no moss. ";

fn corpus_3kb() -> String {
    TEXT.repeat(3000 / TEXT.len() + 1)[..3000].to_string()
}

fn secs(e: &Estimate) -> f64 {
    e.seconds.expect("the estimate has a time")
}

/// The estimate must be within 0.05x..20x of the measured time (floor 10 ms).
fn assert_in_band(what: &str, estimated: f64, measured: f64) {
    let m = measured.max(0.01);
    let ratio = estimated / m;
    println!(
        "REGRESSION {what}: estimated {estimated:.4} s, measured {measured:.4} s, ratio {ratio:.2}"
    );
    assert!(
        (0.05..=20.0).contains(&ratio),
        "{what}: estimated {estimated:.4} s vs measured {measured:.4} s (ratio {ratio:.2})"
    );
}

struct Dir(std::path::PathBuf);
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn dir(tag: &str) -> Dir {
    let d = std::env::temp_dir().join(format!("ferrum_reg_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    Dir(d)
}

fn cfg(epochs: usize) -> TransformerConfig {
    TransformerConfig {
        context_len: 16,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        vocab_size: 300,
        epochs,
        batch_size: 8,
        lr: 0.01,
        ..TransformerConfig::default()
    }
}

fn train_params(corpus_path: &str, epochs: usize) -> JobParams {
    JobParams {
        corpus_path: corpus_path.into(),
        context_len: 16,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        vocab_size: 300,
        epochs,
        batch_size: 8,
        threads: 1,
        ..JobParams::default()
    }
}

#[test]
fn tiny_training_estimate_matches_the_measured_wall_time() {
    let d = dir("train");
    let corpus = corpus_3kb();
    let path = d.0.join("c.txt");
    std::fs::write(&path, &corpus).unwrap();
    let path = path.to_string_lossy().into_owned();
    let mut previous = 0.0;
    for epochs in [1usize, 30] {
        let t = Instant::now();
        GenerativeSLM::train_transformer_config_threaded(
            &corpus,
            &cfg(epochs),
            1,
            &mut Rng::new(1),
            |_, _| {},
        )
        .unwrap();
        let measured = t.elapsed().as_secs_f64();
        let est = estimate_now(JobKind::Train, train_params(&path, epochs), None);
        assert_in_band(&format!("train {epochs} epoch(s)"), secs(&est), measured);
        assert!(est.fits_in_ram);
        assert!(secs(&est) > previous, "more epochs must estimate longer");
        previous = secs(&est);
    }
}

#[test]
fn tiny_generate_evaluate_and_clean_estimates_match() {
    let d = dir("gen");
    let corpus = corpus_3kb();
    let slm = GenerativeSLM::train_transformer_config_threaded(
        &corpus,
        &cfg(2),
        1,
        &mut Rng::new(1),
        |_, _| {},
    )
    .unwrap();
    let model = d.0.join("m.gguf").to_string_lossy().into_owned();
    slm.save(&model).unwrap();
    let loaded = GenerativeSLM::load(&model).unwrap();

    let t = Instant::now();
    loaded
        .generate("the quick", 1500, 0.8, &mut Rng::new(2))
        .unwrap();
    let measured = t.elapsed().as_secs_f64();
    let est = estimate_now(
        JobKind::Generate,
        JobParams {
            model_path: model.clone(),
            num_chars: 1500,
            ..JobParams::default()
        },
        None,
    );
    assert_in_band("generate 1500 chars", secs(&est), measured);

    let heldout: String = corpus.repeat(20);
    let t = Instant::now();
    loaded.evaluate(&heldout).unwrap();
    let measured = t.elapsed().as_secs_f64();
    let est = estimate_now(
        JobKind::Evaluate,
        JobParams {
            model_path: model,
            text_chars: heldout.len(),
            ..JobParams::default()
        },
        None,
    );
    assert_in_band("evaluate 60 KB", secs(&est), measured);

    let big = TEXT.repeat(20_000); // ~1.6 MB
    let opts = ferrum_core::CleanOptions::default();
    let t = Instant::now();
    ferrum_core::clean_corpus(&big, &opts);
    let measured = t.elapsed().as_secs_f64();
    let est = estimate_now(
        JobKind::CleanText,
        JobParams {
            text_chars: big.len(),
            ..JobParams::default()
        },
        None,
    );
    assert_in_band("clean 1.6 MB", secs(&est), measured);
}

#[test]
fn tiny_gguf_run_finetune_export_estimates_match() {
    let d = dir("llama");
    let src = d.0.join("base.gguf").to_string_lossy().into_owned();
    // ~0.3M parameters: big enough to take measurable time, still instant to build.
    std::fs::write(&src, test_support::tiny_llama_gguf(64, 4, 4, 128, 64)).unwrap();
    let g = Gguf::open(&src).unwrap();

    // Run: load + a short generation.
    let t = Instant::now();
    let model = g.load_llama_prec(Some(ferrum_core::QKind::Int8)).unwrap();
    model
        .generate(
            &[0, 1, 2, 3],
            16,
            &SamplingParams::with_temperature(0.8),
            &[],
            &mut Rng::new(1),
        )
        .unwrap();
    let measured = t.elapsed().as_secs_f64();
    let est = estimate_now(
        JobKind::GgufRun,
        JobParams {
            model_path: src.clone(),
            quant: "int8".into(),
            max_new: 16,
            prompt_chars: 4,
            ..JobParams::default()
        },
        None,
    );
    assert_in_band("gguf run", secs(&est), measured);

    // Fine-tune: one epoch over ~250 windows.
    let text = "hi abc hi abc ".repeat(20);
    let corpus = d.0.join("c.txt");
    std::fs::write(&corpus, &text).unwrap();
    let tokens: Vec<usize> = (0..text.len() / 4 + 1).map(|i| i % 10).collect();
    let mut tr = LlamaTrainer::new(g.load_llama_prec(None).unwrap()).unwrap();
    tr.set_optimizer(Adam::new(1e-3));
    let t = Instant::now();
    tr.finetune_epoch_threaded(&tokens, 16, 4, &mut Rng::new(1), 1)
        .unwrap();
    let measured = t.elapsed().as_secs_f64();
    let est = estimate_now(
        JobKind::GgufFinetune,
        JobParams {
            model_path: src.clone(),
            corpus_path: corpus.to_string_lossy().into_owned(),
            epochs: 1,
            batch_size: 4,
            seq_len: 16,
            threads: 1,
            ..JobParams::default()
        },
        None,
    );
    // The estimate counts a token for every 4 bytes; the test's token count is the same.
    assert_in_band("gguf finetune 1 epoch", secs(&est), measured);

    // Export.
    let out = d.0.join("out.gguf").to_string_lossy().into_owned();
    let t = Instant::now();
    ferrum_core::write_llama_gguf(
        &g.load_llama_prec(None).unwrap(),
        &g,
        ferrum_core::GgufQuant::from_str("q8_0").unwrap(),
        &out,
    )
    .unwrap();
    let measured = t.elapsed().as_secs_f64();
    let est = estimate_now(
        JobKind::GgufExport,
        JobParams {
            model_path: src,
            quant: "q8_0".into(),
            ..JobParams::default()
        },
        None,
    );
    assert_in_band("gguf export", secs(&est), measured);
}
