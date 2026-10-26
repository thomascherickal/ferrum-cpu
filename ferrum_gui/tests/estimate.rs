//! Integration tests for `estimate_job`'s window-free core: real files, the
//! live hardware, tiny models. No display or Tauri runtime needed.
#[path = "../src/test_support.rs"]
mod test_support;

use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};
use ferrum_gui_lib::estimate::{Estimate, JobKind, JobParams};
use ferrum_gui_lib::estimate_cmd::{estimate_now, gguf_facts, slm_facts};
use std::path::PathBuf;

struct Dir(PathBuf);
impl Dir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ferrum_est_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn path(&self, f: &str) -> String {
        self.0.join(f).to_string_lossy().into_owned()
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const TEXT: &str = "the quick brown fox jumps over the lazy dog. a stitch in time saves nine. ";

fn tiny_slm(dir: &Dir) -> String {
    let corpus = TEXT.repeat(40);
    let cfg = TransformerConfig {
        context_len: 8,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        vocab_size: 300,
        epochs: 1,
        batch_size: 8,
        ..TransformerConfig::default()
    };
    let slm = GenerativeSLM::train_transformer_config_threaded(
        &corpus,
        &cfg,
        1,
        &mut Rng::new(1),
        |_, _| {},
    )
    .unwrap();
    let path = dir.path("tiny.gguf");
    slm.save(&path).unwrap();
    path
}

fn run(kind: JobKind, p: JobParams) -> Estimate {
    estimate_now(kind, p, None)
}

#[test]
fn reads_facts_from_a_real_ferrum_model_and_a_real_llama_gguf() {
    let dir = Dir::new("facts");
    let slm = slm_facts(&tiny_slm(&dir)).expect("ferrum gguf facts");
    assert!(slm.params > 5_000 && slm.matmul_params < slm.params);
    assert_eq!(slm.vocab, 300);
    assert!(slm.file_bytes > 0);

    let llama = dir.path("llama.gguf");
    std::fs::write(&llama, test_support::tiny_llama_gguf(8, 2, 2, 16, 32)).unwrap();
    let g = gguf_facts(&llama).expect("llama facts");
    assert_eq!(
        (g.n_layers, g.context_len, g.embed_params),
        (2, 128, 8 * 32)
    );
    assert!(g.has_output_head && g.params > g.embed_params);

    // Unusable paths give no facts instead of an error.
    for bad in ["", "/no/such/file.gguf", "/dev/null", "/proc/cpuinfo"] {
        assert!(
            slm_facts(bad).is_none() && gguf_facts(bad).is_none(),
            "{bad:?}"
        );
    }
    let not_gguf = dir.path("x.gguf");
    std::fs::write(&not_gguf, b"not a gguf").unwrap();
    assert!(gguf_facts(&not_gguf).is_none());
}

#[test]
fn every_tab_gets_an_estimate_from_real_files() {
    let dir = Dir::new("tabs");
    let model = tiny_slm(&dir);
    let corpus = dir.path("corpus.txt");
    std::fs::write(&corpus, TEXT.repeat(40)).unwrap();
    let llama = dir.path("llama.gguf");
    std::fs::write(&llama, test_support::tiny_llama_gguf(16, 2, 2, 32, 64)).unwrap();

    let train = JobParams {
        corpus_path: corpus.clone(),
        context_len: 16,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        epochs: 1,
        batch_size: 8,
        vocab_size: 300,
        ..JobParams::default()
    };
    let t = run(JobKind::Train, train.clone());
    assert!(
        t.seconds.unwrap() > 0.0 && t.seconds.unwrap() < 60.0,
        "{t:?}"
    );
    assert!(t.fits_in_ram && t.peak_ram_mb.unwrap() > 0.0 && !t.uses_gpu);
    // The same job with a missing corpus says so and still estimates.
    let missing = run(
        JobKind::Train,
        JobParams {
            corpus_path: dir.path("nope.txt"),
            ..train
        },
    );
    assert!(missing.notes.iter().any(|n| n.contains("Corpus not found")));

    let g = run(
        JobKind::Generate,
        JobParams {
            model_path: model.clone(),
            num_chars: 100,
            ..JobParams::default()
        },
    );
    assert!(g.seconds.unwrap() < 5.0 && g.peak_ram_mb.unwrap() > 0.0);
    let e = run(
        JobKind::Evaluate,
        JobParams {
            model_path: model.clone(),
            text_path: corpus.clone(),
            ..JobParams::default()
        },
    );
    assert!(e.seconds.unwrap() > 0.0);
    let mi = run(
        JobKind::ModelInfo,
        JobParams {
            model_path: model,
            ..JobParams::default()
        },
    );
    assert!(mi.seconds.unwrap() < 1.0);

    let gp = JobParams {
        model_path: llama.clone(),
        prompt_chars: 40,
        max_new: 8,
        quant: "int8".into(),
        ..JobParams::default()
    };
    assert!(run(JobKind::GgufRun, gp.clone()).seconds.unwrap() > 0.0);
    let ft = JobParams {
        corpus_path: corpus,
        epochs: 1,
        batch_size: 4,
        seq_len: 16,
        ..gp.clone()
    };
    let f = run(JobKind::GgufFinetune, ft);
    assert!(f.seconds.is_some_and(|s| s > 0.0), "{f:?}");
    let ex = run(
        JobKind::GgufExport,
        JobParams {
            quant: "q8_0".into(),
            ..gp.clone()
        },
    );
    assert!(ex.peak_ram_mb.unwrap() > 0.0);
    assert!(run(JobKind::GgufInfo, gp).seconds.unwrap() < 1.0);

    // A model that was not chosen yet is a note, not a crash.
    assert!(run(JobKind::GgufRun, JobParams::default())
        .seconds
        .is_none());
}

#[test]
fn estimate_follows_the_live_machine() {
    // The estimate reads the real thread count and RAM, not constants.
    let hw = ferrum_gui_lib::hardware::detect();
    let d = run(
        JobKind::Download,
        JobParams {
            max_mb: 10.0,
            ..JobParams::default()
        },
    );
    assert!(d.threads_used >= 1 && d.threads_used <= hw.logical_cores);
    let avail = d.ram_available_mb.expect("RAM is known on this platform");
    assert!(avail > 0.0 && avail <= hw.ram_total_mb.unwrap() as f64);
    assert!(!d.uses_gpu);
    // The thread count follows the request, capped by the machine.
    let one = run(
        JobKind::Train,
        JobParams {
            threads: 1,
            ..tiny_train()
        },
    );
    assert_eq!(one.threads_used, 1);
}

fn tiny_train() -> JobParams {
    JobParams {
        context_len: 16,
        embed_dim: 16,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 32,
        epochs: 1,
        batch_size: 8,
        corpus_bytes: Some(3000),
        ..JobParams::default()
    }
}
