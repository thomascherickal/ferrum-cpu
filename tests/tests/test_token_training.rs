//! Training from a `ferrum-tokens` file streams windows from disk and must give exactly the
//! model that training on the same ids in memory gives: same RNG draws, same window order, at
//! any thread count, with a token budget, dropout, tied weights, and across a checkpoint resume.

use ferrum_core::{
    write_token_file, ByteBpeTokenizer, GenerativeSLM, Rng, TokenFile, TokenFileMeta,
    TransformerConfig, ValidationConfig, DOC_SEP,
};
use std::path::PathBuf;

/// About 2.5 KB: 40 documents over an 18-word vocabulary, from a fixed LCG.
fn corpus() -> String {
    let words: Vec<&str> =
        "the cat dog sat ran on a mat log hill and then it saw big small red old"
            .split(' ')
            .collect();
    let mut state = 4242u64;
    let mut next = move |n: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize % n
    };
    (0..40)
        .map(|_| {
            let body: Vec<&str> = (0..8 + next(9)).map(|_| words[next(words.len())]).collect();
            format!("{}.", body.join(" "))
        })
        .collect::<Vec<_>>()
        .join(DOC_SEP)
}

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("ferrum_tok_train_{tag}_{}", std::process::id()));
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

/// A tokenizer file, its parsed form, and a token file of the corpus made with it.
struct Fixture {
    dir: Tmp,
    tok: ByteBpeTokenizer,
    ids: Vec<u32>,
}

impl Fixture {
    fn new(tag: &str) -> Fixture {
        let dir = Tmp::new(tag);
        let text = corpus();
        let tok = ByteBpeTokenizer::train_with_docs(&text, 300).unwrap();
        std::fs::write(dir.s("tok.gguf"), tok.to_gguf_bytes()).unwrap();
        let ids: Vec<u32> = tok.encode_docs(&text).iter().map(|&i| i as u32).collect();
        let f = Fixture { dir, tok, ids };
        f.write_tokens("t.tok.gguf", &f.ids);
        f
    }

    fn write_tokens(&self, name: &str, ids: &[u32]) {
        let meta = TokenFileMeta {
            vocab_size: self.tok.vocab_size(),
            eos_id: self.tok.eos_id().map(|e| e as u32),
            tokenizer_sha256: "test".into(),
        };
        write_token_file(&self.dir.s(name), &meta, |s| s.push(ids)).unwrap();
    }

    fn cfg(&self, epochs: usize) -> TransformerConfig {
        TransformerConfig {
            context_len: 16,
            embed_dim: 16,
            num_heads: 2,
            num_blocks: 2,
            hidden_dim: 32,
            epochs,
            batch_size: 8,
            vocab_size: 300,
            tokenizer_path: self.dir.s("tok.gguf"),
            ..TransformerConfig::default()
        }
    }

    fn open(&self, name: &str) -> TokenFile {
        TokenFile::open(&self.dir.s(name)).unwrap()
    }

    /// Model bytes trained in memory from the corpus text, and streamed from the token file.
    fn both(&self, cfg: &TransformerConfig, threads: usize) -> (Vec<u8>, Vec<u8>) {
        let saved = |m: GenerativeSLM, name: &str| {
            m.save(&self.dir.s(name)).unwrap();
            std::fs::read(self.dir.s(name)).unwrap()
        };
        let mut epochs_mem = Vec::new();
        let mem = GenerativeSLM::train_transformer_config_threaded(
            &corpus(),
            cfg,
            threads,
            &mut Rng::new(11),
            |ep, loss| epochs_mem.push((ep, loss.to_bits())),
        )
        .unwrap();
        let mut file = self.open("t.tok.gguf");
        let mut epochs_file = Vec::new();
        let streamed = GenerativeSLM::train_transformer_tokens(
            &mut file,
            &self.tok,
            cfg,
            threads,
            &mut Rng::new(11),
            |ep, loss| epochs_file.push((ep, loss.to_bits())),
        )
        .unwrap();
        assert_eq!(epochs_mem, epochs_file, "per-epoch losses differ");
        // Streaming holds one window of ids, never the corpus.
        assert_eq!(file.buffered_bytes(), (cfg.context_len + 1) * 2);
        (saved(mem, "mem.gguf"), saved(streamed, "file.gguf"))
    }
}

#[test]
fn serial_training_is_bit_identical() {
    let f = Fixture::new("serial");
    let (a, b) = f.both(&f.cfg(3), 1);
    assert!(a == b, "model bytes differ ({} vs {})", a.len(), b.len());
}

#[test]
fn threaded_training_is_bit_identical_at_every_shard_count() {
    let f = Fixture::new("threads");
    for threads in [2, 3] {
        let (a, b) = f.both(&f.cfg(2), threads);
        assert!(a == b, "threads {threads}");
    }
}

#[test]
fn strides_budgets_and_per_step_rng_use_stay_identical() {
    let f = Fixture::new("knobs");
    // Stride 5 draws a random phase per epoch; dropout draws a seed per step; a budget ends mid-epoch.
    let mut cfg = f.cfg(3);
    cfg.window_stride = 5;
    cfg.batch_size = 4;
    cfg.dropout = 0.1;
    cfg.weight_tying = true;
    cfg.cosine_lr = true;
    cfg.warmup_steps = 3;
    cfg.grad_clip = 1.0;
    for threads in [1, 2] {
        let (a, b) = f.both(&cfg, threads);
        assert!(a == b, "stride 5, threads {threads}");
    }
    cfg.window_stride = 1;
    cfg.max_tokens = 37 * 4 * 16;
    let (a, b) = f.both(&cfg, 2);
    assert!(a == b, "token budget");
}

#[test]
fn resuming_from_a_checkpoint_gives_the_same_model_both_ways() {
    let f = Fixture::new("resume");
    let run = |from_file: bool, tag: &str| -> (Vec<u8>, Vec<u8>) {
        let ck = f.dir.s(&format!("{tag}.fckp"));
        for (epochs, resume) in [(2, false), (4, true)] {
            let mut cfg = f.cfg(epochs);
            cfg.checkpoint_path = ck.clone();
            cfg.resume = resume;
            let mut rng = Rng::new(5);
            let m = if from_file {
                let mut file = f.open("t.tok.gguf");
                GenerativeSLM::train_transformer_tokens(
                    &mut file,
                    &f.tok,
                    &cfg,
                    2,
                    &mut rng,
                    |_, _| {},
                )
            } else {
                GenerativeSLM::train_transformer_config_threaded(
                    &corpus(),
                    &cfg,
                    2,
                    &mut rng,
                    |_, _| {},
                )
            }
            .unwrap();
            if epochs == 4 {
                m.save(&f.dir.s(&format!("{tag}.gguf"))).unwrap();
            }
        }
        (
            std::fs::read(f.dir.s(&format!("{tag}.gguf"))).unwrap(),
            std::fs::read(&ck).unwrap(),
        )
    };
    let (model_mem, ck_mem) = run(false, "mem");
    let (model_file, ck_file) = run(true, "file");
    assert!(model_mem == model_file, "resumed models differ");
    assert!(ck_mem == ck_file, "final checkpoints differ");
}

#[test]
fn validation_holds_out_the_token_tail_and_trains_on_the_head() {
    let f = Fixture::new("val");
    let n = f.ids.len();
    let val = ValidationConfig {
        val_fraction: 0.2,
        patience: 0,
    };
    let held = (n as f32 * 0.2).round() as usize;
    // One epoch: the best (only) model is the one trained on the head alone.
    f.write_tokens("head.tok.gguf", &f.ids[..n - held]);
    let cfg = f.cfg(1);
    let mut seen = Vec::new();
    let mut file = f.open("t.tok.gguf");
    let validated = GenerativeSLM::train_transformer_tokens_validated(
        &mut file,
        &f.tok,
        &cfg,
        1,
        &val,
        &mut Rng::new(3),
        |p| seen.push((p.epoch, p.is_best, p.val.perplexity)),
    )
    .unwrap();
    let mut head = f.open("head.tok.gguf");
    let plain = GenerativeSLM::train_transformer_tokens(
        &mut head,
        &f.tok,
        &cfg,
        1,
        &mut Rng::new(3),
        |_, _| {},
    )
    .unwrap();
    validated.save(&f.dir.s("v.gguf")).unwrap();
    plain.save(&f.dir.s("p.gguf")).unwrap();
    assert!(std::fs::read(f.dir.s("v.gguf")).unwrap() == std::fs::read(f.dir.s("p.gguf")).unwrap());
    assert_eq!(seen.len(), 1);
    assert!(
        seen[0].1 && seen[0].2.is_finite() && seen[0].2 > 1.0,
        "{seen:?}"
    );

    // Several epochs: epochs are reported in order, the first is the best so far, and the
    // perplexity equals evaluating the held-out ids directly.
    let mut seen = Vec::new();
    let mut file = f.open("t.tok.gguf");
    let best = GenerativeSLM::train_transformer_tokens_validated(
        &mut file,
        &f.tok,
        &f.cfg(4),
        2,
        &val,
        &mut Rng::new(3),
        |p| seen.push((p.epoch, p.is_best, p.val.cross_entropy)),
    )
    .unwrap();
    assert_eq!(seen.iter().map(|s| s.0).collect::<Vec<_>>(), [1, 2, 3, 4]);
    assert!(seen[0].1);
    let lowest = seen.iter().map(|s| s.2).fold(f32::INFINITY, f32::min);
    let tail: Vec<usize> = f.ids[n - held..].iter().map(|&i| i as usize).collect();
    assert_eq!(best.evaluate_ids(&tail).unwrap().cross_entropy, lowest);
}

#[test]
fn a_mismatched_tokenizer_a_short_file_and_a_bad_split_are_refused() {
    let f = Fixture::new("errs");
    let cfg = f.cfg(1);
    let val = |v: f32| ValidationConfig {
        val_fraction: v,
        patience: 0,
    };
    let train = |tok: &ByteBpeTokenizer, name: &str| {
        let mut file = f.open(name);
        GenerativeSLM::train_transformer_tokens(
            &mut file,
            tok,
            &cfg,
            1,
            &mut Rng::new(1),
            |_, _| {},
        )
        .map(|_| ())
        .map_err(|e| e.to_string())
    };
    let other = ByteBpeTokenizer::train_with_docs(&corpus(), 280).unwrap();
    assert!(train(&other, "t.tok.gguf")
        .unwrap_err()
        .contains("not the tokenizer"));
    f.write_tokens("short.tok.gguf", &f.ids[..16]);
    assert!(train(&f.tok, "short.tok.gguf")
        .unwrap_err()
        .contains("more tokens than the context"));
    for (v, want) in [
        (0.0, "open interval"),
        (1.0, "open interval"),
        (0.99, "lower --val"),
    ] {
        let mut file = f.open("t.tok.gguf");
        let e = GenerativeSLM::train_transformer_tokens_validated(
            &mut file,
            &f.tok,
            &cfg,
            1,
            &val(v),
            &mut Rng::new(1),
            |_| {},
        )
        .err()
        .expect("must fail")
        .to_string();
        assert!(e.contains(want), "{v}: {e}");
    }
}
