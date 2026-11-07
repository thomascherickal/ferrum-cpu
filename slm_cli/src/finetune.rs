//! `finetune-gguf`: AdamW fine-tuning of an imported GGUF.

use crate::args::*;
use crate::memory::*;
use crate::util::*;
use crate::validate::*;
use std::time::Instant;

/// Fine-tune an imported GGUF (f32) with the full AdamW stack and write a
/// checkpoint that `run-gguf --resume` (or a later `finetune-gguf --resume`) can
/// apply back over the base GGUF.
pub(crate) fn cmd_finetune_gguf(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use ferrum_core::{
        Adam, FinetuneRun, Gguf, GgufTokenizer, LlamaTrainer, LrSchedule, Rng, SamplingParams,
    };

    if args.positional.len() < 3 {
        return Err(
            "usage: train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> \
                    [--epochs N] [--lr F] [--batch N] [--seq N] [--warmup N] [--clip F] \
                    [--weight_decay F] [--dropout F] [--threads N] [--qat] [--seed N] \
                    [--resume ckpt.flck] [--sample]"
                .into(),
        );
    }
    require_resume_file(args.value("resume"), ".flck")?;
    if args.has("sample") {
        check_temp(args.get("temp", 0.8))?; // before hours of training, not after
    }
    check_lr(args.get("lr", 1e-4))?;
    check_epochs(args.get("epochs", 3), false)?;
    check_batch_threads(args.get("batch", 8), args.get("threads", 0))?;
    check_regularizers(
        args.get("dropout", 0.0),
        args.get("weight_decay", 0.0),
        args.get("clip", 1.0),
    )?;
    let gguf_path = &args.positional[0];
    let corpus_path = &args.positional[1];
    let out_path = &args.positional[2];
    check_distinct_paths(&[
        ("the output", out_path),
        ("the model", gguf_path),
        ("the corpus", corpus_path),
    ])?;
    check_writable(out_path)?; // before hours of training, not after
    let stop = crate::stop::condition(args, out_path)?;

    // 1. Open + f32-load the base model (training needs full-precision masters).
    println!("Opening {gguf_path} (streamed)…");
    let g = crate::util::fmt_err(Gguf::open(gguf_path), &format!("cannot open GGUF {gguf_path}"))?;
    println!(
        "  GGUF v{}   architecture = {}",
        g.version,
        g.architecture().unwrap_or("?")
    );
    let est = estimate_resident_bytes(&g, None); // f32
    println!("  estimated resident (f32) ≈ {:.2} GB", est as f64 / 1e9);
    if let Some(e) = available_memory_bytes().and_then(|avail| {
        println!("  available memory         ≈ {:.2} GB", avail as f64 / 1e9);
        let train_est = finetune_peak_bytes(est);
        println!(
            "  estimated training RAM   ≈ {:.2} GB (weights + grad + Adam m/v + checkpoint save)",
            train_est as f64 / 1e9
        );
        finetune_memory_error(train_est, avail, args.has("force"))
    }) {
        return Err(e.into());
    }

    let tok = GgufTokenizer::from_gguf(&g)
        .map_err(|_| "this GGUF has no tokenizer; text fine-tuning needs one")?;
    println!(
        "  tokenizer = {:?}  (vocab {})",
        tok.model(),
        tok.vocab_size()
    );

    println!("Loading weights (f32)…");
    let t0 = Instant::now();
    let model = crate::util::fmt_err(g.load_llama_prec(None), "cannot load model")?;
    println!(
        "  loaded in {:.2}s: {} layers, dim {}, vocab {}, ctx {}",
        t0.elapsed().as_secs_f32(),
        model.cfg.n_layers,
        model.cfg.model_dim,
        model.cfg.vocab_size,
        model.cfg.context_len,
    );

    // 2. Read + tokenize the corpus.
    let text = crate::util::fmt_err(std::fs::read_to_string(corpus_path), &format!("cannot read corpus {corpus_path}"))?;
    let mut tokens: Vec<usize> = Vec::new();
    if let Some(bos) = tok.prefix_bos() {
        tokens.push(bos);
    }
    tokens.extend(tok.encode(&text));
    // The corpus identity the checkpoint records and a resume checks.
    let corpus_id = crate::util::fmt_err(slm_cli::hash::file_id(corpus_path), &format!("cannot read corpus {corpus_path}"))?;
    println!("  corpus: {} chars → {} tokens", text.len(), tokens.len());

    // 3. Hyperparameters.
    let epochs: usize = args.get("epochs", 3);
    let lr: f32 = args.get("lr", 1e-4);
    check_lr(lr)?;
    let batch: usize = args.get("batch", 8);
    let seq: usize = args.get("seq", 64).min(model.cfg.context_len).max(2);
    let weight_decay: f32 = args.get("weight_decay", 0.0);
    let dropout: f32 = args.get("dropout", 0.0);
    let clip: f32 = args.get("clip", 1.0);
    let warmup: u64 = args.get("warmup", 0);
    let qat = args.has("qat");
    let seed: u64 = args.get("seed", 1337);
    let threads_req: usize = args.get("threads", 0);
    let threads = if threads_req == 0 {
        ferrum_core::num_threads()
    } else {
        threads_req
    };

    if tokens.len() < seq {
        return Err(format!(
            "corpus has {} tokens but --seq is {seq}; supply more text or a smaller --seq",
            tokens.len()
        )
        .into());
    }

    // 4. Build + configure the trainer.
    let mut tr = crate::util::fmt_err(LlamaTrainer::new(model), "cannot build trainer")?;
    tr.set_optimizer(Adam::new(lr));
    tr.set_weight_decay(weight_decay);
    tr.set_dropout(dropout);
    tr.set_grad_clip(if clip > 0.0 { Some(clip) } else { None });
    tr.set_qat(qat);

    let num_windows = tokens.len() - seq + 1;
    let steps_per_epoch = num_windows.div_ceil(batch.max(1)) as u64;

    // 5. Resume optimizer state if requested (continues where a prior run stopped).
    let mut rng = if let Some(ckpt) = args.value("resume") {
        let bytes = crate::util::fmt_err(std::fs::read(ckpt), &format!("cannot read checkpoint {ckpt}"))?;
        let r = crate::util::fmt_err(tr.load_checkpoint_from_vec(bytes), &format!("cannot resume from {ckpt}"))?;
        println!("  resumed from {ckpt} at step {}", tr.step_count());
        r
    } else {
        Rng::new(seed)
    };
    // Refuses a resume on another corpus, batch or --seq; notes changed lr settings. The
    // returned step is where this run's first epoch began (an interrupted epoch is replayed).
    let run = FinetuneRun {
        corpus_id: Some(corpus_id),
        batch,
        seq,
        lr,
        warmup,
        clip,
        weight_decay,
        dropout,
    };
    let base = tr.begin_run(&run, tokens.len())?;

    // The schedule spans the *cumulative* step timeline (after any resume), so a
    // resumed run does not start past total_steps — which would pin the LR at 0.
    if warmup > 0 {
        let total = base + steps_per_epoch * epochs as u64;
        tr.set_lr_schedule(Some(LrSchedule::warmup_cosine(
            lr,
            warmup,
            total.max(warmup + 1),
        )));
    }

    let capped = finetune_threads(est, available_memory_bytes(), threads);
    if let Some(note) = thread_cap_note(capped, threads) {
        println!("{note}");
    }
    let threads = capped;

    println!(
        "\nFine-tuning: {epochs} epochs · seq {seq} · batch {batch} · lr {lr} · \
         wd {weight_decay} · dropout {dropout} · clip {clip} · {}{} · {threads} threads",
        if warmup > 0 {
            format!("warmup {warmup} ")
        } else {
            String::new()
        },
        if qat { "QAT int8" } else { "f32" },
    );
    println!("  {num_windows} windows · {steps_per_epoch} steps/epoch");

    // 6. Train. A stop request ends the run after a step (never the last one);
    // the checkpoint below is still written, so the run resumes.
    let total_steps = base + steps_per_epoch * epochs as u64;
    let mut stopped = None;
    for e in 0..epochs {
        let te = Instant::now();
        let (loss, halted) =
            tr.finetune_epoch_until(&tokens, seq, batch, &mut rng, threads, &mut |step| {
                let cause = if step < total_steps {
                    stop.should_stop()
                } else {
                    None
                };
                if cause.is_some() {
                    stopped = cause.map(|c| (c, step));
                }
                cause.is_some()
            })?;
        let secs = te.elapsed().as_secs_f32();
        let toks = (num_windows * seq) as f32;
        println!(
            "  epoch {:>3}/{epochs}  loss {loss:.4}  ppl {:.2}  [{:.1}s, {:.0} tok/s, lr {:.2e}]",
            e + 1,
            loss.exp(),
            secs,
            toks / secs.max(1e-6),
            tr.lr_schedule()
                .map(|s| s.lr_at(tr.step_count()))
                .unwrap_or(lr),
        );
        if halted {
            break;
        }
    }

    // 7. Save the checkpoint (weights + optimizer moments + RNG + step).
    let bytes = tr.save_checkpoint(&rng);
    // Atomic: a second Ctrl-C (which exits at once) never leaves a torn checkpoint behind.
    crate::util::fmt_err(ferrum_core::atomic_write(out_path, &bytes), "cannot write")?;
    println!(
        "\nSaved fine-tune checkpoint → {out_path} ({:.2} MB)\n  \
         run it with:  train_transformer run-gguf {gguf_path} \"<prompt>\" --resume {out_path}",
        bytes.len() as f64 / 1e6
    );
    crate::stop::report(&stop, stopped, Some(out_path));

    // 8. Optional sample generation from the fine-tuned model (not after a Ctrl-C).
    if args.has("sample") && !crate::interrupt::is_set() {
        let prompt = tok.bos().into_iter().collect::<Vec<_>>();
        let prompt = if prompt.is_empty() {
            vec![tokens[0]]
        } else {
            prompt
        };
        let params = SamplingParams::with_temperature(args.get("temp", 0.8));
        let stops: Vec<usize> = tok.eos().into_iter().collect();
        let mut grng = Rng::new(time_seed());
        let out = tr
            .model
            .generate(&prompt, args.get("max", 48), &params, &stops, &mut grng)?;
        println!("\n── sample ──\n{}\n────────────", tok.decode(&out));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// export-gguf: re-quantize / export an imported llama/qwen2 GGUF checkpoint
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{letter_vocab, tiny_llama_gguf};
    use crate::test_util::{args, err, Tmp};

    fn finetune(raw: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        cmd_finetune_gguf(&args("finetune-gguf", raw))
    }

    #[test]
    fn hyperparameters_and_paths_are_checked_before_the_model_is_opened() {
        assert!(err(finetune(&["m.gguf", "c.txt"]))
            .starts_with("usage: train_transformer finetune-gguf"));
        let t = Tmp::new("finetune_args");
        let (m, c, o) = (t.path("none.gguf"), t.path("c.txt"), t.path("o.flck"));
        let bad = |extra: &[&str]| {
            let mut raw = vec![m.as_str(), c.as_str(), o.as_str()];
            raw.extend_from_slice(extra);
            err(finetune(&raw))
        };
        assert!(bad(&["--resume", "/nonexistent/x.flck"]).contains("does not exist"));
        assert!(bad(&["--sample", "--temp", "-1"]).contains("--temp"));
        assert!(bad(&["--lr", "0"]).contains("--lr"));
        assert!(bad(&["--epochs", "0"]).contains("--epochs"));
        assert!(bad(&["--batch", "0"]).contains("--batch must be at least 1"));
        assert!(bad(&["--dropout", "1"]).contains("--dropout"));
        let e = err(finetune(&[&m, &c, &m]));
        assert!(e.contains("the output") && e.contains("the model"), "{e}");
        assert!(bad(&[]).starts_with("cannot open GGUF "));
        assert!(
            !std::path::Path::new(&o).exists(),
            "nothing written on an error"
        );
    }

    #[test]
    fn the_corpus_must_fill_one_window_capped_at_the_context() {
        let t = Tmp::new("finetune_seq");
        let model = t.write("m.gguf", tiny_llama_gguf(Some(&letter_vocab()), &[]));
        let bare = t.write("bare.gguf", tiny_llama_gguf(None, &[]));
        let corpus = t.write("c.txt", "abcdefghij"); // 10 tokens, context 16
        let out = t.path("o.flck");
        let e = err(finetune(&[&bare, &corpus, &out]));
        assert_eq!(e, "this GGUF has no tokenizer; text fine-tuning needs one");
        let e = err(finetune(&[&model, &corpus, &out, "--seq", "1000"]));
        assert!(
            e.starts_with("corpus has 10 tokens but --seq is 16;"),
            "{e}"
        );
        let one = t.write("one.txt", "a");
        let e = err(finetune(&[&model, &one, &out, "--seq", "1"]));
        assert!(e.starts_with("corpus has 1 tokens but --seq is 2;"), "{e}");
    }

    /// One epoch on the stub model writes a checkpoint that a resumed run and `run-gguf` accept.
    #[test]
    fn writes_a_checkpoint_that_resumes_and_runs() {
        let t = Tmp::new("finetune_ok");
        let model = t.write("m.gguf", tiny_llama_gguf(Some(&letter_vocab()), &[]));
        let corpus = t.write("c.txt", "abc abc abc abc abc abc");
        let out = t.path("o.flck");
        let common = ["--batch", "2", "--seq", "6", "--threads", "1"];
        let mut raw = vec![
            model.as_str(),
            corpus.as_str(),
            out.as_str(),
            "--epochs",
            "1",
        ];
        raw.extend_from_slice(&common);
        finetune(&raw).unwrap();
        let first = std::fs::read(&out).unwrap();
        assert!(!first.is_empty());
        let out2 = t.path("o2.flck");
        let mut raw = vec![
            model.as_str(),
            corpus.as_str(),
            out2.as_str(),
            "--epochs",
            "2",
        ];
        raw.extend_from_slice(&common);
        raw.extend_from_slice(&["--resume", &out, "--warmup", "2"]);
        finetune(&raw).unwrap();
        assert_ne!(
            std::fs::read(&out2).unwrap(),
            first,
            "the resumed run trained further"
        );
        crate::run_gguf::cmd_run_gguf(&args(
            "run-gguf",
            &[
                &model, "abc", "--resume", &out2, "--max", "2", "--temp", "0", "--quant", "int8",
            ],
        ))
        .unwrap();
        crate::eval_gguf::cmd_eval_gguf(&args(
            "eval-gguf",
            &[&model, &corpus, "--resume", &out2, "--quant", "int8", "--ctx", "6"],
        ))
        .unwrap();
        crate::export::cmd_export_gguf(&args(
            "export-gguf",
            &[&model, &t.path("exp.gguf"), "--resume", &out2, "--quant", "f32"],
        ))
        .unwrap();

        // RAM refusal
        let e = crate::memory::with_available_memory_override(10, || err(finetune(&raw)));
        assert!(e.contains("training memory"));

        // No tokenizer refusal
        let no_tok = t.write("no_tok.gguf", tiny_llama_gguf(None, &[]));
        let e = err(finetune(&[&no_tok, &corpus, &out, "--epochs", "1", "--batch", "2", "--seq", "6", "--threads", "1"]));
        assert!(e.contains("this GGUF has no tokenizer"), "{e}");

        // Thread capping note
        let mut raw_threads = raw.clone();
        raw_threads.retain(|&s| s != "--threads" && s != "1");
        raw_threads.extend_from_slice(&["--threads", "16", "--force"]);
        crate::memory::with_available_memory_override(100_000, || finetune(&raw_threads).unwrap());
    }
}
