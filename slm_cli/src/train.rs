//! `train` (and the train-or-load step `run` shares): corpus -> trained GGUF model.

use crate::args::*;
use crate::util::*;
use crate::validate::*;
use ferrum_core::{ByteBpeTokenizer, GenerativeSLM, Rng, TokenFile, ValidationConfig};
use std::time::Instant;

/// What `train` learns from: text held in memory, or a token file streamed from disk (with the
/// tokenizer it was made with).
pub(crate) enum Input<'a> {
    Text(&'a str),
    Tokens(&'a mut TokenFile, &'a ByteBpeTokenizer),
}

/// Train a transformer SLM on `corpus` (QAT, int8) and save it to
/// `model_path`. Returns the trained model.
pub(crate) fn train_and_save(
    corpus: &str,
    corpus_path: &str,
    model_path: &str,
    args: &Args,
) -> Result<GenerativeSLM, Box<dyn std::error::Error>> {
    train_and_save_input(Input::Text(corpus), corpus_path, model_path, args)
}

/// [`train_and_save`] over either kind of [`Input`].
pub(crate) fn train_and_save_input(
    input: Input,
    corpus_path: &str,
    model_path: &str,
    args: &Args,
) -> Result<GenerativeSLM, Box<dyn std::error::Error>> {
    // Check before spending time training.
    let quant = args.value("quant").unwrap_or("int8");
    if !matches!(quant, "int8" | "int4" | "f32") {
        return Err(format!("--quant must be int8|int4|f32 (got '{quant}')").into());
    }
    if let Some(tok) = reused_tokenizer_with_val(args) {
        println!(
            "note: reusing {tok}; if it was trained without --val it has seen the validation text"
        );
    }
    let preset = crate::preset::selected(args)?;
    let mut cfg = args.config();
    // Only a run that writes a checkpoint needs the corpus identity.
    if !cfg.checkpoint_path.is_empty() {
        cfg.corpus_id = Some(
            slm_cli::hash::file_id(corpus_path)
                .map_err(|e| format!("cannot read corpus {corpus_path}: {e}"))?,
        );
    }
    cfg.stop = crate::stop::condition(args, model_path)?;
    check_lr(cfg.lr)?;
    check_regularizers(cfg.dropout, cfg.weight_decay, cfg.grad_clip)?;
    let budget = args.get("tokens", 0.0f64);
    check_token_budget(budget)?;
    if budget == 0.0 {
        check_epochs(cfg.epochs, true)?; // with a budget the epoch count is ignored
    }
    check_batch_threads(cfg.batch_size, args.get("threads", 0))?;
    // No two files may be one file, and every output must be writable before hours of training.
    let ck_label = if cfg.resume {
        "--resume"
    } else {
        "--checkpoint"
    };
    check_distinct_paths(&[
        ("the model output", model_path),
        (ck_label, &cfg.checkpoint_path),
        ("--tokenizer", &cfg.tokenizer_path),
        ("the corpus", corpus_path),
        ("--metrics-csv", &cfg.metrics_csv),
    ])?;
    check_writable(model_path)?;
    if !cfg.checkpoint_path.is_empty() {
        check_writable(&cfg.checkpoint_path)?;
    }
    if !cfg.metrics_csv.is_empty() {
        check_writable(&cfg.metrics_csv)?;
    }
    if cfg.vocab_size > 0
        && !cfg.tokenizer_path.is_empty()
        && !std::path::Path::new(&cfg.tokenizer_path).exists()
    {
        check_writable(&cfg.tokenizer_path)?;
    }
    let resumed = cfg
        .resume
        .then(|| crate::resume::read_info(&cfg.checkpoint_path))
        .flatten();
    if let Some(info) = &resumed {
        crate::resume::apply(args, &mut cfg, info);
    }
    let seed: u64 = args.get("seed", 1337);
    // 0 = auto-detect the machine's parallelism; 1 = serial.
    let threads: usize = args.get("threads", 0);
    let mut rng = Rng::new(seed);
    let corpus_desc = match &input {
        Input::Text(c) => format!("{} chars", c.chars().filter(|&ch| ch != '\r').count()),
        Input::Tokens(f, _) => format!(
            "{} tokens, {} documents, streamed from disk",
            f.len(),
            f.documents()
        ),
    };

    println!("╔══════════════════════════════════════════════════════════╗");
    let mode = match (cfg.qat, quant) {
        (true, "int8") => "int8 QAT".to_string(),
        (true, q) => format!("int8 QAT, saved as {q}"),
        (false, q) => format!("fp32, no QAT, saved as {q}"),
    };
    println!("  ferrum transformer SLM trainer ({mode})");
    if let Some(p) = preset {
        println!("  Preset  : {} (explicit flags override it)", p.name);
    }
    println!("  Corpus  : {corpus_path}  ({corpus_desc})");
    let tok_desc = match &input {
        Input::Tokens(_, tok) => format!("byte-level BPE (vocab {})", tok.vocab_size()),
        Input::Text(_) if cfg.vocab_size == 0 => "character-level".to_string(),
        Input::Text(_) => format!("byte-level BPE (vocab {})", cfg.vocab_size),
    };
    println!(
        "  Context : {}   Embed: {}   Hidden: {}",
        cfg.context_len, cfg.embed_dim, cfg.hidden_dim
    );
    println!("  Heads   : {}   Blocks: {}", cfg.num_heads, cfg.num_blocks);
    println!("  Tokenizer: {tok_desc}");
    let epochs_desc = if cfg.max_tokens > 0 {
        "budget".to_string()
    } else {
        cfg.epochs.to_string()
    };
    let seed_desc = if resumed.is_some() {
        "from checkpoint".to_string()
    } else {
        seed.to_string()
    };
    println!(
        "  Epochs  : {epochs_desc}   LR: {}   Batch: {}   Seed: {seed_desc}",
        cfg.lr, cfg.batch_size
    );
    let resolved_threads = if threads == 0 {
        ferrum_core::num_threads()
    } else {
        threads
    };
    println!("  Threads : {resolved_threads} (data-parallel minibatch training)");
    if cfg.max_tokens > 0 {
        println!("  Budget  : {} tokens (overrides --epochs)", cfg.max_tokens);
    }
    if cfg.resume {
        println!("  Resume  : {}", cfg.checkpoint_path);
    }
    if !cfg.metrics_csv.is_empty() {
        println!("  Metrics : {}", cfg.metrics_csv);
    }
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let t0 = Instant::now();
    let epochs = cfg.epochs;
    let budgeted = cfg.max_tokens > 0;
    let report_every = (epochs / 20).max(1);
    // Budgeted runs have no known epoch count: report at most every 5 s.
    let last_report: std::cell::Cell<Option<Instant>> = std::cell::Cell::new(None);
    let report = |ep: usize, line: String| {
        let due = if budgeted {
            last_report.get().is_none_or(|t| t.elapsed().as_secs() >= 5)
        } else {
            ep == 1 || ep.is_multiple_of(report_every) || ep == epochs
        };
        if due {
            last_report.set(Some(Instant::now()));
            println!("{line}");
        }
    };
    let total = if budgeted {
        String::new()
    } else {
        format!("/{epochs}")
    };

    let reused_tokenizer =
        !cfg.tokenizer_path.is_empty() && std::path::Path::new(&cfg.tokenizer_path).exists();
    // Early stopping, tracked from the per-epoch callback: (last epoch, best epoch, epochs since best).
    let (mut last_epoch, mut best_epoch, mut stale) = (0, 0, 0);
    let patience: usize = args.get("patience", 0);
    let mut on_epoch =
        |ep: usize, loss: f32| report(ep, format!("  epoch {ep:>5}{total}   loss = {loss:.6}"));
    let mut on_val = |p: &ferrum_core::ValidationProgress| {
        (last_epoch, stale) = (p.epoch, if p.is_best { 0 } else { stale + 1 });
        if p.is_best {
            best_epoch = p.epoch;
        }
        let best = if p.is_best { "  *best" } else { "" };
        report(
            p.epoch,
            format!(
                "  epoch {:>5}{total}   loss = {:.6}   val ppl = {:.3}{best}",
                p.epoch, p.train_loss, p.val.perplexity
            ),
        );
    };
    let val = args.value("val").map(|_| ValidationConfig {
        val_fraction: args.get("val", 0.1),
        patience,
    });
    let slm = match (input, val) {
        (Input::Text(corpus), Some(val)) => GenerativeSLM::train_transformer_config_validated(
            corpus,
            &cfg,
            threads,
            &val,
            &mut rng,
            &mut on_val,
        )?,
        (Input::Text(corpus), None) => GenerativeSLM::train_transformer_config_threaded(
            corpus,
            &cfg,
            threads,
            &mut rng,
            &mut on_epoch,
        )?,
        (Input::Tokens(file, tok), Some(val)) => GenerativeSLM::train_transformer_tokens_validated(
            file,
            tok,
            &cfg,
            threads,
            &val,
            &mut rng,
            &mut on_val,
        )?,
        (Input::Tokens(file, tok), None) => GenerativeSLM::train_transformer_tokens(
            file,
            tok,
            &cfg,
            threads,
            &mut rng,
            &mut on_epoch,
        )?,
    };

    println!("\nTrained in {:.2}s.", t0.elapsed().as_secs_f32());
    let stopped = cfg.stop.stopped();
    if patience > 0 && stale >= patience {
        println!("early stop at epoch {last_epoch}, best epoch {best_epoch} kept");
    }
    let vocab_kind = if slm.meta.tokenizer_state.is_empty() {
        "chars"
    } else {
        "BPE tokens"
    };
    println!(
        "  vocab = {} {}   context = {}   output_dim = {}",
        slm.meta.output_dim, vocab_kind, slm.meta.input_dim, slm.meta.output_dim
    );

    if cfg.vocab_size > 0 && slm.meta.output_dim != cfg.vocab_size {
        let why = if reused_tokenizer {
            "fixed by the reused tokenizer"
        } else {
            "limited by the corpus"
        };
        println!(
            "  vocab: {} (requested {}, {why})",
            slm.meta.output_dim, cfg.vocab_size
        );
    }

    match quant {
        "int8" => slm.save(model_path),
        "int4" => slm.save_int4(model_path),
        _ => slm.save_f32(model_path),
    }
    .map_err(|e| format!("cannot write {model_path}: {e}"))?;
    let size = std::fs::metadata(model_path)?.len();
    println!("Saved {size} bytes → {model_path} ({quant} GGUF)");
    if let Some(w) = slm_cli::runtime_export::plain_gguf_warning(std::path::Path::new(model_path)) {
        eprintln!("{w}");
    }

    // Verify the file roundtrips.
    let reloaded = GenerativeSLM::load(model_path)?;
    println!("Reload check: OK ({} layers).", reloaded.model.len());
    if let Some((cause, step)) = stopped {
        cfg.stop.clear_stop_file();
        let ck = Some(cfg.checkpoint_path.as_str()).filter(|p| !p.is_empty());
        crate::stop::announce(cause, step, ck);
    }
    Ok(reloaded)
}

pub(crate) fn print_sample(
    slm: &GenerativeSLM,
    corpus: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let context_len = slm.meta.input_dim;
    let seed_text: String = corpus
        .chars()
        .filter(|&c| c != '\r')
        .take(context_len)
        .collect();
    print_sample_from(slm, &seed_text)
}

/// [`print_sample`] with the seed text already chosen.
pub(crate) fn print_sample_from(
    slm: &GenerativeSLM,
    seed_text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut g_rng = Rng::new(time_seed());
    let sample = slm.generate(seed_text, 120, 0.7, &mut g_rng)?;
    println!("\n── sample ──\n{sample}\n────────────");
    Ok(())
}

pub(crate) fn cmd_train(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.positional.len() < 2 {
        return Err(
            "usage: train_transformer train <corpus.txt | tokens.gguf> <model.gguf> [options]"
                .into(),
        );
    }
    // Before the model-exists shortcut too, so a typo'd --resume never goes unnoticed.
    require_resume_file(args.value("resume"), ".fckp")?;
    let corpus_path = &args.positional[0];
    let model_path = &args.positional[1];
    // A GGUF corpus is a pre-tokenized file (see `tokenize`); detected by content, not extension.
    let tokens = ferrum_core::is_gguf_file(corpus_path);
    let corpus = if tokens {
        String::new()
    } else {
        read_corpus(corpus_path)?
    };
    let mut token_input = tokens
        .then(|| crate::token_train::open(args, corpus_path))
        .transpose()?;

    let model_exists = std::path::Path::new(model_path).exists() && !args.has("force");
    let slm = if model_exists {
        let slm = GenerativeSLM::load(model_path)?;
        println!(
            "Model {model_path} already exists — loaded saved weights ({} layers, vocab {}).",
            slm.model.len(),
            slm.meta.output_dim
        );
        println!("Skipping training. Pass --force to retrain from scratch.");
        slm
    } else if let Some((file, tok)) = token_input.as_mut() {
        train_and_save_input(Input::Tokens(file, tok), corpus_path, model_path, args)?
    } else {
        train_and_save(&corpus, corpus_path, model_path, args)?
    };

    if args.has("sample") {
        match token_input.as_mut() {
            Some((file, tok)) => {
                print_sample_from(&slm, &crate::token_train::sample_seed(file, tok)?)?
            }
            None => print_sample(&slm, &corpus)?,
        }
    }
    Ok(())
}
