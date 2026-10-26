//! `train_transformer` — corpus → causal transformer SLM trainer & generator.
//!
//! Trains a character-level decoder-only **causal Transformer** Small
//! Language Model from any raw text corpus — entirely on CPU, with a
//! `std`-only engine (two pure-Rust Unicode table crates). Training is int8 quantization-aware (QAT) and models are
//! exported as int8 GGUF (≈4× smaller than f32) unless `--quant int4|f32` says otherwise.
//!
//! Trained weights are cached on disk: if the model file already exists,
//! `train` and `run` load it instead of retraining (use `--force` to retrain).
//!
//! ## Usage
//!
//! ```text
//! train_transformer train    <corpus.txt|tokens.gguf> <model.gguf> [options]
//! train_transformer run      <corpus.txt> <model.gguf> <seed text> [options]
//! train_transformer generate <model.gguf>  <seed text> [options]
//! train_transformer run-gguf  <model.gguf> [prompt] [--resume ckpt.flck] [options]
//! train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> [options]
//! train_transformer export-gguf <in.gguf> <out.gguf> [--quant q8_0|q4_0|q4_k|…] [--resume ckpt.flck]
//! train_transformer eval-gguf <model.gguf> <text.txt> [--ctx N] [--quant f32|int8|int4] [--resume ckpt.flck] [--json out.json]
//! train_transformer info     <model.gguf>
//! train_transformer prepare  --input <file|dir> --out <dir> [options]
//! train_transformer tokenize <corpus.txt|dir> --tokenizer <tok.gguf> --out <train.tok.gguf> [--doc-sep S]
//! ```
//!
//! ### `train` / `run` options
//! ```text
//!   --context <N>   context window in characters   (default 16)
//!   --embed   <N>   embedding dimension            (default 32)
//!   --heads   <N>   attention heads                (default 4)
//!   --blocks  <N>   transformer blocks             (default 2)
//!   --hidden  <N>   FFN hidden width               (default 64)
//!   --epochs  <N>   training epochs                (default 100)
//!   --lr      <F>   Adam learning rate             (default 0.01)
//!   --batch   <N>   minibatch size                 (default 16)
//!   --vocab   <N>   BPE vocab size (0 = char-level)(default 512)
//!   --seed    <N>   RNG seed                       (default 1337)
//!   --threads <N>   data-parallel worker threads   (default 0 = auto)
//!   --stride  <N>   window stride (0 = context)    (default 0)
//!   --tokens  <N>   token budget, overrides epochs (default 0 = off)
//!   --cosine / --warmup <N>  warmup + cosine LR decay
//!   --clip <F>  --tie  --no-qat  --tokenizer <path>
//!   --quant <int8|int4|f32>  on-disk precision      (default int8)
//!   --checkpoint <path>  --checkpoint-every <N>  --resume <path>
//!   --max-seconds <F>  stop cleanly after the step that crosses the budget (also: touch <model>.stop)
//!   --val <F>  --patience <N>   validation split + early stopping
//!   --preset <tiny|small|medium>  starting sizes from Manual §6.7 (explicit flags win)
//!   --force         retrain even if the model file exists
//!   --sample        print a short sample after training
//!   --metrics-csv <file>  one CSV row per optimizer step (loss, lr, grad norm, tokens/s)
//!   --verbose | -v  print a progress line every 10 steps (tokens/s, grad norm, elapsed, ETA)
//!   --debug         print all engine internals
//! ```
//!
//! ### `generate` / `run` options
//! ```text
//!   --chars <N>  characters to generate            (default 200)
//!   --temp  <F>  sampling temperature              (default 0.8)
//!   --gen-seed <N>  generation RNG seed            (default time-based)
//!   --stream     print the completion live as it is generated (also with --stop/--max-tokens;
//!                only a possible stop-string prefix is held back)
//!   --top-k <N>  --top-p <F>  --repetition-penalty <F>   sampling knobs (off by default)
//!   --max-tokens <N>  token budget (default --chars); --stop <S> ends the
//!                     output before S (repeatable). Either flag selects the
//!                     token-budgeted path: stop strings are excluded and
//!                     --chars is ignored.
//! ```

#![forbid(unsafe_code)]

mod args;
mod cmd_prepare;
mod cmd_tokenize;
mod eval;
mod eval_gguf;
mod export;
mod finetune;
mod generate;
mod info;
mod memory;
mod preset;
mod resume;
mod run_gguf;
mod stop;
mod token_train;
mod train;
mod usage;
mod util;
mod validate;

use crate::args::*;
use crate::cmd_prepare::cmd_prepare;
use crate::cmd_tokenize::cmd_tokenize;
use crate::eval::*;
use crate::eval_gguf::cmd_eval_gguf;
use crate::export::*;
use crate::finetune::*;
use crate::generate::*;
use crate::info::*;
use crate::run_gguf::*;
use crate::train::*;
use crate::usage::*;

pub(crate) fn main() {
    // `train_transformer info m.gguf | head -1`: `println!` panics on a closed pipe. The
    // lazy fix for every print site at once: a panic whose message is that failure exits 0.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map_or("", |s| s.as_str());
        if msg.starts_with("failed printing to stdout") && msg.contains("Broken pipe") {
            std::process::exit(0);
        }
        default_hook(info);
    }));
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        print_usage();
        std::process::exit(1);
    }
    let cmd = argv[1].clone();
    let Some(spec) = spec_for(&cmd) else {
        if matches!(cmd.as_str(), "-h" | "--help" | "help") {
            print_usage();
            return Ok(());
        }
        if matches!(cmd.as_str(), "-V" | "--version") {
            println!("train_transformer {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        eprintln!("unknown command: {cmd} (run 'train_transformer --help' for usage)");
        std::process::exit(1);
    };
    if argv[2..].iter().any(|a| a == "--help" || a == "-h") {
        print!("{}", command_usage(&cmd, spec));
        return Ok(());
    }
    let args = Args::parse_for(&cmd, &argv[2..], spec)?;
    if let Some(extra) = max_positional(&cmd).and_then(|n| args.positional.get(n)) {
        return Err(format!("unexpected argument '{extra}' for '{cmd}'").into());
    }
    let raw_args = &argv[2..];
    preset::selected(&args)?; // an unknown --preset fails before anything else, even if the model exists
    if engine_debug(&cmd, &args) {
        ferrum_core::set_verbose(true);
    }
    match cmd.as_str() {
        "train" => cmd_train(&args),
        "run" => cmd_run(&args),
        "generate" | "gen" => cmd_generate(&args),
        "run-gguf" | "gguf" => cmd_run_gguf(&args),
        "finetune-gguf" | "finetune" => cmd_finetune_gguf(&args),
        "export-gguf" | "export" => cmd_export_gguf(&args),
        "export-ollama" => slm_cli::runtime_export::cmd_export_ollama(
            &args.positional,
            args.value("bin"),
            args.value("modelfile-only"),
        )
        .map_err(Into::into),
        "export-lmstudio" => slm_cli::runtime_export::cmd_export_lmstudio(
            &args.positional,
            args.value("repo"),
            args.value("bin"),
        )
        .map_err(Into::into),
        "eval" => cmd_eval(&args),
        "eval-gguf" => cmd_eval_gguf(&args),
        "prepare" => cmd_prepare(&args, raw_args),
        "tokenize" => cmd_tokenize(&args),
        "info" => cmd_info(&args),
        _ => unreachable!("spec_for covers every command"),
    }
}
