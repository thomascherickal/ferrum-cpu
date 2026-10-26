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
//! train_transformer train    <corpus.txt> <model.gguf> [options]
//! train_transformer run      <corpus.txt> <model.gguf> <seed text> [options]
//! train_transformer generate <model.gguf>  <seed text> [options]
//! train_transformer run-gguf  <model.gguf> [prompt] [--resume ckpt.flck] [options]
//! train_transformer finetune-gguf <model.gguf> <corpus.txt> <out.flck> [options]
//! train_transformer export-gguf <in.gguf> <out.gguf> [--quant q8_0|q4_0|q4_k|…] [--resume ckpt.flck]
//! train_transformer info     <model.gguf>
//! train_transformer prepare  --input <file|dir> --out <dir> [options]
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
//!   --val <F>  --patience <N>   validation split + early stopping
//!   --force         retrain even if the model file exists
//!   --sample        print a short sample after training
//!   --verbose | -v  print a progress line every 10 steps (tokens/s, elapsed)
//!   --debug         print all engine internals
//! ```
//!
//! ### `generate` / `run` options
//! ```text
//!   --chars <N>  characters to generate            (default 200)
//!   --temp  <F>  sampling temperature              (default 0.8)
//!   --gen-seed <N>  generation RNG seed            (default time-based)
//!   --stream     print the completion live as it is generated
//!   --top-k <N>  --top-p <F>  --repetition-penalty <F>   sampling knobs (off by default)
//!   --max-tokens <N>  token budget (default --chars); --stop <S> ends the
//!                     output before S (repeatable). Either flag selects the
//!                     token-budgeted path: stop strings are excluded and
//!                     --chars is ignored.
//! ```

#![forbid(unsafe_code)]

mod args;
mod eval;
mod export;
mod finetune;
mod generate;
mod info;
mod memory;
mod prepare;
mod resume;
mod run_gguf;
mod runtime_export;
mod train;
mod usage;
mod util;
mod validate;

use crate::args::*;
use crate::eval::*;
use crate::export::*;
use crate::finetune::*;
use crate::generate::*;
use crate::info::*;
use crate::prepare::cmd_prepare;
use crate::run_gguf::*;
use crate::train::*;
use crate::usage::*;

pub(crate) fn main() {
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
        eprintln!("unknown command: {cmd}\n");
        print_usage();
        std::process::exit(1);
    };
    let args = Args::parse_for(&cmd, &argv[2..], spec)?;
    let raw_args = &argv[2..];
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
        "export-ollama" => runtime_export::cmd_export_ollama(
            &args.positional,
            args.value("bin"),
            args.value("modelfile-only"),
        )
        .map_err(Into::into),
        "export-lmstudio" => runtime_export::cmd_export_lmstudio(
            &args.positional,
            args.value("repo"),
            args.value("bin"),
        )
        .map_err(Into::into),
        "eval" => cmd_eval(&args),
        "prepare" => cmd_prepare(&args, raw_args),
        "info" => cmd_info(&args),
        _ => unreachable!("spec_for covers every command"),
    }
}
