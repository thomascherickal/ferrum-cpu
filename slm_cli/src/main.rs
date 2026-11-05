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
mod interrupt;
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

/// The integration tests' synthetic tiny-llama GGUF builder, reused as the stub model of the
/// subcommand modules' unit tests (as `slm_server/src/test_util.rs` does).
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_fixtures;

/// Helpers shared by the subcommand modules' unit tests.
#[cfg(test)]
mod test_util {
    use crate::args::{spec_for, Args};

    /// `raw` parsed exactly as `train_transformer <cmd> raw...` would parse it.
    pub(crate) fn args(cmd: &str, raw: &[&str]) -> Args {
        let raw: Vec<String> = raw.iter().map(|s| s.to_string()).collect();
        Args::parse_for(cmd, &raw, spec_for(cmd).unwrap()).unwrap()
    }

    /// The error text of a command that must fail.
    pub(crate) fn err<T>(r: Result<T, Box<dyn std::error::Error>>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    /// A fresh scratch directory under the system temp dir, removed on drop.
    pub(crate) struct Tmp(std::path::PathBuf);

    impl Tmp {
        pub(crate) fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!("ferrum_cli_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
        /// The path of `name` inside the directory.
        pub(crate) fn path(&self, name: &str) -> String {
            self.0.join(name).to_string_lossy().into_owned()
        }
        /// Write `bytes` to `name` and return its path.
        pub(crate) fn write(&self, name: &str, bytes: impl AsRef<[u8]>) -> String {
            let p = self.path(name);
            std::fs::write(&p, bytes).unwrap();
            p
        }
    }

    /// The flags of a micro model (context 8, embed 8, one block, one epoch, characters): the
    /// smallest model `train` builds, the stub for the commands that need a Ferrum model.
    pub(crate) const MICRO: [&str; 18] = [
        "--context",
        "8",
        "--embed",
        "8",
        "--heads",
        "2",
        "--blocks",
        "1",
        "--hidden",
        "16",
        "--epochs",
        "1",
        "--vocab",
        "0",
        "--threads",
        "1",
        "--seed",
        "1",
    ];

    /// Train the micro model on a short corpus into `t/name` and return its path.
    pub(crate) fn micro_model(t: &Tmp, name: &str) -> String {
        let corpus = t.write(&format!("{name}.txt"), "abc abd abe ".repeat(8));
        let model = t.path(name);
        let mut raw = vec![corpus.as_str(), model.as_str()];
        raw.extend_from_slice(&MICRO);
        crate::train::cmd_train(&args("train", &raw)).unwrap();
        model
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

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

/// Whether a panic message is `println!` failing on a reader that went away: "Broken pipe"
/// on Unix, "The pipe is being closed. (os error 232)" (or error 109/233) on Windows.
fn is_closed_stdout(msg: &str) -> bool {
    msg.starts_with("failed printing to stdout")
        && [
            "Broken pipe",
            "(os error 232)",
            "(os error 109)",
            "(os error 233)",
        ]
        .iter()
        .any(|m| msg.contains(m))
}

pub(crate) fn main() {
    // `train_transformer info m.gguf | head -1`: `println!` panics on a closed pipe. The
    // lazy fix for every print site at once: a panic whose message is that failure exits 0.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map_or("", |s| s.as_str());
        if is_closed_stdout(msg) {
            std::process::exit(0);
        }
        default_hook(info);
    }));
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    // A run stopped by Ctrl-C has written its checkpoint; say so with the shell's status.
    if interrupt::is_set() {
        std::process::exit(interrupt::EXIT_STATUS);
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
    if interrupt::wants_handler(&cmd) {
        if let Err(e) = interrupt::install() {
            eprintln!("warning: {e}; Ctrl-C will end the run without a checkpoint");
        }
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
        // ponytail: `spec_for` accepted `cmd`, so after the arms above only `info` is left.
        _ => cmd_info(&args),
    }
}

#[cfg(test)]
mod tests {
    use super::is_closed_stdout;
    use super::test_util::{args, err, Tmp};

    #[test]
    fn closed_stdout_messages_by_platform() {
        for (msg, want) in [
            ("failed printing to stdout: Broken pipe (os error 32)", true),
            ("failed printing to stdout: The pipe is being closed. (os error 232)", true),
            ("failed printing to stdout: The pipe has been ended. (os error 109)", true),
            ("failed printing to stdout: No process is on the other end of the pipe. (os error 233)", true),
            ("failed printing to stdout: No space left on device (os error 28)", false),
            ("Broken pipe", false),
            ("", false),
        ] {
            assert_eq!(is_closed_stdout(msg), want, "{msg}");
        }
    }

    #[test]
    fn test_util_parses_like_the_binary_and_cleans_up() {
        let a = args("eval", &["m.gguf", "t.txt", "--json", "o.json"]);
        assert_eq!(a.positional, ["m.gguf", "t.txt"]);
        assert_eq!(a.value("json"), Some("o.json"));
        assert_eq!(err::<()>(Err("boom".into())), "boom");
        let t = Tmp::new("selftest");
        let p = t.write("x.txt", "hi");
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "hi");
        let dir = std::path::PathBuf::from(t.path(""));
        drop(t);
        assert!(!dir.exists(), "the scratch directory is removed on drop");
    }
}
