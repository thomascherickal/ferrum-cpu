//! Flag tables and the strict `positional --flag value` parser shared by every subcommand.

use ferrum_core::TransformerConfig;

/// How a `--flag` is parsed. Values are validated at parse time, so a typo
/// like `--context 12x` is an error instead of a silent default.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    Bool,
    Usize,
    U64,
    F32,
    /// f64, so budgets like `5e7` parse.
    F64,
    Str,
    /// Repeatable string flag (`--stop a --stop b`).
    StrList,
}

/// A subcommand's accepted flags.
pub(crate) type Spec = &'static [(&'static str, Kind)];

pub(crate) const TRAIN_FLAGS: Spec = &[
    ("context", Kind::Usize),
    ("embed", Kind::Usize),
    ("heads", Kind::Usize),
    ("blocks", Kind::Usize),
    ("hidden", Kind::Usize),
    ("epochs", Kind::Usize),
    ("lr", Kind::F32),
    ("batch", Kind::Usize),
    ("vocab", Kind::Usize),
    ("seed", Kind::U64),
    ("threads", Kind::Usize),
    ("weight_decay", Kind::F32),
    ("dropout", Kind::F32),
    ("stride", Kind::Usize),
    ("tokens", Kind::F64),
    ("cosine", Kind::Bool),
    ("warmup", Kind::U64),
    ("clip", Kind::F32),
    ("tie", Kind::Bool),
    ("no-qat", Kind::Bool),
    ("quant", Kind::Str),
    ("debug", Kind::Bool),
    ("tokenizer", Kind::Str),
    ("checkpoint", Kind::Str),
    ("checkpoint-every", Kind::U64),
    ("resume", Kind::Str),
    ("val", Kind::F32),
    ("patience", Kind::Usize),
    ("force", Kind::Bool),
    ("sample", Kind::Bool),
    ("max-seconds", Kind::F64),
    ("preset", Kind::Str),
    ("metrics-csv", Kind::Str),
];

pub(crate) const GENERATE_FLAGS: Spec = &[
    ("chars", Kind::Usize),
    ("temp", Kind::F32),
    ("gen-seed", Kind::U64),
    ("stream", Kind::Bool),
    ("top-k", Kind::Usize),
    ("top-p", Kind::F32),
    ("repetition-penalty", Kind::F32),
    ("max-tokens", Kind::Usize),
    ("stop", Kind::StrList),
];

pub(crate) const RUN_GGUF_FLAGS: Spec = &[
    ("quant", Kind::Str),
    ("max", Kind::Usize),
    ("temp", Kind::F32),
    ("gen-seed", Kind::U64),
    ("top-k", Kind::Usize),
    ("top-p", Kind::F32),
    ("rep", Kind::F32),
    ("ids", Kind::Str),
    ("raw", Kind::Bool),
    ("force", Kind::Bool),
    ("resume", Kind::Str),
];

pub(crate) const FINETUNE_FLAGS: Spec = &[
    ("epochs", Kind::Usize),
    ("lr", Kind::F32),
    ("batch", Kind::Usize),
    ("seq", Kind::Usize),
    ("weight_decay", Kind::F32),
    ("dropout", Kind::F32),
    ("clip", Kind::F32),
    ("warmup", Kind::U64),
    ("qat", Kind::Bool),
    ("seed", Kind::U64),
    ("threads", Kind::Usize),
    ("resume", Kind::Str),
    ("sample", Kind::Bool),
    ("temp", Kind::F32),
    ("max", Kind::Usize),
    ("force", Kind::Bool),
    ("max-seconds", Kind::F64),
];

pub(crate) const EXPORT_FLAGS: Spec = &[
    ("quant", Kind::Str),
    ("resume", Kind::Str),
    ("force", Kind::Bool),
];

pub(crate) const EXPORT_OLLAMA_FLAGS: Spec = &[("bin", Kind::Str), ("modelfile-only", Kind::Str)];

pub(crate) const EXPORT_LMSTUDIO_FLAGS: Spec = &[("repo", Kind::Str), ("bin", Kind::Str)];

pub(crate) const PREPARE_FLAGS: Spec = &[
    ("input", Kind::StrList),
    ("out", Kind::Str),
    ("doc-sep", Kind::Str),
    ("lossy", Kind::Bool),
    ("nfkc", Kind::Bool),
    ("lowercase", Kind::Bool),
    ("strip-gutenberg", Kind::Bool),
    ("min-chars", Kind::Usize),
    ("max-chars", Kind::Usize),
    ("dedup", Kind::Bool),
    ("val-frac", Kind::F64),
    ("test-frac", Kind::F64),
    ("seed", Kind::U64),
];

pub(crate) const TOKENIZE_FLAGS: Spec = &[
    ("tokenizer", Kind::Str),
    ("out", Kind::Str),
    ("doc-sep", Kind::Str),
];

pub(crate) const EVAL_FLAGS: Spec = &[("json", Kind::Str)];

pub(crate) const EVAL_GGUF_FLAGS: Spec = &[
    ("ctx", Kind::Usize),
    ("quant", Kind::Str),
    ("json", Kind::Str),
    ("resume", Kind::Str),
    ("force", Kind::Bool),
];

pub(crate) const NO_FLAGS: Spec = &[];

/// `run` takes the union of train and generate flags.
pub(crate) fn run_flags() -> Vec<(&'static str, Kind)> {
    TRAIN_FLAGS.iter().chain(GENERATE_FLAGS).copied().collect()
}

pub(crate) fn spec_for(cmd: &str) -> Option<Spec> {
    static RUN: std::sync::OnceLock<Vec<(&'static str, Kind)>> = std::sync::OnceLock::new();
    Some(match cmd {
        "train" => TRAIN_FLAGS,
        "run" => RUN.get_or_init(run_flags).as_slice(),
        "generate" | "gen" => GENERATE_FLAGS,
        "run-gguf" | "gguf" => RUN_GGUF_FLAGS,
        "finetune-gguf" | "finetune" => FINETUNE_FLAGS,
        "export-gguf" | "export" => EXPORT_FLAGS,
        "export-ollama" => EXPORT_OLLAMA_FLAGS,
        "export-lmstudio" => EXPORT_LMSTUDIO_FLAGS,
        "prepare" => PREPARE_FLAGS,
        "tokenize" => TOKENIZE_FLAGS,
        "eval" => EVAL_FLAGS,
        "eval-gguf" => EVAL_GGUF_FLAGS,
        "info" => NO_FLAGS,
        _ => return None,
    })
}

/// The most positional arguments a command takes; `None` for the free-text commands
/// (`run`, `generate`, `run-gguf` join the extra words into the prompt).
pub(crate) fn max_positional(cmd: &str) -> Option<usize> {
    match cmd {
        "train" | "export-gguf" | "export" | "export-ollama" | "eval" | "eval-gguf" => Some(2),
        "finetune-gguf" | "finetune" => Some(3),
        "export-lmstudio" | "info" => Some(1),
        "prepare" => Some(0),
        "tokenize" => Some(1),
        _ => None,
    }
}

/// Parsed `positional --flag value` arguments for one subcommand.
pub(crate) struct Args {
    pub(crate) positional: Vec<String>,
    pub(crate) verbose: bool,
    pub(crate) flags: std::collections::HashMap<String, Vec<String>>,
    pub(crate) bools: std::collections::HashSet<String>,
}

impl Args {
    #[cfg(test)]
    pub(crate) fn parse(raw: &[String], spec: Spec) -> Result<Self, String> {
        Self::parse_for("", raw, spec)
    }

    pub(crate) fn parse_for(cmd: &str, raw: &[String], spec: Spec) -> Result<Self, String> {
        let mut a = Args {
            positional: Vec::new(),
            verbose: false,
            flags: Default::default(),
            bools: Default::default(),
        };
        let mut i = 0;
        while i < raw.len() {
            let arg = &raw[i];
            if arg == "--verbose" || arg == "-v" {
                a.verbose = true;
            } else if let Some(name) = arg.strip_prefix("--") {
                let kind = spec
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, k)| *k)
                    .ok_or_else(|| {
                        let valid: Vec<String> =
                            spec.iter().map(|(n, _)| format!("--{n}")).collect();
                        let for_cmd = if cmd.is_empty() {
                            String::new()
                        } else {
                            format!(" for '{cmd}'")
                        };
                        format!(
                            "unknown flag --{name}{for_cmd} (valid: {})",
                            valid.join(" ")
                        )
                    })?;
                if kind == Kind::Bool {
                    a.bools.insert(name.to_string());
                } else {
                    let val = raw
                        .get(i + 1)
                        .ok_or_else(|| format!("--{name} needs a value"))?;
                    let ok = match kind {
                        Kind::Usize => val.parse::<usize>().is_ok(),
                        Kind::U64 => val.parse::<u64>().is_ok(),
                        Kind::F32 => val.parse::<f32>().is_ok(),
                        Kind::F64 => val.parse::<f64>().is_ok(),
                        Kind::Str | Kind::StrList | Kind::Bool => true,
                    };
                    if !ok {
                        return Err(format!(
                            "--{name}: cannot parse '{val}' as {}",
                            kind_name(kind)
                        ));
                    }
                    a.flags
                        .entry(name.to_string())
                        .or_default()
                        .push(val.clone());
                    i += 1;
                }
            } else {
                a.positional.push(arg.clone());
            }
            i += 1;
        }
        Ok(a)
    }

    /// Last value of `key` (values were validated at parse time).
    pub(crate) fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.value(key)
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }
    pub(crate) fn value(&self, key: &str) -> Option<&str> {
        self.flags
            .get(key)
            .and_then(|v| v.last())
            .map(String::as_str)
    }
    pub(crate) fn get_all(&self, key: &str) -> Vec<String> {
        self.flags.get(key).cloned().unwrap_or_default()
    }
    pub(crate) fn has(&self, key: &str) -> bool {
        self.bools.contains(key)
    }

    pub(crate) fn config(&self) -> TransformerConfig {
        let d = match crate::preset::selected(self) {
            Ok(Some(p)) => p.defaults(TransformerConfig::default()),
            _ => TransformerConfig::default(), // an unknown name is reported by `train_and_save`
        };
        TransformerConfig {
            context_len: self.get("context", d.context_len),
            embed_dim: self.get("embed", d.embed_dim),
            num_heads: self.get("heads", d.num_heads),
            num_blocks: self.get("blocks", d.num_blocks),
            hidden_dim: self.get("hidden", d.hidden_dim),
            epochs: self.get("epochs", d.epochs),
            lr: self.get("lr", d.lr),
            batch_size: self.get("batch", d.batch_size),
            vocab_size: self.get("vocab", d.vocab_size),
            weight_decay: self.get("weight_decay", d.weight_decay),
            dropout: self.get("dropout", d.dropout),
            window_stride: self.get("stride", d.window_stride),
            // f64 so budgets like `5e7` parse.
            max_tokens: self.get("tokens", 0.0f64) as u64,
            qat: !self.has("no-qat"),
            grad_clip: self.get("clip", d.grad_clip),
            cosine_lr: self.has("cosine"),
            warmup_steps: self.get("warmup", d.warmup_steps),
            weight_tying: self.has("tie"),
            tokenizer_path: self.get("tokenizer", d.tokenizer_path),
            // `--resume ckpt` resumes from and keeps writing to that file.
            checkpoint_path: self
                .value("resume")
                .or_else(|| self.value("checkpoint"))
                .map(String::from)
                .unwrap_or_default(),
            checkpoint_every: self.get("checkpoint-every", d.checkpoint_every),
            resume: self.value("resume").is_some(),
            metrics_csv: self.get("metrics-csv", String::new()),
            // `-v` prints a progress line every 10 optimizer steps.
            log_every_steps: if self.verbose { 10 } else { 0 },
            stop: d.stop,
        }
    }
}

/// The `--tokenizer` path when `--val` is set and that file already exists: a
/// reused tokenizer may have been trained on the held-out text too.
pub(crate) fn reused_tokenizer_with_val(args: &Args) -> Option<&str> {
    args.value("val")?;
    args.value("tokenizer")
        .filter(|p| std::path::Path::new(p).exists())
}

/// Whether to turn on the engine's internal logs. `train` and `run` use `-v`
/// for periodic progress lines instead and need `--debug` for the internals.
pub(crate) fn engine_debug(cmd: &str, args: &Args) -> bool {
    args.has("debug") || (args.verbose && !matches!(cmd, "train" | "run"))
}

pub(crate) fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Usize | Kind::U64 => "a non-negative integer",
        Kind::F32 | Kind::F64 => "a number",
        _ => "a string",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }
    const SPEC: Spec = &[
        ("stop", Kind::StrList),
        ("temp", Kind::F32),
        ("stream", Kind::Bool),
        ("n", Kind::Usize),
        ("seed", Kind::U64),
        ("budget", Kind::F64),
        ("name", Kind::Str),
    ];

    fn err(raw: &[&str]) -> String {
        Args::parse(&s(raw), SPEC)
            .err()
            .expect("expected parse error")
    }

    #[test]
    fn repeated_flags_keep_all_values() {
        let a = Args::parse(&s(&["--stop", "a", "--stop", "b"]), SPEC).unwrap();
        assert_eq!(a.get_all("stop"), vec!["a", "b"]);
        assert_eq!(a.value("stop"), Some("b"));
    }

    #[test]
    fn repeated_scalar_flag_last_wins() {
        let a = Args::parse(&s(&["--n", "1", "--n", "2"]), SPEC).unwrap();
        assert_eq!(a.get("n", 0usize), 2);
    }

    #[test]
    fn negative_number_is_a_value() {
        let a = Args::parse(&s(&["--temp", "-1"]), SPEC).unwrap();
        assert_eq!(a.get("temp", 0.0f32), -1.0);
    }

    #[test]
    fn bool_and_positionals_mix() {
        let a = Args::parse(&s(&["m.gguf", "--stream", "hello", "-v"]), SPEC).unwrap();
        assert!(a.has("stream") && a.verbose);
        assert_eq!(a.positional, vec!["m.gguf", "hello"]);
    }

    #[test]
    fn bool_does_not_consume_next_argument() {
        let a = Args::parse(&s(&["--stream", "7"]), SPEC).unwrap();
        assert!(a.has("stream"));
        assert_eq!(a.positional, vec!["7"]);
        assert!(a.value("stream").is_none());
    }

    #[test]
    fn long_verbose_and_absent_flags() {
        let a = Args::parse(&s(&["--verbose"]), SPEC).unwrap();
        assert!(a.verbose);
        let a = Args::parse(&s(&[]), SPEC).unwrap();
        assert!(!a.verbose && !a.has("stream") && a.positional.is_empty());
        assert_eq!(a.get("n", 9usize), 9);
        assert!(a.get_all("stop").is_empty());
        assert_eq!(a.value("n"), None);
    }

    #[test]
    fn single_dash_is_positional() {
        let a = Args::parse(&s(&["-x", "-"]), SPEC).unwrap();
        assert_eq!(a.positional, vec!["-x", "-"]);
    }

    #[test]
    fn unknown_flag_lists_valid_ones() {
        let e = err(&["--nope"]);
        assert!(e.contains("unknown flag --nope"), "{e}");
        assert!(e.contains("--stop") && e.contains("--temp"), "{e}");
        assert!(!e.contains(" for '"), "{e}");
    }

    #[test]
    fn unknown_flag_names_the_command() {
        let e = Args::parse_for("info", &s(&["--epochs", "3"]), NO_FLAGS)
            .err()
            .unwrap();
        assert!(e.contains("unknown flag --epochs for 'info'"), "{e}");
    }

    #[test]
    fn bare_double_dash_is_unknown() {
        assert!(err(&["--"]).contains("unknown flag --"));
    }

    #[test]
    fn missing_value_at_end_is_an_error() {
        assert_eq!(err(&["--temp"]), "--temp needs a value");
        assert_eq!(err(&["--name"]), "--name needs a value");
        assert_eq!(err(&["--stop"]), "--stop needs a value");
    }

    #[test]
    fn each_kind_rejects_unparsable_values() {
        for (flag, bad, what) in [
            ("n", "12x", "a non-negative integer"),
            ("n", "-3", "a non-negative integer"),
            ("seed", "1.5", "a non-negative integer"),
            ("temp", "hot", "a number"),
            ("budget", "5e", "a number"),
        ] {
            let e = err(&[&format!("--{flag}"), bad]);
            assert_eq!(e, format!("--{flag}: cannot parse '{bad}' as {what}"));
        }
    }

    #[test]
    fn each_kind_accepts_valid_values() {
        let a = Args::parse(
            &s(&[
                "--n",
                "5",
                "--seed",
                "18446744073709551615",
                "--temp",
                "0.5",
                "--budget",
                "5e7",
                "--name",
                "-weird",
                "--stop",
                "",
            ]),
            SPEC,
        )
        .unwrap();
        assert_eq!(a.get("n", 0usize), 5);
        assert_eq!(a.get("seed", 0u64), u64::MAX);
        assert_eq!(a.get("temp", 0.0f32), 0.5);
        assert_eq!(a.get("budget", 0.0f64), 5e7);
        assert_eq!(a.value("name"), Some("-weird"));
        assert_eq!(a.get_all("stop"), vec![""]);
    }

    #[test]
    fn get_falls_back_when_type_mismatches() {
        // A Str flag read as a number falls back to the default.
        let a = Args::parse(&s(&["--name", "abc"]), SPEC).unwrap();
        assert_eq!(a.get("name", 4usize), 4);
    }

    #[test]
    fn kind_names() {
        assert_eq!(kind_name(Kind::Usize), "a non-negative integer");
        assert_eq!(kind_name(Kind::U64), "a non-negative integer");
        assert_eq!(kind_name(Kind::F32), "a number");
        assert_eq!(kind_name(Kind::F64), "a number");
        assert_eq!(kind_name(Kind::Str), "a string");
        assert_eq!(kind_name(Kind::StrList), "a string");
        assert_eq!(kind_name(Kind::Bool), "a string");
    }

    #[test]
    fn prepare_accepts_repeated_input_and_every_documented_flag() {
        let a = Args::parse_for(
            "prepare",
            &s(&[
                "--input",
                "a.txt",
                "--input",
                "dir",
                "--out",
                "o",
                "--doc-sep",
                "<|endoftext|>",
                "--lossy",
                "--nfkc",
                "--lowercase",
                "--strip-gutenberg",
                "--dedup",
                "--min-chars",
                "10",
                "--max-chars",
                "500",
                "--val-frac",
                "0.1",
                "--test-frac",
                "0.2",
                "--seed",
                "7",
            ]),
            PREPARE_FLAGS,
        )
        .unwrap();
        assert_eq!(a.get_all("input"), vec!["a.txt", "dir"]);
        assert_eq!(a.value("out"), Some("o"));
        assert!(a.has("lossy") && a.has("nfkc") && a.has("dedup"));
        assert_eq!(a.get("val-frac", 0.0f64), 0.1);
        assert_eq!(a.get("seed", 0u64), 7);
    }

    #[test]
    fn prepare_rejects_unknown_and_malformed_flags() {
        let e = Args::parse_for("prepare", &s(&["--inptu", "x"]), PREPARE_FLAGS)
            .err()
            .unwrap();
        assert!(e.contains("unknown flag --inptu"), "{e}");
        let e = Args::parse_for("prepare", &s(&["--val-frac", "abc"]), PREPARE_FLAGS)
            .err()
            .unwrap();
        assert!(e.contains("--val-frac"), "{e}");
        let e = Args::parse_for("prepare", &s(&["--out"]), PREPARE_FLAGS)
            .err()
            .unwrap();
        assert!(e.contains("--out needs a value"), "{e}");
    }

    #[test]
    fn eval_takes_only_json() {
        let a = Args::parse_for(
            "eval",
            &s(&["m.gguf", "t.txt", "--json", "o.json"]),
            EVAL_FLAGS,
        )
        .unwrap();
        assert_eq!(a.value("json"), Some("o.json"));
        assert_eq!(a.positional, vec!["m.gguf", "t.txt"]);
        assert!(Args::parse_for("eval", &s(&["--epochs", "1"]), EVAL_FLAGS).is_err());
    }

    #[test]
    fn spec_for_known_commands_and_aliases() {
        for c in [
            "train",
            "run",
            "generate",
            "gen",
            "run-gguf",
            "gguf",
            "finetune-gguf",
            "finetune",
            "export-gguf",
            "export",
            "export-ollama",
            "export-lmstudio",
            "prepare",
            "eval",
            "info",
        ] {
            assert!(spec_for(c).is_some(), "{c}");
        }
        for c in ["", "help", "-h", "bogus", "TRAIN"] {
            assert!(spec_for(c).is_none(), "{c}");
        }
        assert_eq!(spec_for("eval").unwrap().len(), EVAL_FLAGS.len());
        assert!(spec_for("eval").unwrap().iter().any(|(n, _)| *n == "json"));
        assert!(spec_for("info").unwrap().is_empty());
        assert_eq!(
            spec_for("gen").unwrap().len(),
            spec_for("generate").unwrap().len()
        );
    }

    #[test]
    fn run_accepts_train_and_generate_flags_but_not_gguf_ones() {
        let spec = spec_for("run").unwrap();
        for f in TRAIN_FLAGS.iter().chain(GENERATE_FLAGS) {
            assert!(spec.iter().any(|(n, _)| *n == f.0), "{}", f.0);
        }
        assert!(!spec.iter().any(|(n, _)| *n == "ids" || *n == "raw"));
        assert_eq!(spec.len(), run_flags().len());
    }

    #[test]
    fn every_flag_each_command_reads_is_accepted() {
        // Mirrors the `args.get/has/value` calls in each cmd_* function.
        let cases: &[(&str, &[&str])] = &[
            (
                "train",
                &[
                    "context",
                    "embed",
                    "heads",
                    "blocks",
                    "hidden",
                    "epochs",
                    "lr",
                    "batch",
                    "vocab",
                    "seed",
                    "threads",
                    "weight_decay",
                    "dropout",
                    "stride",
                    "tokens",
                    "cosine",
                    "warmup",
                    "clip",
                    "tie",
                    "no-qat",
                    "quant",
                    "debug",
                    "tokenizer",
                    "checkpoint",
                    "checkpoint-every",
                    "resume",
                    "val",
                    "patience",
                    "force",
                    "sample",
                    "max-seconds",
                ],
            ),
            (
                "generate",
                &[
                    "chars",
                    "temp",
                    "gen-seed",
                    "stream",
                    "top-k",
                    "top-p",
                    "repetition-penalty",
                    "max-tokens",
                    "stop",
                ],
            ),
            (
                "run",
                &["force", "stream", "chars", "context", "max-tokens", "stop"],
            ),
            (
                "run-gguf",
                &[
                    "quant", "max", "temp", "gen-seed", "top-k", "top-p", "rep", "ids", "raw",
                    "force", "resume",
                ],
            ),
            (
                "finetune-gguf",
                &[
                    "epochs",
                    "lr",
                    "batch",
                    "seq",
                    "weight_decay",
                    "dropout",
                    "clip",
                    "warmup",
                    "qat",
                    "seed",
                    "threads",
                    "resume",
                    "sample",
                    "temp",
                    "max",
                    "force",
                    "max-seconds",
                ],
            ),
            ("export-gguf", &["quant", "resume", "force"]),
            ("export-ollama", &["bin", "modelfile-only"]),
            ("export-lmstudio", &["repo", "bin"]),
        ];
        for (cmd, flags) in cases {
            let spec = spec_for(cmd).unwrap();
            for f in *flags {
                let kind = spec.iter().find(|(n, _)| n == f).map(|(_, k)| *k);
                assert!(kind.is_some(), "{cmd} --{f}");
            }
        }
    }

    #[test]
    fn config_reads_flags_and_defaults() {
        let spec = spec_for("train").unwrap();
        let d = TransformerConfig::default();
        let a = Args::parse(&s(&[]), spec).unwrap();
        let c = a.config();
        assert_eq!(c.context_len, d.context_len);
        assert!(c.qat && !c.cosine_lr && !c.weight_tying && !c.resume);
        assert_eq!(c.max_tokens, 0);
        assert_eq!(c.checkpoint_path, "");

        let a = Args::parse(
            &s(&[
                "--context",
                "8",
                "--tokens",
                "5e7",
                "--no-qat",
                "--cosine",
                "--tie",
                "--checkpoint",
                "a.fckp",
            ]),
            spec,
        )
        .unwrap();
        let c = a.config();
        assert_eq!(c.context_len, 8);
        assert_eq!(c.max_tokens, 50_000_000);
        assert!(!c.qat && c.cosine_lr && c.weight_tying && !c.resume);
        assert_eq!(c.checkpoint_path, "a.fckp");
    }

    #[test]
    fn resume_wins_over_checkpoint_and_sets_resume() {
        let spec = spec_for("train").unwrap();
        let a = Args::parse(&s(&["--checkpoint", "a", "--resume", "b"]), spec).unwrap();
        let c = a.config();
        assert_eq!(c.checkpoint_path, "b");
        assert!(c.resume);
    }

    #[test]
    fn verbose_means_progress_lines_for_train_and_run_and_internals_elsewhere() {
        let train = spec_for("train").unwrap();
        let plain = Args::parse(&s(&[]), train).unwrap();
        let v = Args::parse(&s(&["-v"]), train).unwrap();
        let dbg = Args::parse(&s(&["--debug"]), train).unwrap();
        assert_eq!(plain.config().log_every_steps, 0);
        assert_eq!(v.config().log_every_steps, 10);
        for cmd in ["train", "run"] {
            assert!(!engine_debug(cmd, &plain), "{cmd}");
            assert!(!engine_debug(cmd, &v), "{cmd} -v is progress only");
            assert!(engine_debug(cmd, &dbg), "{cmd} --debug");
        }
        for cmd in ["generate", "run-gguf", "info", "eval"] {
            assert!(!engine_debug(cmd, &plain), "{cmd}");
            assert!(engine_debug(cmd, &v), "{cmd} -v keeps the old meaning");
        }
    }

    #[test]
    fn tokenizer_reuse_is_flagged_only_with_val_and_an_existing_file() {
        let spec = spec_for("train").unwrap();
        let dir = std::env::temp_dir().join(format!("ferrum_tokreuse_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tok = dir.join("t.gguf");
        let tok_s = tok.to_str().unwrap().to_string();
        let missing = dir.join("nope.gguf").to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&tok);
        let parse = |a: &[&str]| Args::parse(&s(a), spec).unwrap();
        let no_val = parse(&["--tokenizer", &tok_s]);
        let no_tok = parse(&["--val", "0.1"]);
        let absent = parse(&["--val", "0.1", "--tokenizer", &missing]);
        std::fs::write(&tok, b"x").unwrap();
        let both = parse(&["--val", "0.1", "--tokenizer", &tok_s]);
        assert_eq!(reused_tokenizer_with_val(&no_val), None);
        assert_eq!(reused_tokenizer_with_val(&no_tok), None);
        assert_eq!(reused_tokenizer_with_val(&absent), None);
        assert_eq!(reused_tokenizer_with_val(&both), Some(tok_s.as_str()));
    }
}
