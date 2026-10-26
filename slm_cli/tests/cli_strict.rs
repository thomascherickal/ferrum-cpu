//! The CLI must reject what it does not understand instead of silently
//! falling back to defaults (review defects #3 and #4).

use std::process::Command;

fn run(args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(args)
        .output()
        .expect("spawn train_transformer");
    let text =
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout);
    (out.status.code().unwrap_or(-1), text)
}

/// A corpus file private to the calling test (a shared path races: `fs::write` truncates it
/// while a parallel test is reading it, which showed up as "corpus is empty").
fn corpus() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "ferrum_cli_strict_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("c.txt");
    std::fs::write(&p, "abc abc abc\n".repeat(50)).unwrap();
    p
}

#[test]
fn unknown_flag_is_an_error() {
    let c = corpus();
    let (code, text) = run(&[
        "train",
        c.to_str().unwrap(),
        "/nonexistent/m.gguf",
        "--contxt",
        "8",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("unknown flag --contxt"), "{text}");
}

#[test]
fn unparsable_value_is_an_error() {
    let c = corpus();
    let (code, text) = run(&[
        "train",
        c.to_str().unwrap(),
        "/nonexistent/m.gguf",
        "--context",
        "12x",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("--context") && text.contains("12x"), "{text}");
}

#[test]
fn missing_value_is_an_error() {
    let (code, text) = run(&["generate", "m.gguf", "hi", "--temp"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("--temp needs a value"), "{text}");
}

#[test]
fn missing_resume_checkpoint_is_an_error() {
    let c = corpus();
    let (code, text) = run(&[
        "train",
        c.to_str().unwrap(),
        "/nonexistent/m.gguf",
        "--resume",
        "/nonexistent/nope.fckp",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(
        text.contains("checkpoint /nonexistent/nope.fckp does not exist"),
        "{text}"
    );
}

/// Regression (deferred minor): when the model file already exists, `train`/`run` skipped
/// training and never looked at `--resume`, so a typo'd checkpoint went unnoticed.
#[test]
fn missing_resume_checkpoint_is_an_error_even_when_the_model_exists() {
    let c = corpus();
    let existing = c.to_str().unwrap(); // any existing file: the check runs before it is loaded
    for cmd in ["train", "run"] {
        let mut args = vec![cmd, existing, existing];
        if cmd == "run" {
            args.push("seed");
        }
        args.extend(["--resume", "/nonexistent/nope.flck"]);
        let (code, text) = run(&args);
        assert_eq!(code, 1, "{cmd}: {text}");
        assert!(
            text.contains("checkpoint /nonexistent/nope.flck does not exist"),
            "{cmd}: {text}"
        );
    }
}

#[test]
fn flag_valid_elsewhere_is_rejected_for_this_command() {
    let (code, text) = run(&["info", "m.gguf", "--epochs", "3"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("unknown flag --epochs for 'info'"), "{text}");
}

#[test]
fn unknown_command_is_an_error_with_usage() {
    let (code, text) = run(&["frobnicate"]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("unknown command: frobnicate"), "{text}");
    assert!(
        text.contains("--help"),
        "names the way to the usage: {text}"
    );
    assert!(!text.contains("USAGE"), "no usage dump: {text}");
    assert_eq!(text.trim().lines().count(), 1, "{text}");
}

#[test]
fn extra_positional_arguments_are_errors_naming_the_token() {
    for args in [
        &["info", "m.gguf", "extra"][..],
        &["eval", "m.gguf", "t.txt", "extra"][..],
        &["prepare", "--input", "x", "--out", "y", "extra"][..],
        &["prepare", "extra"][..],
        &["train", "c.txt", "m.gguf", "extra"][..],
        &["finetune-gguf", "m.gguf", "c.txt", "o.flck", "extra"][..],
        &["export-gguf", "a.gguf", "b.gguf", "extra"][..],
        &["export-ollama", "m.gguf", "name", "extra"][..],
        &["export-lmstudio", "m.gguf", "extra"][..],
    ] {
        let (code, text) = run(args);
        assert_eq!(code, 1, "{args:?}: {text}");
        assert!(
            text.contains("unexpected argument 'extra'"),
            "{args:?}: {text}"
        );
    }
}

#[test]
fn prompt_commands_still_join_their_extra_words() {
    // generate/run/run-gguf take free text: this must get past the argument check.
    let (_, text) = run(&["generate", "/nonexistent/m.gguf", "two", "words"]);
    assert!(!text.contains("unexpected argument"), "{text}");
}

#[test]
fn version_flag_prints_name_and_version() {
    for f in ["--version", "-V"] {
        let o = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
            .arg(f)
            .output()
            .unwrap();
        assert_eq!(o.status.code(), Some(0), "{f}");
        assert_eq!(
            String::from_utf8_lossy(&o.stdout).trim(),
            format!("train_transformer {}", env!("CARGO_PKG_VERSION"))
        );
    }
}

#[test]
fn command_help_prints_that_commands_usage_on_stdout() {
    for cmd in [
        "train",
        "run",
        "generate",
        "run-gguf",
        "finetune-gguf",
        "export-gguf",
        "export-ollama",
        "export-lmstudio",
        "eval",
        "prepare",
        "info",
    ] {
        for h in ["--help", "-h"] {
            let o = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
                .args([cmd, h])
                .output()
                .unwrap();
            let out = String::from_utf8_lossy(&o.stdout);
            assert_eq!(o.status.code(), Some(0), "{cmd} {h}");
            assert!(
                out.contains(&format!("usage: train_transformer {cmd}")),
                "{cmd}: {out}"
            );
        }
    }
    // the options come from the same flag table the parser uses
    let o = Command::new(env!("CARGO_BIN_EXE_train_transformer"))
        .args(["train", "--help"])
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(
        out.contains("--context") && out.contains("--no-qat"),
        "{out}"
    );
    assert!(!out.contains("--stop"), "train has no --stop: {out}");
}

#[test]
fn no_arguments_prints_usage_and_fails() {
    let (code, text) = run(&[]);
    assert_eq!(code, 1, "{text}");
    assert!(text.contains("USAGE"), "{text}");
}

#[test]
fn help_variants_print_usage_and_succeed() {
    for h in ["help", "-h", "--help"] {
        let (code, text) = run(&[h]);
        assert_eq!(code, 0, "{h}: {text}");
        assert!(text.contains("USAGE"), "{h}: {text}");
    }
}

#[test]
fn valid_flags_pass_the_parser_and_reach_the_command() {
    // Parses fine, then fails later on the missing model file (not the parser).
    let (code, text) = run(&[
        "generate",
        "/nonexistent/m.gguf",
        "hi",
        "--temp",
        "-1",
        "-v",
    ]);
    assert_eq!(code, 1, "{text}");
    assert!(!text.contains("unknown flag"), "{text}");
    assert!(!text.contains("needs a value"), "{text}");
}

#[test]
fn aliases_use_their_commands_flag_table() {
    // `gguf` accepts --rep; `gen` rejects it (--top-k is shared with generate now).
    let (_, text) = run(&["gguf", "/nonexistent.gguf", "--rep", "3"]);
    assert!(!text.contains("unknown flag"), "{text}");
    let (code, text) = run(&["gen", "m.gguf", "hi", "--rep", "3"]);
    assert_eq!(code, 1);
    assert!(text.contains("unknown flag --rep for 'gen'"), "{text}");
}

#[test]
fn eval_and_info_dispatch_past_the_parser() {
    for args in [
        ["eval", "/nonexistent/m.gguf", "/nonexistent/t.txt"],
        ["info", "/nonexistent/m.gguf", "-v"],
    ] {
        let (code, text) = run(&args);
        assert_eq!(code, 1, "{text}");
        assert!(!text.contains("unknown"), "{text}");
    }
}

#[test]
fn info_reports_gguf_format_and_rejects_non_gguf() {
    let dir = std::env::temp_dir().join(format!("ferrum_cli_info_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let corpus_path = dir.join("c.txt");
    std::fs::write(&corpus_path, "abc abc abc\n".repeat(50)).unwrap();
    let model = dir.join("m.gguf");
    let (code, text) = run(&[
        "train",
        corpus_path.to_str().unwrap(),
        model.to_str().unwrap(),
        "--epochs",
        "1",
        "--context",
        "8",
        "--embed",
        "16",
        "--heads",
        "2",
        "--blocks",
        "1",
        "--hidden",
        "32",
    ]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("(int8 GGUF)"), "{text}");
    let (code, text) = run(&["info", model.to_str().unwrap()]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("Format    : GGUF v3 (gpt2, int8)"), "{text}");

    let junk = dir.join("junk.gguf");
    std::fs::write(&junk, b"FINF\x05\x00\x00\x00").unwrap();
    let (code, text) = run(&["info", junk.to_str().unwrap()]);
    assert_ne!(code, 0);
    assert!(text.contains("old Ferrum .bin (FINF) model"), "{text}");
    let (code, text) = run(&["info"]);
    assert_ne!(code, 0);
    assert!(
        text.contains("usage: train_transformer info <model.gguf>"),
        "{text}"
    );
}

#[test]
fn gguf_commands_name_a_missing_resume_checkpoint_before_loading_anything() {
    for args in [
        vec![
            "run-gguf",
            "/nonexistent/m.gguf",
            "hi",
            "--resume",
            "/nonexistent/r.flck",
        ],
        vec![
            "export-gguf",
            "/nonexistent/m.gguf",
            "/tmp/o.gguf",
            "--resume",
            "/nonexistent/r.flck",
        ],
        vec![
            "finetune-gguf",
            "/nonexistent/m.gguf",
            "c.txt",
            "o.flck",
            "--resume",
            "/nonexistent/r.flck",
        ],
    ] {
        let (code, text) = run(&args);
        assert_eq!(code, 1, "{text}");
        assert!(
            text.contains("checkpoint /nonexistent/r.flck does not exist"),
            "{args:?}: {text}"
        );
    }
}

#[test]
fn bad_sampling_values_are_rejected_before_any_model_is_loaded() {
    for (args, want) in [
        (
            vec!["generate", "/nope.gguf", "hi", "--temp", "-1"],
            "--temp must be",
        ),
        (
            vec!["generate", "/nope.gguf", "hi", "--temp", "nan"],
            "--temp must be",
        ),
        (
            vec!["generate", "/nope.gguf", "hi", "--top-p", "2"],
            "--top-p must be",
        ),
        (
            vec!["generate", "/nope.gguf", "hi", "--repetition-penalty", "0"],
            "--repetition-penalty must be",
        ),
        (
            vec!["run-gguf", "/nope.gguf", "hi", "--rep", "-3"],
            "--rep must be",
        ),
        (
            vec!["run", "/nope.txt", "/nope.gguf", "hi", "--top-p", "9"],
            "--top-p must be",
        ),
        (
            vec!["run-gguf", "/nope.gguf", "hi", "--temp", "inf"],
            "--temp must be",
        ),
        (
            vec![
                "finetune-gguf",
                "/nope.gguf",
                "c.txt",
                "o.flck",
                "--sample",
                "--temp",
                "-2",
            ],
            "--temp must be",
        ),
    ] {
        let (code, text) = run(&args);
        assert_eq!(code, 1, "{args:?}: {text}");
        assert!(text.contains(want), "{args:?}: {text}");
    }
}

#[test]
fn nonsense_training_hyperparameters_fail_before_training_starts() {
    let c = corpus();
    let m = std::env::temp_dir().join(format!("ferrum_strict_hp_{}.gguf", std::process::id()));
    let _ = std::fs::remove_file(&m);
    for (extra, want) in [
        (vec!["--dropout", "5"], "--dropout must be"),
        (vec!["--dropout", "nan"], "--dropout must be"),
        (vec!["--weight_decay", "-1"], "--weight_decay must be"),
        (vec!["--clip", "nan"], "--clip must be"),
        (vec!["--tokens", "inf"], "--tokens must be"),
        (vec!["--tokens", "-5"], "--tokens must be"),
        (vec!["--embed", "0", "--heads", "1"], "must be > 0"),
        (vec!["--hidden", "0"], "must be > 0"),
        (
            vec!["--embed", "100000000", "--heads", "1", "--hidden", "1"],
            "model too large",
        ),
    ] {
        let mut args = vec!["train", c.to_str().unwrap(), m.to_str().unwrap(), "--force"];
        args.extend(["--vocab", "300"]);
        args.extend(&extra);
        let (code, text) = run(&args);
        assert_eq!(code, 1, "{extra:?}: {text}");
        assert!(text.contains(want), "{extra:?}: {text}");
        assert!(!m.exists(), "{extra:?} must not save a model");
    }
    // finetune validates its hyperparameters before opening the (missing) model.
    for (extra, want) in [
        (vec!["--dropout", "1"], "--dropout must be"),
        (vec!["--clip", "-1"], "--clip must be"),
        (vec!["--lr", "0"], "--lr must be"),
    ] {
        let mut args = vec!["finetune-gguf", "/nope.gguf", "c.txt", "o.flck"];
        args.extend(&extra);
        let (code, text) = run(&args);
        assert_eq!(code, 1, "{extra:?}: {text}");
        assert!(text.contains(want), "{extra:?}: {text}");
    }
}
