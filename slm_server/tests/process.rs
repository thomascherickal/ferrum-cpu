//! The real `ferrum-serve` binary: usage and exit codes, serving over TCP, and
//! graceful shutdown on SIGINT (what `main.rs` wires up).

mod common;

use common::{http, sigint_and_wait, spawn, BIN};
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};
use std::path::PathBuf;
use std::process::{Command, Output};

fn model_file(name: &str) -> PathBuf {
    let cfg = TransformerConfig {
        context_len: 8,
        embed_dim: 8,
        num_heads: 2,
        num_blocks: 1,
        hidden_dim: 16,
        epochs: 1,
        vocab_size: 0,
        ..TransformerConfig::default()
    };
    let m = GenerativeSLM::train_transformer_config(
        &"abc ".repeat(40),
        &cfg,
        &mut Rng::new(1),
        |_, _| {},
    )
    .unwrap();
    let p = std::env::temp_dir().join(format!("ferrum-serve-proc-{}-{name}", std::process::id()));
    std::fs::write(&p, m.to_bytes_quantized().unwrap()).unwrap();
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().unwrap()
}

#[test]
fn help_prints_usage_and_exits_zero() {
    let o = run(&["--help"]);
    assert!(o.status.success());
    assert!(String::from_utf8_lossy(&o.stdout).contains("usage: ferrum-serve --model"));
}

#[test]
fn version_prints_name_and_version_and_exits_zero() {
    for f in ["--version", "-V"] {
        let o = run(&[f]);
        assert!(o.status.success(), "{f}");
        let out = String::from_utf8_lossy(&o.stdout);
        assert_eq!(
            out.trim(),
            format!("ferrum-serve {}", env!("CARGO_PKG_VERSION"))
        );
    }
}

#[test]
fn bad_arguments_exit_one_with_error_on_stderr() {
    for args in [
        &["--bogus", "1"][..],
        &[][..],
        &["--model", "/nonexistent/m.gguf"][..],
    ] {
        let o = run(args);
        assert_eq!(o.status.code(), Some(1), "{args:?}");
        assert!(
            String::from_utf8_lossy(&o.stderr).starts_with("error: "),
            "{args:?}"
        );
    }
}

/// Regression: `--model -h` was accepted as a model path named "-h".
#[test]
fn model_dash_h_is_a_usage_error() {
    let o = run(&["--model", "-h"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("--model needs a value"));
}

#[test]
fn serves_over_tcp_and_stops_gracefully_on_sigint() {
    let model = model_file("sigint");
    let (mut child, _err, addr) = spawn(&model, &[]);
    assert_eq!(http(&addr, "GET", "/health", "").0, 200);
    let (status, body) = http(
        &addr,
        "POST",
        "/v1/loglikelihood",
        r#"{"requests":[{"context":"abc","continuation":" abc"}]}"#,
    );
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("logprob"), "{body}");
    assert_eq!(http(&addr, "GET", "/nope", "").0, 404);

    let status = sigint_and_wait(&mut child);
    std::fs::remove_file(model).ok();
    assert!(status.success(), "{status:?}");
}
