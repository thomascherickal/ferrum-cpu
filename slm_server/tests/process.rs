//! The real `ferrum-serve` binary: usage and exit codes, serving over TCP, and
//! graceful shutdown on SIGINT (what `main.rs` wires up).

mod common;

use common::{assert_graceful, http, http_with, sigint_and_wait, spawn, spawn_env, BIN};
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
    Command::new(BIN)
        .args(args)
        .env_remove("FERRUM_TOKEN")
        .output()
        .unwrap()
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
    assert_graceful(status);
}

/// Next stderr line (the auth line follows the listening line).
fn next_line(err: &mut impl std::io::BufRead) -> String {
    let mut l = String::new();
    err.read_line(&mut l).unwrap();
    l
}

const LL: &str = r#"{"requests":[{"context":"abc","continuation":" abc"}]}"#;

fn check_token_gate(addr: &str, token: &str) {
    let bearer = format!("Bearer {token}");
    // /health stays open; everything else, including unknown routes, needs the token.
    assert_eq!(http(addr, "GET", "/health", "").0, 200);
    for (m, path) in [
        ("GET", "/v1/info"),
        ("POST", "/v1/loglikelihood"),
        ("GET", "/nope"),
    ] {
        let (s, head, body) = http_with(addr, m, path, LL, &[]);
        assert_eq!(s, 401, "{m} {path}");
        assert_eq!(body, r#"{"error":"missing or invalid bearer token"}"#);
        assert!(
            head.to_ascii_lowercase()
                .contains("www-authenticate: bearer"),
            "{head}"
        );
    }
    let wrong = http_with(
        addr,
        "POST",
        "/v1/loglikelihood",
        LL,
        &[("authorization", "Bearer nope")],
    );
    assert_eq!(wrong.0, 401);
    let (s, _, body) = http_with(
        addr,
        "POST",
        "/v1/loglikelihood",
        LL,
        &[("authorization", &bearer)],
    );
    assert_eq!(s, 200, "{body}");
    assert!(body.contains("logprob"), "{body}");
    assert_eq!(
        http_with(addr, "GET", "/v1/info", "", &[("authorization", &bearer)]).0,
        200
    );
}

#[test]
fn bearer_token_from_the_environment_gates_every_route_but_health() {
    let model = model_file("tok-env");
    let (mut child, mut err, addr) = spawn_env(&model, &[], &[("FERRUM_TOKEN", "s3cret-env")]);
    assert!(next_line(&mut err).contains("auth ON"), "banner");
    check_token_gate(&addr, "s3cret-env");
    let status = sigint_and_wait(&mut child);
    std::fs::remove_file(model).ok();
    assert_graceful(status);
}

#[test]
fn token_flag_wins_over_the_environment() {
    let model = model_file("tok-flag");
    let (mut child, _err, addr) = spawn_env(
        &model,
        &["--token", "from-flag"],
        &[("FERRUM_TOKEN", "from-env")],
    );
    check_token_gate(&addr, "from-flag");
    assert_eq!(
        http_with(
            &addr,
            "GET",
            "/v1/info",
            "",
            &[("authorization", "Bearer from-env")]
        )
        .0,
        401
    );
    child.kill().ok();
    child.wait().ok();
    std::fs::remove_file(model).ok();
}

#[test]
fn loopback_without_a_token_is_open_and_says_so() {
    let model = model_file("tok-off");
    let (mut child, mut err, addr) = spawn(&model, &[]);
    assert!(next_line(&mut err).contains("auth OFF (loopback only)"));
    assert_eq!(http(&addr, "GET", "/v1/info", "").0, 200);
    child.kill().ok();
    child.wait().ok();
    std::fs::remove_file(model).ok();
}

#[test]
fn non_loopback_host_without_a_token_is_refused_at_startup() {
    let model = model_file("tok-refuse");
    let m = model.to_str().unwrap();
    for host in ["0.0.0.0", "::", "192.0.2.1"] {
        let o = run(&["--model", m, "--host", host, "--port", "0"]);
        assert_eq!(o.status.code(), Some(1), "{host}");
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(
            err.starts_with("error: refusing to listen on non-loopback host")
                && err.contains("FERRUM_TOKEN")
                && err.contains("--allow-unauthenticated"),
            "{host}: {err}"
        );
    }
    // A token (from the environment) or the explicit opt-out lets it start.
    let (mut child, mut err, _addr) =
        spawn_env(&model, &["--host", "0.0.0.0"], &[("FERRUM_TOKEN", "t")]);
    assert!(next_line(&mut err).contains("auth ON"));
    child.kill().ok();
    child.wait().ok();
    let (mut child, mut err, addr) =
        spawn(&model, &["--host", "0.0.0.0", "--allow-unauthenticated"]);
    assert!(next_line(&mut err).contains("auth OFF (--allow-unauthenticated"));
    let port = addr.rsplit(':').next().unwrap();
    assert_eq!(
        http(&format!("127.0.0.1:{port}"), "GET", "/v1/info", "").0,
        200
    );
    child.kill().ok();
    child.wait().ok();
    std::fs::remove_file(model).ok();
}

#[test]
fn empty_token_flag_is_a_usage_error() {
    let o = run(&["--model", "m", "--token", ""]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("--token must not be empty"));
}
