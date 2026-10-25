//! The real `ferrum-serve` binary: usage and exit codes, serving over TCP, and
//! graceful shutdown on SIGINT (what `main.rs` wires up).

use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_ferrum-serve");

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

// The caller owns (and waits on) the child; clippy cannot see that through the returned tuple.
#[allow(clippy::zombie_processes)]
/// Start the server on an ephemeral port; returns the child, its stderr reader and its address.
fn spawn(model: &Path) -> (Child, BufReader<std::process::ChildStderr>, String) {
    let mut child = Command::new(BIN)
        .args(["--model", model.to_str().unwrap(), "--port", "0"])
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    loop {
        line.clear();
        if err.read_line(&mut line).unwrap() == 0 {
            child.wait().ok();
            panic!("server exited before listening");
        }
        if let Some(i) = line.find(" on http://") {
            let addr = line[i + " on http://".len()..].trim().to_string();
            return (child, err, addr);
        }
    }
}

/// Minimal HTTP/1.1 client: (status, body).
fn http(addr: &str, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).unwrap();
    let (head, body) = raw.split_once("\r\n\r\n").unwrap();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, body.to_string())
}

#[test]
fn help_prints_usage_and_exits_zero() {
    let o = run(&["--help"]);
    assert!(o.status.success());
    assert!(String::from_utf8_lossy(&o.stdout).contains("usage: ferrum-serve --model"));
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
    let (mut child, _err, addr) = spawn(&model);
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

    let kill = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(kill.success());
    let deadline = Instant::now() + Duration::from_secs(20);
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        if Instant::now() > deadline {
            child.kill().ok();
            panic!("server did not exit after SIGINT");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    std::fs::remove_file(model).ok();
    assert!(status.success(), "{status:?}");
}
