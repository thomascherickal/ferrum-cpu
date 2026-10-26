//! Helpers for tests that run the real `ferrum-serve` binary.
#![allow(dead_code)] // each test crate uses a different subset

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

pub const BIN: &str = env!("CARGO_BIN_EXE_ferrum-serve");

/// Start the server on an ephemeral port with `extra` flags; returns the child, its stderr
/// reader (keep it alive so the server never gets SIGPIPE) and its address.
// The caller owns (and waits on) the child; clippy cannot see that through the returned tuple.
#[allow(clippy::zombie_processes)]
pub fn spawn(
    model: &Path,
    extra: &[&str],
) -> (Child, BufReader<std::process::ChildStderr>, String) {
    let mut child = Command::new(BIN)
        .args(["--model", model.to_str().unwrap(), "--port", "0"])
        .args(extra)
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

/// Minimal HTTP/1.1 client: (status, body). A server that answers before it has read the
/// whole body (413) may reset the connection on our remaining writes; the response is still read.
pub fn http(addr: &str, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    let _ = write!(
        s,
        "{method} {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut raw = String::new();
    let _ = s.read_to_string(&mut raw);
    let (head, body) = raw.split_once("\r\n\r\n").expect("no HTTP response");
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, body.to_string())
}

/// SIGINT the server and wait (up to 20 s) for it to exit.
pub fn sigint_and_wait(child: &mut Child) -> ExitStatus {
    let kill = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(kill.success());
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(s) = child.try_wait().unwrap() {
            return s;
        }
        if Instant::now() > deadline {
            child.kill().ok();
            panic!("server did not exit after SIGINT");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
