use super::*;
use crate::ipc_tests::{app, call, collect, msg, tmp, wait_for};
use ferrum_core::{GenerativeSLM, Rng, TransformerConfig};
use serde_json::{json, Value};
use std::io::{Read, Write};
use tauri::Manager;

fn form(model: &str) -> ServeParams {
    serde_json::from_value(json!({
        "modelPath": model, "host": "127.0.0.1", "port": 0, "maxBatch": 64, "maxChars": 65536,
        "maxBody": 8388608, "maxTokens": 4096
    }))
    .unwrap()
}

#[test]
fn the_form_is_checked_like_the_cli_and_becomes_the_servers_config() {
    let dir = tmp("serve_validate");
    let model = dir.join("m.gguf");
    std::fs::write(&model, b"x").unwrap();
    let m = model.to_string_lossy().into_owned();

    let c = validate(form(&m)).unwrap();
    assert_eq!(
        (c.host.as_str(), c.port, c.token.clone()),
        ("127.0.0.1", 0, None)
    );
    assert_eq!(c.limits, Limits::default());
    assert!(!c.allow_unauthenticated);

    // A blank token is no token; a token is kept as typed.
    let mut p = form(&m);
    p.token = Some(String::new());
    assert_eq!(validate(p).unwrap().token, None);
    let mut p = form(&m);
    p.token = Some("abc123".into());
    assert_eq!(validate(p).unwrap().token.as_deref(), Some("abc123"));

    // Zero limits, as `--max-batch 0` etc.
    for field in ["max batch", "max chars", "max body", "max tokens"] {
        let mut p = form(&m);
        match field {
            "max batch" => p.max_batch = 0,
            "max chars" => p.max_chars = 0,
            "max body" => p.max_body = 0,
            _ => p.max_tokens = 0,
        }
        assert_eq!(
            validate(p).unwrap_err(),
            format!("{field} must be at least 1 (got 0)")
        );
    }
    let mut p = form(&m);
    p.host = "  ".into();
    assert!(validate(p).unwrap_err().contains("host is required"));

    // Off this machine only with a token.
    for host in ["0.0.0.0", "::", "192.168.1.5", "example.com"] {
        let mut p = form(&m);
        p.host = host.into();
        let e = validate(p).unwrap_err();
        assert!(e.contains("bearer token is required"), "{host}: {e}");
        let mut p = form(&m);
        p.host = host.into();
        p.token = Some("t".into());
        assert!(validate(p).is_ok(), "{host}");
    }
    for host in ["127.0.0.1", "::1", "localhost", " LocalHost "] {
        let mut p = form(&m);
        p.host = host.into();
        assert!(validate(p).is_ok(), "{host}");
    }
    // A token a client could not send.
    for t in ["a b", "a\nb", "tab\t"] {
        let mut p = form(&m);
        p.token = Some(t.into());
        assert!(validate(p).unwrap_err().contains("token must be"), "{t:?}");
    }
    // The model path goes through the path checks.
    let mut p = form("/nope/m.gguf");
    p.host = "127.0.0.1".into();
    assert!(validate(p).unwrap_err().contains("does not exist"));
    assert!(validate(form("/dev/null"))
        .unwrap_err()
        .contains("not allowed"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn every_validation_message_has_a_useful_error_code() {
    use crate::infra::error::classify;
    for (msg, code) in [
        ("max batch must be at least 1 (got 0)", "invalid"),
        ("host is required (127.0.0.1 keeps the server on this machine)", "invalid"),
        ("host 0.0.0.0 is not this machine, so a bearer token is required: type one or press Generate", "invalid"),
        ("token must be one word without spaces or control characters", "invalid"),
        ("a model server is already running", "busy"),
    ] {
        assert_eq!(classify(msg), code, "{msg}");
    }
}

#[test]
fn a_wildcard_host_is_shown_as_loopback_and_ipv6_is_bracketed() {
    let url = |s: &str| display_url(s.parse().unwrap());
    assert_eq!(url("127.0.0.1:8080"), "http://127.0.0.1:8080");
    assert_eq!(url("0.0.0.0:9"), "http://127.0.0.1:9");
    assert_eq!(url("[::]:9"), "http://[::1]:9");
    assert_eq!(url("[::1]:9"), "http://[::1]:9");
    assert_eq!(url("192.168.1.5:80"), "http://192.168.1.5:80");
}

fn tiny_model_file(dir: &std::path::Path) -> String {
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
    let p = dir.join("tiny.gguf");
    std::fs::write(&p, m.to_bytes_quantized().unwrap()).unwrap();
    p.to_string_lossy().into_owned()
}

/// One HTTP/1.1 request over a real socket: (status, body).
fn http(
    url: &str,
    method: &str,
    path: &str,
    auth: Option<&str>,
    body: Option<&str>,
) -> (u16, String) {
    let addr = url.strip_prefix("http://").unwrap();
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    let auth = auth.map_or(String::new(), |a| format!("Authorization: {a}\r\n"));
    let body = body.unwrap_or("");
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    (
        head.split(' ').nth(1).unwrap().parse().unwrap(),
        body.to_string(),
    )
}

fn start_body(model: &str, token: Option<&str>) -> Value {
    json!({"params": {
        "modelPath": model, "host": "127.0.0.1", "port": 0, "maxBatch": 64, "maxChars": 65536,
        "maxBody": 8388608, "maxTokens": 4096, "token": token
    }})
}

#[test]
fn serve_start_status_requests_logs_and_stop_through_ipc_over_a_real_socket() {
    let (app, w) = app();
    let dir = tmp("serve_e2e");
    let model = tiny_model_file(&dir);
    let logs = collect(&app, "serve-log");

    // Nothing is running yet; stopping nothing is not an error.
    assert_eq!(
        call(&w, "serve_status", json!({})).unwrap()["running"],
        false
    );
    assert_eq!(call(&w, "serve_stop", json!({})).unwrap()["running"], false);

    let st = call(&w, "serve_start", start_body(&model, None)).unwrap();
    assert_eq!(st["running"], true);
    let url = st["url"].as_str().unwrap().to_string();
    assert!(
        url.starts_with("http://127.0.0.1:") && !url.ends_with(":0"),
        "{url}"
    );

    // A second start is refused while this one runs, and so is a stop-less restart.
    let e = call(&w, "serve_start", start_body(&model, None)).unwrap_err();
    assert_eq!(e["code"], "busy", "{e}");
    assert_eq!(msg(&e), "a model server is already running");

    // The real routes over a real socket.
    assert_eq!(
        http(&url, "GET", "/health", None, None),
        (200, "\"ok\"".into())
    );
    let (code, body) = http(
        &url,
        "POST",
        "/v1/generate",
        None,
        Some(r#"{"requests":[{"prompt":"ab","max_tokens":4}]}"#),
    );
    assert_eq!(code, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert!(v["results"][0]["text"].is_string(), "{v}");
    let (code, body) = http(&url, "GET", "/v1/info", None, None);
    assert_eq!(code, 200);
    assert!(body.contains("model_sha256"), "{body}");
    assert_eq!(http(&url, "GET", "/nope", None, None).0, 404);

    // The counter counts all four, and the log holds the banner and one line per request.
    let status = call(&w, "serve_status", json!({})).unwrap();
    assert_eq!(
        (status["running"].clone(), status["requests"].clone()),
        (json!(true), json!(4))
    );
    assert_eq!(status["url"], url.as_str());
    wait_for(&logs, 6); // two banner lines, four requests
    let lines: Vec<String> = logs
        .lock()
        .unwrap()
        .iter()
        .map(|l| serde_json::from_str::<String>(l).unwrap())
        .collect();
    assert!(
        lines[0].starts_with("ferrum-serve: ") && lines[0].contains(&url[7..]),
        "{lines:?}"
    );
    assert!(lines[1].contains("auth OFF (loopback only)"), "{lines:?}");
    assert_eq!(
        &lines[2..],
        [
            "GET /health -> 200",
            "POST /v1/generate -> 200",
            "GET /v1/info -> 200",
            "GET /nope -> 404"
        ]
    );

    // Stop is graceful and complete: the port is closed and the slot is free again.
    let stopped = call(&w, "serve_stop", json!({})).unwrap();
    assert_eq!(
        (stopped["running"].clone(), stopped["requests"].clone()),
        (json!(false), json!(4))
    );
    assert!(std::net::TcpStream::connect(url.strip_prefix("http://").unwrap()).is_err());
    assert_eq!(
        call(&w, "serve_status", json!({})).unwrap()["running"],
        false
    );
    wait_for(&logs, 7);
    assert_eq!(
        logs.lock().unwrap().last().unwrap(),
        "\"ferrum-serve: stopped\""
    );

    // And it starts again (a token this time: 401 without it, 200 with it, /health stays open).
    let st = call(&w, "serve_start", start_body(&model, Some("s3cret-token"))).unwrap();
    let url = st["url"].as_str().unwrap().to_string();
    assert_eq!(http(&url, "GET", "/health", None, None).0, 200);
    assert_eq!(http(&url, "GET", "/v1/info", None, None).0, 401);
    assert_eq!(
        http(&url, "GET", "/v1/info", Some("Bearer wrong"), None).0,
        401
    );
    assert_eq!(
        http(&url, "GET", "/v1/info", Some("Bearer s3cret-token"), None).0,
        200
    );
    // The token is nowhere in what the page can see.
    wait_for(&logs, 7 + 2 + 4);
    let all = logs.lock().unwrap().join("\n");
    assert!(
        all.contains("auth ON") && !all.contains("s3cret-token"),
        "{all}"
    );
    assert!(!call(&w, "serve_status", json!({}))
        .unwrap()
        .to_string()
        .contains("s3cret"));
    call(&w, "serve_stop", json!({})).unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn start_errors_reach_the_ui_as_structured_errors_and_leave_the_slot_free() {
    let (_app, w) = app();
    let dir = tmp("serve_errors");
    let model = tiny_model_file(&dir);

    let mut body = start_body(&model, None);
    body["params"]["host"] = json!("0.0.0.0");
    let e = call(&w, "serve_start", body).unwrap_err();
    assert_eq!(e["code"], "invalid", "{e}");
    assert!(msg(&e).contains("bearer token is required"), "{e}");

    let mut body = start_body(&model, None);
    body["params"]["maxBatch"] = json!(0);
    assert!(
        msg(&call(&w, "serve_start", body).unwrap_err()).contains("max batch must be at least 1")
    );

    let e = call(&w, "serve_start", start_body("/nope/m.gguf", None)).unwrap_err();
    assert_eq!(e["code"], "not_found", "{e}");

    // A file that is not a model fails while loading, and the slot is free afterwards.
    let junk = dir.join("junk.gguf");
    std::fs::write(&junk, b"not a model").unwrap();
    let e = call(&w, "serve_start", start_body(&junk.to_string_lossy(), None)).unwrap_err();
    assert!(!msg(&e).is_empty(), "{e}");
    // A port that cannot be bound (taken by this very listener) is an error naming it.
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let mut body = start_body(&model, None);
    body["params"]["port"] = json!(taken.local_addr().unwrap().port());
    let e = call(&w, "serve_start", body).unwrap_err();
    assert!(msg(&e).contains("cannot bind"), "{e}");
    // Nothing is left running, so a good start works.
    assert_eq!(
        call(&w, "serve_status", json!({})).unwrap()["running"],
        false
    );
    let st = call(&w, "serve_start", start_body(&model, None)).unwrap();
    assert_eq!(st["running"], true);
    call(&w, "serve_stop", json!({})).unwrap();
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_last_log_line_says_how_the_server_ended() {
    assert_eq!(stopped_line(Ok::<(), String>(())), "ferrum-serve: stopped");
    assert_eq!(
        stopped_line(Err("address in use")),
        "ferrum-serve: stopped with an error: address in use"
    );
}

#[test]
fn a_server_kept_busy_by_a_request_is_cut_off_after_the_wait() {
    // A task that never ends stands in for a request that does not finish.
    let stuck = tauri::async_runtime::spawn(std::future::pending::<()>());
    let e = wait_for_end(&stuck, Duration::from_secs(1)).unwrap_err();
    assert_eq!(
        e,
        "the server did not stop within 1 s (a request was still running); it was cut off"
    );
    let done = tauri::async_runtime::spawn(async {});
    assert!(wait_for_end(&done, Duration::from_secs(5)).is_ok());
}

#[test]
fn a_server_that_ended_by_itself_is_forgotten_by_the_next_status() {
    let (app, w) = app();
    let state = app.state::<crate::AppState>();
    let (stop, _stopped) = tokio::sync::oneshot::channel::<()>();
    let task = tauri::async_runtime::spawn(async {});
    while !task.inner().is_finished() {
        std::thread::sleep(Duration::from_millis(5));
    }
    *state.serve.slot() = Some(Running {
        url: "http://127.0.0.1:1".into(),
        auth: false,
        requests: Arc::new(AtomicU64::new(3)),
        stop,
        task,
    });
    let s = call(&w, "serve_status", json!({})).unwrap();
    assert_eq!(
        s,
        json!({"running": false, "url": null, "requests": 0, "auth": false})
    );
    assert!(state.serve.slot().is_none());

    // And a server task that refuses to finish when `serve_stop` is called returns an IPC error.
    let (stop2, _stopped2) = tokio::sync::oneshot::channel::<()>();
    let stuck = tauri::async_runtime::spawn(std::future::pending::<()>());
    *state.serve.slot() = Some(Running {
        url: "http://127.0.0.1:1".into(),
        auth: false,
        requests: Arc::new(AtomicU64::new(0)),
        stop: stop2,
        task: stuck,
    });
    let err = call(&w, "serve_stop", json!({})).unwrap_err();
    assert!(msg(&err).contains("was cut off"), "{err}");
}
