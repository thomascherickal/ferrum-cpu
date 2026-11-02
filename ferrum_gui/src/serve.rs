//! The Serve tab: run the `ferrum-serve` HTTP server for one model inside the app.
//!
//! The server is the `slm_server` library (the same code the `ferrum-serve` binary runs), spawned
//! on Tauri's async runtime, so no child process has to be found or bundled. One server at a time
//! (the [`Job::Serve`] slot). `serve_start` checks the form like the CLI does (limits of at least
//! 1, no non-loopback host without a token), loads the model, binds, and answers with the real
//! URL (port 0 picks a free one). The startup banner and one line per request reach the webview
//! as `serve-log` events; `serve_status` reports the request count. `serve_stop` shuts the server
//! down gracefully and waits for it, so the model's memory is free when it returns.
//!
//! The bearer token lives only in this process's memory for the life of the server: it is not in
//! any event, log line or status, and the page never stores it.

use crate::error::{blocking, CmdError};
use crate::jobs::Job;
use crate::paths::checked_path;
use crate::AppState;
use serde::{Deserialize, Serialize};
use slm_server::cli::{is_loopback, Config};
use slm_server::{bind, Limits};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::async_runtime::JoinHandle;
use tauri::{AppHandle, Emitter, Runtime, State};

/// How long `serve_stop` waits for in-flight requests before it cuts the server off.
const STOP_WAIT: Duration = Duration::from_secs(10);

/// The form of the Serve tab.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServeParams {
    pub model_path: String,
    pub host: String,
    /// `0` asks the system for a free port.
    pub port: u16,
    pub max_batch: usize,
    pub max_chars: usize,
    pub max_body: usize,
    pub max_tokens: usize,
    /// Bearer token; blank or absent means no authentication (loopback only).
    #[serde(default)]
    pub token: Option<String>,
}

/// What the tab shows about the server.
#[derive(Serialize, Default, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ServeStatus {
    pub running: bool,
    /// Where to reach it from this machine, e.g. `http://127.0.0.1:8080`.
    pub url: Option<String>,
    /// Requests answered so far, rejected ones included.
    pub requests: u64,
    /// Does the server demand a bearer token? (The token itself is never reported.)
    pub auth: bool,
}

struct Running {
    url: String,
    auth: bool,
    requests: Arc<AtomicU64>,
    stop: tokio::sync::oneshot::Sender<()>,
    task: JoinHandle<()>,
}

impl Running {
    fn alive(&self) -> bool {
        !self.task.inner().is_finished()
    }
    fn status(&self) -> ServeStatus {
        ServeStatus {
            running: self.alive(),
            url: Some(self.url.clone()),
            requests: self.requests.load(Ordering::Relaxed),
            auth: self.auth,
        }
    }
}

/// The running server, if any (a finished one is dropped on the next look).
#[derive(Default)]
pub struct Serve(Mutex<Option<Running>>);

impl Serve {
    fn slot(&self) -> std::sync::MutexGuard<'_, Option<Running>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Check the form and turn it into the server's own `Config`.
fn validate(p: ServeParams) -> Result<Config, String> {
    checked_path("model path", &p.model_path, true)?;
    let host = p.host.trim().to_string();
    if host.is_empty() {
        return Err("host is required (127.0.0.1 keeps the server on this machine)".into());
    }
    for (name, v) in [
        ("max batch", p.max_batch),
        ("max chars", p.max_chars),
        ("max body", p.max_body),
        ("max tokens", p.max_tokens),
    ] {
        if v == 0 {
            return Err(format!("{name} must be at least 1 (got 0)"));
        }
    }
    let token = p.token.filter(|t| !t.is_empty());
    if let Some(t) = &token {
        if t.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err("token must be one word without spaces or control characters".into());
        }
    }
    if token.is_none() && !is_loopback(&host) {
        return Err(format!(
            "host {host} is not this machine, so a bearer token is required: type one or press Generate"
        ));
    }
    Ok(Config {
        model: p.model_path,
        host,
        port: p.port,
        limits: Limits {
            max_batch: p.max_batch,
            max_chars: p.max_chars,
            max_body: p.max_body,
            max_tokens: p.max_tokens,
        },
        token,
        allow_unauthenticated: false,
    })
}

/// The address to type on this machine: a wildcard bind is reached through loopback.
fn display_url(addr: SocketAddr) -> String {
    let ip = match addr.ip() {
        ip if ip.is_unspecified() && ip.is_ipv4() => "127.0.0.1".parse().expect("literal"),
        ip if ip.is_unspecified() => "::1".parse().expect("literal"),
        ip => ip,
    };
    format!("http://{}", SocketAddr::new(ip, addr.port()))
}

/// The last `serve-log` line of a server that ended.
fn stopped_line<E: std::fmt::Display>(result: Result<(), E>) -> String {
    match result {
        Ok(()) => "ferrum-serve: stopped".to_string(),
        Err(e) => format!("ferrum-serve: stopped with an error: {e}"),
    }
}

/// Wait up to `wait` for the server task to end; cut it off when a request keeps it busy.
fn wait_for_end(task: &JoinHandle<()>, wait: Duration) -> Result<(), String> {
    let deadline = Instant::now() + wait;
    while !task.inner().is_finished() {
        if Instant::now() > deadline {
            task.abort();
            return Err(format!(
                "the server did not stop within {} s (a request was still running); it was cut off",
                wait.as_secs()
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}

/// Load the model, start the server and answer with its URL. Fails at once when one is running.
#[tauri::command]
pub async fn serve_start<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    params: ServeParams,
) -> Result<ServeStatus, CmdError> {
    let cfg = validate(params)?;
    let auth = cfg.token.is_some();
    let guard = state.jobs.start(Job::Serve)?;
    let log = {
        let app = app.clone();
        move |line: String| {
            let _ = app.emit("serve-log", line);
        }
    };
    let request_log = log.clone();
    let bound = blocking(move || {
        // Loading reads and parses the whole model, so it stays off the async threads.
        tauri::async_runtime::block_on(bind(cfg, Some(Arc::new(request_log))))
    })
    .await?;
    for line in &bound.banner {
        log(line.clone());
    }
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let url = display_url(bound.addr);
    let requests = bound.requests.clone();
    let mut slot = state.serve.slot();
    let task = tauri::async_runtime::spawn(async move {
        let _guard = guard; // frees the job slot when the server ends, however it ends
        let result = bound
            .serve(async {
                let _ = stopped.await;
            })
            .await;
        log(stopped_line(result));
    });
    let running = Running {
        url,
        auth,
        requests,
        stop,
        task,
    };
    let status = running.status();
    *slot = Some(running);
    Ok(status)
}

/// Stop the server (in-flight requests finish first) and wait until the model is released.
/// Answers `running: false` also when none was running.
#[tauri::command]
pub async fn serve_stop(state: State<'_, AppState>) -> Result<ServeStatus, CmdError> {
    let running = state.serve.slot().take();
    let Some(r) = running else {
        return Ok(ServeStatus::default());
    };
    let requests = r.requests.load(Ordering::Relaxed);
    let Running { stop, task, .. } = r;
    let _ = stop.send(());
    blocking(move || wait_for_end(&task, STOP_WAIT)).await?;
    Ok(ServeStatus {
        running: false,
        url: None,
        requests,
        auth: false,
    })
}

/// Is a server running, where, and how many requests has it answered.
#[tauri::command]
pub fn serve_status(state: State<'_, AppState>) -> Result<ServeStatus, CmdError> {
    let mut slot = state.serve.slot();
    if slot.as_ref().is_some_and(|r| !r.alive()) {
        *slot = None;
    }
    Ok(slot.as_ref().map(Running::status).unwrap_or_default())
}

#[cfg(test)]
mod tests;
