//! Bind-then-serve: everything `ferrum-serve` does before it accepts a request (the exposure
//! guard, loading the model, binding the socket) separated from serving, so a front end such as
//! the GUI can learn the real address (port 0), show the banner and count requests before it
//! starts serving.

use crate::cli::{is_loopback, Config};
use crate::{app, with_bearer_token, Info};
use axum::extract::{Request, State};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::Response;
use axum::Router;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Receives one line per finished request (`POST /v1/generate -> 200`).
pub type RequestLog = Arc<dyn Fn(String) + Send + Sync>;

/// A loaded model on a bound socket, not yet serving.
pub struct Bound {
    /// The address actually bound (the real port when `--port 0` was asked for).
    pub addr: SocketAddr,
    pub info: Info,
    /// The startup lines `ferrum-serve` prints: model and address, then the auth state.
    pub banner: Vec<String>,
    /// Requests answered so far, rejected ones (401, 404, 503) included.
    pub requests: Arc<AtomicU64>,
    listener: tokio::net::TcpListener,
    router: Router,
}

#[derive(Clone)]
struct Tally {
    requests: Arc<AtomicU64>,
    log: Option<RequestLog>,
}

/// Count every request and, when asked, log it. The outermost layer, so it sees what the token
/// gate rejects too.
async fn tally(State(t): State<Tally>, req: Request, next: Next) -> Response {
    let (method, path) = (req.method().clone(), req.uri().path().to_string());
    let resp = next.run(req).await;
    t.requests.fetch_add(1, Ordering::Relaxed);
    if let Some(log) = &t.log {
        // The path is the client's; keep a hostile one from flooding a log view.
        let path: String = path.chars().take(120).collect();
        log(format!("{method} {path} -> {}", resp.status().as_u16()));
    }
    resp
}

/// Check that `cfg` may listen where it asks, load the model and bind. Blocking work (reading
/// and parsing the model) happens here, before the socket opens.
pub async fn bind(cfg: Config, request_log: Option<RequestLog>) -> Result<Bound, String> {
    if cfg.token.is_none() && !cfg.allow_unauthenticated && !is_loopback(&cfg.host) {
        return Err(format!(
            "refusing to listen on non-loopback host {} without a bearer token: set FERRUM_TOKEN \
(or --token), or pass --allow-unauthenticated",
            cfg.host
        ));
    }
    let bytes = std::fs::read(&cfg.model).map_err(|e| format!("cannot read {}: {e}", cfg.model))?;
    let (router, info) = app(bytes, cfg.limits.clone())?;
    let router = match &cfg.token {
        Some(t) => with_bearer_token(router, t),
        None => router,
    };
    let requests = Arc::new(AtomicU64::new(0));
    let router = router.layer(from_fn_with_state(
        Tally {
            requests: requests.clone(),
            log: request_log,
        },
        tally,
    ));
    let listener = tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port))
        .await
        .map_err(|e| format!("cannot bind {}:{}: {e}", cfg.host, cfg.port))?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    let auth = match (&cfg.token, is_loopback(&cfg.host)) {
        (Some(_), _) => "ON (bearer token required, except GET /health)",
        (None, true) => "OFF (loopback only)",
        (None, false) => "OFF (--allow-unauthenticated: anyone who can reach this port can use it)",
    };
    let banner = vec![
        format!(
            "ferrum-serve: {} ({} params, context {}, vocab {}) on http://{addr}",
            cfg.model, info.params, info.context_len, info.vocab
        ),
        format!("ferrum-serve: auth {auth}"),
    ];
    Ok(Bound {
        addr,
        info,
        banner,
        requests,
        listener,
        router,
    })
}

impl Bound {
    /// Serve until `shutdown` resolves; in-flight requests finish first.
    pub async fn serve(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), String> {
        axum::serve(self.listener, self.router)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use std::io::{Read, Write};
    use std::sync::Mutex;

    fn model_file(name: &str) -> String {
        let p = std::env::temp_dir().join(format!("ferrum-bind-{}-{name}", std::process::id()));
        let m = crate::test_util::model(0);
        std::fs::write(&p, m.to_bytes_quantized().unwrap()).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn cfg(model: String, token: Option<&str>) -> Config {
        Config {
            model,
            host: "127.0.0.1".into(),
            port: 0,
            limits: Limits::default(),
            token: token.map(String::from),
            allow_unauthenticated: false,
        }
    }

    /// One blocking HTTP/1.1 request; returns the status code.
    fn status(addr: SocketAddr, method: &str, path: &str, auth: Option<&str>) -> u16 {
        let mut s = std::net::TcpStream::connect(addr).unwrap();
        let auth = auth.map_or(String::new(), |a| format!("Authorization: {a}\r\n"));
        write!(
            s,
            "{method} {path} HTTP/1.1\r\nHost: x\r\n{auth}Content-Length: 0\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out.split(' ').nth(1).unwrap().parse().unwrap()
    }

    #[tokio::test]
    async fn bind_reports_the_real_port_and_the_banner_serve_counts_and_logs_requests() {
        let path = model_file("count");
        let lines = Arc::new(Mutex::new(Vec::new()));
        let sink = lines.clone();
        let log: RequestLog = Arc::new(move |l| sink.lock().unwrap().push(l));
        let bound = bind(cfg(path.clone(), Some("tok")), Some(log))
            .await
            .unwrap();
        assert_ne!(bound.addr.port(), 0, "port 0 must resolve to a real port");
        assert_eq!(bound.banner.len(), 2);
        assert!(
            bound.banner[0].starts_with("ferrum-serve: ")
                && bound.banner[0].contains(&bound.addr.to_string())
        );
        assert!(bound.banner[1].contains("auth ON"));
        let (addr, requests) = (bound.addr, bound.requests.clone());
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(bound.serve(async {
            let _ = rx.await;
        }));
        let codes = tokio::task::spawn_blocking(move || {
            vec![
                status(addr, "GET", "/health", None),
                status(addr, "GET", "/v1/info", None), // no token
                status(addr, "GET", "/v1/info", Some("Bearer tok")),
                status(addr, "GET", "/nope", Some("Bearer tok")),
            ]
        })
        .await
        .unwrap();
        assert_eq!(codes, [200, 401, 200, 404]);
        assert_eq!(requests.load(Ordering::Relaxed), 4);
        assert_eq!(
            *lines.lock().unwrap(),
            [
                "GET /health -> 200",
                "GET /v1/info -> 401",
                "GET /v1/info -> 200",
                "GET /nope -> 404"
            ]
        );
        tx.send(()).unwrap();
        assert_eq!(server.await.unwrap(), Ok(()));
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn bind_without_a_log_still_counts_and_refuses_an_exposed_host_without_a_token() {
        let path = model_file("nolog");
        let b = bind(cfg(path.clone(), None), None).await.unwrap();
        assert!(b.banner[1].contains("auth OFF (loopback only)"));
        let mut open = cfg(path.clone(), None);
        open.host = "0.0.0.0".into();
        let e = bind(open.clone(), None).await.err().unwrap();
        assert!(
            e.contains("refusing to listen on non-loopback host 0.0.0.0"),
            "{e}"
        );
        // Explicitly allowed, it listens and the banner says that nothing guards the port.
        open.allow_unauthenticated = true;
        let b = bind(open, None).await.unwrap();
        assert!(
            b.banner[1].contains("auth OFF (--allow-unauthenticated"),
            "{:?}",
            b.banner
        );
        std::fs::remove_file(path).ok();
    }
}
