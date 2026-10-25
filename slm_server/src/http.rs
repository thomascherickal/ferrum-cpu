//! HTTP layer: routing, request validation, limits and error mapping. Model
//! work is queued to the model thread (`model_thread`).

use crate::info::Info;
use crate::limits::{Limits, MAX_INFLIGHT, MAX_STOPS, MAX_STOP_BYTES};
use crate::model_thread::{spawn_model_thread, GenRequest, Job, Reply};
use axum::extract::{rejection::JsonRejection, DefaultBodyLimit, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{mpsc, Arc, Weak};
use tokio::sync::{oneshot, Semaphore};

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    info: Arc<Info>,
    /// Dead once the model thread has exited (it holds the only strong ref).
    alive: Weak<()>,
    max_batch: usize,
    max_chars: usize,
    max_tokens: usize,
    inflight: Arc<Semaphore>,
}

/// Load the model in `bytes` (a model file) and build the router, spawning the
/// model thread. Also returns the model's [`Info`].
pub fn app(bytes: Vec<u8>, limits: Limits) -> Result<(Router, Info), String> {
    let (jobs, info, alive) = spawn_model_thread(bytes)?;
    let state = AppState {
        jobs,
        info: Arc::new(info.clone()),
        alive,
        max_batch: limits.max_batch,
        max_chars: limits.max_chars,
        max_tokens: limits.max_tokens,
        inflight: Arc::new(Semaphore::new(MAX_INFLIGHT)),
    };
    let posts = Router::new()
        .route("/v1/loglikelihood", post(loglikelihood))
        .route("/v1/loglikelihood_rolling", post(loglikelihood_rolling))
        .route("/v1/generate", post(generate))
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            limit_inflight,
        ));
    let router = Router::new()
        .route("/health", get(health))
        .route(
            "/v1/info",
            get(|State(s): State<AppState>| async move { Json((*s.info).clone()) }),
        )
        .merge(posts)
        .layer(DefaultBodyLimit::max(limits.max_body))
        .fallback(|| async { ApiError(StatusCode::NOT_FOUND, "not found".into()) })
        .method_not_allowed_fallback(|| async {
            ApiError(StatusCode::METHOD_NOT_ALLOWED, "method not allowed".into())
        })
        .with_state(state);
    Ok((router, info))
}

/// Admit a POST only while fewer than [`MAX_INFLIGHT`] are in flight.
async fn limit_inflight(State(s): State<AppState>, req: Request, next: Next) -> Response {
    match s.inflight.clone().try_acquire_owned() {
        Ok(_permit) => next.run(req).await,
        Err(_) => ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("too many requests in flight (max {MAX_INFLIGHT})"),
        )
        .into_response(),
    }
}

/// Liveness: 503 once the model thread is gone.
async fn health(State(s): State<AppState>) -> Result<Json<&'static str>, ApiError> {
    if s.alive.strong_count() == 0 {
        return Err(ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "model thread stopped".into(),
        ));
    }
    Ok(Json("ok"))
}

/// JSON error response.
#[derive(Debug)]
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

/// Map axum's JSON rejection: oversized body -> 413, anything else -> 400.
fn reject(e: JsonRejection) -> ApiError {
    let status = if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    };
    ApiError(status, e.body_text())
}

/// Reject a batch of `n` texts with the given char lengths if it exceeds the limits.
fn check(s: &AppState, n: usize, lens: impl Iterator<Item = usize>) -> Result<(), ApiError> {
    if n > s.max_batch {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            format!("batch of {n} exceeds max_batch {}", s.max_batch),
        ));
    }
    if let Some(len) = lens.into_iter().find(|&l| l > s.max_chars) {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            format!("text of {len} chars exceeds max_chars {}", s.max_chars),
        ));
    }
    Ok(())
}

/// Queue a job on the model thread and await its reply.
async fn submit(s: &AppState, make: impl FnOnce(Reply) -> Job) -> Result<Json<Value>, ApiError> {
    let stopped = || {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "model thread stopped".into(),
        )
    };
    let (tx, rx) = oneshot::channel();
    s.jobs.send(make(tx)).map_err(|_| stopped())?;
    match rx.await {
        Ok(Ok(results)) => Ok(Json(json!({"results": results}))),
        Ok(Err(e)) => Err(ApiError(StatusCode::UNPROCESSABLE_ENTITY, e)),
        Err(_) => Err(stopped()),
    }
}

#[derive(Deserialize)]
struct LlRequest {
    context: String,
    continuation: String,
}
#[derive(Deserialize)]
struct LlBody {
    requests: Vec<LlRequest>,
}

async fn loglikelihood(
    State(s): State<AppState>,
    body: Result<Json<LlBody>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(b) = body.map_err(reject)?;
    let lens = b
        .requests
        .iter()
        .map(|r| r.context.chars().count() + r.continuation.chars().count());
    check(&s, b.requests.len(), lens)?;
    let reqs = b
        .requests
        .into_iter()
        .map(|r| (r.context, r.continuation))
        .collect();
    submit(&s, |tx| Job::Score(reqs, tx)).await
}

#[derive(Deserialize)]
struct RollingBody {
    texts: Vec<String>,
}

async fn loglikelihood_rolling(
    State(s): State<AppState>,
    body: Result<Json<RollingBody>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(b) = body.map_err(reject)?;
    check(&s, b.texts.len(), b.texts.iter().map(|t| t.chars().count()))?;
    submit(&s, |tx| Job::Rolling(b.texts, tx)).await
}

#[derive(Deserialize)]
struct GenBody {
    requests: Vec<GenRequest>,
}

/// Boundary checks for one generate request.
fn validate_gen(r: &GenRequest, max_tokens: usize) -> Result<(), ApiError> {
    let bad = |m: String| Err(ApiError(StatusCode::BAD_REQUEST, m));
    if r.max_tokens > max_tokens {
        return bad(format!(
            "max_tokens {} exceeds max_tokens limit {max_tokens}",
            r.max_tokens
        ));
    }
    if !(r.temperature.is_finite() && r.temperature >= 0.0) {
        return bad(format!(
            "temperature must be finite and >= 0, got {}",
            r.temperature
        ));
    }
    if !(r.top_p > 0.0 && r.top_p <= 1.0) {
        return bad(format!("top_p must be in (0, 1], got {}", r.top_p));
    }
    if r.stop.len() > MAX_STOPS {
        return bad(format!(
            "{} stop strings exceed the limit of {MAX_STOPS}",
            r.stop.len()
        ));
    }
    if let Some(s) = r.stop.iter().find(|s| s.len() > MAX_STOP_BYTES) {
        return bad(format!(
            "a stop string of {} bytes exceeds the limit of {MAX_STOP_BYTES}",
            s.len()
        ));
    }
    Ok(())
}

async fn generate(
    State(s): State<AppState>,
    body: Result<Json<GenBody>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(b) = body.map_err(reject)?;
    check(
        &s,
        b.requests.len(),
        b.requests.iter().map(|r| r.prompt.chars().count()),
    )?;
    b.requests
        .iter()
        .try_for_each(|r| validate_gen(r, s.max_tokens))?;
    submit(&s, |tx| Job::Generate(b.requests, tx)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::info::info_for;
    use crate::test_util::model;

    fn state(jobs: mpsc::Sender<Job>) -> AppState {
        AppState {
            jobs,
            info: Arc::new(info_for(&model(0), b"")),
            alive: Weak::new(),
            max_batch: 2,
            max_chars: 3,
            max_tokens: 5,
            inflight: Arc::new(Semaphore::new(MAX_INFLIGHT)),
        }
    }

    #[test]
    fn check_accepts_limits_and_rejects_over() {
        let (tx, _rx) = mpsc::channel();
        let s = state(tx);
        assert!(check(&s, 0, std::iter::empty()).is_ok());
        assert!(check(&s, 2, [3, 1].into_iter()).is_ok());
        let e = check(&s, 3, std::iter::empty()).unwrap_err();
        assert_eq!(e.0, StatusCode::BAD_REQUEST);
        assert!(e.1.contains("max_batch 2"));
        let e = check(&s, 1, [4].into_iter()).unwrap_err();
        assert_eq!(e.0, StatusCode::BAD_REQUEST);
        assert!(e.1.contains("4 chars") && e.1.contains("max_chars 3"));
    }

    #[tokio::test]
    async fn api_error_renders_json_body() {
        use http_body_util::BodyExt;
        let resp = ApiError(StatusCode::IM_A_TEAPOT, "boom".into()).into_response();
        assert_eq!(resp.status(), StatusCode::IM_A_TEAPOT);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap(),
            json!({"error": "boom"})
        );
    }

    #[tokio::test]
    async fn reject_maps_413_and_everything_else_to_400() {
        use axum::extract::FromRequest;
        use axum::http::Request;
        // A body over the limit: the length-limited body errors while being read.
        let too_big = Request::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::new(http_body_util::Limited::new(
                axum::body::Body::from("x".repeat(100)),
                10,
            )))
            .unwrap();
        let rej = Json::<Value>::from_request(too_big, &()).await.unwrap_err();
        let e = reject(rej);
        assert_eq!(e.0, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(!e.1.is_empty());
        let bad = Request::builder()
            .header("content-type", "application/json")
            .body(axum::body::Body::from("{nope"))
            .unwrap();
        let rej = Json::<Value>::from_request(bad, &()).await.unwrap_err();
        assert_eq!(reject(rej).0, StatusCode::BAD_REQUEST);
        let no_ct = Request::builder()
            .body(axum::body::Body::from("{}"))
            .unwrap();
        let rej = Json::<Value>::from_request(no_ct, &()).await.unwrap_err();
        assert_eq!(reject(rej).0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn submit_reports_stopped_model_thread() {
        // Receiver gone: send fails.
        let (tx, rx) = mpsc::channel();
        drop(rx);
        let e = submit(&state(tx), |r| Job::Rolling(vec![], r))
            .await
            .unwrap_err();
        assert_eq!(
            (e.0, e.1.as_str()),
            (StatusCode::SERVICE_UNAVAILABLE, "model thread stopped")
        );
        // Thread takes the job but drops the reply without answering.
        let (tx, rx) = mpsc::channel();
        let t = std::thread::spawn(move || drop(rx.recv().unwrap()));
        let e = submit(&state(tx), |r| Job::Rolling(vec![], r))
            .await
            .unwrap_err();
        t.join().unwrap();
        assert_eq!(
            (e.0, e.1.as_str()),
            (StatusCode::SERVICE_UNAVAILABLE, "model thread stopped")
        );
    }

    #[tokio::test]
    async fn submit_maps_ok_and_model_error() {
        let (tx, rx) = mpsc::channel::<Job>();
        let t = std::thread::spawn(move || {
            for job in rx {
                let (texts, reply) = match job {
                    Job::Rolling(t, r) => (t, r),
                    Job::Score(_, r) | Job::Generate(_, r) | Job::Panic(r) | Job::Pool(_, r) => {
                        (vec!["other".into()], r)
                    }
                };
                let _ = reply.send(if texts.is_empty() {
                    Ok(vec![json!(1)])
                } else {
                    Err("bad".into())
                });
            }
        });
        let s = state(tx);
        let Json(v) = submit(&s, |r| Job::Rolling(vec![], r)).await.unwrap();
        assert_eq!(v, json!({"results": [1]}));
        let e = submit(&s, |r| Job::Rolling(vec!["x".into()], r))
            .await
            .unwrap_err();
        assert_eq!(
            (e.0, e.1.as_str()),
            (StatusCode::UNPROCESSABLE_ENTITY, "bad")
        );
        let e = submit(&s, |r| Job::Score(vec![], r)).await.unwrap_err();
        assert_eq!(
            (e.0, e.1.as_str()),
            (StatusCode::UNPROCESSABLE_ENTITY, "bad")
        );
        let e = submit(&s, |r| Job::Generate(vec![], r)).await.unwrap_err();
        assert_eq!(e.0, StatusCode::UNPROCESSABLE_ENTITY);
        let e = submit(&s, Job::Panic).await.unwrap_err();
        assert_eq!(e.0, StatusCode::UNPROCESSABLE_ENTITY);
        let e = submit(&s, |r| Job::Pool(false, r)).await.unwrap_err();
        assert_eq!(e.0, StatusCode::UNPROCESSABLE_ENTITY);
        drop(s);
        t.join().unwrap();
    }

    #[tokio::test]
    async fn model_thread_survives_a_panicking_job() {
        let m = model(0);
        let (tx, _info, alive) = spawn_model_thread(m.to_bytes_quantized().unwrap()).unwrap();
        let mut s = state(tx);
        s.alive = alive;
        let e = submit(&s, Job::Panic).await.unwrap_err();
        assert_eq!(e.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(e.1.contains("model panicked: boom"), "{}", e.1);
        // next request still served, and health still ok
        let Json(v) = submit(&s, |r| Job::Rolling(vec!["abc".into()], r))
            .await
            .unwrap();
        assert_eq!(v["results"].as_array().unwrap().len(), 1);
        assert!(health(State(s)).await.is_ok());
    }

    /// Regression: a panic on a `parallel.rs` pool worker (not the model thread)
    /// failed that request with a clean 422, but killed the worker, so every later
    /// parallel matmul failed too. The server must answer an error, not hang, and
    /// keep serving afterwards.
    #[tokio::test]
    async fn pool_worker_panic_is_an_error_and_the_server_keeps_serving() {
        let m = model(0);
        let (tx, _info, alive) = spawn_model_thread(m.to_bytes_quantized().unwrap()).unwrap();
        let mut s = state(tx);
        s.alive = alive;
        for _ in 0..(2 * ferrum_core::parallel::num_threads() + 2) {
            let e = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                submit(&s, |r| Job::Pool(true, r)),
            )
            .await
            .expect("must not hang")
            .unwrap_err();
            assert_eq!(e.0, StatusCode::UNPROCESSABLE_ENTITY);
            assert!(e.1.contains("panicked"), "{}", e.1);
            let Json(v) = submit(&s, |r| Job::Pool(false, r)).await.unwrap();
            assert_eq!(v["results"][0].as_f64(), Some(2048.0));
        }
        assert!(health(State(s)).await.is_ok());
    }

    #[tokio::test]
    async fn health_is_503_when_model_thread_gone() {
        let (tx, _rx) = mpsc::channel();
        let e = health(State(state(tx))).await.unwrap_err();
        assert_eq!(
            (e.0, e.1.as_str()),
            (StatusCode::SERVICE_UNAVAILABLE, "model thread stopped")
        );
    }

    fn gen_req(max_tokens: usize, temperature: f32, top_p: f32) -> GenRequest {
        GenRequest {
            prompt: String::new(),
            max_tokens,
            stop: vec![],
            temperature,
            top_k: 0,
            top_p,
            seed: 0,
        }
    }

    #[test]
    fn validate_gen_boundaries() {
        assert!(validate_gen(&gen_req(5, 0.0, 1.0), 5).is_ok());
        assert!(validate_gen(&gen_req(0, 2.0, 0.001), 5).is_ok());
        let e = validate_gen(&gen_req(6, 0.0, 1.0), 5).unwrap_err();
        assert!(e.1.contains("max_tokens 6") && e.1.contains("limit 5"));
        for t in [-0.1, f32::NAN, f32::INFINITY] {
            let e = validate_gen(&gen_req(1, t, 1.0), 5).unwrap_err();
            assert!(e.1.contains("temperature"), "{}", e.1);
        }
        for p in [0.0, -1.0, 1.01, f32::NAN] {
            let e = validate_gen(&gen_req(1, 0.0, p), 5).unwrap_err();
            assert_eq!(e.0, StatusCode::BAD_REQUEST);
            assert!(e.1.contains("top_p"), "{}", e.1);
        }
    }

    #[test]
    fn validate_gen_bounds_stop_strings() {
        let mut r = gen_req(1, 0.0, 1.0);
        r.stop = vec!["s".repeat(MAX_STOP_BYTES); MAX_STOPS];
        assert!(validate_gen(&r, 5).is_ok(), "exactly at both limits");
        r.stop.push("x".into());
        let e = validate_gen(&r, 5).unwrap_err();
        assert!(
            e.1.contains("17 stop strings") && e.1.contains("16"),
            "{}",
            e.1
        );
        r.stop = vec!["s".repeat(MAX_STOP_BYTES + 1)];
        let e = validate_gen(&r, 5).unwrap_err();
        assert_eq!(e.0, StatusCode::BAD_REQUEST);
        assert!(e.1.contains("1025 bytes"), "{}", e.1);
    }

    #[tokio::test]
    async fn limit_inflight_sheds_load_and_releases_permits() {
        use tower::ServiceExt;
        let (tx, _rx) = mpsc::channel();
        let mk = |permits: usize| {
            let mut st = state(tx.clone());
            st.inflight = Arc::new(Semaphore::new(permits));
            let r = Router::new()
                .route("/x", post(|| async { "ok" }))
                .route_layer(axum::middleware::from_fn_with_state(
                    st.clone(),
                    limit_inflight,
                ));
            (r, st)
        };
        let req = || {
            axum::http::Request::builder()
                .method("POST")
                .uri("/x")
                .body(axum::body::Body::empty())
                .unwrap()
        };
        let (r, st) = mk(0);
        let resp = r.oneshot(req()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let (r, st1) = mk(1);
        // One permit serves any number of sequential requests (it is returned each time).
        for _ in 0..3 {
            let resp = r.clone().oneshot(req()).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }
        assert_eq!(st1.inflight.available_permits(), 1);
        assert_eq!(st.inflight.available_permits(), 0);
    }
}
