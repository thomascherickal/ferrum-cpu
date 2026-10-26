//! Bearer-token gate: every route except `GET /health` needs `Authorization: Bearer <token>`.

use axum::extract::{Request, State};
use axum::http::{header, Method, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Wrap `router` so that every request but `GET /health` must carry `token`. The gate is the
/// outermost layer, so a rejected request is never read, queued or counted in flight.
pub fn with_bearer_token(router: Router, token: &str) -> Router {
    let want = Arc::new(Sha256::digest(token.as_bytes()).to_vec());
    router.layer(from_fn_with_state(want, gate))
}

/// Compare SHA-256 digests with no early exit: the time does not depend on how much of the
/// token matched (nor on its length).
fn matches(given: &str, want: &[u8]) -> bool {
    let d = Sha256::digest(given.as_bytes());
    d.len() == want.len() && d.iter().zip(want).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

async fn gate(State(want): State<Arc<Vec<u8>>>, req: Request, next: Next) -> Response {
    let health = req.method() == Method::GET && req.uri().path() == "/health";
    let authed = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split_once(' '))
        .is_some_and(|(scheme, t)| scheme.eq_ignore_ascii_case("bearer") && matches(t, &want));
    if health || authed {
        return next.run(req).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        Json(json!({"error": "missing or invalid bearer token"})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::routing::{get, post};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn router() -> Router {
        with_bearer_token(
            Router::new()
                .route("/health", get(|| async { "ok" }))
                .route("/health", post(|| async { "posted" }))
                .route("/v1/x", post(|| async { "secret" })),
            "s3cret",
        )
    }

    async fn call(method: &str, uri: &str, auth: Option<&str>) -> (StatusCode, Option<String>) {
        let mut b = axum::http::Request::builder().method(method).uri(uri);
        if let Some(a) = auth {
            b = b.header("authorization", a);
        }
        let resp = router()
            .oneshot(b.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let www = resp
            .headers()
            .get("www-authenticate")
            .map(|v| v.to_str().unwrap().to_string());
        (resp.status(), www)
    }

    #[tokio::test]
    async fn health_is_open_everything_else_needs_the_token() {
        assert_eq!(call("GET", "/health", None).await.0, StatusCode::OK);
        // Only GET /health is exempt.
        assert_eq!(
            call("POST", "/health", None).await.0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            call("POST", "/v1/x", None).await.0,
            StatusCode::UNAUTHORIZED
        );
        // Unknown routes do not leak a 404 to an unauthenticated caller.
        assert_eq!(call("GET", "/nope", None).await.0, StatusCode::UNAUTHORIZED);
        assert_eq!(
            call("POST", "/v1/x", Some("Bearer s3cret")).await.0,
            StatusCode::OK
        );
        assert_eq!(
            call("POST", "/v1/x", Some("bearer s3cret")).await.0,
            StatusCode::OK,
            "the scheme is case-insensitive"
        );
    }

    #[tokio::test]
    async fn wrong_or_malformed_credentials_are_401_with_challenge() {
        for auth in [
            "Bearer wrong",
            "Bearer s3cre",
            "Bearer s3cret2",
            "Bearer ",
            "Bearer",
            "s3cret",
            "Basic czNjcmV0",
        ] {
            let (s, www) = call("POST", "/v1/x", Some(auth)).await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "{auth}");
            assert_eq!(www.as_deref(), Some("Bearer"), "{auth}");
        }
    }

    #[tokio::test]
    async fn the_401_body_is_the_documented_json_error() {
        let resp = router()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/x")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
            json!({"error": "missing or invalid bearer token"})
        );
    }

    #[test]
    fn matches_compares_the_whole_token() {
        let want = Sha256::digest(b"abc").to_vec();
        assert!(matches("abc", &want));
        assert!(!matches("abd", &want));
        assert!(!matches("", &want));
        assert!(!matches("abc", &want[..31]));
    }
}
