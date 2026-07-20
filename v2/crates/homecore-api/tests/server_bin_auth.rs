//! HC-WS-08 (ADR-161): the `homecore-api-server` bin must honor the
//! `HOMECORE_TOKENS` env whitelist instead of unconditionally accepting
//! any non-empty bearer.
//!
//! `main()` is not directly callable, so this reproduces the bin's exact
//! token-provisioning path (`LongLivedTokenStore::from_env()` when
//! `HOMECORE_TOKENS` is set) and drives a real HTTP request through the
//! router. On the pre-fix bin — which used `SharedState::new()` →
//! `allow_any_non_empty()` with NO env path — a wrong bearer was
//! accepted; this test asserts it is now rejected with 401.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use homecore::HomeCore;
use homecore_api::{router, LongLivedTokenStore, SharedState};
use tower::ServiceExt; // for `oneshot`

/// Build the same state the bin builds when HOMECORE_TOKENS is set.
async fn provisioned_state(valid: &str) -> SharedState {
    // Mirror `from_env()` deterministically without mutating process
    // env (which would race other tests): an `empty()` store with the
    // one provisioned token registered is exactly what
    // `from_env()` produces for `HOMECORE_TOKENS=<valid>`.
    let store = LongLivedTokenStore::empty();
    store.register(valid).await;
    SharedState::with_tokens(HomeCore::new(), "Home", "test", store)
}

#[tokio::test]
async fn provisioned_bin_rejects_wrong_bearer() {
    let app = router(provisioned_state("the_real_token").await);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/api/states")
                .header("Authorization", "Bearer the_wrong_token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "a provisioned token store must reject a wrong bearer (HC-WS-08)"
    );
}

#[tokio::test]
async fn provisioned_bin_accepts_correct_bearer() {
    let app = router(provisioned_state("the_real_token").await);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/api/states")
                .header("Authorization", "Bearer the_real_token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn from_env_path_enforces_whitelist() {
    // Exercise the literal `from_env()` constructor the bin uses, under
    // a serialized env mutation, to prove the env path itself enforces.
    std::env::set_var("HOMECORE_TOKENS", "env_token_1, env_token_2");
    let store = LongLivedTokenStore::from_env();
    std::env::remove_var("HOMECORE_TOKENS");

    assert!(store.is_valid("env_token_1").await);
    assert!(store.is_valid("env_token_2").await);
    assert!(!store.is_valid("not_in_whitelist").await);
    assert!(!store.is_dev_mode().await, "from_env must NOT be dev mode");
}

// ─── HC-API-AUTH-01: `GET /api/` must be auth-gated like every sibling ───
//
// HA's `APIStatusView` inherits `requires_auth = True`, so `/api/` returns
// 401 for a missing/wrong bearer and 200 only for a valid one. The pre-fix
// `api_root` took no headers and unconditionally returned 200 — these two
// tests FAIL on that code.

#[tokio::test]
async fn api_root_rejects_missing_bearer() {
    let app = router(provisioned_state("the_real_token").await);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/api/")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "GET /api/ with NO bearer must be 401 (HC-API-AUTH-01) — HA's \
         APIStatusView requires_auth=True; a 200 here lets an \
         unauthenticated party confirm a live endpoint and tells a \
         token-validation probe a bad token is good"
    );
}

#[tokio::test]
async fn api_root_rejects_wrong_bearer() {
    let app = router(provisioned_state("the_real_token").await);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/api/")
                .header("Authorization", "Bearer the_wrong_token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::UNAUTHORIZED,
        "GET /api/ with a WRONG bearer must be 401 (HC-API-AUTH-01)"
    );
}

#[tokio::test]
async fn api_root_accepts_correct_bearer() {
    let app = router(provisioned_state("the_real_token").await);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/api/")
                .header("Authorization", "Bearer the_real_token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /api/ with the correct bearer must still return 200 (API running.)"
    );
}
