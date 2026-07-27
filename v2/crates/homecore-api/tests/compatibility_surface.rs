use axum::body::Body;
use axum::http::{Request, StatusCode};
use homecore::HomeCore;
use homecore_api::{router, LongLivedTokenStore, SharedState};
use http_body_util::BodyExt;
use tower::ServiceExt;

async fn app() -> (axum::Router, HomeCore) {
    let homecore = HomeCore::new();
    let tokens = LongLivedTokenStore::empty();
    tokens.register("test-token").await;
    let state = SharedState::with_tokens(homecore.clone(), "Test", "test", tokens);
    (router(state), homecore)
}

fn post(uri: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("authorization", "Bearer test-token")
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

#[tokio::test]
async fn rest_event_is_delivered_to_domain_bus() {
    let (app, homecore) = app().await;
    let mut receiver = homecore.bus().subscribe_domain();
    let response = app
        .oneshot(post("/api/events/test_event", r#"{"answer":42}"#))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let event = receiver.recv().await.unwrap();
    assert_eq!(event.event_type, "test_event");
    assert_eq!(event.event_data["answer"], 42);
}

#[tokio::test]
async fn rest_template_uses_live_state_environment() {
    let (app, _) = app().await;
    let response = app
        .oneshot(post("/api/template", r#"{"template":"{{ 6 * 7 }}"}"#))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[..], b"42");
}

#[tokio::test]
async fn compatibility_matrix_is_authenticated() {
    let (app, _) = app().await;
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/homecore/compatibility")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}
