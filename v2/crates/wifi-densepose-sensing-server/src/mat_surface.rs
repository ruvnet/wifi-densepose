//! Host the existing MAT router without creating demo incidents or detections.
//! Incident state is in-memory; CSI ingest must be configured explicitly.

/// Attach before host, bearer, companion and privacy middleware are applied.
pub fn router() -> axum::Router {
    wifi_densepose_mat::api::create_router(wifi_densepose_mat::api::AppState::new())
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn real_mat_router_starts_empty_and_reports_optional_ml_truthfully() {
        let app = super::router();
        for path in [
            "/api/v1/mat/events?page=0&page_size=100",
            "/api/v1/mat/scan/status",
        ] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            let body = to_bytes(response.into_body(), 128 * 1024).await.unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            if path.contains("events") {
                assert_eq!(value["events"].as_array().unwrap().len(), 0);
                assert_eq!(value["total"], 0);
            } else {
                assert_eq!(value["scanning"], false);
                assert_eq!(value["ml_enabled"], false);
                assert_eq!(value["ml_ready"], false);
            }
        }
    }

    #[tokio::test]
    async fn mat_reads_mutations_and_stream_remain_behind_host_auth_layers() {
        let app = super::router().layer(axum::middleware::from_fn_with_state(
            crate::bearer_auth::AuthState::from_token("mat-test-fixture-token"),
            crate::bearer_auth::require_bearer,
        ));
        for (method, path) in [
            ("GET", "/api/v1/mat/events"),
            ("POST", "/api/v1/mat/scan/control"),
            ("GET", "/ws/mat/stream"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "{method} {path}"
            );
        }
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/mat/scan/status")
                    .header("authorization", "Bearer mat-test-fixture-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn real_incident_create_and_read_use_only_explicit_test_state() {
        let app = super::router();
        let body = r#"{"event_type":"Earthquake","latitude":0,"longitude":0,"description":"EXPLICIT ISOLATED TEST FIXTURE"}"#;
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/mat/events")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/mat/events?page=0&page_size=100")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = to_bytes(response.into_body(), 128 * 1024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["total"], 1);
        assert_eq!(value["events"][0]["survivor_count"], 0);
        assert_eq!(
            value["events"][0]["description"],
            "EXPLICIT ISOLATED TEST FIXTURE"
        );
    }
}
