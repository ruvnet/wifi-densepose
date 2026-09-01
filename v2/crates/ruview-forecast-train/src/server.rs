//! fal Direct Server adapter. The only accepted training body is the closed,
//! synthetic [`HostedSyntheticPayload`](crate::fal::HostedSyntheticPayload).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use tokio::sync::Semaphore;

use crate::{
    cancel::{CancelToken, Cancellation},
    fal::{HostedSyntheticPayload, HostedTrainingOutcome},
};

const REQUEST_ID_HEADER: &str = "x-fal-request-id";
const MAX_SERVER_BODY: usize = 64 * 1024;
const MAX_JOBS: usize = 1024;
const MAX_CONCURRENT_JOBS: usize = 1;

/// Synchronous typed executor used inside `spawn_blocking`. Production binds
/// this to the same Burn trainer as the local CLI; tests use a deterministic
/// fake without compiling a tensor backend.
pub trait SyntheticJobExecutor: Send + Sync + 'static {
    /// Execute exactly one validated synthetic plan.
    fn execute(
        &self,
        payload: HostedSyntheticPayload,
        cancellation: &CancelToken,
    ) -> Result<HostedTrainingOutcome, String>;
}

#[derive(Clone)]
struct ServerState {
    executor: Arc<dyn SyntheticJobExecutor>,
    jobs: Arc<Mutex<HashMap<String, JobState>>>,
    execution_slots: Arc<Semaphore>,
}

#[derive(Clone)]
enum JobState {
    Running {
        digest: crate::config::Sha256Digest,
        cancel: CancelToken,
        expires_at_ms: u64,
    },
    Complete {
        digest: crate::config::Sha256Digest,
        result: Box<HostedTrainingOutcome>,
    },
    Failed {
        digest: crate::config::Sha256Digest,
        expires_at_ms: u64,
    },
}

/// Health response used by fal deployment probes.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Service readiness.
    pub status: &'static str,
    /// Protocol limitation.
    pub training_mode: &'static str,
}

/// Constructs the Direct Server router.
pub fn router(executor: Arc<dyn SyntheticJobExecutor>) -> Router {
    let state = ServerState {
        executor,
        jobs: Arc::new(Mutex::new(HashMap::new())),
        execution_slots: Arc::new(Semaphore::new(MAX_CONCURRENT_JOBS)),
    };
    Router::new()
        .route("/health", get(health))
        .route("/train", post(train))
        .route("/train/cancel", post(cancel))
        .layer(DefaultBodyLimit::max(MAX_SERVER_BODY))
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        training_mode: "synthetic_only",
    })
}

async fn train(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let request_id = match request_id(&headers) {
        Ok(value) => value,
        Err(status) => return status.into_response(),
    };
    let payload: HostedSyntheticPayload = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid synthetic payload").into_response(),
    };
    if payload.validate_for_worker(unix_time_millis()).is_err() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            "rejected synthetic payload",
        )
            .into_response();
    }
    let (cancel, execution_slot) = {
        let mut jobs = match state.jobs.lock() {
            Ok(value) => value,
            Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        };
        let now_ms = unix_time_millis();
        jobs.retain(|_, job| match job {
            JobState::Running { .. } => true,
            JobState::Complete { result, .. } => result.artifacts_expire_at_ms() > now_ms,
            JobState::Failed { expires_at_ms, .. } => *expires_at_ms > now_ms,
        });
        match jobs.get(&request_id) {
            Some(JobState::Complete { digest, result }) if *digest == payload.request_digest => {
                return Json(result.as_ref().clone()).into_response();
            }
            Some(JobState::Running { digest, .. }) if *digest == payload.request_digest => {
                return (StatusCode::CONFLICT, "request already running").into_response();
            }
            Some(JobState::Failed { digest, .. }) if *digest == payload.request_digest => {
                // X-Fal-No-Retry makes fal itself avoid retrying; a repeated
                // delivery receives the same terminal answer.
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "request previously failed",
                )
                    .into_response();
            }
            Some(_) => return (StatusCode::CONFLICT, "request id digest conflict").into_response(),
            None if jobs.len() >= MAX_JOBS => return StatusCode::TOO_MANY_REQUESTS.into_response(),
            None => {}
        }
        let execution_slot = match Arc::clone(&state.execution_slots).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => return StatusCode::TOO_MANY_REQUESTS.into_response(),
        };
        let token = CancelToken::new();
        jobs.insert(
            request_id.clone(),
            JobState::Running {
                digest: payload.request_digest,
                cancel: token.clone(),
                expires_at_ms: payload.expires_at_ms,
            },
        );
        (token, execution_slot)
    };

    let executor = Arc::clone(&state.executor);
    let execute_cancel = cancel.clone();
    let _execution_slot = execution_slot;
    let result =
        tokio::task::spawn_blocking(move || executor.execute(payload, &execute_cancel)).await;
    let mut jobs = match state.jobs.lock() {
        Ok(value) => value,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    match result {
        Ok(Ok(value)) => {
            let digest = match jobs.get(&request_id) {
                Some(JobState::Running { digest, .. }) => *digest,
                _ => return StatusCode::CONFLICT.into_response(),
            };
            jobs.insert(
                request_id,
                JobState::Complete {
                    digest,
                    result: Box::new(value.clone()),
                },
            );
            Json(value).into_response()
        }
        Ok(Err(_)) => {
            if let Some(JobState::Running {
                digest,
                expires_at_ms,
                ..
            }) = jobs.get(&request_id)
            {
                let (digest, expires_at_ms) = (*digest, *expires_at_ms);
                jobs.insert(
                    request_id,
                    JobState::Failed {
                        digest,
                        expires_at_ms,
                    },
                );
            }
            if cancel.is_cancelled() {
                (
                    StatusCode::from_u16(499).expect("valid cancellation status"),
                    "training cancelled",
                )
                    .into_response()
            } else {
                (StatusCode::UNPROCESSABLE_ENTITY, "training failed").into_response()
            }
        }
        Err(_) => {
            if let Some(JobState::Running {
                digest,
                expires_at_ms,
                ..
            }) = jobs.get(&request_id)
            {
                let (digest, expires_at_ms) = (*digest, *expires_at_ms);
                jobs.insert(
                    request_id,
                    JobState::Failed {
                        digest,
                        expires_at_ms,
                    },
                );
            }
            (StatusCode::INTERNAL_SERVER_ERROR, "training task failed").into_response()
        }
    }
}

async fn cancel(State(state): State<ServerState>, headers: HeaderMap) -> impl IntoResponse {
    let request_id = match request_id(&headers) {
        Ok(value) => value,
        Err(status) => return status.into_response(),
    };
    let jobs = match state.jobs.lock() {
        Ok(value) => value,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    match jobs.get(&request_id) {
        Some(JobState::Running { cancel, .. }) => {
            cancel.cancel();
            StatusCode::ACCEPTED.into_response()
        }
        Some(JobState::Complete { .. } | JobState::Failed { .. }) => StatusCode::OK.into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn request_id(headers: &HeaderMap) -> Result<String, StatusCode> {
    let value = headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;
    if value.is_empty()
        || value.len() > 160
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(value.to_owned())
}

fn unix_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    struct NeverCalled;
    impl SyntheticJobExecutor for NeverCalled {
        fn execute(
            &self,
            _: HostedSyntheticPayload,
            _: &CancelToken,
        ) -> Result<HostedTrainingOutcome, String> {
            panic!("invalid request reached executor")
        }
    }

    #[tokio::test]
    async fn direct_server_train_requires_request_id_header() {
        let response = router(Arc::new(NeverCalled))
            .oneshot(Request::post("/train").body(Body::from("{}")).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn request_id_rejects_path_injection() {
        let response = router(Arc::new(NeverCalled))
            .oneshot(
                Request::post("/train")
                    .header(REQUEST_ID_HEADER, "bad/path")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unknown_cancel_is_not_success() {
        let response = router(Arc::new(NeverCalled))
            .oneshot(
                Request::post("/train/cancel")
                    .header(REQUEST_ID_HEADER, "req-1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn privacy_external_dataset_payload_is_denied() {
        let body = r#"{"dataset_path":"/data/customer.jsonl","tenant":"x"}"#;
        let response = router(Arc::new(NeverCalled))
            .oneshot(
                Request::post("/train")
                    .header(REQUEST_ID_HEADER, "req-2")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn server_allows_one_training_execution() {
        let slots = Arc::new(Semaphore::new(MAX_CONCURRENT_JOBS));
        let first = Arc::clone(&slots).try_acquire_owned().unwrap();
        assert!(Arc::clone(&slots).try_acquire_owned().is_err());
        drop(first);
        assert!(slots.try_acquire_owned().is_ok());
    }
}
