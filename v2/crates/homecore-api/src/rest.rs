use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use serde::{Deserialize, Serialize};

use homecore::{Context, EntityId};

use crate::auth::BearerAuth;
use crate::error::{ApiError, ApiResult};
use crate::state::SharedState;

#[derive(Serialize)]
pub struct ApiRunning { message: &'static str }

/// `GET /api/` — the HA `APIStatusView` ("API running." ping).
///
/// Security (HC-API-AUTH-01): HA's `APIStatusView` inherits
/// `requires_auth = True` from `HomeAssistantView`, so an unauthenticated
/// (or wrong-token) request to `/api/` returns **401**, not 200. HA
/// clients (and the companion app) rely on this status route as a
/// *token-validation probe* — a 200 here would tell a client a bad token
/// is good, and would let an unauthenticated party confirm a live
/// HOMECORE-API endpoint. The P2 handler skipped the bearer gate that
/// every sibling route applies; this restores wire-compat by validating
/// the bearer like `get_config`/`get_states` before replying.
pub async fn api_root(headers: HeaderMap, State(s): State<SharedState>) -> ApiResult<Json<ApiRunning>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    Ok(Json(ApiRunning { message: "API running." }))
}

#[derive(Serialize)]
pub struct ApiConfig {
    location_name: String,
    version: String,
    state: &'static str,
    components: Vec<String>,
}

pub async fn get_config(headers: HeaderMap, State(s): State<SharedState>) -> ApiResult<Json<ApiConfig>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    Ok(Json(ApiConfig {
        location_name: s.location_name().to_string(),
        version: s.version().to_string(),
        state: "RUNNING",
        components: vec![],
    }))
}

#[derive(Serialize)]
pub struct StateView {
    pub entity_id: String,
    pub state: String,
    pub attributes: serde_json::Value,
    pub last_changed: String,
    pub last_updated: String,
    pub context: ContextView,
}

#[derive(Serialize)]
pub struct ContextView {
    pub id: String,
    pub user_id: Option<String>,
    pub parent_id: Option<String>,
}

impl StateView {
    pub fn from_state(s: &homecore::State) -> Self {
        Self {
            entity_id: s.entity_id.as_str().to_string(),
            state: s.state.clone(),
            attributes: s.attributes.clone(),
            last_changed: s.last_changed.to_rfc3339(),
            last_updated: s.last_updated.to_rfc3339(),
            context: ContextView {
                id: s.context.id.to_string(),
                user_id: s.context.user_id.clone(),
                parent_id: s.context.parent_id.map(|p| p.to_string()),
            },
        }
    }
}

pub async fn get_states(headers: HeaderMap, State(s): State<SharedState>) -> ApiResult<Json<Vec<StateView>>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let snapshots = s.homecore().states().all();
    Ok(Json(snapshots.iter().map(|x| StateView::from_state(x)).collect()))
}

pub async fn get_state(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path(entity_id): Path<String>,
) -> ApiResult<Json<StateView>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let id = EntityId::parse(entity_id.clone()).map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let st = s.homecore().states().get(&id).ok_or_else(|| ApiError::NotFound(entity_id))?;
    Ok(Json(StateView::from_state(&st)))
}

#[derive(Deserialize)]
pub struct SetStateRequest {
    pub state: String,
    #[serde(default)]
    pub attributes: serde_json::Value,
}

/// DELETE /api/states/:entity_id — remove an entity from the state
/// machine. Idempotent: returns 204 whether or not the entity existed,
/// matching HA's removal semantics. 4xx only for malformed entity_id or
/// auth failure.
pub async fn delete_state(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path(entity_id): Path<String>,
) -> ApiResult<StatusCode> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let id = EntityId::parse(entity_id).map_err(|e| ApiError::BadRequest(e.to_string()))?;
    s.homecore().states().remove(&id);
    Ok(StatusCode::NO_CONTENT)
}

pub async fn set_state(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path(entity_id): Path<String>,
    Json(body): Json<SetStateRequest>,
) -> ApiResult<(StatusCode, Json<StateView>)> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let id = EntityId::parse(entity_id).map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let existed = s.homecore().states().get(&id).is_some();
    let attrs = if body.attributes.is_null() { serde_json::json!({}) } else { body.attributes };
    let snap = s.homecore().states().set(id, body.state, attrs, Context::new());
    let status = if existed { StatusCode::OK } else { StatusCode::CREATED };
    Ok((status, Json(StateView::from_state(&snap))))
}

#[derive(Serialize)]
pub struct ServiceDomainView {
    pub domain: String,
    pub services: serde_json::Value,
}

pub async fn get_services(headers: HeaderMap, State(s): State<SharedState>) -> ApiResult<Json<Vec<ServiceDomainView>>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let services = s.homecore().services().registered_services().await;
    let mut by_domain: std::collections::HashMap<String, serde_json::Map<String, serde_json::Value>> =
        std::collections::HashMap::new();
    for sv in services {
        by_domain.entry(sv.domain.clone()).or_default().insert(sv.service.clone(), serde_json::json!({}));
    }
    Ok(Json(by_domain.into_iter().map(|(domain, services)| ServiceDomainView {
        domain, services: serde_json::Value::Object(services),
    }).collect()))
}

pub async fn call_service(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path((domain, service)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> ApiResult<Json<serde_json::Value>> {
    use homecore::{ServiceCall, ServiceName};
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let call = ServiceCall {
        name: ServiceName::new(domain.clone(), service.clone()),
        data: body,
        context: Context::new(),
    };
    let resp = s.homecore().services().call(call).await.map_err(|e| match e {
        homecore::ServiceError::NotRegistered { .. } => ApiError::ServiceNotRegistered { domain, service },
        other => ApiError::Internal(other.to_string()),
    })?;
    Ok(Json(resp))
}
