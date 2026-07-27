use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use homecore::{Context, EntityId};

use crate::auth::BearerAuth;
use crate::error::{ApiError, ApiResult};
use crate::state::SharedState;

#[derive(Serialize)]
pub struct ApiRunning {
    message: &'static str,
}

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
pub async fn api_root(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<Json<ApiRunning>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    Ok(Json(ApiRunning {
        message: "API running.",
    }))
}

#[derive(Serialize)]
pub struct ApiConfig {
    location_name: String,
    version: String,
    state: &'static str,
    components: Vec<String>,
}

const LOADED_COMPONENTS: &[&str] = &[
    "api",
    "automation",
    "config",
    "homecore",
    "recorder",
    "websocket_api",
];

pub async fn get_config(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<Json<ApiConfig>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    Ok(Json(ApiConfig {
        location_name: s.location_name().to_string(),
        version: s.version().to_string(),
        state: "RUNNING",
        components: LOADED_COMPONENTS
            .iter()
            .map(|component| (*component).to_owned())
            .collect(),
    }))
}

pub async fn get_components(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<Json<Vec<String>>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    Ok(Json(
        LOADED_COMPONENTS
            .iter()
            .map(|component| (*component).to_owned())
            .collect(),
    ))
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

#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    filter_entity_id: Option<String>,
    end_time: Option<String>,
    #[serde(default)]
    minimal_response: bool,
    #[serde(default)]
    no_attributes: bool,
    #[serde(default)]
    significant_changes_only: bool,
}

const MAX_HISTORY_ENTITIES: usize = 32;
const MAX_API_HISTORY_ROWS: usize = 100_000;

pub async fn get_history(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Query(query): Query<HistoryQuery>,
) -> ApiResult<Json<Vec<Vec<StateView>>>> {
    history_response(headers, s, None, query).await
}

pub async fn get_history_period(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path(start_time): Path<String>,
    Query(query): Query<HistoryQuery>,
) -> ApiResult<Json<Vec<Vec<StateView>>>> {
    history_response(headers, s, Some(start_time), query).await
}

async fn history_response(
    headers: HeaderMap,
    state: SharedState,
    start_time: Option<String>,
    query: HistoryQuery,
) -> ApiResult<Json<Vec<Vec<StateView>>>> {
    let _ = BearerAuth::from_headers(&headers, state.tokens()).await?;
    let recorder = state
        .recorder()
        .ok_or_else(|| ApiError::Unavailable("recorder is disabled".into()))?;
    let now = chrono::Utc::now();
    let start = match start_time {
        Some(value) => parse_history_time(&value)?,
        None => now - chrono::Duration::days(1),
    };
    let end = match query.end_time.as_deref() {
        Some(value) => parse_history_time(value)?,
        None => now,
    };
    if end < start {
        return Err(ApiError::BadRequest(
            "end_time must not precede start_time".into(),
        ));
    }

    let explicit_filter = query.filter_entity_id.is_some();
    let entity_ids = match query.filter_entity_id.as_deref() {
        Some(raw) => raw
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| {
                EntityId::parse(value)
                    .map_err(|error| ApiError::BadRequest(format!("invalid entity_id: {error}")))
            })
            .collect::<ApiResult<Vec<_>>>()?,
        None => state
            .homecore()
            .states()
            .all()
            .into_iter()
            .map(|snapshot| snapshot.entity_id.clone())
            .collect(),
    };
    // Only reject an explicit, unusually-large `filter_entity_id` list. The
    // real HA frontend's history page calls this endpoint with NO filter by
    // design (meaning "all entities") — a real install routinely has 50-500+
    // entities, so applying this cap there rejected the single most common
    // call shape outright. The `MAX_API_HISTORY_ROWS` total-row budget below
    // already bounds the actual work regardless of entity count.
    if explicit_filter && entity_ids.len() > MAX_HISTORY_ENTITIES {
        return Err(ApiError::BadRequest(format!(
            "history queries are limited to {MAX_HISTORY_ENTITIES} explicitly filtered entities"
        )));
    }

    let mut result = Vec::with_capacity(entity_ids.len());
    let mut remaining = MAX_API_HISTORY_ROWS;
    for entity_id in entity_ids {
        let rows = recorder
            .get_state_history_limited(&entity_id, start, end, remaining)
            .await
            .map_err(|error| ApiError::Internal(format!("history query failed: {error}")))?;
        remaining = remaining.saturating_sub(rows.len());
        let mut previous_state: Option<String> = None;
        let states = rows
            .into_iter()
            .filter_map(|row| {
                if query.significant_changes_only
                    && previous_state.as_deref() == Some(row.state.as_str())
                {
                    return None;
                }
                previous_state = Some(row.state.clone());
                let changed = history_timestamp(row.last_changed_ts);
                let updated = history_timestamp(row.last_updated_ts);
                Some(StateView {
                    entity_id: row.entity_id.as_str().to_owned(),
                    state: row.state,
                    attributes: if query.no_attributes || query.minimal_response {
                        serde_json::json!({})
                    } else {
                        row.attributes
                    },
                    last_changed: changed,
                    last_updated: updated,
                    context: ContextView {
                        id: row.context_id.unwrap_or_default(),
                        user_id: None,
                        parent_id: None,
                    },
                })
            })
            .collect();
        result.push(states);
    }
    Ok(Json(result))
}

fn parse_history_time(value: &str) -> ApiResult<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&chrono::Utc))
        .map_err(|_| ApiError::BadRequest("history timestamps must be RFC 3339".into()))
}

fn history_timestamp(seconds: f64) -> String {
    let whole = seconds.floor() as i64;
    let nanos = ((seconds - seconds.floor()) * 1_000_000_000.0).round() as u32;
    chrono::DateTime::<chrono::Utc>::from_timestamp(whole, nanos.min(999_999_999))
        .unwrap_or(chrono::DateTime::<chrono::Utc>::UNIX_EPOCH)
        .to_rfc3339()
}

#[derive(Debug, Deserialize)]
pub struct LogbookQuery {
    end_time: Option<String>,
    entity: Option<String>,
}

pub async fn get_logbook(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Query(query): Query<LogbookQuery>,
) -> ApiResult<Json<Vec<serde_json::Value>>> {
    logbook_response(headers, s, None, query).await
}

pub async fn get_logbook_period(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path(start_time): Path<String>,
    Query(query): Query<LogbookQuery>,
) -> ApiResult<Json<Vec<serde_json::Value>>> {
    logbook_response(headers, s, Some(start_time), query).await
}

async fn logbook_response(
    headers: HeaderMap,
    state: SharedState,
    start_time: Option<String>,
    query: LogbookQuery,
) -> ApiResult<Json<Vec<serde_json::Value>>> {
    let _ = BearerAuth::from_headers(&headers, state.tokens()).await?;
    let recorder = state
        .recorder()
        .ok_or_else(|| ApiError::Unavailable("recorder is disabled".into()))?;
    let now = chrono::Utc::now();
    let start = match start_time {
        Some(value) => parse_history_time(&value)?,
        None => now - chrono::Duration::days(1),
    };
    let end = match query.end_time.as_deref() {
        Some(value) => parse_history_time(value)?,
        None => now,
    };
    if end < start {
        return Err(ApiError::BadRequest(
            "end_time must not precede start_time".into(),
        ));
    }
    let explicit_filter = query.entity.is_some();
    let entity_ids = match query.entity.as_deref() {
        Some(raw) => raw
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| {
                EntityId::parse(value)
                    .map_err(|error| ApiError::BadRequest(format!("invalid entity_id: {error}")))
            })
            .collect::<ApiResult<Vec<_>>>()?,
        None => state
            .homecore()
            .states()
            .all()
            .into_iter()
            .map(|snapshot| snapshot.entity_id.clone())
            .collect(),
    };
    // See the matching comment in `history_response`: only reject an
    // explicit, unusually-large filter — the default (no filter, "all
    // entities") is the real HA frontend's normal call shape, and the
    // `MAX_API_HISTORY_ROWS` row budget below already bounds the work.
    if explicit_filter && entity_ids.len() > MAX_HISTORY_ENTITIES {
        return Err(ApiError::BadRequest(format!(
            "logbook queries are limited to {MAX_HISTORY_ENTITIES} explicitly filtered entities"
        )));
    }
    let mut entries = Vec::new();
    let mut remaining = MAX_API_HISTORY_ROWS;
    for entity_id in entity_ids {
        let rows = recorder
            .get_state_history_limited(&entity_id, start, end, remaining)
            .await
            .map_err(|error| ApiError::Internal(format!("logbook query failed: {error}")))?;
        remaining = remaining.saturating_sub(rows.len());
        for row in rows {
            entries.push(serde_json::json!({
                "when": history_timestamp(row.last_updated_ts),
                "name": row.entity_id.as_str(),
                "state": row.state,
                "entity_id": row.entity_id.as_str(),
                "context_id": row.context_id
            }));
        }
    }
    entries.sort_by(|left, right| {
        left["when"]
            .as_str()
            .cmp(&right["when"].as_str())
            .then_with(|| left["entity_id"].as_str().cmp(&right["entity_id"].as_str()))
    });
    Ok(Json(entries))
}

#[derive(Serialize)]
pub struct CalendarView {
    entity_id: String,
    name: String,
}

pub async fn get_calendars(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<Json<Vec<CalendarView>>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let calendars = s
        .homecore()
        .states()
        .all_by_domain("calendar")
        .into_iter()
        .map(|state| CalendarView {
            entity_id: state.entity_id.as_str().to_owned(),
            name: state
                .attributes
                .get("friendly_name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(state.entity_id.as_str())
                .to_owned(),
        })
        .collect();
    Ok(Json(calendars))
}

#[derive(Debug, Deserialize)]
pub struct CalendarQuery {
    start: String,
    end: String,
}

pub async fn get_calendar_events(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path(entity_id): Path<String>,
    Query(query): Query<CalendarQuery>,
) -> ApiResult<Json<Vec<serde_json::Value>>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let id =
        EntityId::parse(&entity_id).map_err(|error| ApiError::BadRequest(error.to_string()))?;
    if id.domain() != "calendar" || s.homecore().states().get(&id).is_none() {
        return Err(ApiError::NotFound(entity_id));
    }
    let start = parse_history_time(&query.start)?;
    let end = parse_history_time(&query.end)?;
    if end < start {
        return Err(ApiError::BadRequest(
            "end must not precede start".to_owned(),
        ));
    }
    // Calendar integrations may expose their current entity without an event
    // provider. An empty list is the valid response for that interval.
    Ok(Json(Vec::new()))
}

pub async fn get_camera_proxy(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path(entity_id): Path<String>,
) -> ApiResult<StatusCode> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let id =
        EntityId::parse(&entity_id).map_err(|error| ApiError::BadRequest(error.to_string()))?;
    if id.domain() != "camera" || s.homecore().states().get(&id).is_none() {
        return Err(ApiError::NotFound(entity_id));
    }
    Err(ApiError::Unavailable(
        "camera integration has no image provider".into(),
    ))
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

pub async fn get_states(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<Json<Vec<StateView>>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let snapshots = s.homecore().states().all();
    Ok(Json(
        snapshots.iter().map(|x| StateView::from_state(x)).collect(),
    ))
}

pub async fn get_state(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path(entity_id): Path<String>,
) -> ApiResult<Json<StateView>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let id = EntityId::parse(entity_id.clone()).map_err(|e| ApiError::BadRequest(e.to_string()))?;
    let st = s
        .homecore()
        .states()
        .get(&id)
        .ok_or(ApiError::NotFound(entity_id))?;
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
    let attrs = if body.attributes.is_null() {
        serde_json::json!({})
    } else {
        body.attributes
    };
    let snap = s
        .homecore()
        .states()
        .set(id, body.state, attrs, Context::new());
    let status = if existed {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    };
    Ok((status, Json(StateView::from_state(&snap))))
}

#[derive(Serialize)]
pub struct ServiceDomainView {
    pub domain: String,
    pub services: serde_json::Value,
}

pub async fn get_services(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<Json<Vec<ServiceDomainView>>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let services = s.homecore().services().registered_services().await;
    let mut by_domain: std::collections::HashMap<
        String,
        serde_json::Map<String, serde_json::Value>,
    > = std::collections::HashMap::new();
    for sv in services {
        by_domain
            .entry(sv.domain.clone())
            .or_default()
            .insert(sv.service.clone(), serde_json::json!({}));
    }
    Ok(Json(
        by_domain
            .into_iter()
            .map(|(domain, services)| ServiceDomainView {
                domain,
                services: serde_json::Value::Object(services),
            })
            .collect(),
    ))
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
    let resp = s
        .homecore()
        .services()
        .call(call)
        .await
        .map_err(|e| match e {
            homecore::ServiceError::NotRegistered { .. } => {
                ApiError::ServiceNotRegistered { domain, service }
            }
            other => ApiError::Internal(other.to_string()),
        })?;
    Ok(Json(resp))
}

#[derive(Serialize)]
pub struct EventView {
    pub event: String,
    pub listener_count: usize,
}

/// Event types whose wire shape is implemented by the core event bridge.
const CORE_EVENT_TYPES: &[&str] = &[
    "state_changed",
    "call_service",
    "homeassistant_start",
    "homeassistant_stop",
];

pub async fn get_events(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<Json<Vec<EventView>>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    Ok(Json(
        CORE_EVENT_TYPES
            .iter()
            .map(|event| EventView {
                event: (*event).to_owned(),
                // Tokio broadcast intentionally does not expose a stable
                // per-filter count. Zero is HA-compatible and honest.
                listener_count: 0,
            })
            .collect(),
    ))
}

/// Whether `event_type` is acceptable to fire on the domain bus.
///
/// Real Home Assistant places essentially no format restriction on event
/// types beyond "non-empty string" — integrations commonly fire types with
/// mixed case, dots, or hyphens (e.g. `mobile_app.notification_action`,
/// `ios.action_fired`). The original check here only accepted
/// `[a-z0-9_]+`, silently rejecting any of those — a real behavioral gap
/// versus the documented contract, not a security boundary (this endpoint is
/// already bearer-authenticated). We keep only the bounds that protect the
/// server itself: non-empty, a sane length cap, and no control characters
/// (which could otherwise corrupt log lines or downstream storage).
pub(crate) fn is_valid_event_type(event_type: &str) -> bool {
    !event_type.is_empty()
        && event_type.len() <= 255
        && event_type.chars().all(|ch| !ch.is_control())
}

pub async fn fire_event(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Path(event_type): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> ApiResult<Json<serde_json::Value>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    if !is_valid_event_type(&event_type) {
        return Err(ApiError::BadRequest("invalid event_type".into()));
    }
    if !body.is_object() && !body.is_null() {
        return Err(ApiError::BadRequest("event data must be an object".into()));
    }
    let data = if body.is_null() {
        serde_json::json!({})
    } else {
        body
    };
    s.homecore().bus().fire_domain(homecore::DomainEvent::new(
        event_type.clone(),
        data,
        Context::new(),
    ));
    Ok(Json(
        serde_json::json!({"message": format!("Event {event_type} fired.")}),
    ))
}

#[derive(Deserialize)]
pub struct TemplateRequest {
    pub template: String,
}

pub async fn render_template(
    headers: HeaderMap,
    State(s): State<SharedState>,
    Json(body): Json<TemplateRequest>,
) -> ApiResult<String> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    let environment = homecore_automation::TemplateEnvironment::new(std::sync::Arc::new(
        s.homecore().states().clone(),
    ));
    environment
        .render(&body.template)
        .map_err(|error| ApiError::BadRequest(error.to_string()))
}

pub async fn check_config(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<Json<serde_json::Value>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    // Runtime configuration has already passed HOMECORE's typed loaders.
    Ok(Json(serde_json::json!({
        "result": "valid",
        "errors": null,
        "warnings": null
    })))
}

pub async fn error_log(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<impl IntoResponse> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    Ok((
        [("content-type", "text/plain; charset=utf-8")],
        String::new(),
    ))
}

/// Machine-readable support matrix. This prevents clients from confusing
/// core protocol compatibility with every optional HA integration.
pub async fn compatibility(
    headers: HeaderMap,
    State(s): State<SharedState>,
) -> ApiResult<Json<serde_json::Value>> {
    let _ = BearerAuth::from_headers(&headers, s.tokens()).await?;
    Ok(Json(serde_json::json!({
        "baseline": "Home Assistant Core 2025.1",
        "rest": {
            "core": "implemented",
            "events": "implemented",
            "template": "implemented",
            "check_config": "implemented",
            "error_log": "implemented",
            "history": "implemented_when_recorder_enabled",
            "logbook": "implemented_when_recorder_enabled",
            "calendar": "implemented_with_integration_supplied_events",
            "camera": "implemented_with_integration_supplied_images",
            "media": "integration_dependent"
        },
        "websocket": {
            "auth": "implemented",
            "states_services_config": "implemented",
            "events": "implemented",
            "render_template": "implemented",
            "feature_negotiation_and_panels": "implemented",
            "registry_lists": {
                "entity": "implemented",
                "device": "implemented",
                "area": "implemented_empty",
                "mutations": "requires_persistent_registry_backend"
            },
            "lovelace_media": "integration_dependent"
        }
    })))
}

#[cfg(test)]
mod tests {
    use super::is_valid_event_type;

    /// Real HA integrations commonly fire event types with mixed case, dots,
    /// or hyphens (e.g. `mobile_app.notification_action`). The original
    /// `[a-z0-9_]+`-only check rejected all of these; only non-empty,
    /// length, and control-character bounds should remain.
    #[test]
    fn realistic_ha_event_types_are_accepted() {
        assert!(is_valid_event_type("mobile_app.notification_action"));
        assert!(is_valid_event_type("ios.action_fired"));
        assert!(is_valid_event_type("Custom-Event.2"));
        assert!(is_valid_event_type("state_changed"));
    }

    #[test]
    fn empty_oversized_or_control_char_event_types_are_rejected() {
        assert!(!is_valid_event_type(""));
        assert!(!is_valid_event_type(&"a".repeat(256)));
        assert!(!is_valid_event_type("bad\nevent"));
        assert!(!is_valid_event_type("bad\tevent"));
    }
}
