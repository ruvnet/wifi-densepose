//! WebSocket handler — `/api/websocket`. ADR-130 §2.2 P2 command subset.
//!
//! Protocol mirrors HA's WS API:
//!   server → `{"type":"auth_required","ha_version":"<v>"}`
//!   client → `{"type":"auth","access_token":"<token>"}`
//!   server → `{"type":"auth_ok","ha_version":"<v>"}`
//!   client → `{"id":1,"type":"get_states"}`
//!   server → `{"id":1,"type":"result","success":true,"result":[...]}`
//!
//! `ha_version` is the homecore version string — see ADR-130 Q1 for the
//! companion-app feature-detect concern.
//!
//! ## Security (ADR-161)
//!
//! The `auth` token is validated against [`crate::tokens::LongLivedTokenStore`]
//! via `state.tokens().is_valid()` — the *same* store the REST path uses
//! (`auth::BearerAuth`). A wrong token receives `auth_invalid` and the socket
//! is closed. (HC-WS-01 closed the prior bypass where any non-empty token was
//! accepted.) Command replies are transmitted by a dedicated writer task that
//! drains the response channel onto the socket (HC-WS-02 closed the prior
//! reply-theater where responses were logged and discarded).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::warn;

use homecore::{Context, ServiceCall, ServiceName, SystemEvent};

use crate::rest::StateView;
use crate::state::SharedState;

/// WebSocket upgrade entry point. Mounted on `/api/websocket`.
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: SharedState) {
    // Phase 1 — auth handshake.
    let auth_req = serde_json::json!({
        "type": "auth_required",
        "ha_version": state.version(),
    });
    if socket.send(Message::Text(auth_req.to_string())).await.is_err() {
        return;
    }

    let token = match socket.recv().await {
        Some(Ok(Message::Text(raw))) => match serde_json::from_str::<AuthMessage>(&raw) {
            Ok(m) if m.kind == "auth" => m.access_token,
            _ => {
                let _ = socket
                    .send(Message::Text(
                        serde_json::json!({"type":"auth_invalid","message":"expected auth"}).to_string(),
                    ))
                    .await;
                return;
            }
        },
        _ => return,
    };

    // Validate the bearer token against the same store the REST path
    // uses (`state.tokens().is_valid()` — see `rest.rs` /
    // `auth::BearerAuth`). Before the HC-WS-01 fix this checked only
    // `token.trim().is_empty()` and accepted ANY non-empty token even
    // with a provisioned `HOMECORE_TOKENS` whitelist — a full WS auth
    // bypass. `is_valid()` rejects the empty token internally and, in
    // DEV (`allow_any`) mode, still accepts any non-empty bearer (with
    // a warn) so smoke tests keep working.
    if !state.tokens().is_valid(&token).await {
        let _ = socket
            .send(Message::Text(
                serde_json::json!({"type":"auth_invalid","message":"invalid token"}).to_string(),
            ))
            .await;
        return;
    }
    let auth_ok = serde_json::json!({"type":"auth_ok","ha_version": state.version()});
    if socket.send(Message::Text(auth_ok.to_string())).await.is_err() {
        return;
    }

    // Phase 2 — command loop.
    let conn = Connection::new(state.clone());
    conn.run(socket).await;
}

#[derive(Deserialize)]
struct AuthMessage {
    #[serde(rename = "type")]
    kind: String,
    access_token: String,
}

#[derive(Deserialize)]
struct WsCommand {
    id: u64,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    event_type: Option<String>,
    #[serde(default)]
    subscription: Option<u64>,
    #[serde(default)]
    entity_id: Option<String>,
    #[serde(default)]
    domain: Option<String>,
    #[serde(default)]
    service: Option<String>,
    #[serde(default)]
    service_data: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ResultMessage<'a> {
    id: u64,
    #[serde(rename = "type")]
    kind: &'static str,
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorView<'a>>,
}

#[derive(Serialize)]
struct ErrorView<'a> {
    code: &'static str,
    message: &'a str,
}

struct Connection {
    state: SharedState,
    next_sub_id: AtomicU64,
    subs: Arc<dashmap::DashMap<u64, SubscriptionHandle>>,
}

struct SubscriptionHandle {
    abort: tokio::task::AbortHandle,
}

impl Connection {
    fn new(state: SharedState) -> Self {
        Self {
            state,
            next_sub_id: AtomicU64::new(1),
            subs: Arc::new(dashmap::DashMap::new()),
        }
    }

    async fn run(self, socket: WebSocket) {
        use futures_util::{SinkExt, StreamExt};

        let conn = Arc::new(self);
        // Split the socket so a dedicated writer task can drain `rx` onto
        // the wire while the reader task processes commands concurrently.
        // Before the HC-WS-02 fix the socket was moved into a recv-only
        // task and the only `rx` consumer just `debug!`-logged and
        // DISCARDED every message — so no `result`/`pong`/`event` ever
        // reached the client. Now `rx` feeds `socket.send`.
        let (mut sink, mut stream) = socket.split();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        // Writer task: drain replies onto the socket. A `__pong:<n>`
        // sentinel maps to a binary Pong control frame; everything else
        // is a JSON text frame.
        let writer_task = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let send_result = if let Some(n) = msg.strip_prefix("__pong:") {
                    let len: usize = n.parse().unwrap_or(0);
                    sink.send(Message::Pong(vec![0u8; len])).await
                } else {
                    sink.send(Message::Text(msg)).await
                };
                if send_result.is_err() {
                    break;
                }
            }
        });

        // Reader task: parse and dispatch commands; responses are pushed
        // into `tx` and transmitted by the writer task above.
        let reader_tx = tx.clone();
        {
            let conn = Arc::clone(&conn);
            while let Some(frame) = stream.next().await {
                match frame {
                    Ok(Message::Text(raw)) => {
                        let cmd: WsCommand = match serde_json::from_str(&raw) {
                            Ok(c) => c,
                            Err(e) => {
                                warn!("bad ws command: {e}");
                                continue;
                            }
                        };
                        conn.handle_cmd(cmd, &reader_tx).await;
                    }
                    Ok(Message::Ping(p)) => {
                        let _ = reader_tx.send(format!("__pong:{}", p.len()));
                    }
                    Ok(Message::Close(_)) | Err(_) => break,
                    _ => {}
                }
            }
            // Cancel all subscriptions on disconnect.
            for entry in conn.subs.iter() {
                entry.value().abort.abort();
            }
        }

        // Reader loop ended → drop the senders so the writer task's `rx`
        // closes and the task exits cleanly.
        drop(tx);
        drop(reader_tx);
        let _ = writer_task.await;
    }

    async fn handle_cmd(&self, cmd: WsCommand, tx: &tokio::sync::mpsc::UnboundedSender<String>) {
        match cmd.kind.as_str() {
            "ping" => {
                let msg = serde_json::json!({"id": cmd.id, "type": "pong"});
                let _ = tx.send(msg.to_string());
            }
            "get_states" => {
                let snapshots = self.state.homecore().states().all();
                let views: Vec<StateView> = snapshots.iter().map(|s| StateView::from_state(s)).collect();
                self.ack(tx, cmd.id, true, Some(serde_json::to_value(views).unwrap()));
            }
            "get_config" => {
                let payload = serde_json::json!({
                    "location_name": self.state.location_name(),
                    "version": self.state.version(),
                    "state": "RUNNING",
                });
                self.ack(tx, cmd.id, true, Some(payload));
            }
            "get_services" => {
                let services = self.state.homecore().services().registered_services().await;
                let mut by_domain: std::collections::HashMap<String, serde_json::Map<String, serde_json::Value>> =
                    std::collections::HashMap::new();
                for s in services {
                    by_domain.entry(s.domain).or_default().insert(s.service, serde_json::json!({}));
                }
                let payload = serde_json::to_value(by_domain).unwrap();
                self.ack(tx, cmd.id, true, Some(payload));
            }
            "call_service" => {
                let (Some(domain), Some(service)) = (cmd.domain.clone(), cmd.service.clone()) else {
                    self.err(tx, cmd.id, "missing_domain_service", "domain and service are required");
                    return;
                };
                let call = ServiceCall {
                    name: ServiceName::new(domain.clone(), service.clone()),
                    data: cmd.service_data.unwrap_or(serde_json::json!({})),
                    context: Context::new(),
                };
                match self.state.homecore().services().call(call).await {
                    Ok(v) => self.ack(tx, cmd.id, true, Some(v)),
                    Err(e) => self.err(tx, cmd.id, "service_error", &e.to_string()),
                }
            }
            "subscribe_events" => {
                let sub_id = self.next_sub_id.fetch_add(1, Ordering::Relaxed);
                let filter = cmd.event_type.clone();
                let tx_clone = tx.clone();
                let mut domain_rx = self.state.homecore().bus().subscribe_domain();
                let mut system_rx = self.state.homecore().bus().subscribe_system();
                let task = tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            evt = system_rx.recv() => match evt {
                                Ok(SystemEvent::StateChanged(sc)) => {
                                    if filter.as_deref() == Some("state_changed") || filter.is_none() {
                                        let payload = serde_json::json!({
                                            "id": sub_id,
                                            "type": "event",
                                            "event": {
                                                "event_type": "state_changed",
                                                "data": {
                                                    "entity_id": sc.entity_id.as_str(),
                                                    "old_state": sc.old_state.as_ref().map(|s| StateView::from_state(s)),
                                                    "new_state": sc.new_state.as_ref().map(|s| StateView::from_state(s)),
                                                },
                                                "origin": "LOCAL",
                                                "time_fired": sc.fired_at.to_rfc3339(),
                                            }
                                        });
                                        if tx_clone.send(payload.to_string()).is_err() { break; }
                                    }
                                }
                                Ok(_) => {}
                                // A slow consumer that falls >4,096 events behind
                                // gets `Lagged(n)`, which is RECOVERABLE: the bus
                                // doc (`bus.rs` §"Lagged receivers must re-sync")
                                // and HA's WS contract both keep the subscription
                                // alive across a lag. The pre-fix `Err(_) => break`
                                // treated `Lagged` as fatal, silently killing the
                                // client's event stream on a burst (HC-WS-LAG-01).
                                // Skip the dropped window and continue; only a
                                // `Closed` sender ends the task.
                                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                                Err(broadcast::error::RecvError::Closed) => break,
                            },
                            evt = domain_rx.recv() => match evt {
                                Ok(de) => {
                                    if filter.as_deref() == Some(de.event_type.as_str()) || filter.is_none() {
                                        let payload = serde_json::json!({
                                            "id": sub_id,
                                            "type": "event",
                                            "event": {
                                                "event_type": de.event_type,
                                                "data": de.event_data,
                                                "origin": format!("{:?}", de.origin).to_uppercase(),
                                                "time_fired": de.fired_at.to_rfc3339(),
                                            }
                                        });
                                        if tx_clone.send(payload.to_string()).is_err() { break; }
                                    }
                                }
                                // Same recoverable-lag handling as the system arm
                                // above (HC-WS-LAG-01): a lagged domain-event
                                // receiver re-syncs and continues; only `Closed`
                                // terminates the subscription.
                                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                                Err(broadcast::error::RecvError::Closed) => break,
                            }
                        }
                    }
                });
                self.subs.insert(
                    sub_id,
                    SubscriptionHandle {
                        abort: task.abort_handle(),
                    },
                );
                self.ack(tx, cmd.id, true, None);
            }
            "unsubscribe_events" => {
                if let Some(sub_id) = cmd.subscription {
                    if let Some((_, handle)) = self.subs.remove(&sub_id) {
                        handle.abort.abort();
                        self.ack(tx, cmd.id, true, None);
                    } else {
                        self.err(tx, cmd.id, "not_found", "subscription_id not found");
                    }
                } else {
                    self.err(tx, cmd.id, "missing_subscription", "subscription is required");
                }
            }
            other => {
                self.err(tx, cmd.id, "unknown_command", &format!("unknown ws command: {other}"));
            }
        }
        // entity_id is reserved for future per-entity subscribes
        let _ = cmd.entity_id;
    }

    fn ack(
        &self,
        tx: &tokio::sync::mpsc::UnboundedSender<String>,
        id: u64,
        success: bool,
        result: Option<serde_json::Value>,
    ) {
        let msg = ResultMessage {
            id,
            kind: "result",
            success,
            result,
            error: None,
        };
        let _ = tx.send(serde_json::to_string(&msg).unwrap());
    }

    fn err(&self, tx: &tokio::sync::mpsc::UnboundedSender<String>, id: u64, code: &'static str, message: &str) {
        let msg = ResultMessage {
            id,
            kind: "result",
            success: false,
            result: None,
            error: Some(ErrorView { code, message }),
        };
        let _ = tx.send(serde_json::to_string(&msg).unwrap());
    }
}

// Suppress unused warnings for placeholder broadcast type
#[allow(dead_code)]
type _UnusedSubBroadcast = broadcast::Sender<()>;
