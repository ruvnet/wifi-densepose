//! Bounded async TCP/HTTP foundation for HAP.
//!
//! The listener parses plaintext pairing HTTP and has authenticated accessory
//! endpoint handlers ready for a future encrypted transport. Because Pair-
//! Setup, Pair-Verify, and HAP frame encryption are not complete, no network
//! request can currently transition a session to authenticated.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use httparse::Status;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, Semaphore};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{timeout, Instant};

use crate::accessory::{HapAccessoryType, HapCharacteristic, HapCharacteristicValue};
use crate::bridge::{CharacteristicEvent, ExposedAccessory, HapBridge};
use crate::error::HapError;
use crate::mdns::MdnsAdvertiser;
use crate::pairing::PairingStore;
use crate::protocol::pairing_unavailable_response;
use crate::session::Session;

const HAP_JSON: &str = "application/hap+json";
const HAP_TLV: &str = "application/pairing+tlv8";

/// Resource bounds for the HAP listener.
#[derive(Debug, Clone)]
pub struct HapServerConfig {
    pub bind_addr: SocketAddr,
    pub max_connections: usize,
    pub max_header_bytes: usize,
    pub max_body_bytes: usize,
    pub request_timeout: Duration,
    pub shutdown_timeout: Duration,
}

impl Default for HapServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: SocketAddr::from(([0, 0, 0, 0], 51826)),
            max_connections: 32,
            max_header_bytes: 16 * 1024,
            max_body_bytes: 64 * 1024,
            request_timeout: Duration::from_secs(10),
            shutdown_timeout: Duration::from_secs(5),
        }
    }
}

impl HapServerConfig {
    fn validate(&self) -> Result<(), HapError> {
        if self.max_connections == 0
            || self.max_header_bytes < 512
            || self.max_body_bytes == 0
            || self.request_timeout.is_zero()
            || self.shutdown_timeout.is_zero()
        {
            return Err(HapError::Server(
                "invalid zero or undersized server limit".into(),
            ));
        }
        Ok(())
    }
}

/// Running server handle. Dropping it aborts the listener; [`shutdown`] also
/// retracts mDNS and waits for connection tasks within the configured bound.
pub struct HapServerHandle {
    local_addr: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<Result<(), HapError>>>,
    shutdown_timeout: Duration,
}

impl HapServerHandle {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub async fn shutdown(mut self) -> Result<(), HapError> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        let Some(mut task) = self.task.take() else {
            return Ok(());
        };
        match timeout(self.shutdown_timeout, &mut task).await {
            Ok(result) => {
                result.map_err(|error| HapError::Server(format!("server task failed: {error}")))?
            }
            Err(_) => {
                task.abort();
                let _ = task.await;
                Err(HapError::Server(
                    "server shutdown timed out; task aborted".into(),
                ))
            }
        }
    }
}

impl Drop for HapServerHandle {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

/// Start the bounded listener and advertise the actual bound port.
pub async fn start_server(
    config: HapServerConfig,
    bridge: HapBridge,
    pairings: Arc<PairingStore>,
    advertiser: Arc<dyn MdnsAdvertiser>,
) -> Result<HapServerHandle, HapError> {
    config.validate()?;
    let listener = TcpListener::bind(config.bind_addr)
        .await
        .map_err(|error| HapError::Server(format!("bind {}: {error}", config.bind_addr)))?;
    let local_addr = listener
        .local_addr()
        .map_err(|error| HapError::Server(format!("read local address: {error}")))?;

    let mut record = bridge.service_record.clone();
    record.port = local_addr.port();
    record.paired = pairings.is_paired()?;
    advertiser.advertise(&record).await?;

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task_config = config.clone();
    let task = tokio::spawn(run_listener(
        listener,
        task_config,
        bridge,
        pairings,
        advertiser,
        record.instance_name,
        shutdown_rx,
    ));
    Ok(HapServerHandle {
        local_addr,
        shutdown: Some(shutdown_tx),
        task: Some(task),
        // The listener owns the configured drain window; the handle allows a
        // small scheduling/retraction margin before enforcing its outer abort.
        shutdown_timeout: config
            .shutdown_timeout
            .saturating_add(Duration::from_secs(1)),
    })
}

async fn run_listener(
    listener: TcpListener,
    config: HapServerConfig,
    bridge: HapBridge,
    pairings: Arc<PairingStore>,
    advertiser: Arc<dyn MdnsAdvertiser>,
    instance_name: String,
    mut shutdown: oneshot::Receiver<()>,
) -> Result<(), HapError> {
    let permits = Arc::new(Semaphore::new(config.max_connections));
    let mut connections = JoinSet::new();

    loop {
        let permit = tokio::select! {
            _ = &mut shutdown => break,
            permit = permits.clone().acquire_owned() => {
                permit.map_err(|_| HapError::Server("connection semaphore closed".into()))?
            }
        };
        let accepted = tokio::select! {
            _ = &mut shutdown => {
                drop(permit);
                break;
            }
            accepted = listener.accept() => accepted
        };
        match accepted {
            Ok((stream, peer)) => {
                let bridge = bridge.clone();
                let pairings = pairings.clone();
                let limits = config.clone();
                connections.spawn(async move {
                    let _permit = permit;
                    if let Err(error) =
                        serve_connection(stream, peer, limits, bridge, pairings).await
                    {
                        tracing::debug!(%peer, %error, "HAP connection closed");
                    }
                });
            }
            Err(error) => {
                tracing::warn!(%error, "HAP accept failed");
            }
        }
        while connections.try_join_next().is_some() {}
    }

    drop(listener);
    let deadline = Instant::now() + config.shutdown_timeout;
    while !connections.is_empty() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() || timeout(remaining, connections.join_next()).await.is_err() {
            connections.abort_all();
            while connections.join_next().await.is_some() {}
            break;
        }
    }
    advertiser.retract(&instance_name).await
}

async fn serve_connection(
    mut stream: TcpStream,
    _peer: SocketAddr,
    config: HapServerConfig,
    bridge: HapBridge,
    pairings: Arc<PairingStore>,
) -> Result<(), HapError> {
    let mut buffer = ConnectionBuffer::default();
    let mut session = Session::new();
    let mut subscriptions = HashSet::new();
    let mut events = bridge.subscribe_events();

    loop {
        tokio::select! {
            request = timeout(
                config.request_timeout,
                read_request(&mut stream, &mut buffer, &config),
            ) => {
                let request = match request {
                    Ok(Ok(Some(request))) => request,
                    Ok(Ok(None)) => break,
                    Ok(Err(error)) => {
                        let response = error_response(&error);
                        write_response(&mut stream, response).await?;
                        break;
                    }
                    Err(_) => {
                        write_response(&mut stream, Response::plain(408, b"request timeout".to_vec())).await?;
                        break;
                    }
                };
                let close = request.connection_close;
                let response = dispatch_request(
                    request,
                    &mut session,
                    &bridge,
                    &pairings,
                    &mut subscriptions,
                );
                write_response(&mut stream, response).await?;
                if close {
                    break;
                }
            }
            event = events.recv(), if session.state().is_authenticated() && !subscriptions.is_empty() => {
                match event {
                    Ok(event) => {
                        if let Some(payload) = event_payload(&bridge, &event, &subscriptions) {
                            write_event(&mut stream, payload).await?;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        // A lagged controller must resynchronize through GET /characteristics.
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
    session.close();
    Ok(())
}

#[derive(Debug)]
struct Request {
    method: String,
    target: String,
    body: Vec<u8>,
    connection_close: bool,
}

#[derive(Default)]
struct ConnectionBuffer {
    bytes: Vec<u8>,
}

async fn read_request(
    stream: &mut TcpStream,
    buffer: &mut ConnectionBuffer,
    config: &HapServerConfig,
) -> Result<Option<Request>, RequestReadError> {
    let header_end = loop {
        if let Some(position) = find_header_end(&buffer.bytes) {
            break position + 4;
        }
        if buffer.bytes.len() >= config.max_header_bytes {
            return Err(RequestReadError::HeadersTooLarge);
        }
        let mut chunk = [0u8; 2048];
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(RequestReadError::Io)?;
        if read == 0 {
            return if buffer.bytes.is_empty() {
                Ok(None)
            } else {
                Err(RequestReadError::Malformed("truncated HTTP headers"))
            };
        }
        buffer.bytes.extend_from_slice(&chunk[..read]);
    };
    if header_end > config.max_header_bytes {
        return Err(RequestReadError::HeadersTooLarge);
    }

    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut parsed = httparse::Request::new(&mut headers);
    match parsed.parse(&buffer.bytes[..header_end]) {
        Ok(Status::Complete(_)) => {}
        Ok(Status::Partial) => return Err(RequestReadError::Malformed("partial HTTP request")),
        Err(error) => return Err(RequestReadError::MalformedOwned(error.to_string())),
    }
    if parsed.version != Some(1) {
        return Err(RequestReadError::Malformed("HTTP/1.1 required"));
    }
    let method = parsed
        .method
        .ok_or(RequestReadError::Malformed("missing method"))?
        .to_owned();
    let target = parsed
        .path
        .ok_or(RequestReadError::Malformed("missing request target"))?
        .to_owned();
    if target.len() > 2048 || !target.starts_with('/') {
        return Err(RequestReadError::Malformed("invalid request target"));
    }

    let mut content_length = None;
    let mut connection_close = false;
    for header in parsed.headers.iter() {
        if header.name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(RequestReadError::Malformed(
                "Transfer-Encoding is unsupported",
            ));
        }
        if header.name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some() {
                return Err(RequestReadError::Malformed("duplicate Content-Length"));
            }
            let value = std::str::from_utf8(header.value)
                .map_err(|_| RequestReadError::Malformed("non-UTF8 Content-Length"))?;
            content_length = Some(
                value
                    .parse::<usize>()
                    .map_err(|_| RequestReadError::Malformed("invalid Content-Length"))?,
            );
        }
        if header.name.eq_ignore_ascii_case("connection")
            && header.value.eq_ignore_ascii_case(b"close")
        {
            connection_close = true;
        }
    }
    let content_length = content_length.unwrap_or(0);
    if content_length > config.max_body_bytes {
        return Err(RequestReadError::BodyTooLarge);
    }
    let request_end = header_end
        .checked_add(content_length)
        .ok_or(RequestReadError::BodyTooLarge)?;
    while buffer.bytes.len() < request_end {
        let remaining = request_end - buffer.bytes.len();
        let mut chunk = vec![0u8; remaining.min(8192)];
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(RequestReadError::Io)?;
        if read == 0 {
            return Err(RequestReadError::Malformed("truncated HTTP body"));
        }
        buffer.bytes.extend_from_slice(&chunk[..read]);
    }
    let body = buffer.bytes[header_end..request_end].to_vec();
    buffer.bytes.drain(..request_end);
    Ok(Some(Request {
        method,
        target,
        body,
        connection_close,
    }))
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

#[derive(Debug)]
enum RequestReadError {
    Io(std::io::Error),
    Malformed(&'static str),
    MalformedOwned(String),
    HeadersTooLarge,
    BodyTooLarge,
}

impl std::fmt::Display for RequestReadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Malformed(message) => formatter.write_str(message),
            Self::MalformedOwned(message) => formatter.write_str(message),
            Self::HeadersTooLarge => formatter.write_str("HTTP headers too large"),
            Self::BodyTooLarge => formatter.write_str("HTTP body too large"),
        }
    }
}

fn error_response(error: &RequestReadError) -> Response {
    match error {
        RequestReadError::HeadersTooLarge => Response::plain(431, error.to_string().into_bytes()),
        RequestReadError::BodyTooLarge => Response::plain(413, error.to_string().into_bytes()),
        _ => Response::plain(400, error.to_string().into_bytes()),
    }
}

struct Response {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

impl Response {
    fn plain(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            content_type: "text/plain; charset=utf-8",
            body,
        }
    }

    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            content_type: HAP_JSON,
            body: serde_json::to_vec(&value).expect("JSON value serialization cannot fail"),
        }
    }
}

fn dispatch_request(
    request: Request,
    session: &mut Session,
    bridge: &HapBridge,
    pairings: &PairingStore,
    subscriptions: &mut HashSet<(u64, u64)>,
) -> Response {
    match (
        request.method.as_str(),
        request.target.split('?').next().unwrap_or(""),
    ) {
        ("POST", "/pair-setup") => {
            let _ = session.begin_pair_setup(pairings.is_paired().unwrap_or(true));
            let response = pairing_unavailable_response(&request.body);
            let _ = session.reset_pairing();
            match response {
                Ok(body) => Response {
                    status: 200,
                    content_type: HAP_TLV,
                    body,
                },
                Err(error) => Response::plain(400, error.to_string().into_bytes()),
            }
        }
        ("POST", "/pair-verify") => {
            let _ = session.begin_pair_verify();
            let response = pairing_unavailable_response(&request.body);
            let _ = session.reset_pairing();
            match response {
                Ok(body) => Response {
                    status: 200,
                    content_type: HAP_TLV,
                    body,
                },
                Err(error) => Response::plain(400, error.to_string().into_bytes()),
            }
        }
        _ if !session.state().is_authenticated() => Response::json(
            470,
            json!({"status": -70401, "message": "Connection Authorization Required"}),
        ),
        ("GET", "/accessories") => Response::json(200, accessories_json(bridge)),
        ("GET", "/characteristics") => characteristics_response(&request.target, bridge),
        ("PUT", "/characteristics") => {
            characteristic_subscription_response(&request.body, subscriptions)
        }
        ("POST", "/pairings") => Response::json(
            501,
            json!({"status": -70406, "message": "Pairings management awaits encrypted transport"}),
        ),
        _ => Response::plain(404, b"not found".to_vec()),
    }
}

fn accessories_json(bridge: &HapBridge) -> Value {
    let accessories = indexed_accessories(bridge);
    let mut output = vec![json!({
        "aid": 1,
        "services": [accessory_information(1, "HOMECORE Bridge")]
    })];
    for (aid, accessory) in accessories {
        let mut characteristics = Vec::new();
        for (index, (kind, value)) in accessory.mapping.characteristics.iter().enumerate() {
            characteristics.push(json!({
                "iid": 8 + index as u64,
                "type": characteristic_type(*kind),
                "perms": ["pr", "ev"],
                "format": characteristic_format(value),
                "value": characteristic_value(value),
            }));
        }
        output.push(json!({
            "aid": aid,
            "services": [
                accessory_information(1, accessory.entity_id.as_str()),
                {
                    "iid": 7,
                    "type": service_type(accessory.accessory_type),
                    "primary": true,
                    "characteristics": characteristics,
                }
            ]
        }));
    }
    json!({ "accessories": output })
}

fn accessory_information(iid: u64, name: &str) -> Value {
    json!({
        "iid": iid,
        "type": "3E",
        "characteristics": [
            {"iid": iid + 1, "type": "23", "perms": ["pr"], "format": "string", "value": name},
            {"iid": iid + 2, "type": "20", "perms": ["pr"], "format": "string", "value": "HOMECORE"},
            {"iid": iid + 3, "type": "21", "perms": ["pr"], "format": "string", "value": "HOMECORE HAP Bridge"},
            {"iid": iid + 4, "type": "30", "perms": ["pr"], "format": "string", "value": name},
            {"iid": iid + 5, "type": "52", "perms": ["pr"], "format": "string", "value": env!("CARGO_PKG_VERSION")}
        ]
    })
}

fn indexed_accessories(bridge: &HapBridge) -> Vec<(u64, ExposedAccessory)> {
    let mut accessories = bridge.running_accessories();
    accessories.sort_by(|left, right| left.entity_id.as_str().cmp(right.entity_id.as_str()));
    accessories
        .into_iter()
        .enumerate()
        .map(|(index, accessory)| (index as u64 + 2, accessory))
        .collect()
}

fn characteristics_response(target: &str, bridge: &HapBridge) -> Response {
    let Some(query) = target.split_once('?').map(|(_, query)| query) else {
        return Response::plain(400, b"missing characteristic query".to_vec());
    };
    let Some(ids) = query.split('&').find_map(|part| part.strip_prefix("id=")) else {
        return Response::plain(400, b"missing id query".to_vec());
    };
    if ids.len() > 4096 || ids.split(',').count() > 128 {
        return Response::plain(400, b"characteristic query too large".to_vec());
    }
    let accessories = indexed_accessories(bridge);
    let mut values = Vec::new();
    for id in ids.split(',') {
        let Some((aid, iid)) = parse_aid_iid(id) else {
            return Response::plain(400, b"invalid aid.iid".to_vec());
        };
        let value = accessories
            .iter()
            .find(|(candidate, _)| *candidate == aid)
            .and_then(|(_, accessory)| {
                accessory
                    .mapping
                    .characteristics
                    .get(iid.saturating_sub(8) as usize)
            })
            .map(|(_, value)| characteristic_value(value));
        values.push(match value {
            Some(value) => json!({"aid": aid, "iid": iid, "value": value}),
            None => json!({"aid": aid, "iid": iid, "status": -70409}),
        });
    }
    Response::json(207, json!({"characteristics": values}))
}

fn characteristic_subscription_response(
    body: &[u8],
    subscriptions: &mut HashSet<(u64, u64)>,
) -> Response {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return Response::plain(400, b"invalid characteristic JSON".to_vec());
    };
    let Some(items) = value.get("characteristics").and_then(Value::as_array) else {
        return Response::plain(400, b"missing characteristics array".to_vec());
    };
    if items.len() > 128 {
        return Response::plain(400, b"too many characteristic writes".to_vec());
    }
    for item in items {
        let (Some(aid), Some(iid), Some(enabled)) = (
            item.get("aid").and_then(Value::as_u64),
            item.get("iid").and_then(Value::as_u64),
            item.get("ev").and_then(Value::as_bool),
        ) else {
            // Entity writes are not yet connected to HOMECORE service calls.
            return Response::json(207, json!({"characteristics": [{"status": -70405}]}));
        };
        if enabled {
            subscriptions.insert((aid, iid));
        } else {
            subscriptions.remove(&(aid, iid));
        }
    }
    Response {
        status: 204,
        content_type: HAP_JSON,
        body: Vec::new(),
    }
}

fn event_payload(
    bridge: &HapBridge,
    event: &CharacteristicEvent,
    subscriptions: &HashSet<(u64, u64)>,
) -> Option<Vec<u8>> {
    let (aid, _) = indexed_accessories(bridge)
        .into_iter()
        .find(|(_, accessory)| accessory.entity_id == event.entity_id)?;
    let values: Vec<Value> = event
        .characteristics
        .iter()
        .enumerate()
        .filter_map(|(index, (_, value))| {
            let iid = index as u64 + 8;
            subscriptions
                .contains(&(aid, iid))
                .then(|| json!({"aid": aid, "iid": iid, "value": characteristic_value(value)}))
        })
        .collect();
    (!values.is_empty())
        .then(|| serde_json::to_vec(&json!({"characteristics": values})).expect("serialize event"))
}

fn parse_aid_iid(value: &str) -> Option<(u64, u64)> {
    let (aid, iid) = value.split_once('.')?;
    Some((aid.parse().ok()?, iid.parse().ok()?))
}

fn characteristic_value(value: &HapCharacteristicValue) -> Value {
    match value {
        HapCharacteristicValue::Bool(value) => json!(value),
        HapCharacteristicValue::UInt8(value) => json!(value),
        HapCharacteristicValue::Float(value) => json!(value),
    }
}

fn characteristic_format(value: &HapCharacteristicValue) -> &'static str {
    match value {
        HapCharacteristicValue::Bool(_) => "bool",
        HapCharacteristicValue::UInt8(_) => "uint8",
        HapCharacteristicValue::Float(_) => "float",
    }
}

fn service_type(kind: HapAccessoryType) -> &'static str {
    match kind {
        HapAccessoryType::Lightbulb => "43",
        HapAccessoryType::Switch => "49",
        HapAccessoryType::OccupancySensor => "86",
        HapAccessoryType::MotionSensor => "85",
        HapAccessoryType::TemperatureSensor => "8A",
        HapAccessoryType::HumiditySensor => "82",
        HapAccessoryType::LeakSensor => "83",
        HapAccessoryType::ContactSensor => "80",
        HapAccessoryType::Door => "81",
        HapAccessoryType::Lock => "45",
        HapAccessoryType::SecuritySystem => "7E",
    }
}

fn characteristic_type(kind: HapCharacteristic) -> &'static str {
    match kind {
        HapCharacteristic::On => "25",
        HapCharacteristic::Brightness => "8",
        HapCharacteristic::CurrentTemperature => "11",
        HapCharacteristic::CurrentRelativeHumidity => "10",
        HapCharacteristic::OccupancyDetected => "71",
        HapCharacteristic::MotionDetected => "22",
        HapCharacteristic::LeakDetected => "70",
        HapCharacteristic::ContactSensorState => "6A",
        HapCharacteristic::CurrentDoorState => "E",
        HapCharacteristic::LockCurrentState => "1D",
        HapCharacteristic::SecuritySystemCurrentState => "66",
    }
}

async fn write_response(stream: &mut TcpStream, response: Response) -> Result<(), HapError> {
    let reason = match response.status {
        200 => "OK",
        204 => "No Content",
        207 => "Multi-Status",
        400 => "Bad Request",
        404 => "Not Found",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        470 => "Connection Authorization Required",
        501 => "Not Implemented",
        _ => "Error",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n",
        response.status,
        reason,
        response.content_type,
        response.body.len()
    );
    stream
        .write_all(header.as_bytes())
        .await
        .map_err(|error| HapError::Server(format!("write response header: {error}")))?;
    stream
        .write_all(&response.body)
        .await
        .map_err(|error| HapError::Server(format!("write response body: {error}")))
}

async fn write_event(stream: &mut TcpStream, body: Vec<u8>) -> Result<(), HapError> {
    let header = format!(
        "EVENT/1.0 200 OK\r\nContent-Type: {HAP_JSON}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .await
        .map_err(|error| HapError::Server(format!("write event header: {error}")))?;
    stream
        .write_all(&body)
        .await
        .map_err(|error| HapError::Server(format!("write event body: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdns::{HapServiceRecord, NullAdvertiser};
    use homecore::entity::{EntityId, State};
    use homecore::event::Context;

    fn bridge() -> HapBridge {
        let bridge = HapBridge::new(HapServiceRecord::bridge(
            "RuView Sense",
            51826,
            "AA:BB:CC:DD:EE:FF",
        ));
        let entity_id = EntityId::parse("binary_sensor.room_occupancy").unwrap();
        let state = State::new(
            entity_id.clone(),
            "on",
            json!({"device_class": "occupancy"}),
            Context::default(),
        );
        bridge.add_accessory(&entity_id, &state).unwrap();
        bridge
    }

    async fn server() -> (HapServerHandle, tempfile::TempDir) {
        let directory = tempfile::tempdir().unwrap();
        let pairings =
            Arc::new(PairingStore::open(directory.path().join("pairings.json")).unwrap());
        let config = HapServerConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            request_timeout: Duration::from_secs(1),
            shutdown_timeout: Duration::from_secs(1),
            ..HapServerConfig::default()
        };
        let handle = start_server(config, bridge(), pairings, Arc::new(NullAdvertiser))
            .await
            .unwrap();
        (handle, directory)
    }

    async fn exchange(addr: SocketAddr, request: &[u8]) -> Vec<u8> {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(request).await.unwrap();
        stream.shutdown().await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        response
    }

    #[tokio::test]
    async fn lifecycle_binds_and_shuts_down() {
        let (server, _directory) = server().await;
        assert_ne!(server.local_addr().port(), 0);
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_remains_bounded_with_idle_connection() {
        let (server, _directory) = server().await;
        let _idle = TcpStream::connect(server.local_addr()).await.unwrap();
        timeout(Duration::from_secs(3), server.shutdown())
            .await
            .expect("shutdown exceeded its outer bound")
            .unwrap();
    }

    #[tokio::test]
    async fn unauthenticated_accessory_request_is_gated() {
        let (server, _directory) = server().await;
        let response = exchange(
            server.local_addr(),
            b"GET /accessories HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        .await;
        assert!(response.starts_with(b"HTTP/1.1 470"));
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn pairing_endpoint_returns_explicit_unavailable_tlv() {
        let (server, _directory) = server().await;
        let response = exchange(
            server.local_addr(),
            b"POST /pair-setup HTTP/1.1\r\nHost: localhost\r\nContent-Length: 3\r\nConnection: close\r\n\r\n\x06\x01\x01",
        )
        .await;
        assert!(response.starts_with(b"HTTP/1.1 200"));
        assert!(response.ends_with(b"\x06\x01\x02\x07\x01\x06"));
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn malformed_and_oversized_requests_are_rejected() {
        let (server, _directory) = server().await;
        let malformed = exchange(
            server.local_addr(),
            b"GET / HTTP/1.0\r\nConnection: close\r\n\r\n",
        )
        .await;
        assert!(malformed.starts_with(b"HTTP/1.1 400"));
        let oversized = exchange(
            server.local_addr(),
            b"POST /pair-setup HTTP/1.1\r\nContent-Length: 999999\r\nConnection: close\r\n\r\n",
        )
        .await;
        assert!(oversized.starts_with(b"HTTP/1.1 413"));
        server.shutdown().await.unwrap();
    }

    #[test]
    fn authenticated_internal_dispatch_exposes_accessories_and_events() {
        let bridge = bridge();
        let directory = tempfile::tempdir().unwrap();
        let pairings = PairingStore::open(directory.path().join("pairings.json")).unwrap();
        let mut session = Session::authenticated_for_test(true);
        let mut subscriptions = HashSet::new();
        let response = dispatch_request(
            Request {
                method: "GET".into(),
                target: "/accessories".into(),
                body: Vec::new(),
                connection_close: false,
            },
            &mut session,
            &bridge,
            &pairings,
            &mut subscriptions,
        );
        assert_eq!(response.status, 200);
        let body: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(body["accessories"].as_array().unwrap().len(), 2);

        let response = dispatch_request(
            Request {
                method: "PUT".into(),
                target: "/characteristics".into(),
                body: br#"{"characteristics":[{"aid":2,"iid":8,"ev":true}]}"#.to_vec(),
                connection_close: false,
            },
            &mut session,
            &bridge,
            &pairings,
            &mut subscriptions,
        );
        assert_eq!(response.status, 204);
        assert!(subscriptions.contains(&(2, 8)));
    }
}
