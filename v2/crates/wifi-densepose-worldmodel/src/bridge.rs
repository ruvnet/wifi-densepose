//! Async Unix-socket client that sends an [`OccupancyWorldModelRequest`] to
//! the OccWorld Python inference server and receives an
//! [`OccupancyWorldModelResponse`] (ADR-147).
//!
//! ## Protocol
//! Communication uses newline-delimited JSON over a Unix-domain stream socket:
//! 1. Connect to the socket path.
//! 2. Write the JSON-serialised request followed by a single `\n` byte.
//! 3. Read bytes until the first `\n`; decode as JSON response.
//!
//! A hard 30-second wall-clock timeout wraps the entire operation.

use std::path::PathBuf;
use std::time::Duration;

#[cfg(unix)]
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::time::timeout;

use crate::error::WorldModelError;
use crate::{OccupancyWorldModelRequest, OccupancyWorldModelResponse};

/// Hard deadline applied to each inference round-trip.
const TIMEOUT_S: u64 = 30;

/// Maximum number of bytes accepted for a single response line.
///
/// 200×200×16 future frames × 15 steps × ~1 byte/voxel = ~9.6 MB in the
/// worst case; set a generous 64 MB ceiling to stay safe without allocating
/// it up front. (Only used by the unix socket reader.)
#[cfg(unix)]
const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// Thin async client for the OccWorld Unix-socket inference server.
///
/// Instances are cheap to clone (they only hold a [`PathBuf`]) and are safe
/// to share across threads.  A fresh TCP-free connection is established for
/// every [`OccWorldBridge::predict`] call so the server can restart between
/// requests without invalidating a long-lived connection handle.
#[derive(Debug, Clone)]
pub struct OccWorldBridge {
    /// Path to the Unix-domain socket served by the OccWorld Python process.
    pub socket_path: PathBuf,
}

impl OccWorldBridge {
    /// Creates a new bridge pointing at the given Unix-domain socket path.
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
    }

    /// Sends `request` to the OccWorld server and returns the decoded
    /// response, or an error if the connection fails, times out, or the
    /// response is malformed.
    pub async fn predict(
        &self,
        request: OccupancyWorldModelRequest,
    ) -> Result<OccupancyWorldModelResponse, WorldModelError> {
        timeout(
            Duration::from_secs(TIMEOUT_S),
            self.send_recv(request),
        )
        .await
        .map_err(|_| WorldModelError::Timeout { timeout_s: TIMEOUT_S })?
    }

    /// Non-unix platforms have no Unix-domain sockets. The OccWorld bridge is a
    /// Linux-appliance feature (the Python inference server runs on the GPU host),
    /// so on Windows/other targets the crate still compiles but `predict` fails
    /// fast with a clear error instead of silently degrading.
    #[cfg(not(unix))]
    async fn send_recv(
        &self,
        _request: OccupancyWorldModelRequest,
    ) -> Result<OccupancyWorldModelResponse, WorldModelError> {
        Err(WorldModelError::Protocol(
            "OccWorld Unix-socket bridge is only supported on unix targets".into(),
        ))
    }

    /// Internal: connect, write request, read response — no timeout here;
    /// the outer [`timeout`] in [`predict`] handles that.
    #[cfg(unix)]
    async fn send_recv(
        &self,
        request: OccupancyWorldModelRequest,
    ) -> Result<OccupancyWorldModelResponse, WorldModelError> {
        let stream = self.connect().await?;

        // Split into reader/writer halves so we can write and then read
        // without fully consuming the stream.
        let (reader_half, mut writer_half) = stream.into_split();

        // Encode request as a single newline-terminated JSON line.
        let mut payload = serde_json::to_vec(&request)?;
        payload.push(b'\n');

        writer_half
            .write_all(&payload)
            .await
            .map_err(|e| WorldModelError::Protocol(format!("write error: {e}")))?;

        // Flush the write half so the server sees the complete line.
        writer_half
            .flush()
            .await
            .map_err(|e| WorldModelError::Protocol(format!("flush error: {e}")))?;

        // Read exactly one newline-delimited JSON line from the server.
        let mut line = String::new();
        let mut buf_reader = BufReader::new(reader_half);

        buf_reader
            .read_line(&mut line)
            .await
            .map_err(|e| WorldModelError::Protocol(format!("read error: {e}")))?;

        if line.is_empty() {
            return Err(WorldModelError::Protocol(
                "server closed connection before sending a response".into(),
            ));
        }

        if line.len() > MAX_RESPONSE_BYTES {
            return Err(WorldModelError::Protocol(format!(
                "response line too large ({} bytes > {} byte limit)",
                line.len(),
                MAX_RESPONSE_BYTES
            )));
        }

        let response: OccupancyWorldModelResponse = serde_json::from_str(line.trim())?;

        // Propagate any VRAM error signalled by the server via a dedicated
        // sentinel in the model_id field (convention agreed in ADR-147).
        if response.model_id.starts_with("error:vram:") {
            return Err(WorldModelError::VramUnavailable(
                response.model_id["error:vram:".len()..].to_owned(),
            ));
        }

        Ok(response)
    }

    /// Establishes a [`UnixStream`] connection to `self.socket_path`.
    #[cfg(unix)]
    async fn connect(&self) -> Result<UnixStream, WorldModelError> {
        UnixStream::connect(&self.socket_path)
            .await
            .map_err(|e| WorldModelError::SocketConnect {
                path: self.socket_path.display().to_string(),
                source: e,
            })
    }
}

/// Returns the default Unix socket path used by the OccWorld Python server
/// as specified in ADR-147.
pub fn default_socket_path() -> PathBuf {
    PathBuf::from("/tmp/occworld.sock")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_new_stores_path() {
        let b = OccWorldBridge::new("/tmp/test.sock");
        assert_eq!(b.socket_path, PathBuf::from("/tmp/test.sock"));
    }

    #[test]
    fn default_socket_path_is_deterministic() {
        assert_eq!(default_socket_path(), PathBuf::from("/tmp/occworld.sock"));
    }

    /// Verify that a missing socket returns `SocketConnect` and not a panic.
    /// Unix-only: non-unix targets return a `Protocol` "unsupported" error instead.
    #[cfg(unix)]
    #[tokio::test]
    async fn connect_to_missing_socket_returns_error() {
        let bridge = OccWorldBridge::new("/tmp/__occworld_nonexistent_test__.sock");
        use crate::{OccupancyGrid3D, OccupancyWorldModelRequest, SceneBoundsJson};
        let req = OccupancyWorldModelRequest {
            past_frames: vec![OccupancyGrid3D {
                width: 200,
                height: 200,
                depth: 16,
                voxels: vec![17u8; 200 * 200 * 16],
            }],
            voxel_resolution_m: 0.1,
            scene_bounds: SceneBoundsJson {
                min_e: -10.0,
                min_n: -10.0,
                max_e: 10.0,
                max_n: 10.0,
            },
            prediction_steps: 1,
        };
        let err = bridge.predict(req).await.unwrap_err();
        assert!(
            matches!(err, WorldModelError::SocketConnect { .. }),
            "expected SocketConnect, got {err:?}"
        );
    }
}
