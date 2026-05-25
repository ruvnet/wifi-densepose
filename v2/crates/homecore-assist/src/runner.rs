//! RufloRunner trait + NoopRunner (P1 stub).
//!
//! The ruflo agent is a Node.js process that exposes an MCP-over-stdio
//! interface for LLM-grade intent disambiguation. HOMECORE-ASSIST manages
//! a long-lived subprocess via `tokio::process::Child`.
//!
//! ## P1 scope
//!
//! Only the trait + `NoopRunner` stub ship in P1. No subprocess is spawned.
//!
//! ## P2 scope
//!
//! Real subprocess management with Windows-safe teardown per ADR-133 §Q3:
//! - `Child` wrapped in `Arc<Mutex<Option<Child>>>`.
//! - Explicit `async shutdown()` calls `child.kill().await` before drop.
//! - `tokio::signal` handler registered for `Ctrl+C`/`SIGINT` that calls
//!   `shutdown()` before exit.
//! - Windows job object approach (option 3 per Q3) deferred to P3.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::intent::Intent;

/// Error type for the assist pipeline (runner + pipeline-level errors).
#[derive(Error, Debug)]
pub enum AssistError {
    #[error("runner not started")]
    NotStarted,
    #[error("runner IO error: {0}")]
    Io(String),
    #[error("runner response parse error: {0}")]
    ParseError(String),
    #[error("recognizer error: {0}")]
    Recognizer(#[from] crate::recognizer::RecognizerError),
    #[error("handler error: {0}")]
    Handler(#[from] crate::handler::HandlerError),
    #[error("no handler registered for intent: {0}")]
    NoHandler(String),
}

/// Configuration for launching the ruflo agent subprocess.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RufloRunnerOpts {
    /// Path to the `ruflo-agent.js` entry point.
    pub script_path: String,
    /// Additional environment variables to pass to the subprocess.
    pub env: std::collections::HashMap<String, String>,
    /// Request timeout in milliseconds (default 5000).
    pub timeout_ms: u64,
}

impl Default for RufloRunnerOpts {
    fn default() -> Self {
        Self {
            script_path: "ruflo-agent.js".into(),
            env: Default::default(),
            timeout_ms: 5000,
        }
    }
}

/// JSON response from the ruflo agent subprocess.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RufloResponse {
    /// Recognised intent, if the LLM resolved one.
    pub intent: Option<Intent>,
    /// Spoken text from the LLM, if any.
    pub speech: Option<String>,
}

/// Trait for the ruflo agent subprocess runner.
///
/// P1 ships only this trait + `NoopRunner`. The real subprocess runner
/// lands in P2 with Windows-safe teardown (ADR-133 §Q3).
#[async_trait]
pub trait RufloRunner: Send + Sync + 'static {
    /// Spawn (or reconnect to) the ruflo agent subprocess.
    async fn spawn(&mut self, opts: RufloRunnerOpts) -> Result<(), AssistError>;

    /// Send an utterance payload to the agent and await a response.
    ///
    /// `payload` is an arbitrary JSON object; at minimum it should include
    /// `{ "utterance": "...", "language": "..." }`.
    async fn send_request(
        &self,
        payload: serde_json::Value,
    ) -> Result<RufloResponse, AssistError>;

    /// Gracefully shut down the subprocess.
    ///
    /// Must be idempotent — calling `shutdown` on an already-stopped runner
    /// must return `Ok(())` rather than an error.
    async fn shutdown(&mut self) -> Result<(), AssistError>;
}

/// P1 no-op implementation. Spawn/send/shutdown are all immediate Ok.
///
/// `send_request` returns an empty `RufloResponse` (no intent, no speech),
/// which causes the pipeline to fall through to the regex recognizer path.
#[derive(Default)]
pub struct NoopRunner {
    started: bool,
}

impl NoopRunner {
    pub fn new() -> Self {
        Self { started: false }
    }
}

#[async_trait]
impl RufloRunner for NoopRunner {
    async fn spawn(&mut self, _opts: RufloRunnerOpts) -> Result<(), AssistError> {
        self.started = true;
        tracing::debug!("NoopRunner: spawn called (P1 stub — no subprocess started)");
        Ok(())
    }

    async fn send_request(
        &self,
        _payload: serde_json::Value,
    ) -> Result<RufloResponse, AssistError> {
        // P1 stub: always returns empty response so the pipeline falls through
        // to the regex recognizer.
        Ok(RufloResponse {
            intent: None,
            speech: None,
        })
    }

    async fn shutdown(&mut self) -> Result<(), AssistError> {
        // Idempotent: Ok whether or not spawn was called.
        self.started = false;
        tracing::debug!("NoopRunner: shutdown called (idempotent no-op in P1)");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_runner_spawn_returns_ok() {
        let mut runner = NoopRunner::new();
        let result = runner.spawn(RufloRunnerOpts::default()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn noop_runner_send_request_returns_empty_response() {
        let runner = NoopRunner::new();
        let resp = runner
            .send_request(serde_json::json!({"utterance": "turn on the light", "language": "en"}))
            .await
            .unwrap();
        assert!(resp.intent.is_none());
        assert!(resp.speech.is_none());
    }

    #[tokio::test]
    async fn noop_runner_shutdown_is_idempotent() {
        let mut runner = NoopRunner::new();
        // First shutdown without spawn — must not error.
        assert!(runner.shutdown().await.is_ok());
        // Spawn then shutdown — must not error.
        runner.spawn(RufloRunnerOpts::default()).await.unwrap();
        assert!(runner.shutdown().await.is_ok());
        // Second shutdown — must still not error.
        assert!(runner.shutdown().await.is_ok());
    }
}
