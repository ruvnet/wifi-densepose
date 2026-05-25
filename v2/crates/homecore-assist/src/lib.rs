//! HOMECORE-ASSIST — Voice/intent pipeline + ruflo agent bridge.
//!
//! Implements [ADR-133](../../../docs/adr/ADR-133-homecore-assist-ruflo.md):
//! the Assist pipeline that takes a voice utterance through intent
//! recognition, intent handling, and response synthesis.
//!
//! ## Module layout (P1 scaffold)
//!
//! - [`intent`] — `IntentName`, `Intent`, `IntentResponse`, `Card`
//! - [`recognizer`] — `IntentRecognizer` trait + `RegexIntentRecognizer` (P1)
//! - [`handler`] — `IntentHandler` trait + 5 built-in HA-mirroring handlers
//! - [`runner`] — `RufloRunner` trait + `NoopRunner` (P1 stub)
//! - [`pipeline`] — `AssistPipeline`: wires recognizer → handler → response
//!
//! ## P1 scope
//!
//! - Regex-based intent recognition (HA classic intent matching).
//! - Built-in handlers: `HassTurnOn`, `HassTurnOff`, `HassLightSet`,
//!   `HassNevermind`, `HassCancelAll`.
//! - `RufloRunner` trait surface only; `NoopRunner` stub for P1.
//!
//! ## What's NOT here yet (deferred to P2+)
//!
//! - Real `tokio::process::Child` subprocess runner for `node ruflo-agent.js`
//!   (Windows-safe teardown per ADR-133 §Q3 lands in P2).
//! - `SemanticIntentRecognizer` using ruvector HNSW embeddings (P2).
//! - STT/TTS bridge and satellite protocol (P3).

pub mod intent;
pub mod recognizer;
pub mod handler;
pub mod runner;
pub mod pipeline;

pub use intent::{Card, Intent, IntentName, IntentResponse};
pub use recognizer::{IntentRecognizer, RecognizerError, RegexIntentRecognizer};
pub use handler::{
    HandlerError, HassCancelAll, HassLightSet, HassNevermind, HassTurnOff, HassTurnOn,
    IntentHandler,
};
pub use runner::{AssistError, NoopRunner, RufloResponse, RufloRunner, RufloRunnerOpts};
pub use pipeline::AssistPipeline;
