//! AssistPipeline — wires recognizer → handler → response.
//!
//! The pipeline is the public entry point for the HOMECORE-ASSIST subsystem.
//! The HOMECORE-API WebSocket `assist` command will call
//! `pipeline.process(utterance, language, &hc).await`.
//!
//! ## Processing flow
//!
//! 1. Call `recognizer.recognize(utterance, language)`.
//! 2. If no intent matched → return `IntentResponse::not_understood()`.
//! 3. Look up the handler by intent name.
//! 4. Call `handler.handle(intent, hc)`.
//! 5. Return the `IntentResponse`.
//!
//! The `RufloRunner` is reserved for a P2 LLM disambiguation pass that
//! fires between steps 1 and 2 when the regex recognizer returns `None`.

use std::collections::HashMap;
use std::sync::Arc;

use homecore::HomeCore;
use tracing::debug;

use crate::handler::IntentHandler;
use crate::intent::IntentResponse;
use crate::recognizer::IntentRecognizer;
use crate::runner::AssistError;

/// Boxed type alias so the pipeline can hold heterogeneous handlers.
type BoxedHandler = Arc<dyn IntentHandler>;

/// The main Assist pipeline.
///
/// Construct with `AssistPipeline::new(recognizer)`, register handlers
/// with `register_handler`, then call `process`.
pub struct AssistPipeline<R: IntentRecognizer> {
    recognizer: R,
    handlers: HashMap<String, BoxedHandler>,
}

impl<R: IntentRecognizer> AssistPipeline<R> {
    /// Create a new pipeline with the given recognizer and no handlers.
    pub fn new(recognizer: R) -> Self {
        Self {
            recognizer,
            handlers: HashMap::new(),
        }
    }

    /// Register an intent handler.  If a handler for the same intent name
    /// was already registered, it is replaced.
    pub fn register_handler<H: IntentHandler>(&mut self, handler: H) {
        self.handlers
            .insert(handler.intent_name().to_owned(), Arc::new(handler));
    }

    /// Process an utterance through the full pipeline.
    ///
    /// # Errors
    ///
    /// Returns `AssistError` only for unexpected internal failures.
    /// Unknown intents and unrecognised utterances are returned as
    /// `IntentResponse::not_understood()` — not as errors — so the caller
    /// (WebSocket handler) can always synthesise a speech reply.
    pub async fn process(
        &self,
        utterance: &str,
        language: &str,
        hc: &HomeCore,
    ) -> Result<IntentResponse, AssistError> {
        debug!(%utterance, %language, "AssistPipeline: processing utterance");

        let intent = match self.recognizer.recognize(utterance, language).await {
            Ok(Some(i)) => i,
            Ok(None) => {
                debug!("no intent recognised — returning not_understood");
                return Ok(IntentResponse::not_understood());
            }
            Err(e) => return Err(AssistError::Recognizer(e)),
        };

        let name = intent.name.as_str().to_owned();
        let handler = self.handlers.get(&name).cloned();

        match handler {
            Some(h) => h
                .handle(intent, hc)
                .await
                .map_err(AssistError::Handler),
            None => {
                debug!(%name, "no handler registered for intent");
                Ok(IntentResponse::not_understood())
            }
        }
    }

    /// Convenience: count of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }
}

/// Builder that pre-wires the standard set of built-in HA intent handlers.
///
/// Use this when you want all 5 P1 built-ins registered without listing
/// them individually.
pub fn default_pipeline(
    recognizer: impl IntentRecognizer,
) -> AssistPipeline<impl IntentRecognizer> {
    use crate::handler::{HassCancelAll, HassLightSet, HassNevermind, HassTurnOff, HassTurnOn};
    let mut pipeline = AssistPipeline::new(recognizer);
    pipeline.register_handler(HassTurnOn);
    pipeline.register_handler(HassTurnOff);
    pipeline.register_handler(HassLightSet);
    pipeline.register_handler(HassNevermind);
    pipeline.register_handler(HassCancelAll);
    pipeline
}

#[cfg(test)]
mod tests {
    use homecore::service::FnHandler;
    use homecore::{HomeCore, ServiceName};

    use crate::handler::{HassTurnOff, HassTurnOn};
    use crate::recognizer::RegexIntentRecognizer;

    use super::*;

    async fn build_test_pipeline() -> (AssistPipeline<RegexIntentRecognizer>, HomeCore) {
        let r = RegexIntentRecognizer::new();
        r.register(
            "HassTurnOn",
            r"turn on (?:the )?(?P<entity_id>[a-z_][a-z0-9_ ]*(?:\.[a-z0-9_]+)?)",
            "*",
        )
        .await
        .unwrap();
        r.register(
            "HassTurnOff",
            r"turn off (?:the )?(?P<entity_id>[a-z_][a-z0-9_ ]*(?:\.[a-z0-9_]+)?)",
            "*",
        )
        .await
        .unwrap();
        r.register("HassNevermind", r"never ?mind|cancel that", "*")
            .await
            .unwrap();

        let mut pipeline = AssistPipeline::new(r);
        pipeline.register_handler(HassTurnOn);
        pipeline.register_handler(HassTurnOff);
        pipeline.register_handler(crate::handler::HassNevermind);

        let hc = HomeCore::new();
        // Register spy handlers so service calls don't return NotRegistered.
        hc.services()
            .register(
                ServiceName::new("homeassistant", "turn_on"),
                FnHandler(|_| async { Ok(serde_json::json!({})) }),
            )
            .await;
        hc.services()
            .register(
                ServiceName::new("homeassistant", "turn_off"),
                FnHandler(|_| async { Ok(serde_json::json!({})) }),
            )
            .await;
        (pipeline, hc)
    }

    #[tokio::test]
    async fn pipeline_turn_on_end_to_end() {
        let (pipeline, hc) = build_test_pipeline().await;
        let resp = pipeline
            .process("turn on light.kitchen", "en", &hc)
            .await
            .unwrap();
        assert!(resp.speech.contains("light.kitchen"));
    }

    #[tokio::test]
    async fn pipeline_turn_off_end_to_end() {
        let (pipeline, hc) = build_test_pipeline().await;
        let resp = pipeline
            .process("turn off switch.fan", "en", &hc)
            .await
            .unwrap();
        assert!(resp.speech.to_lowercase().contains("off") || resp.speech.contains("switch.fan"));
    }

    #[tokio::test]
    async fn pipeline_unknown_utterance_returns_not_understood() {
        let (pipeline, hc) = build_test_pipeline().await;
        let resp = pipeline
            .process("what is the weather like", "en", &hc)
            .await
            .unwrap();
        assert!(resp.speech.contains("not sure") || resp.speech.contains("I'm not"));
    }

    #[tokio::test]
    async fn pipeline_recognized_but_no_handler_returns_not_understood() {
        // Register a pattern but NOT its handler.
        let r = RegexIntentRecognizer::new();
        r.register("HassGetState", r"what is (?P<entity_id>\S+)", "*")
            .await
            .unwrap();
        let pipeline = AssistPipeline::new(r);
        let hc = HomeCore::new();
        let resp = pipeline
            .process("what is light.kitchen", "en", &hc)
            .await
            .unwrap();
        assert!(resp.speech.contains("not sure") || resp.speech.contains("I'm not"));
    }

    #[tokio::test]
    async fn default_pipeline_registers_five_handlers() {
        let r = RegexIntentRecognizer::new();
        let pipeline = default_pipeline(r);
        assert_eq!(pipeline.handler_count(), 5);
    }

    #[tokio::test]
    async fn pipeline_nevermind_response() {
        let (pipeline, hc) = build_test_pipeline().await;
        let resp = pipeline
            .process("never mind", "en", &hc)
            .await
            .unwrap();
        assert!(
            resp.speech.to_lowercase().contains("okay")
                || resp.speech.to_lowercase().contains("never")
                || resp.speech.to_lowercase().contains("cancel")
        );
    }

    #[tokio::test]
    async fn pipeline_use_homecore_service_fn_handler() {
        use homecore::service::FnHandler;
        let hc = HomeCore::new();
        hc.services()
            .register(
                ServiceName::new("homeassistant", "turn_on"),
                FnHandler(|_| async { Ok(serde_json::json!({"ok": true})) }),
            )
            .await;
        let r = RegexIntentRecognizer::new();
        r.register(
            "HassTurnOn",
            r"on (?P<entity_id>\S+)",
            "*",
        )
        .await
        .unwrap();
        let mut pipeline = AssistPipeline::new(r);
        pipeline.register_handler(HassTurnOn);
        let resp = pipeline.process("on light.bed", "en", &hc).await.unwrap();
        assert!(resp.speech.contains("light.bed"));
    }
}
