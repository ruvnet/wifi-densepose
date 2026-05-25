//! Intent handler trait + built-in HA-mirroring handlers.
//!
//! Mirrors `homeassistant.helpers.intent.IntentHandler`. Each handler
//! receives a recognised `Intent` and a `HomeCore` handle, dispatches the
//! appropriate service call, and returns an `IntentResponse`.
//!
//! ## Built-in handlers (P1)
//!
//! | Handler | HA service | Slots |
//! |---------|-----------|-------|
//! | `HassTurnOn` | `homeassistant.turn_on` | `entity_id` |
//! | `HassTurnOff` | `homeassistant.turn_off` | `entity_id` |
//! | `HassLightSet` | `light.turn_on` | `entity_id`, `brightness`, `color_name` |
//! | `HassNevermind` | — (no-op) | — |
//! | `HassCancelAll` | — (domain event) | — |

use async_trait::async_trait;
use thiserror::Error;

use homecore::{Context, HomeCore, ServiceCall, ServiceName};

use crate::intent::{Intent, IntentResponse};

#[derive(Error, Debug)]
pub enum HandlerError {
    #[error("service call failed: {0}")]
    ServiceFailed(String),
    #[error("missing required slot: {0}")]
    MissingSlot(String),
    #[error("handler internal error: {0}")]
    Internal(String),
}

/// Core trait every intent handler must implement.
#[async_trait]
pub trait IntentHandler: Send + Sync + 'static {
    /// The intent name(s) this handler accepts.
    fn intent_name(&self) -> &str;

    /// Handle the intent and return a response.
    async fn handle(&self, intent: Intent, hc: &HomeCore)
        -> Result<IntentResponse, HandlerError>;
}

// ---- HassTurnOn ----

/// Dispatches `homeassistant.turn_on` (domain-agnostic) for the entity.
pub struct HassTurnOn;

#[async_trait]
impl IntentHandler for HassTurnOn {
    fn intent_name(&self) -> &str {
        "HassTurnOn"
    }

    async fn handle(
        &self,
        intent: Intent,
        hc: &HomeCore,
    ) -> Result<IntentResponse, HandlerError> {
        let entity_id = intent
            .entity_id()
            .ok_or_else(|| HandlerError::MissingSlot("entity_id".into()))?
            .to_owned();
        let call = ServiceCall {
            name: ServiceName::new("homeassistant", "turn_on"),
            data: serde_json::json!({ "entity_id": entity_id }),
            context: Context::new(),
        };
        hc.services()
            .call(call)
            .await
            .map_err(|e| HandlerError::ServiceFailed(e.to_string()))?;
        Ok(IntentResponse::speech_only(format!("Turned on {entity_id}.")))
    }
}

// ---- HassTurnOff ----

/// Dispatches `homeassistant.turn_off` for the entity.
pub struct HassTurnOff;

#[async_trait]
impl IntentHandler for HassTurnOff {
    fn intent_name(&self) -> &str {
        "HassTurnOff"
    }

    async fn handle(
        &self,
        intent: Intent,
        hc: &HomeCore,
    ) -> Result<IntentResponse, HandlerError> {
        let entity_id = intent
            .entity_id()
            .ok_or_else(|| HandlerError::MissingSlot("entity_id".into()))?
            .to_owned();
        let call = ServiceCall {
            name: ServiceName::new("homeassistant", "turn_off"),
            data: serde_json::json!({ "entity_id": entity_id }),
            context: Context::new(),
        };
        hc.services()
            .call(call)
            .await
            .map_err(|e| HandlerError::ServiceFailed(e.to_string()))?;
        Ok(IntentResponse::speech_only(format!("Turned off {entity_id}.")))
    }
}

// ---- HassLightSet ----

/// Dispatches `light.turn_on` with optional `brightness` and `color_name`.
pub struct HassLightSet;

#[async_trait]
impl IntentHandler for HassLightSet {
    fn intent_name(&self) -> &str {
        "HassLightSet"
    }

    async fn handle(
        &self,
        intent: Intent,
        hc: &HomeCore,
    ) -> Result<IntentResponse, HandlerError> {
        let entity_id = intent
            .entity_id()
            .ok_or_else(|| HandlerError::MissingSlot("entity_id".into()))?
            .to_owned();
        let mut data = serde_json::json!({ "entity_id": entity_id });
        if let Some(b) = intent.slots.get("brightness") {
            data["brightness"] = b.clone();
        }
        if let Some(c) = intent.slots.get("color_name") {
            data["color_name"] = c.clone();
        }
        let call = ServiceCall {
            name: ServiceName::new("light", "turn_on"),
            data,
            context: Context::new(),
        };
        hc.services()
            .call(call)
            .await
            .map_err(|e| HandlerError::ServiceFailed(e.to_string()))?;
        Ok(IntentResponse::speech_only(format!("Done, adjusted {entity_id}.")))
    }
}

// ---- HassNevermind ----

/// No-op — acknowledges the cancellation without a service call.
pub struct HassNevermind;

#[async_trait]
impl IntentHandler for HassNevermind {
    fn intent_name(&self) -> &str {
        "HassNevermind"
    }

    async fn handle(
        &self,
        _intent: Intent,
        _hc: &HomeCore,
    ) -> Result<IntentResponse, HandlerError> {
        Ok(IntentResponse::speech_only("Okay, never mind."))
    }
}

// ---- HassCancelAll ----

/// Fires a domain event to cancel all running scripts/automations.
pub struct HassCancelAll;

#[async_trait]
impl IntentHandler for HassCancelAll {
    fn intent_name(&self) -> &str {
        "HassCancelAll"
    }

    async fn handle(
        &self,
        _intent: Intent,
        hc: &HomeCore,
    ) -> Result<IntentResponse, HandlerError> {
        use homecore::{Context, DomainEvent};
        let event = DomainEvent::new(
            "homeassistant_stop_all_scripts",
            serde_json::json!({}),
            Context::new(),
        );
        // fire_domain is synchronous and infallible (returns receiver count).
        let _receivers = hc.bus().fire_domain(event);
        Ok(IntentResponse::speech_only("Cancelled all running automations."))
    }
}

#[cfg(test)]
mod tests {
    use homecore::service::FnHandler;
    use homecore::ServiceName;

    use super::*;

    /// Build a `HomeCore` pre-registered with a spy handler for the given
    /// service.  Returns `(HomeCore, Arc<AtomicBool>)` so tests can assert
    /// the handler was called.
    async fn hc_with_spy(domain: &str, service: &str) -> (HomeCore, std::sync::Arc<std::sync::atomic::AtomicBool>) {
        let hc = HomeCore::new();
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called2 = called.clone();
        hc.services()
            .register(
                ServiceName::new(domain, service),
                FnHandler(move |_call| {
                    let c = called2.clone();
                    async move {
                        c.store(true, std::sync::atomic::Ordering::SeqCst);
                        Ok(serde_json::json!({}))
                    }
                }),
            )
            .await;
        (hc, called)
    }

    #[tokio::test]
    async fn turn_on_dispatches_service() {
        let (hc, called) = hc_with_spy("homeassistant", "turn_on").await;
        let intent = Intent::with_entity("HassTurnOn", "light.kitchen", "en");
        let resp = HassTurnOn.handle(intent, &hc).await.unwrap();
        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        assert!(resp.speech.contains("light.kitchen"));
    }

    #[tokio::test]
    async fn turn_off_dispatches_service() {
        let (hc, called) = hc_with_spy("homeassistant", "turn_off").await;
        let intent = Intent::with_entity("HassTurnOff", "switch.fan", "en");
        let resp = HassTurnOff.handle(intent, &hc).await.unwrap();
        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        assert!(resp.speech.contains("switch.fan"));
    }

    #[tokio::test]
    async fn light_set_dispatches_light_turn_on() {
        let (hc, called) = hc_with_spy("light", "turn_on").await;
        let mut intent = Intent::with_entity("HassLightSet", "light.living", "en");
        intent
            .slots
            .insert("brightness".into(), serde_json::json!(128));
        let resp = HassLightSet.handle(intent, &hc).await.unwrap();
        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
        assert!(resp.speech.contains("light.living"));
    }

    #[tokio::test]
    async fn nevermind_returns_ok_response() {
        let hc = HomeCore::new();
        let intent = Intent {
            name: crate::intent::IntentName::new("HassNevermind"),
            slots: Default::default(),
            language: "en".into(),
        };
        let resp = HassNevermind.handle(intent, &hc).await.unwrap();
        assert!(resp.speech.to_lowercase().contains("never mind")
            || resp.speech.to_lowercase().contains("nevermind")
            || resp.speech.to_lowercase().contains("okay"));
    }

    #[tokio::test]
    async fn cancel_all_fires_domain_event() {
        let hc = HomeCore::new();
        // Subscribe before firing so the sender has a live receiver.
        let mut rx = hc.bus().subscribe_domain();
        let intent = Intent {
            name: crate::intent::IntentName::new("HassCancelAll"),
            slots: Default::default(),
            language: "en".into(),
        };
        let resp = HassCancelAll.handle(intent, &hc).await.unwrap();
        assert!(resp.speech.to_lowercase().contains("cancel"));
        // Domain event should have been broadcast.
        let event = rx.recv().await.unwrap();
        assert_eq!(event.event_type, "homeassistant_stop_all_scripts");
    }
}
