//! Intent types for the HOMECORE-ASSIST pipeline.
//!
//! Mirrors `homeassistant.helpers.intent.Intent` and
//! `homeassistant.helpers.intent.IntentResponse`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Newtype wrapping the intent name string (e.g. `"HassTurnOn"`).
///
/// Kept as a newtype rather than a raw `String` so that call sites can
/// pattern-match on well-known constant values without stringly-typed bugs.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct IntentName(pub String);

impl IntentName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for IntentName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A recognised user intent with extracted slot values.
///
/// Mirrors `homeassistant.helpers.intent.Intent`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Intent {
    /// The intent name, e.g. `HassTurnOn`.
    pub name: IntentName,
    /// Extracted slots as a JSON-value map.  Keys are slot names
    /// (e.g. `"entity_id"`, `"brightness"`); values are typed by the
    /// recognizer.
    pub slots: HashMap<String, serde_json::Value>,
    /// BCP-47 language tag of the utterance (e.g. `"en"`, `"en-US"`).
    pub language: String,
}

impl Intent {
    /// Convenience constructor for single-slot intents.
    pub fn with_entity(name: impl Into<String>, entity_id: impl Into<String>, lang: &str) -> Self {
        let mut slots = HashMap::new();
        slots.insert(
            "entity_id".into(),
            serde_json::Value::String(entity_id.into()),
        );
        Self {
            name: IntentName::new(name),
            slots,
            language: lang.to_owned(),
        }
    }

    /// Return the `entity_id` slot as a `&str`, if present.
    pub fn entity_id(&self) -> Option<&str> {
        self.slots.get("entity_id").and_then(|v| v.as_str())
    }
}

/// Optional card displayed in the HA frontend alongside the speech response.
///
/// Mirrors `homeassistant.helpers.intent.IntentResponseType.ACTION_DONE`
/// card payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Card {
    pub title: String,
    pub content: String,
}

/// The full response produced by an intent handler.
///
/// Mirrors `homeassistant.helpers.intent.IntentResponse`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntentResponse {
    /// Spoken text to synthesise (TTS) or display.
    pub speech: String,
    /// Optional rich card for dashboard display.
    pub card: Option<Card>,
    /// Optional structured data for programmatic callers.
    pub data: Option<serde_json::Value>,
}

impl IntentResponse {
    /// Quick constructor for a plain speech-only response.
    pub fn speech_only(text: impl Into<String>) -> Self {
        Self {
            speech: text.into(),
            card: None,
            data: None,
        }
    }

    /// Default "not understood" response, mirroring HA's fallback text.
    pub fn not_understood() -> Self {
        Self::speech_only("I'm not sure how to help with that.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_name_display() {
        let n = IntentName::new("HassTurnOn");
        assert_eq!(format!("{n}"), "HassTurnOn");
    }

    #[test]
    fn intent_with_entity_sets_slot() {
        let intent = Intent::with_entity("HassTurnOn", "light.kitchen", "en");
        assert_eq!(intent.entity_id(), Some("light.kitchen"));
        assert_eq!(intent.name.as_str(), "HassTurnOn");
    }

    #[test]
    fn not_understood_response_text() {
        let r = IntentResponse::not_understood();
        assert!(r.speech.contains("not sure"));
        assert!(r.card.is_none());
    }
}
