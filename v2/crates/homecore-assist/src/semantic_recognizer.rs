//! `SemanticIntentRecognizer` — embedding-based semantic intent matching.
//!
//! Embeds utterances with [`crate::embedding`] (deterministic feature hashing)
//! and runs an **exact in-memory cosine k-NN** over enrolled intent exemplars.
//! On a match above the similarity threshold the exemplar's intent is returned,
//! with slots extracted from the incoming utterance via an optional paired
//! regex. Below threshold (or with an empty index) it delegates to the inner
//! [`RegexIntentRecognizer`](crate::recognizer::RegexIntentRecognizer).
//!
//! For the small intent vocabularies HOMECORE deals with, an exact cosine scan
//! is both faster and far more robust than an external ANN index — it has no
//! storage backend, no cross-crate feature coupling, and is fully deterministic.
//! Embeddings are L2-normalised, so cosine similarity is a plain dot product.
//!
//! Gated behind the default-on `semantic` feature. When disabled, a thin
//! delegating wrapper keeps the public type available.

use async_trait::async_trait;
#[cfg(feature = "semantic")]
use std::collections::HashMap;

#[cfg(feature = "semantic")]
use regex::Regex;

use crate::intent::Intent;
#[cfg(feature = "semantic")]
use crate::intent::IntentName;
use crate::recognizer::{IntentRecognizer, RecognizerError, RegexIntentRecognizer};

/// Default cosine-similarity threshold above which a semantic match is accepted.
pub const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.75;

/// One enrolled exemplar: a natural-language phrase mapped to an intent, with
/// an optional regex to extract slots from the *incoming* utterance on a hit.
#[cfg(feature = "semantic")]
struct Exemplar {
    name: IntentName,
    language: String,
    /// Optional slot-extraction regex applied to the matched utterance.
    slot_regex: Option<Regex>,
    /// L2-normalised embedding of the enrolled phrase, for cosine k-NN.
    vector: Vec<f32>,
}

/// Semantic recognizer backed by a real ruvector-core HNSW index.
///
/// Enroll exemplar phrases with [`enroll`](Self::enroll); `recognize` embeds
/// the utterance, runs k-NN search over the index, and accepts the nearest
/// exemplar when its similarity clears the threshold. Below threshold (or when
/// the index is empty) it delegates to the inner regex recognizer.
#[cfg(feature = "semantic")]
pub struct SemanticIntentRecognizer {
    fallback: RegexIntentRecognizer,
    index: std::sync::Arc<tokio::sync::RwLock<SemanticIndexInner>>,
    threshold: f32,
}

#[cfg(feature = "semantic")]
struct SemanticIndexInner {
    /// Enrolled exemplars in insertion order; the `Vec` index is the id.
    exemplars: Vec<Exemplar>,
}

#[cfg(feature = "semantic")]
impl SemanticIntentRecognizer {
    /// Build a semantic recognizer wrapping `fallback`, using the default
    /// similarity threshold.
    pub fn new(fallback: RegexIntentRecognizer) -> Self {
        Self::with_threshold(fallback, DEFAULT_SIMILARITY_THRESHOLD)
    }

    /// Build with an explicit similarity threshold in `[0, 1]`.
    pub fn with_threshold(fallback: RegexIntentRecognizer, threshold: f32) -> Self {
        Self {
            fallback,
            index: std::sync::Arc::new(tokio::sync::RwLock::new(SemanticIndexInner {
                exemplars: Vec::new(),
            })),
            threshold,
        }
    }

    /// Enroll an exemplar phrase for `name`/`language`.
    ///
    /// `slot_pattern`, if given, is a regex whose named capture groups are
    /// extracted from the *incoming* utterance when this exemplar wins, so
    /// semantic matches still produce slots (e.g. `entity_id`).
    pub async fn enroll(
        &self,
        name: impl Into<String>,
        phrase: &str,
        language: impl Into<String>,
        slot_pattern: Option<&str>,
    ) -> Result<(), RecognizerError> {
        let slot_regex = match slot_pattern {
            Some(p) => Some(Regex::new(p).map_err(|e| RecognizerError::BadPattern(e.to_string()))?),
            None => None,
        };
        let vector = crate::embedding::embed(phrase);

        let mut inner = self.index.write().await;
        inner.exemplars.push(Exemplar {
            name: IntentName::new(name),
            language: language.into(),
            slot_regex,
            vector,
        });
        Ok(())
    }

    /// Embed `utterance` and return the best `(exemplar_id, similarity)` whose
    /// exemplar matches `language`, or `None` if the index is empty.
    async fn nearest(&self, utterance: &str, language: &str) -> Option<(usize, f32)> {
        let normalised = utterance.trim().to_lowercase();
        let query = crate::embedding::embed(&normalised);

        // Exact in-memory cosine k-NN. Embeddings are L2-normalised, so cosine
        // similarity is a plain dot product (see `crate::embedding`). Returns the
        // best language-eligible exemplar, or `None` for an empty index.
        let inner = self.index.read().await;
        inner
            .exemplars
            .iter()
            .enumerate()
            .filter(|(_, e)| e.language == "*" || e.language == language)
            .map(|(id, e)| (id, crate::embedding::cosine_similarity(&query, &e.vector)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Like [`recognize`](IntentRecognizer::recognize) but also returns the
    /// cosine similarity of the winning exemplar (or the best below-threshold
    /// candidate). Exposed so callers/tests can see the real match score.
    pub async fn recognize_scored(
        &self,
        utterance: &str,
        language: &str,
    ) -> Result<(Option<Intent>, Option<f32>), RecognizerError> {
        // Fail-closed on an over-length utterance before embedding/scanning.
        // Untrusted input must not force an unbounded `to_lowercase` clone +
        // full tokenisation/embedding. Mirrors the regex recognizer's bound.
        if utterance.len() > crate::recognizer::MAX_UTTERANCE_BYTES {
            return Ok((None, None));
        }
        if let Some((id, similarity)) = self.nearest(utterance, language).await {
            if similarity >= self.threshold {
                let inner = self.index.read().await;
                let exemplar = &inner.exemplars[id];
                let mut slots: HashMap<String, serde_json::Value> = HashMap::new();
                if let Some(re) = &exemplar.slot_regex {
                    if let Some(caps) = re.captures(&utterance.trim().to_lowercase()) {
                        for cap_name in re.capture_names().flatten() {
                            if let Some(m) = caps.name(cap_name) {
                                slots.insert(
                                    cap_name.to_owned(),
                                    serde_json::Value::String(m.as_str().to_owned()),
                                );
                            }
                        }
                    }
                }
                return Ok((
                    Some(Intent {
                        name: exemplar.name.clone(),
                        slots,
                        language: language.to_owned(),
                    }),
                    Some(similarity),
                ));
            }
            // Below threshold — fall back to regex but still report the score.
            let regex_hit = self.fallback.recognize(utterance, language).await?;
            return Ok((regex_hit, Some(similarity)));
        }
        // Empty index — pure regex fallback.
        Ok((self.fallback.recognize(utterance, language).await?, None))
    }
}

#[cfg(feature = "semantic")]
#[async_trait]
impl IntentRecognizer for SemanticIntentRecognizer {
    async fn recognize(
        &self,
        utterance: &str,
        language: &str,
    ) -> Result<Option<Intent>, RecognizerError> {
        let (intent, _score) = self.recognize_scored(utterance, language).await?;
        Ok(intent)
    }
}

/// Fallback definition when the `semantic` feature is disabled: a thin
/// delegating wrapper, so downstream code compiles without ruvector-core.
#[cfg(not(feature = "semantic"))]
pub struct SemanticIntentRecognizer {
    fallback: RegexIntentRecognizer,
}

#[cfg(not(feature = "semantic"))]
impl SemanticIntentRecognizer {
    pub fn new(fallback: RegexIntentRecognizer) -> Self {
        Self { fallback }
    }
}

#[cfg(not(feature = "semantic"))]
#[async_trait]
impl IntentRecognizer for SemanticIntentRecognizer {
    async fn recognize(
        &self,
        utterance: &str,
        language: &str,
    ) -> Result<Option<Intent>, RecognizerError> {
        // Without the `semantic` feature there is no embedding/HNSW facility;
        // delegate to regex (honest: no semantic capability compiled in).
        self.fallback.recognize(utterance, language).await
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
    async fn empty_utterance_against_empty_index_no_panic_no_match() {
        // SECURITY (NaN/empty-poisoning): an empty (zero-vector) query against an
        // empty index must not panic and must yield no intent — the recognizer
        // falls through to the (also empty) regex fallback. Proves the empty-
        // iterator `max_by` path returns None cleanly.
        let semantic = SemanticIntentRecognizer::new(RegexIntentRecognizer::new());
        let result = semantic.recognize("", "en").await.unwrap();
        assert!(result.is_none(), "empty utterance must produce no intent / no action");
    }

    #[tokio::test]
    async fn over_length_utterance_fails_closed_semantic() {
        // SECURITY (DoS / fail-closed): an over-length utterance must short-
        // circuit before embedding/scanning, returning no intent — even if it
        // textually contains an enrolled/fallback-matchable command.
        let semantic = SemanticIntentRecognizer::new(turn_on_recognizer().await);
        let huge = format!(
            "{} turn on the kitchen light",
            "a ".repeat(crate::recognizer::MAX_UTTERANCE_BYTES)
        );
        assert!(huge.len() > crate::recognizer::MAX_UTTERANCE_BYTES);
        let result = semantic.recognize(&huge, "en").await.unwrap();
        assert!(result.is_none(), "over-length utterance must fail closed in semantic path");
    }

    #[tokio::test]
    async fn semantic_recognizer_delegates_to_fallback() {
        // No exemplars enrolled → empty HNSW index → pure regex fallback.
        let semantic = SemanticIntentRecognizer::new(turn_on_recognizer().await);
        let result = semantic
            .recognize("turn on light.kitchen", "en")
            .await
            .unwrap();
        assert!(result.is_some());
    }

    // ── Real HNSW-backed semantic matching (default `semantic` feature) ───────

    #[cfg(feature = "semantic")]
    async fn enrolled_semantic() -> SemanticIntentRecognizer {
        // Regex fallback is empty so any positive result comes from HNSW search.
        let semantic = SemanticIntentRecognizer::new(RegexIntentRecognizer::new());
        semantic
            .enroll(
                "HassTurnOn",
                "turn on the light",
                "en",
                Some(r"(?:turn on|switch on) (?:the )?(?P<entity_id>[a-z_][a-z0-9_ ]*(?:\.[a-z_][a-z0-9_]*)?)"),
            )
            .await
            .unwrap();
        semantic
            .enroll("HassNevermind", "never mind cancel that", "en", None)
            .await
            .unwrap();
        semantic
            .enroll("HassGetWeather", "what is the weather forecast", "en", None)
            .await
            .unwrap();
        semantic
    }

    #[cfg(feature = "semantic")]
    #[tokio::test]
    async fn semantic_matches_enrolled_paraphrase_with_real_score() {
        // FAILS against the old delegate-only stub: regex fallback is empty,
        // so the only way to get a hit is real embedding + HNSW search.
        let semantic = enrolled_semantic().await;
        let (intent, score) = semantic
            .recognize_scored("turn on the kitchen light", "en")
            .await
            .unwrap();

        let intent = intent.expect("paraphrase of an enrolled exemplar must match");
        assert_eq!(intent.name.as_str(), "HassTurnOn");
        let sim = score.expect("a semantic match must report a similarity");
        assert!(
            sim >= DEFAULT_SIMILARITY_THRESHOLD,
            "match similarity {sim:.4} must clear threshold {DEFAULT_SIMILARITY_THRESHOLD}"
        );
        // Slots extracted from the *incoming* utterance via the paired regex.
        assert_eq!(intent.entity_id(), Some("kitchen light"));
    }

    #[cfg(feature = "semantic")]
    #[tokio::test]
    async fn semantic_no_match_for_unknown_utterance_with_real_score() {
        let semantic = enrolled_semantic().await;
        let (intent, score) = semantic
            .recognize_scored("schedule a dentist appointment", "en")
            .await
            .unwrap();

        assert!(intent.is_none(), "unrelated utterance must not match any intent");
        let sim = score.expect("even a no-match reports the best similarity seen");
        assert!(
            sim < DEFAULT_SIMILARITY_THRESHOLD,
            "no-match similarity {sim:.4} must be below threshold {DEFAULT_SIMILARITY_THRESHOLD}"
        );
    }

    #[cfg(feature = "semantic")]
    #[tokio::test]
    async fn semantic_match_outscores_no_match() {
        let semantic = enrolled_semantic().await;
        let (_, hit_score) = semantic
            .recognize_scored("please turn on the lights", "en")
            .await
            .unwrap();
        let (_, miss_score) = semantic
            .recognize_scored("order a pizza for dinner", "en")
            .await
            .unwrap();
        let hit = hit_score.unwrap();
        let miss = miss_score.unwrap();
        assert!(
            hit > miss,
            "enrolled paraphrase ({hit:.4}) must score above unrelated ({miss:.4})"
        );
    }

    #[cfg(feature = "semantic")]
    #[tokio::test]
    async fn semantic_falls_back_to_regex_below_threshold() {
        // Enroll a weak exemplar; arrange a regex fallback that DOES match so we
        // prove the fallback path runs when similarity is below threshold.
        let semantic = SemanticIntentRecognizer::new(turn_on_recognizer().await);
        semantic
            .enroll("HassGetWeather", "what is the weather forecast", "en", None)
            .await
            .unwrap();
        // This utterance is unrelated to the weather exemplar (low similarity)
        // but matches the regex fallback's HassTurnOn pattern.
        let (intent, score) = semantic
            .recognize_scored("turn on light.kitchen", "en")
            .await
            .unwrap();
        let intent = intent.expect("regex fallback must catch this");
        assert_eq!(intent.name.as_str(), "HassTurnOn");
        let sim = score.expect("semantic score still reported on fallback");
        assert!(sim < DEFAULT_SIMILARITY_THRESHOLD, "expected low sim, got {sim:.4}");
    }
}
