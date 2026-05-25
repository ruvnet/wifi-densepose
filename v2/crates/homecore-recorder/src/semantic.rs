//! Ruvector-backed semantic index — ADR-132 P2.
//!
//! ## Embedding strategy (P2 — hash-based)
//!
//! To keep the recorder self-contained and avoid an ML model dependency at P2,
//! state attributes are embedded by a deterministic SHA-256 hash procedure:
//!
//! 1. Canonicalise the state as `"{entity_id}={state}|{attributes_json}"`.
//! 2. SHA-256 hash → 32 bytes.
//! 3. Interpret the 32 bytes as 8 × `i32` (big-endian), cast to `f32`.
//! 4. L2-normalise the resulting 8-element vector.
//!
//! This gives stable, reproducible 8-dimensional unit vectors suitable for
//! cosine-distance HNSW search. Semantic similarity is **not** captured (two
//! states with the same value but different entity IDs will differ). P3 will
//! replace this with a learned sentence-embedding via `ruvector-attention`.
//!
//! ## P3 plan
//!
//! Replace `embed_bytes` with a call to
//! `ruvector_attention::SentenceEmbedding::encode(&text)` for true semantic
//! similarity. Increase `EMBEDDING_DIM` to 384 at that point.

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use homecore::entity::State;
use ruvector_core::{
    types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, VectorEntry},
    VectorDB,
};

use crate::db::SemanticIndex;

/// Dimensionality of the hash-based embedding vectors.
///
/// 8 dimensions: each SHA-256 chunk of 4 bytes becomes one `f32` component.
/// Increase to 384 in P3 when switching to learned embeddings.
pub const EMBEDDING_DIM: usize = 8;

/// Ruvector-backed `SemanticIndex` using in-memory HNSW and hash embeddings.
///
/// The index lives entirely in process memory. A restart clears it; P3 will
/// add persistence via `ruvector-core`'s `storage` feature.
pub struct RuvectorSemanticIndex {
    db: VectorDB,
}

impl RuvectorSemanticIndex {
    /// Create a new in-memory HNSW index with the given `max_elements` capacity.
    ///
    /// Uses cosine distance to match the unit-normalised hash embeddings.
    pub fn new(max_elements: usize) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let options = DbOptions {
            dimensions: EMBEDDING_DIM,
            distance_metric: DistanceMetric::Cosine,
            // storage path is ignored when the `storage` feature is off
            storage_path: ":memory:".to_string(),
            hnsw_config: Some(HnswConfig {
                m: 16,
                ef_construction: 100,
                ef_search: 50,
                max_elements,
            }),
            quantization: None,
        };
        let db = VectorDB::new(options)?;
        Ok(Self { db })
    }

    /// Embed a `State` to a deterministic 8-dimensional unit vector.
    ///
    /// Canonical form: `"{entity_id}={state}|{attributes_json}"`
    /// The attributes JSON is sorted-key (via `serde_json`'s default ordering
    /// of `Map`, which preserves insertion order). For strict canonicalisation
    /// at P3, sort keys explicitly.
    pub fn embed_state(state: &State) -> Vec<f32> {
        let attrs = state.attributes.to_string();
        let input = format!("{}={}|{}", state.entity_id, state.state, attrs);
        Self::embed_str(&input)
    }

    /// Embed an arbitrary string to a deterministic 8-dimensional unit vector.
    pub fn embed_str(input: &str) -> Vec<f32> {
        embed_bytes(input.as_bytes())
    }
}

/// SHA-256 → 8 × f32 unit vector.
///
/// Split the 32-byte digest into 8 chunks of 4 bytes. Interpret each chunk
/// as a big-endian `i32`, cast to `f32`, then L2-normalise.
fn embed_bytes(data: &[u8]) -> Vec<f32> {
    let digest = Sha256::digest(data);
    let mut raw: Vec<f32> = digest
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().expect("chunk is exactly 4 bytes");
            i32::from_be_bytes(bytes) as f32
        })
        .collect();

    // L2-normalise
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut raw {
            *v /= norm;
        }
    }
    raw
}

#[async_trait]
impl SemanticIndex for RuvectorSemanticIndex {
    async fn insert_state(
        &mut self,
        state_id: i64,
        state: &State,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let vector = Self::embed_state(state);
        let entry = VectorEntry {
            id: Some(state_id.to_string()),
            vector,
            metadata: None,
        };
        self.db.insert(entry)?;
        tracing::debug!(state_id, entity_id = %state.entity_id, "semantic index: inserted");
        Ok(())
    }

    async fn search(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<(i64, f32)>, Box<dyn std::error::Error + Send + Sync>> {
        let vector = Self::embed_str(query);
        let results = self.db.search(SearchQuery {
            vector,
            k,
            filter: None,
            ef_search: None,
        })?;
        let hits = results
            .into_iter()
            .filter_map(|r| r.id.parse::<i64>().ok().map(|id| (id, r.score)))
            .collect();
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::RwLock;

    use homecore::entity::{EntityId, State};
    use homecore::event::Context;

    use super::*;
    use crate::db::{Recorder, SemanticIndex};

    fn make_state(entity_id: &str, state_val: &str, attrs: serde_json::Value) -> State {
        let eid = EntityId::parse(entity_id).unwrap();
        let ctx = Context::new();
        State::new(eid, state_val, attrs, ctx)
    }

    // ── embed_state ───────────────────────────────────────────────────────────

    #[test]
    fn embed_state_is_deterministic() {
        let s = make_state("light.kitchen", "on", serde_json::json!({"brightness": 200}));
        let v1 = RuvectorSemanticIndex::embed_state(&s);
        let v2 = RuvectorSemanticIndex::embed_state(&s);
        assert_eq!(v1, v2, "same input must produce identical embedding");
    }

    #[test]
    fn embed_state_is_unit_norm() {
        let s = make_state("sensor.temp", "22.5", serde_json::json!({"unit": "C"}));
        let v = RuvectorSemanticIndex::embed_state(&s);
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-5,
            "embedding must be unit-norm, got norm^2={norm_sq}"
        );
    }

    #[test]
    fn embed_state_dim_is_correct() {
        let s = make_state("binary_sensor.door", "off", serde_json::json!({}));
        let v = RuvectorSemanticIndex::embed_state(&s);
        assert_eq!(v.len(), EMBEDDING_DIM);
    }

    // ── RuvectorSemanticIndex insert + search ─────────────────────────────────

    #[tokio::test]
    async fn insert_then_search_finds_state() {
        let mut idx = RuvectorSemanticIndex::new(1000).unwrap();
        let state = make_state("light.living_room", "on", serde_json::json!({"brightness": 255}));
        idx.insert_state(42, &state).await.unwrap();

        // Query the same canonical string used by embed_state
        let query = format!(
            "{}={}|{}",
            state.entity_id, state.state, state.attributes
        );
        let hits = idx.search(&query, 5).await.unwrap();
        assert!(!hits.is_empty(), "search must return at least one hit");
        assert_eq!(hits[0].0, 42, "top hit must be the inserted state_id");
    }

    #[tokio::test]
    async fn search_ordering_closer_entity_ranks_first() {
        let mut idx = RuvectorSemanticIndex::new(1000).unwrap();

        let s_a = make_state("light.office", "on", serde_json::json!({"brightness": 100}));
        let s_b = make_state("switch.fan", "off", serde_json::json!({}));

        idx.insert_state(1, &s_a).await.unwrap();
        idx.insert_state(2, &s_b).await.unwrap();

        // Query identical to s_a's canonical form → s_a must rank first
        let query_a = format!("{}={}|{}", s_a.entity_id, s_a.state, s_a.attributes);
        let hits = idx.search(&query_a, 2).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0].0, 1,
            "state matching the query must rank first; got {:?}",
            hits
        );
    }

    // ── Recorder end-to-end with RuvectorSemanticIndex ────────────────────────

    #[tokio::test]
    async fn recorder_search_semantic_returns_recorded_state() {
        use homecore::event::StateChangedEvent;
        use chrono::Utc;

        let idx = Arc::new(RwLock::new(
            RuvectorSemanticIndex::new(1000).unwrap(),
        ));
        let semantic: Arc<RwLock<dyn SemanticIndex>> = idx;
        let recorder = Recorder::open_with_index("sqlite::memory:", semantic)
            .await
            .unwrap();

        let state = Arc::new(make_state(
            "sensor.humidity",
            "65",
            serde_json::json!({"unit": "%"}),
        ));
        let event = StateChangedEvent {
            entity_id: state.entity_id.clone(),
            old_state: None,
            new_state: Some(state.clone()),
            fired_at: Utc::now(),
        };
        let state_id = recorder.record_state(&event).await.unwrap().unwrap();

        // Query using the entity prefix — close enough embedding to find it
        let query = format!("{}={}|{}", state.entity_id, state.state, state.attributes);
        let rows = recorder.search_semantic(&query, 5).await.unwrap();
        assert!(!rows.is_empty(), "search_semantic must return at least one row");
        assert_eq!(
            rows[0].state_id, state_id,
            "returned row must match the recorded state"
        );
    }
}
