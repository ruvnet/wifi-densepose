//! Semantic indexing for state attributes — P2 ruvector integration point.
//!
//! This module is **feature-gated** (`--features ruvector`). The trait
//! `SemanticIndex` is defined in [`crate::db`] so it is always available.
//! This module provides the ruvector-backed implementation that will ship
//! once the embedding model boundary is finalised in P2.
//!
//! ## P2 plan
//!
//! 1. Add `ruvector-core` + `ruvector-attention` as optional dependencies.
//! 2. Implement `RuvectorSemanticIndex` here, embedding the serialised
//!    `State.attributes` JSON into a fixed-dimension vector and inserting
//!    it into a ruvector HNSW index keyed by `state_id`.
//! 3. Expose a `search(query: &str, k: usize) -> Vec<StateRow>` helper on
//!    `Recorder` that converts the query string to an embedding and calls
//!    `ruvector_core::HnswIndex::search`.
//!
//! ## Why deferred
//!
//! The embedding model boundary (which model, what dimension, cosine vs
//! dot-product) is still TBD as of ADR-132. Shipping a concrete
//! implementation now would couple the recorder to a specific ruvector
//! version that may need to change once the embedding model is chosen.
//! The no-op `NullSemanticIndex` in P1 keeps the interface stable without
//! locking in that choice.

use std::sync::Arc;

use async_trait::async_trait;

use homecore::entity::State;

use crate::db::SemanticIndex;

/// Stub ruvector-backed semantic index.
///
/// Will be replaced by a real implementation in P2 once the embedding
/// model boundary is confirmed. Currently logs and no-ops.
pub struct RuvectorSemanticIndex;

#[async_trait]
impl SemanticIndex for RuvectorSemanticIndex {
    async fn index_state(
        &self,
        state: &Arc<State>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // P2 TODO: embed state.attributes JSON → f32 vector → ruvector insert.
        tracing::debug!(
            entity_id = %state.entity_id,
            "ruvector semantic index: P2 stub — not yet implemented"
        );
        Ok(())
    }
}
