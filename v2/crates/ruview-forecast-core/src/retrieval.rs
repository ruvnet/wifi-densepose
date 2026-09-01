//! Tenant- and split-scoped analogue retrieval contracts.

use crate::digest::CanonicalWriter;
use crate::series::{check_finite, checked_product, validate_text, MAX_SERIES_VALUES};
use crate::{CanonicalDigest, DataPolicy, ForecastError};
use serde::Serialize;
use std::collections::BTreeSet;

/// Maximum embedding dimension accepted at the core boundary.
pub const MAX_ANALOG_DIMENSION: usize = 4_096;
/// Maximum neighbours in one request.
pub const MAX_ANALOG_K: usize = 64;
const MAX_RECORD_ID_LEN: usize = 128;

/// Immutable tenant/account/workspace/split retrieval partition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RetrievalScope {
    tenant_id: String,
    account_id: String,
    workspace_id: String,
    split_id: String,
}

impl RetrievalScope {
    /// Derive a retrieval scope from an already validated data policy.
    pub fn from_policy(
        policy: &DataPolicy,
        split_id: impl Into<String>,
    ) -> Result<Self, ForecastError> {
        let split_id = split_id.into();
        validate_text("retrieval_split_id", &split_id, MAX_RECORD_ID_LEN, false)?;
        Ok(Self {
            tenant_id: policy.tenant_id().to_owned(),
            account_id: policy.account_id().to_owned(),
            workspace_id: policy.workspace_id().to_owned(),
            split_id,
        })
    }

    /// Tenant identifier.
    #[must_use]
    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    /// Account identifier.
    #[must_use]
    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    /// Workspace identifier.
    #[must_use]
    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    /// Dataset split identifier. Train/test indexes must never be shared.
    #[must_use]
    pub fn split_id(&self) -> &str {
        &self.split_id
    }

    /// Canonical scope digest used to validate every result before use.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"retrieval-scope-v1");
        writer.string(&self.tenant_id);
        writer.string(&self.account_id);
        writer.string(&self.workspace_id);
        writer.string(&self.split_id);
        writer.finish()
    }
}

/// Bounded exact-dimension analogue query.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnalogQuery {
    scope: RetrievalScope,
    embedding: Vec<f32>,
    expected_patch_len: usize,
    k: usize,
}

impl AnalogQuery {
    /// Construct a finite query and bound its total result allocation.
    pub fn new(
        scope: RetrievalScope,
        embedding: Vec<f32>,
        expected_patch_len: usize,
        k: usize,
    ) -> Result<Self, ForecastError> {
        if embedding.is_empty() {
            return Err(ForecastError::EmptyField {
                field: "analog_embedding",
            });
        }
        if embedding.len() > MAX_ANALOG_DIMENSION {
            return Err(ForecastError::LimitExceeded {
                field: "analog_dimension",
                actual: embedding.len(),
                max: MAX_ANALOG_DIMENSION,
            });
        }
        if k == 0 {
            return Err(ForecastError::ZeroValue { field: "analog_k" });
        }
        if k > MAX_ANALOG_K {
            return Err(ForecastError::LimitExceeded {
                field: "analog_k",
                actual: k,
                max: MAX_ANALOG_K,
            });
        }
        if expected_patch_len == 0 {
            return Err(ForecastError::ZeroValue {
                field: "analog_patch_len",
            });
        }
        checked_product(
            "analog_result_cells",
            &[k, expected_patch_len],
            MAX_SERIES_VALUES,
        )?;
        check_finite("analog_embedding", &embedding)?;
        Ok(Self {
            scope,
            embedding,
            expected_patch_len,
            k,
        })
    }

    /// Tenant/workspace/split partition selected before similarity search.
    #[must_use]
    pub fn scope(&self) -> &RetrievalScope {
        &self.scope
    }

    /// Exact finite query embedding.
    #[must_use]
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    /// Required dimension of each retrieved forecast patch.
    #[must_use]
    pub const fn expected_patch_len(&self) -> usize {
        self.expected_patch_len
    }

    /// Maximum returned neighbours.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Deterministic digest of scope, vector, patch shape, and result limit.
    #[must_use]
    pub fn canonical_digest(&self) -> CanonicalDigest {
        let mut writer = CanonicalWriter::new(b"analog-query-v1");
        writer.digest(self.scope.canonical_digest());
        writer.usize(self.embedding.len());
        for value in &self.embedding {
            writer.f32(*value);
        }
        writer.usize(self.expected_patch_len);
        writer.usize(self.k);
        writer.finish()
    }

    /// Validate bound, shape, scope, uniqueness, and deterministic ordering of
    /// results returned by an adapter.
    pub fn validate_results(&self, results: &[AnalogMatch]) -> Result<(), ForecastError> {
        if results.len() > self.k {
            return Err(ForecastError::LimitExceeded {
                field: "analog_results",
                actual: results.len(),
                max: self.k,
            });
        }
        let mut ids = BTreeSet::new();
        for (index, result) in results.iter().enumerate() {
            if result.scope_digest != self.scope.canonical_digest()
                || result.query_digest != self.canonical_digest()
            {
                return Err(ForecastError::RetrievalScopeMismatch);
            }
            if result.forecast_patch.len() != self.expected_patch_len {
                return Err(ForecastError::ShapeMismatch {
                    field: "analog_forecast_patch",
                    expected: self.expected_patch_len,
                    actual: result.forecast_patch.len(),
                });
            }
            if !ids.insert(result.record_id.as_str()) {
                return Err(ForecastError::DuplicateAnalog {
                    record_id: result.record_id.clone(),
                });
            }
            if index > 0 {
                let previous = &results[index - 1];
                let ordered = previous.distance < result.distance
                    || (previous.distance == result.distance
                        && previous.record_id < result.record_id);
                if !ordered {
                    return Err(ForecastError::RetrievalOrder { index });
                }
            }
        }
        Ok(())
    }
}

/// One finite, scope-bound analogue result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnalogMatch {
    record_id: String,
    distance: f32,
    forecast_patch: Vec<f32>,
    metadata_digest: CanonicalDigest,
    scope_digest: CanonicalDigest,
    query_digest: CanonicalDigest,
}

impl AnalogMatch {
    /// Construct a result bound to an exact query.
    pub fn new(
        query: &AnalogQuery,
        record_id: impl Into<String>,
        distance: f32,
        forecast_patch: Vec<f32>,
        metadata_digest: CanonicalDigest,
    ) -> Result<Self, ForecastError> {
        let record_id = record_id.into();
        validate_text("analog_record_id", &record_id, MAX_RECORD_ID_LEN, false)?;
        if !distance.is_finite() || distance < 0.0 {
            return Err(ForecastError::NonFinite {
                field: "analog_distance",
                index: 0,
            });
        }
        if forecast_patch.len() != query.expected_patch_len {
            return Err(ForecastError::ShapeMismatch {
                field: "analog_forecast_patch",
                expected: query.expected_patch_len,
                actual: forecast_patch.len(),
            });
        }
        check_finite("analog_forecast_patch", &forecast_patch)?;
        if metadata_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "analog_metadata_digest",
            });
        }
        Ok(Self {
            record_id,
            distance: if distance == 0.0 { 0.0 } else { distance },
            forecast_patch,
            metadata_digest,
            scope_digest: query.scope.canonical_digest(),
            query_digest: query.canonical_digest(),
        })
    }

    /// Opaque record identifier. It must not encode a subject identity.
    #[must_use]
    pub fn record_id(&self) -> &str {
        &self.record_id
    }

    /// Non-negative finite distance.
    #[must_use]
    pub const fn distance(&self) -> f32 {
        self.distance
    }

    /// Forecast patch with the exact query-declared shape.
    #[must_use]
    pub fn forecast_patch(&self) -> &[f32] {
        &self.forecast_patch
    }

    /// Digest of bounded external metadata, not the metadata itself.
    #[must_use]
    pub const fn metadata_digest(&self) -> CanonicalDigest {
        self.metadata_digest
    }
}

/// Optional analogue retrieval backend.
pub trait AnalogRetriever: Send + Sync {
    /// Exact embedding dimension accepted by this index.
    fn dimension(&self) -> usize;

    /// Retrieve at most `query.k()` results from the already selected scope.
    /// Implementations must call [`AnalogQuery::validate_results`] before
    /// returning success. An empty vector is a valid index result; callers that
    /// require retrieval must convert it to an explicit forecast abstention.
    fn retrieve(&self, query: &AnalogQuery) -> Result<Vec<AnalogMatch>, ForecastError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PrivacyClass, SourceState};

    fn digest(value: &[u8]) -> CanonicalDigest {
        CanonicalDigest::of_bytes(b"retrieval-test", value)
    }

    fn scope() -> RetrievalScope {
        let policy = DataPolicy::new(
            PrivacyClass::P1,
            "tenant",
            "account",
            "workspace",
            "local retrieval",
            digest(b"policy"),
            None,
            None,
            None,
            10_000,
            true,
        )
        .unwrap();
        let _source = SourceState::claimed("fixture").unwrap();
        RetrievalScope::from_policy(&policy, "train").unwrap()
    }

    #[test]
    fn exact_shape_and_deterministic_order_are_enforced() {
        let query = AnalogQuery::new(scope(), vec![1.0, 2.0], 2, 2).unwrap();
        let first = AnalogMatch::new(&query, "a", 0.1, vec![1.0, 2.0], digest(b"a")).unwrap();
        let second = AnalogMatch::new(&query, "b", 0.2, vec![3.0, 4.0], digest(b"b")).unwrap();
        query
            .validate_results(&[first.clone(), second.clone()])
            .unwrap();
        assert!(matches!(
            query.validate_results(&[second, first]),
            Err(ForecastError::RetrievalOrder { .. })
        ));
    }

    #[test]
    fn query_rejects_ragged_or_nonfinite_inputs() {
        assert!(matches!(
            AnalogQuery::new(scope(), vec![f32::NAN], 1, 1),
            Err(ForecastError::NonFinite { .. })
        ));
        let query = AnalogQuery::new(scope(), vec![1.0], 2, 1).unwrap();
        assert!(matches!(
            AnalogMatch::new(&query, "x", 0.0, vec![1.0], digest(b"x")),
            Err(ForecastError::ShapeMismatch { .. })
        ));
    }
}
