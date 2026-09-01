//! Split-isolated, evaluation-only `RuVector` analogue retrieval adapter.

#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the evaluation-only adapter is deliberately nonconstructible downstream"
    )
)]

use std::{collections::BTreeMap, mem::size_of, path::Path, sync::Mutex};

use ruvector_core::{
    types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, VectorEntry},
    VectorDB,
};
use ruview_forecast_core::{
    AnalogMatch, AnalogQuery, AnalogRetriever, CanonicalDigest, ForecastError, RetrievalScope,
    MAX_ANALOG_DIMENSION, MAX_ANALOG_K, MAX_SERIES_VALUES,
};
use tempfile::{Builder, NamedTempFile, TempDir};

// These admission assumptions bind ruvector-core 2.3.0 and the locked
// hnsw_rs 0.3.4 layout. Any dependency update must re-audit this calculation.
const MAX_INDEX_ELEMENTS: usize = 1_024;
const MAX_INDEX_ESTIMATED_BYTES: usize = 128 * 1024 * 1024;
const PEAK_EMBEDDING_COPIES: usize = 8;
const PEAK_PATCH_COPIES: usize = 2;
const HNSW_MAX_LAYERS: usize = 16;
const HNSW_CONNECTIONS: usize = 16;
const HNSW_MAX_LINKS_PER_ELEMENT: usize =
    2 * HNSW_CONNECTIONS + (HNSW_MAX_LAYERS - 1) * HNSW_CONNECTIONS;
// Each live link is budgeted at 128 bytes for its Arc slot, PointWithOrder
// allocation, allocator metadata, and alignment. A further 32 KiB per record
// covers the Point and its sixteen Vec/RwLock layers, five opaque-ID map
// entries, DashMap/BTreeMap nodes, preallocated layer arrays, and bounded
// insertion/search workspaces. Both figures intentionally exceed observed
// 64-bit layouts rather than claiming allocator-exact accounting.
const BUDGET_BYTES_PER_HNSW_LINK: usize = 128;
const BUDGET_FIXED_BYTES_PER_ELEMENT: usize = 32 * 1024;
const OPAQUE_RECORD_ID_BYTES: usize = 64;
const SEARCH_EF: usize = 64;

#[derive(Debug, Clone)]
struct StoredRecord {
    embedding: Vec<f32>,
    forecast_patch: Vec<f32>,
    metadata_digest: CanonicalDigest,
}

struct Inner {
    database: Option<VectorDB>,
    records: BTreeMap<String, StoredRecord>,
}

struct PrivateStorage {
    database_file: Option<NamedTempFile>,
    directory: Option<TempDir>,
}

impl PrivateStorage {
    fn new() -> Result<Self, ForecastError> {
        let directory = Builder::new()
            .prefix("ruview-forecast-ruvector-")
            .tempdir()
            .map_err(|_| invalid_index("ruvector private directory creation failed"))?;
        let database_file = Builder::new()
            .prefix("index-")
            .suffix(".redb")
            .tempfile_in(directory.path())
            .map_err(|_| invalid_index("ruvector private file creation failed"))?;
        enforce_private_permissions(directory.path(), database_file.path())?;
        Ok(Self {
            database_file: Some(database_file),
            directory: Some(directory),
        })
    }

    fn database_path(&self) -> Result<&Path, ForecastError> {
        self.database_file
            .as_ref()
            .map(NamedTempFile::path)
            .ok_or_else(|| invalid_index("ruvector private storage is closed"))
    }

    fn require_memory_only_backend(&self) -> Result<(), ForecastError> {
        let length = self
            .database_file
            .as_ref()
            .ok_or_else(|| invalid_index("ruvector private storage is closed"))?
            .as_file()
            .metadata()
            .map_err(|_| invalid_index("ruvector private file inspection failed"))?
            .len();
        if length != 0 {
            return Err(invalid_index(
                "persistent ruvector backend activation is forbidden",
            ));
        }
        Ok(())
    }

    fn close(&mut self) -> Result<(), ForecastError> {
        let file_result = self
            .database_file
            .take()
            .map_or(Ok(()), NamedTempFile::close)
            .map_err(|_| invalid_index("ruvector private file cleanup failed"));
        let directory_result = self
            .directory
            .take()
            .map_or(Ok(()), TempDir::close)
            .map_err(|_| invalid_index("ruvector private directory cleanup failed"));
        file_result.and(directory_result)
    }
}

/// One bounded in-memory HNSW index permanently bound to a privacy/split scope.
///
/// The service must derive `scope` from its authenticated principal and a
/// validated [`ruview_forecast_core::DataPolicy`], never from request-body
/// identifiers. A distinct value is required for every tenant/account/
/// workspace/split. The dependency is compiled with its memory-only backend.
/// A private temporary directory and empty mode-0600 probe file are retained so
/// construction fails closed if Cargo feature unification unexpectedly enables
/// persistence. [`Self::close`] reports cleanup failure; drop performs the same
/// cleanup as a last resort. Neither path promises memory zeroization.
/// Production retention belongs in a separately reviewed storage adapter. The
/// type has no public constructor, so downstream code cannot wire it to a
/// caller-selected scope.
pub struct RuVectorAnalogIndex {
    scope: RetrievalScope,
    scope_digest: CanonicalDigest,
    dimension: usize,
    patch_len: usize,
    max_k: usize,
    max_elements: usize,
    inner: Mutex<Inner>,
    storage: PrivateStorage,
}

impl RuVectorAnalogIndex {
    /// Construct an empty, bounded, split-isolated Euclidean HNSW index.
    pub(crate) fn new(
        scope: RetrievalScope,
        dimension: usize,
        patch_len: usize,
        max_k: usize,
        max_elements: usize,
    ) -> Result<Self, ForecastError> {
        for (field, value) in [
            ("ruvector_dimension", dimension),
            ("ruvector_patch_len", patch_len),
            ("ruvector_max_k", max_k),
            ("ruvector_max_elements", max_elements),
        ] {
            require_nonzero(field, value)?;
        }
        require_limit("ruvector_dimension", dimension, MAX_ANALOG_DIMENSION)?;
        require_limit("ruvector_max_k", max_k, MAX_ANALOG_K)?;
        require_limit("ruvector_max_elements", max_elements, MAX_INDEX_ELEMENTS)?;

        let estimated_bytes = estimate_index_bytes(dimension, patch_len, max_k, max_elements)?;
        require_limit(
            "ruvector_estimated_bytes",
            estimated_bytes,
            MAX_INDEX_ESTIMATED_BYTES,
        )?;

        let mut storage = PrivateStorage::new()?;
        let database = match make_database(dimension, max_elements, storage.database_path()?) {
            Ok(database) => database,
            Err(error) => {
                return Err(storage.close().err().unwrap_or(error));
            }
        };
        if let Err(error) = storage.require_memory_only_backend() {
            drop(database);
            return Err(storage.close().err().unwrap_or(error));
        }
        let scope_digest = scope.canonical_digest();
        Ok(Self {
            scope,
            scope_digest,
            dimension,
            patch_len,
            max_k,
            max_elements,
            inner: Mutex::new(Inner {
                database: Some(database),
                records: BTreeMap::new(),
            }),
            storage,
        })
    }

    /// Scope permanently selected for this index.
    #[must_use]
    pub const fn scope(&self) -> &RetrievalScope {
        &self.scope
    }

    /// Delete all retained embeddings and forecast patches.
    pub fn clear(&self) -> Result<(), ForecastError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| invalid_index("ruvector index lock poisoned"))?;
        inner.records.clear();
        // ruvector-core's HNSW remove operation retains Point vectors. Drop the
        // complete database before allocating its replacement so clear is a
        // real release boundary and cannot temporarily double the index.
        inner.database.take();
        let database = make_database(
            self.dimension,
            self.max_elements,
            self.storage.database_path()?,
        )?;
        self.storage.require_memory_only_backend()?;
        inner.database = Some(database);
        Ok(())
    }

    /// Close the memory index and remove its private temporary capability.
    pub fn close(mut self) -> Result<(), ForecastError> {
        self.close_resources()
    }

    /// Insert one finite embedding and its same-split forecast patch.
    ///
    /// `record_id` must be a lowercase 32-byte digest encoded as 64 hex bytes;
    /// human identifiers and PII-shaped text are rejected.
    pub fn insert(
        &self,
        record_id: impl Into<String>,
        embedding: Vec<f32>,
        forecast_patch: Vec<f32>,
        metadata_digest: CanonicalDigest,
    ) -> Result<(), ForecastError> {
        let record_id = record_id.into();
        validate_record_id(&record_id)?;
        require_shape("ruvector_embedding", self.dimension, embedding.len())?;
        require_shape(
            "ruvector_forecast_patch",
            self.patch_len,
            forecast_patch.len(),
        )?;
        check_finite("ruvector_embedding", &embedding)?;
        check_finite("ruvector_forecast_patch", &forecast_patch)?;
        if metadata_digest.is_zero() {
            return Err(ForecastError::ZeroDigest {
                field: "ruvector_metadata_digest",
            });
        }

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| invalid_index("ruvector index lock poisoned"))?;
        if inner.records.contains_key(&record_id) {
            return Err(ForecastError::DuplicateAnalog { record_id });
        }
        if inner.records.len() >= self.max_elements {
            return Err(ForecastError::LimitExceeded {
                field: "ruvector_records",
                actual: inner.records.len() + 1,
                max: self.max_elements,
            });
        }
        inner
            .database
            .as_mut()
            .ok_or_else(|| invalid_index("ruvector index is unavailable"))?
            .insert(VectorEntry {
                id: Some(record_id.clone()),
                vector: embedding.clone(),
                metadata: None,
            })
            .map_err(|_| invalid_index("ruvector insertion failed"))?;
        inner.records.insert(
            record_id,
            StoredRecord {
                embedding,
                forecast_patch,
                metadata_digest,
            },
        );
        Ok(())
    }

    fn close_resources(&mut self) -> Result<(), ForecastError> {
        let inner = match self.inner.get_mut() {
            Ok(inner) => inner,
            Err(poisoned) => poisoned.into_inner(),
        };
        inner.records.clear();
        // The backend must be dropped before its capability file and directory.
        inner.database.take();
        self.storage.close()
    }
}

impl Drop for RuVectorAnalogIndex {
    fn drop(&mut self) {
        let _ = self.close_resources();
    }
}

impl AnalogRetriever for RuVectorAnalogIndex {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn retrieve(&self, query: &AnalogQuery) -> Result<Vec<AnalogMatch>, ForecastError> {
        require_shape("analog_embedding", self.dimension, query.embedding().len())?;
        require_shape(
            "analog_forecast_patch",
            self.patch_len,
            query.expected_patch_len(),
        )?;
        require_limit("analog_k", query.k(), self.max_k)?;
        if query.scope().canonical_digest() != self.scope_digest {
            return Err(ForecastError::RetrievalScopeMismatch);
        }

        let inner = self
            .inner
            .lock()
            .map_err(|_| invalid_index("ruvector index lock poisoned"))?;
        if inner.records.is_empty() {
            let results = Vec::new();
            query.validate_results(&results)?;
            return Ok(results);
        }
        // The vendored HNSW implementation has a panic path for a one-record
        // graph. Exact L2 is both safer and cheaper for that case.
        if inner.records.len() == 1 {
            let (record_id, stored) = inner
                .records
                .first_key_value()
                .ok_or_else(|| invalid_index("ruvector record disappeared"))?;
            let result = AnalogMatch::new(
                query,
                record_id.clone(),
                l2_distance(query.embedding(), &stored.embedding)?,
                stored.forecast_patch.clone(),
                stored.metadata_digest,
            )?;
            let results = vec![result];
            query.validate_results(&results)?;
            return Ok(results);
        }
        let hits = inner
            .database
            .as_ref()
            .ok_or_else(|| invalid_index("ruvector index is unavailable"))?
            .search(SearchQuery {
                vector: query.embedding().to_vec(),
                k: query.k(),
                filter: None,
                // k is at most 64; fixed ef bounds approximate-search work
                // independently of the index capacity.
                ef_search: Some(SEARCH_EF),
            })
            .map_err(|_| invalid_index("ruvector search failed"))?;

        // Treat RuVector output as untrusted. Duplicate IDs are collapsed to
        // their smallest finite non-negative distance before ordering.
        let mut unique = BTreeMap::<String, f32>::new();
        for hit in hits {
            if !hit.score.is_finite() || hit.score < 0.0 {
                return Err(invalid_index("ruvector returned an invalid distance"));
            }
            unique
                .entry(hit.id)
                .and_modify(|distance| {
                    if hit.score.total_cmp(distance).is_lt() {
                        *distance = hit.score;
                    }
                })
                .or_insert(hit.score);
        }
        let mut scored = unique
            .into_iter()
            .map(|(record_id, distance)| (distance, record_id))
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        scored.truncate(query.k());

        let mut results = Vec::with_capacity(scored.len());
        for (distance, record_id) in scored {
            let record = inner
                .records
                .get(&record_id)
                .ok_or_else(|| invalid_index("ruvector result is not bound to a record"))?;
            results.push(AnalogMatch::new(
                query,
                record_id,
                distance,
                record.forecast_patch.clone(),
                record.metadata_digest,
            )?);
        }
        query.validate_results(&results)?;
        Ok(results)
    }
}

fn estimate_index_bytes(
    dimension: usize,
    patch_len: usize,
    max_k: usize,
    max_elements: usize,
) -> Result<usize, ForecastError> {
    let embedding_cells = checked_product("ruvector_embedding_cells", dimension, max_elements)?;
    let resident_embedding_cells = checked_product(
        "ruvector_resident_embedding_cells",
        embedding_cells,
        PEAK_EMBEDDING_COPIES,
    )?;
    let search_result_count = max_k.min(max_elements);
    let search_embedding_copies =
        search_result_count
            .checked_add(2)
            .ok_or(ForecastError::SizeOverflow {
                field: "ruvector_search_embedding_copies",
            })?;
    let search_embedding_cells = checked_product(
        "ruvector_search_embedding_cells",
        dimension,
        search_embedding_copies,
    )?;
    let peak_embedding_cells = resident_embedding_cells
        .checked_add(search_embedding_cells)
        .ok_or(ForecastError::SizeOverflow {
            field: "ruvector_peak_embedding_cells",
        })?;
    let embedding_bytes = checked_product(
        "ruvector_embedding_bytes",
        peak_embedding_cells,
        size_of::<f32>(),
    )?;

    let patch_cells = checked_product("ruvector_patch_cells", patch_len, max_elements)?;
    require_limit("ruvector_patch_cells", patch_cells, MAX_SERIES_VALUES)?;
    let resident_patch_cells = checked_product(
        "ruvector_resident_patch_cells",
        patch_cells,
        PEAK_PATCH_COPIES,
    )?;
    let search_patch_cells = checked_product(
        "ruvector_search_patch_cells",
        patch_len,
        search_result_count,
    )?;
    let peak_patch_cells = resident_patch_cells.checked_add(search_patch_cells).ok_or(
        ForecastError::SizeOverflow {
            field: "ruvector_peak_patch_cells",
        },
    )?;
    let patch_bytes = checked_product("ruvector_patch_bytes", peak_patch_cells, size_of::<f32>())?;

    let link_bytes_per_element = checked_product(
        "ruvector_hnsw_link_bytes_per_element",
        HNSW_MAX_LINKS_PER_ELEMENT,
        BUDGET_BYTES_PER_HNSW_LINK,
    )?;
    let graph_bytes_per_element = link_bytes_per_element
        .checked_add(BUDGET_FIXED_BYTES_PER_ELEMENT)
        .ok_or(ForecastError::SizeOverflow {
            field: "ruvector_graph_bytes_per_element",
        })?;
    let graph_bytes = checked_product(
        "ruvector_graph_bytes",
        max_elements,
        graph_bytes_per_element,
    )?;

    embedding_bytes
        .checked_add(patch_bytes)
        .and_then(|value| value.checked_add(graph_bytes))
        .ok_or(ForecastError::SizeOverflow {
            field: "ruvector_estimated_bytes",
        })
}

fn make_database(
    dimension: usize,
    max_elements: usize,
    storage_path: &Path,
) -> Result<VectorDB, ForecastError> {
    let storage_path = storage_path
        .to_str()
        .ok_or_else(|| invalid_index("ruvector private path is not UTF-8"))?;
    VectorDB::new(DbOptions {
        dimensions: dimension,
        distance_metric: DistanceMetric::Euclidean,
        storage_path: storage_path.to_owned(),
        hnsw_config: Some(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: SEARCH_EF,
            max_elements,
        }),
        quantization: None,
    })
    .map_err(|_| invalid_index("ruvector index construction failed"))
}

#[cfg(unix)]
fn enforce_private_permissions(
    directory: &Path,
    database_file: &Path,
) -> Result<(), ForecastError> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))
        .map_err(|_| invalid_index("ruvector private directory permissions failed"))?;
    std::fs::set_permissions(database_file, std::fs::Permissions::from_mode(0o600))
        .map_err(|_| invalid_index("ruvector private file permissions failed"))?;
    let directory_mode = std::fs::metadata(directory)
        .map_err(|_| invalid_index("ruvector private directory inspection failed"))?
        .permissions()
        .mode()
        & 0o777;
    let file_mode = std::fs::metadata(database_file)
        .map_err(|_| invalid_index("ruvector private file inspection failed"))?
        .permissions()
        .mode()
        & 0o777;
    if directory_mode != 0o700 || file_mode != 0o600 {
        return Err(invalid_index("ruvector private permissions are unsafe"));
    }
    Ok(())
}

#[cfg(not(unix))]
fn enforce_private_permissions(
    _directory: &Path,
    _database_file: &Path,
) -> Result<(), ForecastError> {
    Ok(())
}

const fn invalid_index(reason: &'static str) -> ForecastError {
    ForecastError::InvalidSourceState { reason }
}

fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize, ForecastError> {
    left.checked_mul(right)
        .ok_or(ForecastError::SizeOverflow { field })
}

fn require_nonzero(field: &'static str, value: usize) -> Result<(), ForecastError> {
    if value == 0 {
        return Err(ForecastError::ZeroValue { field });
    }
    Ok(())
}

fn require_limit(field: &'static str, value: usize, maximum: usize) -> Result<(), ForecastError> {
    if value > maximum {
        return Err(ForecastError::LimitExceeded {
            field,
            actual: value,
            max: maximum,
        });
    }
    Ok(())
}

fn require_shape(field: &'static str, expected: usize, actual: usize) -> Result<(), ForecastError> {
    if expected != actual {
        return Err(ForecastError::ShapeMismatch {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn check_finite(field: &'static str, values: &[f32]) -> Result<(), ForecastError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ForecastError::NonFinite { field, index });
    }
    Ok(())
}

fn l2_distance(left: &[f32], right: &[f32]) -> Result<f32, ForecastError> {
    require_shape("ruvector_exact_distance", left.len(), right.len())?;
    let squared = left
        .iter()
        .zip(right)
        .try_fold(0.0_f32, |sum, (left, right)| {
            let delta = left - right;
            let next = sum + delta * delta;
            next.is_finite()
                .then_some(next)
                .ok_or_else(|| invalid_index("ruvector exact distance overflowed"))
        })?;
    Ok(squared.sqrt())
}

fn validate_record_id(record_id: &str) -> Result<(), ForecastError> {
    if record_id.len() != OPAQUE_RECORD_ID_BYTES
        || !record_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ForecastError::InvalidText {
            field: "ruvector_record_id",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use ruview_forecast_core::{AnalogRetriever, DataPolicy, PrivacyClass};

    use super::*;

    fn digest(value: &[u8]) -> CanonicalDigest {
        CanonicalDigest::of_bytes(b"ruvector-adapter-test", value)
    }

    fn id(value: &[u8]) -> String {
        digest(value).to_hex()
    }

    fn scope(split: &str) -> RetrievalScope {
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
        RetrievalScope::from_policy(&policy, split).unwrap()
    }

    #[test]
    fn empty_index_is_safe_and_wrong_split_is_rejected() {
        let index = RuVectorAnalogIndex::new(scope("train"), 2, 3, 4, 8).unwrap();
        let query = AnalogQuery::new(scope("train"), vec![0.0, 1.0], 3, 2).unwrap();
        assert!(index.retrieve(&query).unwrap().is_empty());
        let wrong_split = AnalogQuery::new(scope("test"), vec![0.0, 1.0], 3, 2).unwrap();
        assert!(matches!(
            index.retrieve(&wrong_split),
            Err(ForecastError::RetrievalScopeMismatch)
        ));
    }

    #[test]
    fn neighbours_are_scope_bound_and_deterministically_ordered() {
        let train = scope("train");
        let index = RuVectorAnalogIndex::new(train.clone(), 2, 2, 4, 8).unwrap();
        index
            .insert(id(b"b"), vec![1.0, 0.0], vec![2.0, 3.0], digest(b"b"))
            .unwrap();
        index
            .insert(id(b"a"), vec![0.0, 1.0], vec![4.0, 5.0], digest(b"a"))
            .unwrap();
        let query = AnalogQuery::new(train, vec![1.0, 0.0], 2, 2).unwrap();
        let first = index.retrieve(&query).unwrap();
        let second = index.retrieve(&query).unwrap();
        assert_eq!(first, second);
        query.validate_results(&first).unwrap();
    }

    #[test]
    fn capacity_and_pii_shaped_identifiers_fail_before_index_work() {
        assert!(matches!(
            RuVectorAnalogIndex::new(scope("train"), 4_096, 1, 64, MAX_INDEX_ELEMENTS),
            Err(ForecastError::LimitExceeded {
                field: "ruvector_estimated_bytes",
                ..
            })
        ));
        assert!(matches!(
            RuVectorAnalogIndex::new(scope("train"), 2, 2, 1, MAX_INDEX_ELEMENTS + 1),
            Err(ForecastError::LimitExceeded {
                field: "ruvector_max_elements",
                actual,
                max: MAX_INDEX_ELEMENTS,
            }) if actual == MAX_INDEX_ELEMENTS + 1
        ));
        let index = RuVectorAnalogIndex::new(scope("train"), 2, 2, 1, 1).unwrap();
        assert!(matches!(
            index.insert(
                "alice@example.com",
                vec![0.0, 1.0],
                vec![1.0, 2.0],
                digest(b"metadata")
            ),
            Err(ForecastError::InvalidText {
                field: "ruvector_record_id"
            })
        ));
    }

    #[test]
    fn clear_removes_all_retained_records() {
        let train = scope("train");
        let index = RuVectorAnalogIndex::new(train.clone(), 2, 2, 2, 2).unwrap();
        let record_id = id(b"a");
        index
            .insert(
                record_id.clone(),
                vec![0.0, 1.0],
                vec![1.0, 2.0],
                digest(b"a"),
            )
            .unwrap();
        index
            .insert(id(b"b"), vec![1.0, 0.0], vec![3.0, 4.0], digest(b"b"))
            .unwrap();
        index.clear().unwrap();
        let query = AnalogQuery::new(train, vec![0.0, 1.0], 2, 1).unwrap();
        assert!(index.retrieve(&query).unwrap().is_empty());
        index
            .insert(
                record_id.clone(),
                vec![0.0, 1.0],
                vec![9.0, 10.0],
                digest(b"replacement"),
            )
            .unwrap();
        let matches = index.retrieve(&query).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].record_id(), record_id);
        assert_eq!(matches[0].forecast_patch(), &[9.0, 10.0]);
    }

    #[test]
    fn maximum_element_capacity_obeys_the_admission_budget() {
        let bytes = estimate_index_bytes(64, 16, 64, MAX_INDEX_ELEMENTS).unwrap();
        assert_eq!(bytes, 71_455_232);
        assert!(bytes <= MAX_INDEX_ESTIMATED_BYTES);
        let index =
            RuVectorAnalogIndex::new(scope("train"), 64, 16, 64, MAX_INDEX_ELEMENTS).unwrap();
        index.close().unwrap();
    }

    #[test]
    fn one_record_uses_safe_exact_search() {
        let train = scope("train");
        let index = RuVectorAnalogIndex::new(train.clone(), 2, 2, 1, 1).unwrap();
        let record_id = id(b"only");
        index
            .insert(
                record_id.clone(),
                vec![3.0, 4.0],
                vec![1.0, 2.0],
                digest(b"only"),
            )
            .unwrap();
        let query = AnalogQuery::new(train, vec![0.0, 0.0], 2, 1).unwrap();
        let matches = index.retrieve(&query).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].record_id(), record_id);
        assert_eq!(matches[0].distance().to_bits(), 5.0_f32.to_bits());
    }

    #[test]
    fn private_storage_is_mode_limited_and_removed_after_drop() {
        let current_directory = std::env::current_dir().unwrap();
        let legacy_path = current_directory.join(":memory:");
        assert!(!legacy_path.exists());

        let (directory_path, database_path) = {
            let index = RuVectorAnalogIndex::new(scope("train"), 2, 2, 1, 1).unwrap();
            let directory_path = index
                .storage
                .directory
                .as_ref()
                .unwrap()
                .path()
                .to_path_buf();
            let database_path = index.storage.database_path().unwrap().to_path_buf();
            assert!(directory_path.is_dir());
            assert!(database_path.is_file());
            assert!(database_path.starts_with(&directory_path));
            assert!(!directory_path.starts_with(&current_directory));
            assert_eq!(std::fs::metadata(&database_path).unwrap().len(), 0);

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;

                let directory_mode = std::fs::metadata(&directory_path)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777;
                let file_mode = std::fs::metadata(&database_path)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777;
                assert_eq!(directory_mode, 0o700);
                assert_eq!(file_mode, 0o600);
            }

            index
                .insert(
                    id(b"private"),
                    vec![0.0, 1.0],
                    vec![1.0, 2.0],
                    digest(b"private"),
                )
                .unwrap();
            assert_eq!(std::fs::metadata(&database_path).unwrap().len(), 0);
            (directory_path, database_path)
        };

        assert!(!database_path.exists());
        assert!(!directory_path.exists());
        assert!(!legacy_path.exists());
    }

    #[test]
    fn explicit_close_removes_private_storage() {
        let index = RuVectorAnalogIndex::new(scope("train"), 2, 2, 1, 1).unwrap();
        let directory_path = index
            .storage
            .directory
            .as_ref()
            .unwrap()
            .path()
            .to_path_buf();
        let database_path = index.storage.database_path().unwrap().to_path_buf();
        index.close().unwrap();
        assert!(!database_path.exists());
        assert!(!directory_path.exists());
    }

    #[test]
    fn persistence_probe_fails_closed() {
        let mut storage = PrivateStorage::new().unwrap();
        let directory_path = storage.directory.as_ref().unwrap().path().to_path_buf();
        storage
            .database_file
            .as_mut()
            .unwrap()
            .write_all(b"persistent-backend-probe")
            .unwrap();
        assert!(matches!(
            storage.require_memory_only_backend(),
            Err(ForecastError::InvalidSourceState {
                reason: "persistent ruvector backend activation is forbidden"
            })
        ));
        storage.close().unwrap();
        assert!(!directory_path.exists());
    }
}
