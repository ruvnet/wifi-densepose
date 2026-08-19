//! Tenant-scoped spatial history and bounded anomaly explanations (ADR-326).
//!
//! This crate accepts semantic P2/P3 records only. It deliberately has no
//! field for CSI, RF tensors, recordings, biometrics, or identity observations.
//! Tenant and workspace select a physical RuVector index before ANN executes;
//! callers cannot search a shared index and filter the result afterward.

#![forbid(unsafe_code)]

use chacha20poly1305::aead::{Aead, Payload};
use chacha20poly1305::{KeyInit, XChaCha20Poly1305, XNonce};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use wifi_densepose_ruvector::{HnswIndex, HnswParams, Metric};
use zeroize::Zeroizing;

const FILE_MAGIC: &[u8; 8] = b"RVSM01\0\0";
const FORMAT_VERSION: u16 = 1;
const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_RECORDS: usize = 10_000;
const MAX_FEATURES: usize = 1_024;
const MAX_ID_BYTES: usize = 128;
const MAX_DERIVATIONS: usize = 32;
const MAX_KEY_ID_BYTES: usize = 128;
const MAX_K: usize = 100;

/// Errors are fail-closed and avoid embedding tenant data or key material.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    /// A caller-provided record or query violated a bound or invariant.
    #[error("invalid spatial memory input: {0}")]
    Invalid(&'static str),
    /// A message identifier was reused for different content.
    #[error("message identifier reuse conflicts with an existing record")]
    ReplayConflict,
    /// A source sequence did not advance monotonically.
    #[error("source sequence is stale")]
    StaleSequence,
    /// A derivation reference is absent or cyclic.
    #[error("invalid derivation graph")]
    InvalidDerivation,
    /// A destination exists; encrypted state is never overwritten implicitly.
    #[error("destination already exists")]
    AlreadyExists,
    /// The encrypted envelope is malformed, oversized, or unauthenticated.
    #[error("encrypted spatial memory envelope is invalid")]
    InvalidEnvelope,
    /// The requested key identifier is not available.
    #[error("spatial memory key is unavailable")]
    KeyUnavailable,
    /// A bounded filesystem operation failed.
    #[error("spatial memory I/O failed")]
    Io,
}

/// Tenant/workspace isolation key. Both values are authenticated upstream.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PartitionKey {
    /// Cognitum tenant identifier.
    pub tenant_id: String,
    /// Cognitum workspace identifier.
    pub workspace_id: String,
}

/// Honest evidence grade carried into explanations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceGrade {
    /// Synthetic or simulator-only evidence.
    L0,
    /// Laboratory evidence.
    L1,
    /// Deployment evidence without held-out validation.
    L2,
    /// Held-out deployment evidence.
    L3,
}

/// Whether the record is directly observed semantic state or a derivation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecordClass {
    /// Locally normalized P2/P3 observation.
    Observation,
    /// Bounded inference derived from earlier records.
    Inference,
}

/// A bounded semantic spatial record. Arbitrary payload fields are absent by design.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Stable record/message identifier.
    pub record_id: String,
    /// Publisher message identity, deduplicated with `source_id`.
    pub message_id: String,
    /// Authenticated tenant/workspace partition.
    pub partition: PartitionKey,
    /// Site containing this semantic record.
    pub site_id: String,
    /// Optional room/space scope.
    pub space_id: Option<String>,
    /// Versioned feature/schema contract.
    pub schema_version: String,
    /// Stable source identifier used for sequence replay defense.
    pub source_id: String,
    /// Strictly increasing sequence for this source within the partition.
    pub source_sequence: u64,
    /// Observation timestamp in Unix milliseconds.
    pub observed_at_ms: i64,
    /// Query eligibility end time in Unix milliseconds.
    pub expires_at_ms: i64,
    /// Mandatory deletion deadline in Unix milliseconds.
    pub retention_until_ms: i64,
    /// Observation or explicitly derived inference.
    pub class: RecordClass,
    /// Fixed-dimensional semantic feature vector.
    pub features: Vec<f32>,
    /// Calibrated uncertainty in `[0, 1]`.
    pub uncertainty: f32,
    /// Evidence grade, preserved into every explanation.
    pub evidence_grade: EvidenceGrade,
    /// Digest of the normalized source event.
    pub provenance_digest: [u8; 32],
    /// Optional digest of the witness-chain receipt.
    pub witness_digest: Option<[u8; 32]>,
    /// Earlier record identifiers used by an inference.
    pub derived_from: Vec<String>,
}

/// A partition-bound nearest-neighbor query.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchQuery {
    /// Exact tenant/workspace selected before ANN.
    pub partition: PartitionKey,
    /// Query embedding with the configured dimension.
    pub features: Vec<f32>,
    /// Optional site constraint.
    pub site_id: Option<String>,
    /// Optional space constraint.
    pub space_id: Option<String>,
    /// Optional exact schema constraint.
    pub schema_version: Option<String>,
    /// Inclusive observation lower bound.
    pub observed_from_ms: Option<i64>,
    /// Exclusive observation upper bound.
    pub observed_before_ms: Option<i64>,
    /// Maximum results, from 1 through 100.
    pub k: usize,
    /// Injected current time used for expiry checks.
    pub now_ms: i64,
}

/// One tenant-bound historical match.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchMatch {
    /// Matched record identifier.
    pub record_id: String,
    /// RuVector distance; smaller is more similar.
    pub distance: f32,
    /// Original uncertainty.
    pub uncertainty: f32,
    /// Original evidence grade.
    pub evidence_grade: EvidenceGrade,
    /// Original provenance digest.
    pub provenance_digest: [u8; 32],
    /// Optional witness digest.
    pub witness_digest: Option<[u8; 32]>,
}

/// An honest anomaly explanation grounded only in same-partition history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnomalyExplanation {
    /// Exact authenticated boundary used by the search.
    pub partition: PartitionKey,
    /// Timestamp supplied by the caller.
    pub generated_at_ms: i64,
    /// Deliberately modest explanation text; no causal or accuracy claim.
    pub basis: String,
    /// Historical evidence ordered by RuVector distance.
    pub matches: Vec<SearchMatch>,
}

#[derive(Debug)]
struct Partition {
    records: Vec<MemoryRecord>,
    by_id: BTreeMap<String, usize>,
    messages: BTreeMap<(String, String), usize>,
    source_sequences: BTreeMap<String, u64>,
    index: HnswIndex,
}

impl Partition {
    fn empty(dim: usize) -> Self {
        Self {
            records: Vec::new(),
            by_id: BTreeMap::new(),
            messages: BTreeMap::new(),
            source_sequences: BTreeMap::new(),
            index: HnswIndex::new(dim, Metric::Cosine, HnswParams::default()),
        }
    }

    fn rebuild(&mut self, dim: usize) {
        self.by_id.clear();
        self.messages.clear();
        self.source_sequences.clear();
        self.index = HnswIndex::new(dim, Metric::Cosine, HnswParams::default());
        for (position, record) in self.records.iter().enumerate() {
            self.by_id.insert(record.record_id.clone(), position);
            self.messages.insert(
                (record.source_id.clone(), record.message_id.clone()),
                position,
            );
            self.source_sequences
                .entry(record.source_id.clone())
                .and_modify(|value| *value = (*value).max(record.source_sequence))
                .or_insert(record.source_sequence);
            let assigned = self.index.insert(&record.features);
            debug_assert_eq!(assigned as usize, position);
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PersistedState {
    format_version: u16,
    dimension: usize,
    records: Vec<MemoryRecord>,
}

/// In-memory tenant-partitioned RuVector store.
#[derive(Debug)]
pub struct SpatialMemory {
    dimension: usize,
    partitions: BTreeMap<PartitionKey, Partition>,
}

impl SpatialMemory {
    /// Create an empty store with a fixed embedding dimension.
    pub fn new(dimension: usize) -> Result<Self, Error> {
        if dimension == 0 || dimension > MAX_FEATURES {
            return Err(Error::Invalid("feature dimension is out of bounds"));
        }
        Ok(Self {
            dimension,
            partitions: BTreeMap::new(),
        })
    }

    /// Return the fixed embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Count records across all isolated partitions.
    pub fn len(&self) -> usize {
        self.partitions
            .values()
            .map(|partition| partition.records.len())
            .sum()
    }

    /// Return whether no records are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ingest a validated semantic record with replay and derivation defenses.
    pub fn ingest(&mut self, record: MemoryRecord, now_ms: i64) -> Result<(), Error> {
        validate_record(&record, self.dimension, now_ms)?;
        if self.len() >= MAX_RECORDS {
            return Err(Error::Invalid("record capacity reached"));
        }
        let partition = self
            .partitions
            .entry(record.partition.clone())
            .or_insert_with(|| Partition::empty(self.dimension));
        if let Some(position) = partition.by_id.get(&record.record_id) {
            return if partition.records[*position] == record {
                Ok(())
            } else {
                Err(Error::ReplayConflict)
            };
        }
        if let Some(position) = partition
            .messages
            .get(&(record.source_id.clone(), record.message_id.clone()))
        {
            return if partition.records[*position] == record {
                Ok(())
            } else {
                Err(Error::ReplayConflict)
            };
        }
        if partition
            .source_sequences
            .get(&record.source_id)
            .is_some_and(|sequence| record.source_sequence <= *sequence)
        {
            return Err(Error::StaleSequence);
        }
        if record
            .derived_from
            .iter()
            .any(|parent| !partition.by_id.contains_key(parent))
        {
            return Err(Error::InvalidDerivation);
        }
        let position = partition.records.len();
        let assigned = partition.index.insert(&record.features);
        debug_assert_eq!(assigned as usize, position);
        partition
            .source_sequences
            .insert(record.source_id.clone(), record.source_sequence);
        partition.by_id.insert(record.record_id.clone(), position);
        partition.messages.insert(
            (record.source_id.clone(), record.message_id.clone()),
            position,
        );
        partition.records.push(record);
        Ok(())
    }

    /// Search only the exact tenant/workspace index selected by `query`.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchMatch>, Error> {
        validate_partition(&query.partition)?;
        validate_features(&query.features, self.dimension)?;
        if query.k == 0 || query.k > MAX_K {
            return Err(Error::Invalid("search result count is out of bounds"));
        }
        for value in [
            query.site_id.as_deref(),
            query.space_id.as_deref(),
            query.schema_version.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            validate_id(value)?;
        }
        if query
            .observed_from_ms
            .zip(query.observed_before_ms)
            .is_some_and(|(from, before)| from >= before)
        {
            return Err(Error::Invalid("search time range is invalid"));
        }
        let Some(partition) = self.partitions.get(&query.partition) else {
            return Ok(Vec::new());
        };
        // Request all candidates inside this already isolated partition so
        // secondary site/space/time predicates cannot reduce result quality.
        let candidates = partition
            .index
            .search_default(&query.features, partition.records.len());
        let mut matches = Vec::with_capacity(query.k.min(candidates.len()));
        for (id, distance) in candidates {
            let Some(record) = partition.records.get(id as usize) else {
                continue;
            };
            if record.expires_at_ms <= query.now_ms || record.retention_until_ms <= query.now_ms {
                continue;
            }
            if query
                .site_id
                .as_ref()
                .is_some_and(|site| site != &record.site_id)
                || query
                    .space_id
                    .as_ref()
                    .is_some_and(|space| record.space_id.as_ref() != Some(space))
                || query
                    .schema_version
                    .as_ref()
                    .is_some_and(|version| version != &record.schema_version)
                || query
                    .observed_from_ms
                    .is_some_and(|from| record.observed_at_ms < from)
                || query
                    .observed_before_ms
                    .is_some_and(|before| record.observed_at_ms >= before)
            {
                continue;
            }
            matches.push(SearchMatch {
                record_id: record.record_id.clone(),
                distance,
                uncertainty: record.uncertainty,
                evidence_grade: record.evidence_grade,
                provenance_digest: record.provenance_digest,
                witness_digest: record.witness_digest,
            });
            if matches.len() == query.k {
                break;
            }
        }
        Ok(matches)
    }

    /// Build an explanation that states similarity, not cause or certainty.
    pub fn explain(&self, query: &SearchQuery) -> Result<AnomalyExplanation, Error> {
        Ok(AnomalyExplanation {
            partition: query.partition.clone(),
            generated_at_ms: query.now_ms,
            basis: "nearest bounded semantic records in the same authenticated tenant/workspace; similarity is not causation".into(),
            matches: self.search(query)?,
        })
    }

    /// Purge expired records and dependent inferences, then rebuild affected indexes.
    pub fn purge_expired(&mut self, now_ms: i64) -> usize {
        let before = self.len();
        self.partitions.retain(|_, partition| {
            partition.records.retain(|record| {
                record.expires_at_ms > now_ms && record.retention_until_ms > now_ms
            });
            loop {
                let ids: BTreeSet<String> = partition
                    .records
                    .iter()
                    .map(|r| r.record_id.clone())
                    .collect();
                let prior = partition.records.len();
                partition.records.retain(|record| {
                    record
                        .derived_from
                        .iter()
                        .all(|parent| ids.contains(parent))
                });
                if partition.records.len() == prior {
                    break;
                }
            }
            if partition.records.is_empty() {
                false
            } else {
                partition.rebuild(self.dimension);
                true
            }
        });
        before - self.len()
    }

    /// Erase an exact tenant/workspace partition.
    pub fn erase_partition(&mut self, partition: &PartitionKey) -> usize {
        self.partitions
            .remove(partition)
            .map_or(0, |value| value.records.len())
    }

    /// Erase one record plus any transitive inference that derives from it.
    pub fn erase_record(&mut self, partition_key: &PartitionKey, record_id: &str) -> usize {
        let Some(partition) = self.partitions.get_mut(partition_key) else {
            return 0;
        };
        if !partition.by_id.contains_key(record_id) {
            return 0;
        }
        let before = partition.records.len();
        let mut removed = BTreeSet::from([record_id.to_owned()]);
        loop {
            let prior = removed.len();
            for record in &partition.records {
                if record
                    .derived_from
                    .iter()
                    .any(|parent| removed.contains(parent))
                {
                    removed.insert(record.record_id.clone());
                }
            }
            if removed.len() == prior {
                break;
            }
        }
        partition
            .records
            .retain(|record| !removed.contains(&record.record_id));
        let erased = before - partition.records.len();
        if partition.records.is_empty() {
            self.partitions.remove(partition_key);
        } else {
            partition.rebuild(self.dimension);
        }
        erased
    }

    /// Write a new authenticated encrypted snapshot. Existing paths are refused.
    pub fn save_new(&self, path: &Path, key_id: &str, key: &[u8; 32]) -> Result<(), Error> {
        validate_key_id(key_id)?;
        if path.exists() {
            return Err(Error::AlreadyExists);
        }
        let records = self
            .partitions
            .values()
            .flat_map(|partition| partition.records.iter().cloned())
            .collect();
        let plaintext = Zeroizing::new(
            serde_json::to_vec(&PersistedState {
                format_version: FORMAT_VERSION,
                dimension: self.dimension,
                records,
            })
            .map_err(|_| Error::InvalidEnvelope)?,
        );
        if plaintext.len() as u64 > MAX_FILE_BYTES {
            return Err(Error::InvalidEnvelope);
        }
        let mut nonce = [0u8; 24];
        getrandom::getrandom(&mut nonce).map_err(|_| Error::Io)?;
        let aad = envelope_aad(key_id);
        let ciphertext = XChaCha20Poly1305::new(key.into())
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: plaintext.as_slice(),
                    aad: &aad,
                },
            )
            .map_err(|_| Error::InvalidEnvelope)?;
        let mut envelope = Vec::with_capacity(aad.len() + nonce.len() + ciphertext.len());
        envelope.extend_from_slice(FILE_MAGIC);
        envelope.extend_from_slice(&(key_id.len() as u16).to_be_bytes());
        envelope.extend_from_slice(key_id.as_bytes());
        envelope.extend_from_slice(&nonce);
        envelope.extend_from_slice(&ciphertext);
        write_new_atomic(path, &envelope)
    }

    /// Open an authenticated snapshot using a bounded key resolver.
    pub fn load<F>(path: &Path, mut key_for: F, now_ms: i64) -> Result<Self, Error>
    where
        F: FnMut(&str) -> Option<[u8; 32]>,
    {
        let metadata = fs::metadata(path).map_err(|_| Error::Io)?;
        if metadata.len() > MAX_FILE_BYTES || metadata.len() < 8 + 2 + 24 + 16 {
            return Err(Error::InvalidEnvelope);
        }
        let envelope = fs::read(path).map_err(|_| Error::Io)?;
        if envelope.get(..8) != Some(FILE_MAGIC.as_slice()) {
            return Err(Error::InvalidEnvelope);
        }
        let key_len = u16::from_be_bytes([envelope[8], envelope[9]]) as usize;
        if key_len == 0 || key_len > MAX_KEY_ID_BYTES || envelope.len() < 10 + key_len + 24 + 16 {
            return Err(Error::InvalidEnvelope);
        }
        let key_id =
            std::str::from_utf8(&envelope[10..10 + key_len]).map_err(|_| Error::InvalidEnvelope)?;
        validate_key_id(key_id)?;
        let key = Zeroizing::new(key_for(key_id).ok_or(Error::KeyUnavailable)?);
        let nonce_start = 10 + key_len;
        let nonce = XNonce::from_slice(&envelope[nonce_start..nonce_start + 24]);
        let aad = envelope_aad(key_id);
        let plaintext = Zeroizing::new(
            XChaCha20Poly1305::new((&*key).into())
                .decrypt(
                    nonce,
                    Payload {
                        msg: &envelope[nonce_start + 24..],
                        aad: &aad,
                    },
                )
                .map_err(|_| Error::InvalidEnvelope)?,
        );
        let state: PersistedState =
            serde_json::from_slice(plaintext.as_slice()).map_err(|_| Error::InvalidEnvelope)?;
        if state.format_version != FORMAT_VERSION || state.records.len() > MAX_RECORDS {
            return Err(Error::InvalidEnvelope);
        }
        let mut memory = Self::new(state.dimension).map_err(|_| Error::InvalidEnvelope)?;
        for record in state.records {
            memory
                .ingest(record, now_ms)
                .map_err(|_| Error::InvalidEnvelope)?;
        }
        Ok(memory)
    }

    /// Re-encrypt a verified source snapshot to a new, non-existing destination.
    pub fn rotate_key<F>(
        source: &Path,
        destination: &Path,
        key_for: F,
        new_key_id: &str,
        new_key: &[u8; 32],
        now_ms: i64,
    ) -> Result<Self, Error>
    where
        F: FnMut(&str) -> Option<[u8; 32]>,
    {
        let memory = Self::load(source, key_for, now_ms)?;
        memory.save_new(destination, new_key_id, new_key)?;
        Self::load(
            destination,
            |id| (id == new_key_id).then_some(*new_key),
            now_ms,
        )
    }
}

fn validate_record(record: &MemoryRecord, dimension: usize, now_ms: i64) -> Result<(), Error> {
    validate_id(&record.record_id)?;
    validate_id(&record.message_id)?;
    validate_partition(&record.partition)?;
    validate_id(&record.site_id)?;
    if let Some(space) = &record.space_id {
        validate_id(space)?;
    }
    validate_id(&record.schema_version)?;
    validate_id(&record.source_id)?;
    validate_features(&record.features, dimension)?;
    if record.source_sequence == 0 {
        return Err(Error::Invalid("source sequence must be positive"));
    }
    if record.observed_at_ms > now_ms
        || record.expires_at_ms <= now_ms
        || record.retention_until_ms <= now_ms
        || record.expires_at_ms > record.retention_until_ms
    {
        return Err(Error::Invalid("record timestamps are invalid or expired"));
    }
    if !record.uncertainty.is_finite() || !(0.0..=1.0).contains(&record.uncertainty) {
        return Err(Error::Invalid("uncertainty is out of bounds"));
    }
    if record.derived_from.len() > MAX_DERIVATIONS {
        return Err(Error::Invalid("too many derivation references"));
    }
    let mut parents = BTreeSet::new();
    for parent in &record.derived_from {
        validate_id(parent)?;
        if parent == &record.record_id || !parents.insert(parent) {
            return Err(Error::InvalidDerivation);
        }
    }
    match record.class {
        RecordClass::Observation if !record.derived_from.is_empty() => {
            Err(Error::InvalidDerivation)
        }
        RecordClass::Inference if record.derived_from.is_empty() => Err(Error::InvalidDerivation),
        _ => Ok(()),
    }
}

fn validate_partition(partition: &PartitionKey) -> Result<(), Error> {
    validate_id(&partition.tenant_id)?;
    validate_id(&partition.workspace_id)
}

fn validate_id(value: &str) -> Result<(), Error> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(Error::Invalid("identifier is invalid"));
    }
    Ok(())
}

fn validate_features(features: &[f32], dimension: usize) -> Result<(), Error> {
    if features.len() != dimension || features.iter().any(|value| !value.is_finite()) {
        return Err(Error::Invalid("feature vector is invalid"));
    }
    Ok(())
}

fn validate_key_id(key_id: &str) -> Result<(), Error> {
    if key_id.is_empty()
        || key_id.len() > MAX_KEY_ID_BYTES
        || !key_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(Error::InvalidEnvelope);
    }
    Ok(())
}

fn envelope_aad(key_id: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(FILE_MAGIC.len() + 2 + key_id.len());
    aad.extend_from_slice(FILE_MAGIC);
    aad.extend_from_slice(&(key_id.len() as u16).to_be_bytes());
    aad.extend_from_slice(key_id.as_bytes());
    aad
}

fn write_new_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let parent = path.parent().ok_or(Error::Io)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or(Error::Io)?;
    let mut random = [0u8; 8];
    getrandom::getrandom(&mut random).map_err(|_| Error::Io)?;
    let suffix: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
    let temporary: PathBuf = parent.join(format!(".{file_name}.{suffix}.tmp"));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|_| Error::Io)?;
        file.write_all(bytes).map_err(|_| Error::Io)?;
        file.sync_all().map_err(|_| Error::Io)?;
        if path.exists() {
            return Err(Error::AlreadyExists);
        }
        // Atomic create-if-absent: unlike rename-on-Unix this cannot replace a
        // destination created between the earlier existence check and commit.
        fs::hard_link(&temporary, path).map_err(|error| {
            if path.exists() {
                Error::AlreadyExists
            } else {
                let _ = error;
                Error::Io
            }
        })?;
        fs::remove_file(&temporary).map_err(|_| Error::Io)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 2_000_000;

    fn key(tenant: &str, workspace: &str) -> PartitionKey {
        PartitionKey {
            tenant_id: tenant.into(),
            workspace_id: workspace.into(),
        }
    }

    fn record(
        id: &str,
        partition: PartitionKey,
        source: &str,
        sequence: u64,
        features: [f32; 3],
    ) -> MemoryRecord {
        MemoryRecord {
            record_id: id.into(),
            message_id: format!("message-{id}"),
            partition,
            site_id: "site-1".into(),
            space_id: Some("room-1".into()),
            schema_version: "1.0".into(),
            source_id: source.into(),
            source_sequence: sequence,
            observed_at_ms: NOW - 100,
            expires_at_ms: NOW + 1_000,
            retention_until_ms: NOW + 2_000,
            class: RecordClass::Observation,
            features: features.to_vec(),
            uncertainty: 0.2,
            evidence_grade: EvidenceGrade::L1,
            provenance_digest: [7; 32],
            witness_digest: Some([8; 32]),
            derived_from: Vec::new(),
        }
    }

    fn query(partition: PartitionKey) -> SearchQuery {
        SearchQuery {
            partition,
            features: vec![1.0, 0.0, 0.0],
            site_id: None,
            space_id: None,
            schema_version: Some("1.0".into()),
            observed_from_ms: None,
            observed_before_ms: None,
            k: 10,
            now_ms: NOW,
        }
    }

    #[test]
    fn ann_is_physically_partitioned_before_search() {
        let mut memory = SpatialMemory::new(3).unwrap();
        memory
            .ingest(
                record("a", key("tenant-a", "ws"), "sensor-a", 1, [1.0, 0.0, 0.0]),
                NOW,
            )
            .unwrap();
        memory
            .ingest(
                record("b", key("tenant-b", "ws"), "sensor-b", 1, [1.0, 0.0, 0.0]),
                NOW,
            )
            .unwrap();
        memory
            .ingest(
                record(
                    "c",
                    key("tenant-a", "other"),
                    "sensor-c",
                    1,
                    [1.0, 0.0, 0.0],
                ),
                NOW,
            )
            .unwrap();

        let result = memory.search(&query(key("tenant-a", "ws"))).unwrap();
        assert_eq!(
            result
                .iter()
                .map(|item| item.record_id.as_str())
                .collect::<Vec<_>>(),
            ["a"]
        );
        assert_eq!(memory.search(&query(key("missing", "ws"))).unwrap(), []);
    }

    #[test]
    fn replay_sequence_and_derivation_fail_closed() {
        let partition = key("tenant", "ws");
        let mut memory = SpatialMemory::new(3).unwrap();
        let base = record("base", partition.clone(), "sensor", 1, [1.0, 0.0, 0.0]);
        memory.ingest(base.clone(), NOW).unwrap();
        memory.ingest(base.clone(), NOW).unwrap();
        let mut changed = base.clone();
        changed.uncertainty = 0.3;
        assert_eq!(memory.ingest(changed, NOW), Err(Error::ReplayConflict));
        let mut reused_message = record(
            "different-record",
            partition.clone(),
            "sensor",
            2,
            [0.0, 1.0, 0.0],
        );
        reused_message.message_id = base.message_id.clone();
        assert_eq!(
            memory.ingest(reused_message, NOW),
            Err(Error::ReplayConflict)
        );
        assert_eq!(
            memory.ingest(
                record("stale", partition.clone(), "sensor", 1, [0.0, 1.0, 0.0]),
                NOW
            ),
            Err(Error::StaleSequence)
        );

        let mut inference = record("derived", partition, "model", 1, [0.9, 0.1, 0.0]);
        inference.class = RecordClass::Inference;
        inference.derived_from = vec!["missing".into()];
        assert_eq!(memory.ingest(inference, NOW), Err(Error::InvalidDerivation));
    }

    #[test]
    fn explanation_is_bounded_and_preserves_evidence() {
        let partition = key("tenant", "ws");
        let mut memory = SpatialMemory::new(3).unwrap();
        memory
            .ingest(
                record("base", partition.clone(), "sensor", 1, [1.0, 0.0, 0.0]),
                NOW,
            )
            .unwrap();
        let explanation = memory.explain(&query(partition.clone())).unwrap();
        assert_eq!(explanation.partition, partition);
        assert_eq!(explanation.matches[0].evidence_grade, EvidenceGrade::L1);
        assert!(explanation.basis.contains("not causation"));
    }

    #[test]
    fn expiry_cascades_to_derived_records_and_erasure_is_exact() {
        let partition = key("tenant", "ws");
        let mut memory = SpatialMemory::new(3).unwrap();
        let mut base = record("base", partition.clone(), "sensor", 1, [1.0, 0.0, 0.0]);
        base.expires_at_ms = NOW + 10;
        memory.ingest(base, NOW).unwrap();
        let mut derived = record("derived", partition.clone(), "model", 1, [0.9, 0.1, 0.0]);
        derived.class = RecordClass::Inference;
        derived.derived_from = vec!["base".into()];
        memory.ingest(derived, NOW).unwrap();
        assert_eq!(memory.purge_expired(NOW + 11), 2);

        memory
            .ingest(
                record("new", partition.clone(), "sensor", 2, [1.0, 0.0, 0.0]),
                NOW,
            )
            .unwrap();
        assert_eq!(memory.erase_partition(&partition), 1);
        assert!(memory.is_empty());
    }

    #[test]
    fn record_erasure_cascades_to_dependent_inferences_only() {
        let partition = key("tenant", "ws");
        let mut memory = SpatialMemory::new(3).unwrap();
        memory
            .ingest(
                record("base", partition.clone(), "sensor-a", 1, [1.0, 0.0, 0.0]),
                NOW,
            )
            .unwrap();
        memory
            .ingest(
                record(
                    "independent",
                    partition.clone(),
                    "sensor-b",
                    1,
                    [0.0, 1.0, 0.0],
                ),
                NOW,
            )
            .unwrap();
        let mut derived = record("derived", partition.clone(), "model", 1, [0.9, 0.1, 0.0]);
        derived.class = RecordClass::Inference;
        derived.derived_from = vec!["base".into()];
        memory.ingest(derived, NOW).unwrap();
        assert_eq!(memory.erase_record(&partition, "base"), 2);
        let matches = memory.search(&query(partition)).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].record_id, "independent");
    }

    #[test]
    fn encrypted_snapshot_rejects_tamper_wrong_key_and_overwrite() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("memory.rvsm");
        let mut memory = SpatialMemory::new(3).unwrap();
        memory
            .ingest(
                record("base", key("tenant", "ws"), "sensor", 1, [1.0, 0.0, 0.0]),
                NOW,
            )
            .unwrap();
        let old_key = [3u8; 32];
        memory.save_new(&path, "key-1", &old_key).unwrap();
        assert_eq!(
            memory.save_new(&path, "key-1", &old_key),
            Err(Error::AlreadyExists)
        );
        assert_eq!(
            SpatialMemory::load(&path, |_| Some([4u8; 32]), NOW).unwrap_err(),
            Error::InvalidEnvelope
        );

        let loaded =
            SpatialMemory::load(&path, |id| (id == "key-1").then_some(old_key), NOW).unwrap();
        assert_eq!(loaded.len(), 1);
        let mut bytes = fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert_eq!(
            SpatialMemory::load(&path, |_| Some(old_key), NOW).unwrap_err(),
            Error::InvalidEnvelope
        );
    }

    #[test]
    fn key_rotation_writes_new_generation_and_preserves_source() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("old.rvsm");
        let destination = directory.path().join("new.rvsm");
        let mut memory = SpatialMemory::new(3).unwrap();
        memory
            .ingest(
                record("base", key("tenant", "ws"), "sensor", 1, [1.0, 0.0, 0.0]),
                NOW,
            )
            .unwrap();
        let old_key = [5u8; 32];
        let new_key = [6u8; 32];
        memory.save_new(&source, "old-key", &old_key).unwrap();
        let rotated = SpatialMemory::rotate_key(
            &source,
            &destination,
            |_| Some(old_key),
            "new-key",
            &new_key,
            NOW,
        )
        .unwrap();
        assert_eq!(rotated.len(), 1);
        assert!(source.exists());
        assert_eq!(
            SpatialMemory::load(&destination, |id| (id == "new-key").then_some(new_key), NOW)
                .unwrap()
                .len(),
            1
        );
    }
}
