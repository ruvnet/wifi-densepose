//! Bounded adapters for RuVector temporal features, RuField observations, and
//! WorldGraph updates. These are data-plane outputs, not authority to mutate a
//! remote graph; callers remain responsible for tenant and OAuth policy.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::protocol::{
    EvidenceLevel, FrameSource, ModalityContributions, Provenance, TrackEnvelope, TrackState, Vec3,
};

/// One compact trajectory feature suitable for insertion into a RuVector
/// temporal index. It contains no civil identity or raw photon histogram.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalFeature {
    /// Capture session.
    pub session_id: String,
    /// Session-local track id.
    pub track_id: String,
    /// Capture time.
    pub at_unix_ms: u64,
    /// Fixed position, velocity, covariance, confidence, and modality vector.
    pub embedding: [f32; 12],
    /// Evidence ceiling retained with the vector.
    pub evidence_level: EvidenceLevel,
    /// Live/replay/synthetic source retained with the vector.
    pub source: FrameSource,
    /// Algorithm revision retained with the vector.
    pub algorithm_version: String,
    /// Calibration digest retained with the vector.
    pub calibration_hash: String,
    /// Sensor and transport provenance retained with the vector.
    pub provenance: Provenance,
    /// Hard expiry.
    pub expires_at_unix_ms: u64,
}

/// In-process deterministic reference memory. Production deployments can write
/// the same [`TemporalFeature`] records to tenant-scoped RuVector storage.
pub struct TemporalFeatureMemory {
    capacity: usize,
    records: VecDeque<TemporalFeature>,
}

impl TemporalFeatureMemory {
    /// Construct a bounded memory.
    pub fn new(capacity: usize) -> Result<Self, BridgeError> {
        if !(1..=65_536).contains(&capacity) {
            return Err(BridgeError::InvalidCapacity);
        }
        Ok(Self {
            capacity,
            records: VecDeque::with_capacity(capacity.min(1_024)),
        })
    }

    /// Insert every non-UNKNOWN track after validating its envelope.
    pub fn observe(&mut self, envelope: &TrackEnvelope) -> Result<usize, BridgeError> {
        envelope.validate()?;
        self.records
            .retain(|record| record.expires_at_unix_ms >= envelope.captured_at_unix_ms);
        let mut inserted = 0_usize;
        for track in envelope
            .tracks
            .iter()
            .filter(|track| track.state != TrackState::Unknown)
        {
            while self.records.len() >= self.capacity {
                self.records.pop_front();
            }
            self.records.push_back(TemporalFeature {
                session_id: envelope.session_id.clone(),
                track_id: track.track_id.clone(),
                at_unix_ms: envelope.captured_at_unix_ms,
                embedding: [
                    track.position_m.x,
                    track.position_m.y,
                    track.position_m.z,
                    track.velocity_mps.x,
                    track.velocity_mps.y,
                    track.velocity_mps.z,
                    track.covariance_diagonal_m2.x,
                    track.covariance_diagonal_m2.y,
                    track.covariance_diagonal_m2.z,
                    track.confidence,
                    track.modality_contributions.lidar,
                    track.modality_contributions.csi,
                ],
                evidence_level: envelope.evidence_level,
                source: envelope.source,
                algorithm_version: envelope.algorithm_version.clone(),
                calibration_hash: envelope.calibration_hash.clone(),
                provenance: envelope.provenance.clone(),
                expires_at_unix_ms: envelope.expires_at_unix_ms,
            });
            inserted += 1;
        }
        Ok(inserted)
    }

    /// Return the closest same-session feature by cosine similarity.
    #[must_use]
    pub fn nearest(
        &self,
        query: &[f32; 12],
        session_id: &str,
        now_unix_ms: u64,
    ) -> Option<(&TemporalFeature, f32)> {
        if query.iter().any(|value| !value.is_finite()) {
            return None;
        }
        self.records
            .iter()
            .filter(|record| record.session_id == session_id)
            .filter(|record| record.expires_at_unix_ms > now_unix_ms)
            .filter_map(|record| cosine(query, &record.embedding).map(|score| (record, score)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Current bounded record count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no records remain.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Remove all records that are no longer fresh.
    pub fn purge_expired(&mut self, now_unix_ms: u64) -> usize {
        let before = self.records.len();
        self.records
            .retain(|record| record.expires_at_unix_ms > now_unix_ms);
        before - self.records.len()
    }
}

fn cosine(a: &[f32; 12], b: &[f32; 12]) -> Option<f32> {
    let mut dot = 0.0;
    let mut aa = 0.0;
    let mut bb = 0.0;
    for (left, right) in a.iter().zip(b.iter()) {
        dot += left * right;
        aa += left * left;
        bb += right * right;
    }
    let denominator = (aa * bb).sqrt();
    (denominator > f32::EPSILON).then(|| (dot / denominator).clamp(-1.0, 1.0))
}

/// Privacy-safe field observation emitted for every quality-gated track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuFieldObservation {
    /// Field schema.
    pub schema: String,
    /// Session-local observation id.
    pub observation_id: String,
    /// World position.
    pub position_m: Vec3,
    /// Diagonal spatial covariance.
    pub covariance_diagonal_m2: Vec3,
    /// Confidence.
    pub confidence: f32,
    /// Current signal quality.
    pub signal_quality: f32,
    /// Observable modality weights.
    pub modality_contributions: ModalityContributions,
    /// Capture time.
    pub observed_at_unix_ms: u64,
    /// Hard expiry.
    pub expires_at_unix_ms: u64,
    /// Evidence ceiling.
    pub evidence_level: EvidenceLevel,
    /// Source watermark.
    pub source: FrameSource,
    /// Calibration digest.
    pub calibration_hash: String,
    /// Algorithm revision.
    pub algorithm_version: String,
    /// Sensor/signal/transport provenance.
    pub provenance: Provenance,
}

impl RuFieldObservation {
    /// Convert all non-UNKNOWN tracks without retaining raw histograms.
    pub fn from_envelope(envelope: &TrackEnvelope) -> Result<Vec<Self>, BridgeError> {
        envelope.validate()?;
        Ok(envelope
            .tracks
            .iter()
            .filter(|track| track.state != TrackState::Unknown)
            .map(|track| Self {
                schema: "rufield.observation.nlos.v1".into(),
                observation_id: canonical_id(
                    b"rufield-observation-v1",
                    &[
                        envelope.session_id.as_bytes(),
                        track.track_id.as_bytes(),
                        &envelope.sequence.to_le_bytes(),
                    ],
                ),
                position_m: track.position_m,
                covariance_diagonal_m2: track.covariance_diagonal_m2,
                confidence: track.confidence,
                signal_quality: track.signal_quality,
                modality_contributions: track.modality_contributions,
                observed_at_unix_ms: envelope.captured_at_unix_ms,
                expires_at_unix_ms: envelope.expires_at_unix_ms,
                evidence_level: envelope.evidence_level,
                source: envelope.source,
                calibration_hash: envelope.calibration_hash.clone(),
                algorithm_version: envelope.algorithm_version.clone(),
                provenance: envelope.provenance.clone(),
            })
            .collect())
    }
}

/// Minimal idempotent WorldGraph mutation request. A governed writer applies it
/// only after tenant authorization and evidence policy checks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldGraphUpdate {
    /// Mutation schema.
    pub schema: String,
    /// Idempotency key.
    pub idempotency_key: String,
    /// Pseudonymous node id.
    pub node_id: String,
    /// Node kind.
    pub node_kind: String,
    /// World position.
    pub position_m: Vec3,
    /// Estimated velocity.
    pub velocity_mps: Vec3,
    /// Diagonal spatial covariance.
    pub covariance_diagonal_m2: Vec3,
    /// Posterior confidence.
    pub confidence: f32,
    /// Current signal quality.
    pub signal_quality: f32,
    /// Observable modality weights.
    pub modality_contributions: ModalityContributions,
    /// Quality-gated state.
    pub state: TrackState,
    /// Evidence ceiling.
    pub evidence_level: EvidenceLevel,
    /// Live/replay/synthetic source.
    pub source: FrameSource,
    /// Observation time.
    pub observed_at_unix_ms: u64,
    /// Calibration digest.
    pub calibration_hash: String,
    /// Algorithm revision.
    pub algorithm_version: String,
    /// Sensor/signal/transport provenance.
    pub provenance: Provenance,
    /// Hard expiry.
    pub expires_at_unix_ms: u64,
}

impl WorldGraphUpdate {
    /// Convert tracks into idempotent, tenant-neutral mutation requests.
    pub fn from_envelope(envelope: &TrackEnvelope) -> Result<Vec<Self>, BridgeError> {
        envelope.validate()?;
        Ok(envelope
            .tracks
            .iter()
            .map(|track| Self {
                schema: "worldgraph.update.nlos.v1".into(),
                idempotency_key: canonical_id(
                    b"worldgraph-update-v1",
                    &[
                        envelope.session_id.as_bytes(),
                        &envelope.sequence.to_le_bytes(),
                        track.track_id.as_bytes(),
                    ],
                ),
                node_id: canonical_id(
                    b"worldgraph-node-v1",
                    &[envelope.session_id.as_bytes(), track.track_id.as_bytes()],
                ),
                node_kind: "hidden_target_hypothesis".into(),
                position_m: track.position_m,
                velocity_mps: track.velocity_mps,
                covariance_diagonal_m2: track.covariance_diagonal_m2,
                confidence: track.confidence,
                signal_quality: track.signal_quality,
                modality_contributions: track.modality_contributions,
                state: track.state,
                evidence_level: envelope.evidence_level,
                source: envelope.source,
                observed_at_unix_ms: envelope.captured_at_unix_ms,
                calibration_hash: envelope.calibration_hash.clone(),
                algorithm_version: envelope.algorithm_version.clone(),
                provenance: envelope.provenance.clone(),
                expires_at_unix_ms: envelope.expires_at_unix_ms,
            })
            .collect())
    }
}

fn canonical_id(domain: &[u8], fields: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update([0]);
    for field in fields {
        digest.update((field.len() as u32).to_le_bytes());
        digest.update(field);
    }
    format!("nlos-{:x}", digest.finalize())
}

/// Bridge conversion failure.
#[derive(Debug, Error)]
pub enum BridgeError {
    /// Memory capacity was zero or unreasonably large.
    #[error("invalid temporal memory capacity")]
    InvalidCapacity,
    /// Source track contract was invalid.
    #[error(transparent)]
    Contract(#[from] crate::protocol::ContractError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::SyntheticScene;

    #[test]
    fn bridge_retains_evidence_and_never_raw_histograms() {
        let report = SyntheticScene::benchmark(2, 64);
        assert_eq!(report.evidence, "SYNTHETIC_L0");
        let memory = TemporalFeatureMemory::new(8).unwrap();
        assert!(memory.is_empty());
        let serialized = serde_json::to_string(&RuFieldObservation {
            schema: "rufield.observation.nlos.v1".into(),
            observation_id: "s:t:1".into(),
            position_m: Vec3::default(),
            covariance_diagonal_m2: Vec3::default(),
            confidence: 0.0,
            signal_quality: 0.0,
            modality_contributions: ModalityContributions {
                lidar: 1.0,
                csi: 0.0,
            },
            observed_at_unix_ms: 1,
            expires_at_unix_ms: 2,
            evidence_level: EvidenceLevel::L0Synthetic,
            source: FrameSource::Synthetic,
            calibration_hash: "0".repeat(64),
            algorithm_version: "test-v1".into(),
            provenance: crate::protocol::Provenance {
                sensor_id: "sim".into(),
                sensor_model: "sim".into(),
                firmware_version: "sim".into(),
                transient_kind: crate::protocol::TransientKind::Replay,
                histogram_preserved: false,
                transport: "replay".into(),
            },
        })
        .unwrap();
        // The provenance boolean is retained, but neither histogram bins nor
        // raw zones cross this bridge.
        assert!(serialized.contains("histogramPreserved"));
        assert!(!serialized.contains("\"histogram\":"));
        assert!(!serialized.contains("\"zones\":"));
    }
}
