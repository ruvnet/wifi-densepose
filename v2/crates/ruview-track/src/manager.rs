//! [`TrackManager`] — persistent, privacy-preserving probabilistic tracking
//! (ADR-304).
//!
//! # What it does
//!
//! Ingests per-frame [`Detection`]s (a container + 2-D position + a coarse,
//! non-reversible [`CoarseFeature`] + an injected timestamp) and maintains
//! persistent [`Track`](ruview_ontology::Track) entities, each resolved to a
//! pseudonymous [`Person`](ruview_ontology::Person) (`person_7`). It produces
//! per-entity **histories** across zones/rooms
//! (`person_7: kitchen → hallway → bedroom`).
//!
//! # Association (documented, bounded)
//!
//! Per frame it runs gated nearest-neighbour association with a bounded cost:
//!
//! 1. **Topology gate.** A track is a candidate only if the detection's
//!    container is the same as, or [adjacent](crate::Topology) to, the track's
//!    last container.
//! 2. **Value gate + horizon.** Position distance ≤ `gate_position`, feature
//!    distance ≤ `gate_feature`, idle gap ≤ `max_coast_ms`.
//! 3. **Cost.** `w_pos·(pos/gate_pos) + w_feat·(feat/gate_feat)` — bounded to
//!    `[0, w_pos+w_feat]`.
//! 4. **Ambiguity.** If the best and second-best candidates are within
//!    `ambiguity_margin`, the detection is *not* assigned — it spawns a tentative
//!    track. Under-linking, never a wrong join.
//! 5. **Decayed confidence.** `confidence = decay(gap) · similarity`, where
//!    `decay` falls linearly to 0 at `max_coast_ms`. Beyond the horizon the
//!    track has already expired, so a fresh pseudonym is minted.
//!
//! Any detection without a confident, unambiguous match yields an
//! [`Association::Unknown`] outcome and a new tentative track (ADR-297 rule 1:
//! UNKNOWN is first-class, never an error).
//!
//! # Privacy boundary (by construction)
//!
//! - The persistent id is a synthetic pseudonym (`person_7`) with **no** field
//!   or join key to any name, account, MAC, or phone — the ontology
//!   [`Person`](ruview_ontology::Person) schema simply has no such field.
//! - Pseudonyms are **rotatable** via [`TrackManager::rotate_pseudonym`].
//! - Appearance features are coarse and non-reversible by type
//!   ([`CoarseFeature`]); nothing here persists a long-term biometric template.

use std::collections::BTreeMap;

use ruview_ontology::{Container, EvidenceLevel, Person, PersonId, Track, TrackId};

use crate::config::TrackerConfig;
use crate::error::TrackError;
use crate::feature::CoarseFeature;
use crate::topology::Topology;

/// A single per-frame detection handed to the manager.
///
/// Construct with [`Detection::new`], which validates the position at the
/// boundary. The coarse feature is already bounded and non-reversible by type.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Where the detection was observed (space or zone).
    pub container: Container,
    /// A 2-D position within the space frame.
    pub position: [f64; 2],
    /// Coarse, non-identifying appearance descriptor.
    pub feature: CoarseFeature,
    /// Injected capture timestamp (Unix ms). Never sampled from a clock here.
    pub at_unix_ms: i64,
}

impl Detection {
    /// Validate and build a detection, rejecting a non-finite position.
    pub fn new(
        container: Container,
        position: [f64; 2],
        feature: CoarseFeature,
        at_unix_ms: i64,
    ) -> Result<Self, TrackError> {
        if !position[0].is_finite() || !position[1].is_finite() {
            return Err(TrackError::NonFinitePosition);
        }
        Ok(Self {
            container,
            position,
            feature,
            at_unix_ms,
        })
    }
}

/// Lifecycle state of a persistent track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TrackState {
    /// Newly spawned; not yet confirmed by `confirm_after` hits.
    Tentative,
    /// Confirmed and currently observed.
    Active,
    /// Confirmed but idle beyond `lost_after_ms`; still re-identifiable within
    /// `max_coast_ms`.
    Lost,
}

/// One container transition in a track's history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waypoint {
    /// The container entered.
    pub container: Container,
    /// When it was entered (Unix ms).
    pub at_unix_ms: i64,
}

/// Why a detection produced no confident match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownReason {
    /// No existing tracks were candidates.
    NoCandidate,
    /// A nearest track existed but failed the value gate / horizon.
    GateExceeded,
    /// A nearest track existed but was not topologically adjacent.
    TopologyBlocked,
    /// Two tracks were within `ambiguity_margin` — left tentative to avoid a swap.
    Ambiguous,
    /// A candidate track existed but was claimed by a closer detection this frame.
    Contested,
}

/// The association decision for one detection.
#[derive(Clone, Debug, PartialEq)]
pub enum Association {
    /// Matched to an existing track with a decayed confidence in `[0, 1]`.
    Matched {
        /// The track the detection was attributed to.
        track: TrackId,
        /// Decayed continuity confidence (never asserted as certainty).
        confidence: f64,
    },
    /// No confident, unambiguous match: a fresh tentative track was spawned.
    Unknown {
        /// The newly minted tentative track.
        spawned: TrackId,
        /// Why no existing track was chosen.
        reason: UnknownReason,
    },
}

/// The outcome of ingesting one detection.
#[derive(Clone, Debug, PartialEq)]
pub struct IngestOutcome {
    /// The track the detection now belongs to (matched or newly spawned).
    pub track: TrackId,
    /// The persistent pseudonym for that track.
    pub person: PersonId,
    /// The association decision.
    pub association: Association,
}

/// Internal persistent-track record. Not part of the public schema.
#[derive(Clone, Debug)]
struct Entity {
    id: TrackId,
    person: PersonId,
    state: TrackState,
    container: Container,
    position: [f64; 2],
    feature: CoarseFeature,
    last_ms: i64,
    hits: u32,
    history: Vec<Waypoint>,
}

/// Persistent probabilistic tracker producing ADR-303 `Track`/`Person` nodes
/// without civil identity.
#[derive(Clone, Debug)]
pub struct TrackManager {
    config: TrackerConfig,
    topology: Topology,
    entities: BTreeMap<TrackId, Entity>,
    track_counter: u64,
    person_counter: u64,
}

impl TrackManager {
    /// A manager with the given config and topology.
    #[must_use]
    pub fn new(config: TrackerConfig, topology: Topology) -> Self {
        Self {
            config,
            topology,
            entities: BTreeMap::new(),
            track_counter: 0,
            person_counter: 0,
        }
    }

    /// A manager with default policy and an empty topology.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TrackerConfig::default(), Topology::new())
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &TrackerConfig {
        &self.config
    }

    /// Number of live tracks currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether no tracks are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Live track ids, in stable order.
    #[must_use]
    pub fn track_ids(&self) -> Vec<TrackId> {
        self.entities.keys().cloned().collect()
    }

    /// The lifecycle state of a track, if held.
    #[must_use]
    pub fn state(&self, track: &TrackId) -> Option<TrackState> {
        self.entities.get(track).map(|e| e.state)
    }

    /// The pseudonym bound to a track, if held.
    #[must_use]
    pub fn person_of(&self, track: &TrackId) -> Option<&PersonId> {
        self.entities.get(track).map(|e| &e.person)
    }

    /// A track's container history (deduplicated on entry), if held.
    #[must_use]
    pub fn history(&self, track: &TrackId) -> Option<&[Waypoint]> {
        self.entities.get(track).map(|e| e.history.as_slice())
    }

    /// A track's trajectory as an ordered list of containers, if held.
    #[must_use]
    pub fn trajectory(&self, track: &TrackId) -> Option<Vec<Container>> {
        self.entities
            .get(track)
            .map(|e| e.history.iter().map(|w| w.container.clone()).collect())
    }

    /// Advance time to `now_unix_ms`, applying lifecycle decay: mark idle active
    /// tracks lost, and expire (drop) any track idle beyond `max_coast_ms`.
    /// Returns the ids that expired.
    pub fn tick(&mut self, now_unix_ms: i64) -> Vec<TrackId> {
        let mut expired = Vec::new();
        self.entities.retain(|id, e| {
            let gap = (now_unix_ms - e.last_ms).max(0);
            if gap > self.config.max_coast_ms {
                expired.push(id.clone());
                false
            } else {
                if gap > self.config.lost_after_ms && e.state == TrackState::Active {
                    e.state = TrackState::Lost;
                }
                true
            }
        });
        expired
    }

    /// Ingest a single detection. Convenience wrapper over [`Self::ingest_frame`].
    pub fn ingest(&mut self, detection: Detection) -> Result<IngestOutcome, TrackError> {
        let mut out = self.ingest_frame(std::slice::from_ref(&detection))?;
        // Exactly one detection in, exactly one outcome out.
        Ok(out.pop().expect("one detection yields one outcome"))
    }

    /// Ingest a frame of detections, returning one outcome per detection in
    /// input order.
    ///
    /// Association is joint within the frame: each detection matches at most one
    /// track and each track absorbs at most one detection, resolved greedily by
    /// ascending cost. Detections that are unmatched, gated out, topology-blocked,
    /// ambiguous, or contested spawn a fresh tentative track.
    pub fn ingest_frame(
        &mut self,
        detections: &[Detection],
    ) -> Result<Vec<IngestOutcome>, TrackError> {
        // Boundary validation first; malformed input is an error, not UNKNOWN.
        for d in detections {
            if !d.position[0].is_finite() || !d.position[1].is_finite() {
                return Err(TrackError::NonFinitePosition);
            }
        }
        if detections.is_empty() {
            return Ok(Vec::new());
        }

        // Expire stale tracks relative to the frame's latest timestamp so they
        // are not candidates (privacy-safe under-linking beyond the horizon).
        let frame_ms = detections.iter().map(|d| d.at_unix_ms).max().unwrap_or(0);
        self.tick(frame_ms);

        let n = detections.len();
        let ws = self.config.weight_sum();

        // Per-detection scored candidate lists and spawn reasons.
        let mut pairs: Vec<(usize, TrackId, f64)> = Vec::new(); // (det, track, cost)
        let mut reason: Vec<UnknownReason> = vec![UnknownReason::NoCandidate; n];
        let mut ambiguous = vec![false; n];

        for (i, d) in detections.iter().enumerate() {
            let mut scored: Vec<(TrackId, f64)> = Vec::new();
            let mut saw_topo_block = false;
            let mut saw_gate = false;
            let mut saw_any = false;

            for e in self.entities.values() {
                saw_any = true;
                if !self.topology.adjacent(&e.container, &d.container) {
                    saw_topo_block = true;
                    continue;
                }
                let gap = (d.at_unix_ms - e.last_ms).max(0);
                if gap > self.config.max_coast_ms {
                    saw_gate = true;
                    continue;
                }
                let pos = position_distance(d.position, e.position);
                let feat = d.feature.distance(&e.feature);
                if pos > self.config.gate_position || feat > self.config.gate_feature {
                    saw_gate = true;
                    continue;
                }
                let cost = self.config.w_pos * (pos / self.config.gate_position)
                    + self.config.w_feat * (feat / self.config.gate_feature);
                scored.push((e.id.clone(), cost));
            }

            // Deterministic order: cost, then track id.
            scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));

            if scored.len() >= 2 && (scored[1].1 - scored[0].1) < self.config.ambiguity_margin {
                // Two near-equal candidates: refuse to assign, spawn tentative.
                ambiguous[i] = true;
                reason[i] = UnknownReason::Ambiguous;
                continue;
            }

            if scored.is_empty() {
                reason[i] = if !saw_any {
                    UnknownReason::NoCandidate
                } else if saw_gate {
                    UnknownReason::GateExceeded
                } else if saw_topo_block {
                    UnknownReason::TopologyBlocked
                } else {
                    UnknownReason::NoCandidate
                };
            } else {
                // Provisional reason if greedy fails to secure a track.
                reason[i] = UnknownReason::Contested;
                for (tid, cost) in scored {
                    pairs.push((i, tid, cost));
                }
            }
        }

        // Greedy one-to-one assignment by ascending cost.
        pairs.sort_by(|a, b| {
            a.2.total_cmp(&b.2)
                .then_with(|| a.1.as_str().cmp(b.1.as_str()))
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut det_track: Vec<Option<(TrackId, f64)>> = vec![None; n];
        let mut track_used: BTreeMap<TrackId, ()> = BTreeMap::new();
        for (det, track, cost) in pairs {
            if det_track[det].is_some() || track_used.contains_key(&track) {
                continue;
            }
            det_track[det] = Some((track.clone(), cost));
            track_used.insert(track, ());
        }

        // Apply results in detection order (stable pseudonym minting).
        let mut outcomes = Vec::with_capacity(n);
        for (i, d) in detections.iter().enumerate() {
            if let Some((track, cost)) = det_track[i].take() {
                let confidence = self.apply_match(&track, d, cost, ws);
                let person = self.entities[&track].person.clone();
                outcomes.push(IngestOutcome {
                    track: track.clone(),
                    person,
                    association: Association::Matched { track, confidence },
                });
            } else {
                let (track, person) = self.spawn(d)?;
                outcomes.push(IngestOutcome {
                    track: track.clone(),
                    person,
                    association: Association::Unknown {
                        spawned: track,
                        reason: reason[i],
                    },
                });
            }
        }
        Ok(outcomes)
    }

    /// Rotate a track's pseudonym: mint a fresh opaque id and rebind it, keeping
    /// the track and its history intact. Returns the new pseudonym.
    pub fn rotate_pseudonym(&mut self, track: &TrackId) -> Result<PersonId, TrackError> {
        // Mint before the mutable borrow to satisfy the borrow checker.
        let fresh = self.next_person_id()?;
        let e = self
            .entities
            .get_mut(track)
            .ok_or(TrackError::UnknownTrack)?;
        e.person = fresh.clone();
        Ok(fresh)
    }

    /// Project a track to a canonical ADR-303 [`Track`] node, carrying the
    /// pseudonym, evidence level, and provenance. `None` if not held.
    #[must_use]
    pub fn to_track(&self, track: &TrackId) -> Option<Track> {
        let e = self.entities.get(track)?;
        Some(Track {
            id: e.id.clone(),
            person: Some(e.person.clone()),
            located_in: e.container.clone(),
            evidence_level: self.emit_level(e.state),
            provenance: self.config.provenance.clone(),
        })
    }

    /// Project a track's pseudonymous entity to a canonical ADR-303 [`Person`]
    /// node. `None` if not held.
    #[must_use]
    pub fn to_person(&self, track: &TrackId) -> Option<Person> {
        let e = self.entities.get(track)?;
        Some(Person {
            id: e.person.clone(),
            located_in: e.container.clone(),
            evidence_level: self.emit_level(e.state),
            provenance: self.config.provenance.clone(),
        })
    }

    // --- internals ---

    /// Emitted evidence level, floored to `L0` while a track is unconfirmed so a
    /// tentative belief cannot masquerade as corroborated.
    fn emit_level(&self, state: TrackState) -> EvidenceLevel {
        match state {
            TrackState::Tentative => EvidenceLevel::L0,
            _ => self.config.emit_evidence_level,
        }
    }

    fn apply_match(&mut self, track: &TrackId, d: &Detection, cost: f64, ws: f64) -> f64 {
        let confirm_after = self.config.confirm_after;
        let horizon = self.config.max_coast_ms;
        let e = self.entities.get_mut(track).expect("matched track exists");

        let gap = (d.at_unix_ms - e.last_ms).max(0);
        let similarity = (1.0 - cost / ws).clamp(0.0, 1.0);
        let confidence = (decay_factor(gap, horizon) * similarity).clamp(0.0, 1.0);

        if e.container != d.container {
            e.history.push(Waypoint {
                container: d.container.clone(),
                at_unix_ms: d.at_unix_ms,
            });
            e.container = d.container.clone();
        }
        e.position = d.position;
        e.feature = d.feature.clone();
        e.last_ms = d.at_unix_ms;
        e.hits = e.hits.saturating_add(1);
        e.state = match e.state {
            TrackState::Tentative if e.hits >= confirm_after => TrackState::Active,
            TrackState::Lost => TrackState::Active, // re-identified
            other => other,
        };
        confidence
    }

    fn spawn(&mut self, d: &Detection) -> Result<(TrackId, PersonId), TrackError> {
        let id = self.next_track_id()?;
        let person = self.next_person_id()?;
        let confirm_now = self.config.confirm_after <= 1;
        let entity = Entity {
            id: id.clone(),
            person: person.clone(),
            state: if confirm_now {
                TrackState::Active
            } else {
                TrackState::Tentative
            },
            container: d.container.clone(),
            position: d.position,
            feature: d.feature.clone(),
            last_ms: d.at_unix_ms,
            hits: 1,
            history: vec![Waypoint {
                container: d.container.clone(),
                at_unix_ms: d.at_unix_ms,
            }],
        };
        self.entities.insert(id.clone(), entity);
        Ok((id, person))
    }

    fn next_track_id(&mut self) -> Result<TrackId, TrackError> {
        self.track_counter += 1;
        Ok(TrackId::new(format!("track_{}", self.track_counter))?)
    }

    fn next_person_id(&mut self) -> Result<PersonId, TrackError> {
        self.person_counter += 1;
        Ok(PersonId::new(format!("person_{}", self.person_counter))?)
    }
}

/// Euclidean distance between two 2-D positions. Always finite for finite input.
fn position_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Linear time decay: `1` at zero gap, falling to `0` at the horizon and beyond.
/// Confidence in "same entity" falls with the size of the gap.
fn decay_factor(gap_ms: i64, horizon_ms: i64) -> f64 {
    if horizon_ms <= 0 {
        return if gap_ms <= 0 { 1.0 } else { 0.0 };
    }
    (1.0 - gap_ms as f64 / horizon_ms as f64).clamp(0.0, 1.0)
}
