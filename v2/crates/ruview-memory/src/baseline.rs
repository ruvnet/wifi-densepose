//! Configuration, the per-zone learned baseline, and the live observation
//! snapshot (ADR-309 §1–§2 — what "normal" is learned over, on the RuVector
//! temporal substrate; here a bounded in-memory scaffold).
//!
//! **SYNTHETIC / L0.** Every structure here is part of a simulation scaffold. A
//! [`ZoneBaseline`] is a *learned model* of a location's normal physics; it
//! predicts what is normal, it never measures. No value it holds is a hardware,
//! `MEASURED`, or accuracy claim (ADR-282, ADR-297).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use ruview_ontology::{EvidenceLevel, ZoneId};
use ruview_twin::{LinkId, ObservationSet};

use crate::error::MemoryError;
use crate::stat::RunningStat;

/// Hours in the occupancy-by-hour periodicity model (ADR-309 §1).
pub const HOURS_PER_DAY: usize = 24;

/// Upper bound on distinct zones a memory holds. Bounds allocation on untrusted
/// input (CLAUDE.md); construction beyond this is rejected, never truncated.
pub const MAX_ZONES: usize = 4096;

/// Upper bound on learned links per zone.
pub const MAX_LINKS_PER_ZONE: usize = 65_536;

/// Upper bound on modality signature channels per zone.
pub const MAX_MODALITY_CHANNELS: usize = 256;

/// Upper bound, in bytes, on a modality channel identifier.
pub const MAX_CHANNEL_ID_LEN: usize = 256;

/// Tuning of the learned-normal model. All fields are validated at construction
/// so no downstream computation can divide by zero or adapt on a nonsensical
/// factor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Forgetting factor `λ ∈ [0, 1)` — the weight retained on history each
    /// update; learning rate `α = 1 − λ`. Near `1` tracks only slow legitimate
    /// drift; near `0` adapts fast. See [`crate::stat`].
    pub forgetting_factor: f64,
    /// Minimum updates a channel needs before it is scored; below this the
    /// channel is [`Assessment::Unknown`](crate::Assessment::Unknown) — never a
    /// false positive on thin history (ADR-297 rule 1).
    pub min_history: u32,
    /// Significance gate (standard deviations). A channel whose deviation meets
    /// or exceeds this is flagged. Not a calibrated false-alarm rate — a model
    /// gate (cf. ADR-312 `DEFAULT_SIGNIFICANCE_THRESHOLD`).
    pub significance_threshold: f64,
    /// Standard-deviation floor for the occupancy model, so an always-empty hour
    /// (zero variance) yields finite significance rather than a divide-by-zero.
    pub occupancy_floor_std: f64,
    /// Standard-deviation floor (dB) for propagation and modality signatures.
    pub signature_floor_std: f64,
    /// Model version handle stamped into emitted evidence records (ADR-136).
    pub model_version: String,
}

impl MemoryConfig {
    /// A neutral SYNTHETIC default: `λ = 0.9` (learning rate 0.1), `min_history
    /// = 8`, `3σ` gate, occupancy floor `0.1`, signature floor `1.0 dB`. Asserts
    /// nothing about any real environment.
    #[must_use]
    pub fn default_synthetic() -> Self {
        Self {
            forgetting_factor: 0.9,
            min_history: 8,
            significance_threshold: 3.0,
            occupancy_floor_std: 0.1,
            signature_floor_std: 1.0,
            model_version: "ruview-memory-scaffold@0 (SYNTHETIC/L0)".to_string(),
        }
    }

    /// Validate the configuration at the boundary. Never panics.
    ///
    /// # Errors
    /// [`MemoryError::InvalidConfig`] for any out-of-domain field.
    pub fn validate(&self) -> Result<(), MemoryError> {
        if !(self.forgetting_factor.is_finite() && (0.0..1.0).contains(&self.forgetting_factor)) {
            return Err(MemoryError::InvalidConfig {
                what: "forgetting_factor must be finite and in [0, 1)",
            });
        }
        if self.min_history < 1 {
            return Err(MemoryError::InvalidConfig {
                what: "min_history must be >= 1",
            });
        }
        if !(self.significance_threshold.is_finite() && self.significance_threshold > 0.0) {
            return Err(MemoryError::InvalidConfig {
                what: "significance_threshold must be finite and > 0",
            });
        }
        if !(self.occupancy_floor_std.is_finite() && self.occupancy_floor_std > 0.0) {
            return Err(MemoryError::InvalidConfig {
                what: "occupancy_floor_std must be finite and > 0",
            });
        }
        if !(self.signature_floor_std.is_finite() && self.signature_floor_std > 0.0) {
            return Err(MemoryError::InvalidConfig {
                what: "signature_floor_std must be finite and > 0",
            });
        }
        if self.model_version.is_empty() || self.model_version.len() > MAX_CHANNEL_ID_LEN {
            return Err(MemoryError::InvalidConfig {
                what: "model_version must be non-empty and bounded",
            });
        }
        Ok(())
    }
}

/// UTC hour-of-day derived deterministically from an injected Unix-ms timestamp.
/// Pure arithmetic on the caller-supplied value — no wall-clock is read. Handles
/// negative timestamps (pre-1970) via Euclidean remainder.
#[must_use]
pub fn hour_of_day_utc(at_unix_ms: i64) -> u8 {
    let hours = at_unix_ms.div_euclid(3_600_000);
    hours.rem_euclid(HOURS_PER_DAY as i64) as u8
}

/// One learned per-link propagation statistic within a zone. Stored as a `Vec`
/// (not a map) so the whole baseline serializes to JSON — [`LinkId`] is a
/// struct, not a string key.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkStat {
    /// The link this statistic describes.
    pub link: LinkId,
    /// The learned normal RSSI distribution for the link.
    pub stat: RunningStat,
}

/// A live snapshot of a zone used to score against, and then update, its learned
/// normal. Time is injected; nothing here samples a clock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZoneObservation {
    /// The zone this snapshot is for.
    pub zone: ZoneId,
    /// Producer-supplied capture time (Unix ms). Injected; the occupancy hour is
    /// derived from it deterministically.
    pub at_unix_ms: i64,
    /// Occupancy indicator / count for this snapshot (`≥ 0`, finite).
    pub occupancy: f64,
    /// Per-link observed values (reusing the twin's [`ObservationSet`]
    /// vocabulary), scored against the learned propagation baseline.
    pub links: ObservationSet,
    /// Coarse per-modality signature channels (e.g. `"vibration_rms"`), scored
    /// against the learned modality baseline.
    pub modality: BTreeMap<String, f64>,
    /// Evidence level of the source observations. A learned baseline never rises
    /// above the floor of these (ADR-282 no-upgrade).
    pub evidence_level: EvidenceLevel,
}

impl ZoneObservation {
    /// A snapshot with no links or modality channels yet.
    #[must_use]
    pub fn new(zone: ZoneId, at_unix_ms: i64, occupancy: f64, evidence_level: EvidenceLevel) -> Self {
        Self {
            zone,
            at_unix_ms,
            occupancy,
            links: ObservationSet::new(),
            modality: BTreeMap::new(),
            evidence_level,
        }
    }

    /// Add a link observation (builder style).
    #[must_use]
    pub fn with_link(mut self, link: LinkId, value: f64) -> Self {
        self.links = self.links.with(link, value);
        self
    }

    /// Add a modality signature channel (builder style).
    #[must_use]
    pub fn with_modality(mut self, channel: impl Into<String>, value: f64) -> Self {
        self.modality.insert(channel.into(), value);
        self
    }

    /// Validate the snapshot at the boundary: finite, non-negative occupancy;
    /// finite link/modality values; bounded channel count and id length. Never
    /// panics.
    pub(crate) fn validate(&self) -> Result<(), MemoryError> {
        if !self.occupancy.is_finite() {
            return Err(MemoryError::NonFiniteValue { what: "occupancy" });
        }
        if self.occupancy < 0.0 {
            return Err(MemoryError::NegativeOccupancy {
                value: self.occupancy,
            });
        }
        for obs in &self.links.observations {
            if !obs.value.is_finite() {
                return Err(MemoryError::NonFiniteValue { what: "link value" });
            }
        }
        if self.modality.len() > MAX_MODALITY_CHANNELS {
            return Err(MemoryError::TooManyChannels {
                max: MAX_MODALITY_CHANNELS,
            });
        }
        for (channel, value) in &self.modality {
            if channel.len() > MAX_CHANNEL_ID_LEN {
                return Err(MemoryError::ChannelIdTooLong {
                    len: channel.len(),
                    max: MAX_CHANNEL_ID_LEN,
                });
            }
            if !value.is_finite() {
                return Err(MemoryError::NonFiniteValue {
                    what: "modality value",
                });
            }
        }
        Ok(())
    }
}

/// The learned normal physics of one zone (ADR-309 §1): occupancy periodicity,
/// per-link RF propagation, and coarse per-modality signatures.
///
/// **SYNTHETIC / L0.** A learned model of normality, never a measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZoneBaseline {
    /// Occupancy distribution indexed by UTC hour-of-day (`0..24`).
    occupancy_by_hour: [RunningStat; HOURS_PER_DAY],
    /// Learned per-link propagation normal, in deterministic insertion order.
    propagation: Vec<LinkStat>,
    /// Learned per-channel modality normal.
    modality: BTreeMap<String, RunningStat>,
    /// Floor of the evidence levels the baseline was learned from; `None` until
    /// the first observation. A learned normal is never above this.
    evidence_floor: Option<EvidenceLevel>,
    /// Total observations folded into this baseline.
    updates: u64,
}

impl Default for ZoneBaseline {
    fn default() -> Self {
        Self::new()
    }
}

impl ZoneBaseline {
    /// An empty baseline with no history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            occupancy_by_hour: [RunningStat::new(); HOURS_PER_DAY],
            propagation: Vec::new(),
            modality: BTreeMap::new(),
            evidence_floor: None,
            updates: 0,
        }
    }

    /// The occupancy statistic for a UTC hour (`0..24`).
    #[must_use]
    pub fn occupancy_hour(&self, hour: u8) -> &RunningStat {
        &self.occupancy_by_hour[(hour as usize) % HOURS_PER_DAY]
    }

    /// The learned statistic for a link, if any.
    #[must_use]
    pub fn link_stat(&self, link: &LinkId) -> Option<&RunningStat> {
        self.propagation
            .iter()
            .find(|ls| &ls.link == link)
            .map(|ls| &ls.stat)
    }

    /// The learned statistic for a modality channel, if any.
    #[must_use]
    pub fn modality_stat(&self, channel: &str) -> Option<&RunningStat> {
        self.modality.get(channel)
    }

    /// The floor of evidence levels this baseline was learned from, `None`
    /// before any observation.
    #[must_use]
    pub fn evidence_floor(&self) -> Option<EvidenceLevel> {
        self.evidence_floor
    }

    /// Total observations folded into this baseline.
    #[must_use]
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// The learned links, read-only.
    #[must_use]
    pub fn links(&self) -> &[LinkStat] {
        &self.propagation
    }

    /// Seed (or overwrite) a link's baseline from a prior mean/variance — used to
    /// anchor the propagation model on the twin's expected distribution. Bounded.
    pub(crate) fn seed_link(
        &mut self,
        link: LinkId,
        mean: f64,
        variance: f64,
        count: u32,
    ) -> Result<(), MemoryError> {
        let seeded = RunningStat::seeded(mean, variance, count);
        if let Some(ls) = self.propagation.iter_mut().find(|ls| ls.link == link) {
            ls.stat = seeded;
            return Ok(());
        }
        if self.propagation.len() >= MAX_LINKS_PER_ZONE {
            return Err(MemoryError::TooManyLinks {
                max: MAX_LINKS_PER_ZONE,
            });
        }
        self.propagation.push(LinkStat { link, stat: seeded });
        Ok(())
    }

    /// Mutable access to a link statistic, inserting a fresh one if absent.
    /// Bounded — an over-capacity zone is rejected, never grown unbounded.
    pub(crate) fn link_stat_mut(&mut self, link: &LinkId) -> Result<&mut RunningStat, MemoryError> {
        if let Some(pos) = self.propagation.iter().position(|ls| &ls.link == link) {
            return Ok(&mut self.propagation[pos].stat);
        }
        if self.propagation.len() >= MAX_LINKS_PER_ZONE {
            return Err(MemoryError::TooManyLinks {
                max: MAX_LINKS_PER_ZONE,
            });
        }
        self.propagation.push(LinkStat {
            link: link.clone(),
            stat: RunningStat::new(),
        });
        let last = self.propagation.len() - 1;
        Ok(&mut self.propagation[last].stat)
    }

    /// Mutable access to a modality statistic, inserting a fresh one if absent.
    /// Bounded.
    pub(crate) fn modality_stat_mut(
        &mut self,
        channel: &str,
    ) -> Result<&mut RunningStat, MemoryError> {
        if !self.modality.contains_key(channel) && self.modality.len() >= MAX_MODALITY_CHANNELS {
            return Err(MemoryError::TooManyChannels {
                max: MAX_MODALITY_CHANNELS,
            });
        }
        Ok(self
            .modality
            .entry(channel.to_string())
            .or_insert_with(RunningStat::new))
    }

    /// Fold one snapshot's occupancy into the hour bucket.
    pub(crate) fn update_occupancy(&mut self, hour: u8, occupancy: f64, forgetting: f64) {
        self.occupancy_by_hour[(hour as usize) % HOURS_PER_DAY].update(occupancy, forgetting);
    }

    /// Lower the evidence floor to include a prior/source at `level`, without
    /// counting it as an observation. Used to record the twin's SYNTHETIC/L0
    /// prior when seeding a propagation baseline.
    pub(crate) fn record_prior(&mut self, level: EvidenceLevel) {
        self.evidence_floor = Some(match self.evidence_floor {
            Some(existing) => existing.min(level),
            None => level,
        });
    }

    /// Lower the evidence floor to include a new source observation, and bump the
    /// update count.
    pub(crate) fn record_source(&mut self, level: EvidenceLevel) {
        self.record_prior(level);
        self.updates = self.updates.saturating_add(1);
    }
}
