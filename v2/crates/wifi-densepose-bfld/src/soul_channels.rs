//! Per-channel signature container + weight table for the §3.6 matcher.
//!
//! This module ports the channel inventory and default weight table from
//! `docs/research/soul/specification.md` §3.6 into running types. It is the
//! data half of the matcher; the algorithm lives in
//! [`crate::soul_match`].
//!
//! ## What a `SoulChannels` is (and is NOT)
//!
//! A [`SoulChannels`] holds, for one signature, the per-channel feature
//! vectors that §3.6 fuses. Each channel is `Option<...>`: `None` means the
//! channel could not be measured in this window (the matcher treats it as
//! *unavailable* and excludes it from the normalized denominator — graceful
//! degradation, §3.6).
//!
//! The AETHER channel reuses the crate's [`IdentityEmbedding`]
//! ([`crate::embedding`]) so it inherits structural invariant **I2**
//! (in-RAM-only; no `Serialize`/`Clone`/`Copy`; zeroized on `Drop`). As a
//! direct consequence, `SoulChannels` is itself **not `Clone`** — you build a
//! signature once and move it into an enrolled set or use it as a probe.
//!
//! ## Weights are design-intent, not validated
//!
//! The [`MatchWeights::default`] values come from the §3.6 table, which the
//! spec explicitly labels *"open research; these are design intent, not
//! validated"*. They are reproduced faithfully here **with that caveat
//! intact**. Nothing in this crate has tuned them against measured FAR/FRR.

use crate::embedding::IdentityEmbedding;

/// Number of channels fused by the §3.6 matcher.
pub const CHANNEL_COUNT: usize = 8;

/// The eight Soul Signature channels, in the §3.6 table order.
///
/// The enum is the stable index into [`MatchWeights`] and into the
/// per-channel contribution array returned by the matcher. AETHER is index 0
/// (highest design-intent weight); the order otherwise follows the spec table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Channel {
    /// AETHER contrastive embedding (ADR-024). Primary identity anchor.
    AetherEmbedding = 0,
    /// Subcarrier reflection profile — body geometry, angle-stable.
    SubcarrierReflectionProfile = 1,
    /// Cardiac heart-rate profile — physiologically stable in healthy adults.
    CardiacHrProfile = 2,
    /// Gait timing — well-studied, discriminative biometric.
    GaitTiming = 3,
    /// Respiratory pattern — more variable than cardiac.
    RespiratoryPattern = 4,
    /// Skeletal proportions — proxy for body shape; CSI-only is noisy.
    SkeletalProportions = 5,
    /// Body–field coupling — valid only with a room field model
    /// (weight 0.0 single-room).
    BodyFieldCoupling = 6,
    /// Cardiac waveform morphology — supplementary, high-SNR requirement.
    CardiacWaveformMorphology = 7,
}

impl Channel {
    /// All channels in index order. Handy for iterating the matcher.
    pub const ALL: [Channel; CHANNEL_COUNT] = [
        Channel::AetherEmbedding,
        Channel::SubcarrierReflectionProfile,
        Channel::CardiacHrProfile,
        Channel::GaitTiming,
        Channel::RespiratoryPattern,
        Channel::SkeletalProportions,
        Channel::BodyFieldCoupling,
        Channel::CardiacWaveformMorphology,
    ];

    /// Index of this channel (0..[`CHANNEL_COUNT`]).
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// The §3.6 default weights, faithfully reproduced.
///
/// These are **unvalidated design intent** per the spec table. `weights[i]`
/// is the weight of `Channel::ALL[i]`.
///
/// | Channel | Weight |
/// |---|---|
/// | AETHER_Embedding | 0.35 |
/// | Subcarrier_Reflection_Profile | 0.20 |
/// | Cardiac_HR_Profile | 0.15 |
/// | Gait_Timing | 0.15 |
/// | Respiratory_Pattern | 0.10 |
/// | Skeletal_Proportions | 0.05 |
/// | Body_Field_Coupling | 0.00 (single-room) |
/// | Cardiac_Waveform_Morphology | 0.05 |
pub const DEFAULT_WEIGHTS: [f32; CHANNEL_COUNT] =
    [0.35, 0.20, 0.15, 0.15, 0.10, 0.05, 0.00, 0.05];

/// Per-channel fusion weights for the §3.6 score.
///
/// Construct with [`MatchWeights::default`] for the spec table, or
/// [`MatchWeights::new`] for a custom (validated, non-negative, finite)
/// weight vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchWeights {
    weights: [f32; CHANNEL_COUNT],
}

impl MatchWeights {
    /// Build from an explicit weight vector.
    ///
    /// # Errors
    /// Returns [`WeightError`] if any weight is negative, NaN, or infinite, or
    /// if all weights are zero (a degenerate table that can never produce a
    /// defined score).
    pub fn new(weights: [f32; CHANNEL_COUNT]) -> Result<Self, WeightError> {
        let mut any_positive = false;
        for &w in &weights {
            if w.is_nan() || w.is_infinite() {
                return Err(WeightError::NotFinite);
            }
            if w < 0.0 {
                return Err(WeightError::Negative);
            }
            if w > 0.0 {
                any_positive = true;
            }
        }
        if !any_positive {
            return Err(WeightError::AllZero);
        }
        Ok(Self { weights })
    }

    /// Weight of a specific channel.
    #[must_use]
    pub const fn weight(&self, channel: Channel) -> f32 {
        self.weights[channel.index()]
    }

    /// Borrow the raw weight vector (index-aligned to [`Channel::ALL`]).
    #[must_use]
    pub const fn as_array(&self) -> &[f32; CHANNEL_COUNT] {
        &self.weights
    }
}

impl Default for MatchWeights {
    /// The §3.6 default table — **unvalidated design intent**.
    fn default() -> Self {
        Self {
            weights: DEFAULT_WEIGHTS,
        }
    }
}

/// Why a [`MatchWeights`] construction was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WeightError {
    /// A weight was negative — weights must be in `[0, ∞)`.
    #[error("match weight must be non-negative")]
    Negative,
    /// A weight was NaN or infinite.
    #[error("match weight must be finite")]
    NotFinite,
    /// Every weight was zero — the score denominator could never be positive.
    #[error("at least one match weight must be positive")]
    AllZero,
}

/// One signature's per-channel feature vectors.
///
/// `aether` reuses [`IdentityEmbedding`] (invariant I2); the remaining seven
/// channels are plain feature vectors held as fixed-capacity arrays so the
/// type is `no_std`-compatible with no heap allocation. A channel set to
/// `None` is *unavailable* and is excluded from the §3.6 denominator.
///
/// Because `IdentityEmbedding` is intentionally not `Clone`, `SoulChannels`
/// is not `Clone` either — build it once, then move it into the enrolled set
/// or hand it to the matcher as a probe.
pub struct SoulChannels {
    /// AETHER embedding channel (in-RAM-only; I2). `None` if not enrolled/measured.
    pub aether: Option<IdentityEmbedding>,
    /// The seven non-AETHER channels, index-aligned to `Channel` 1..=7.
    /// `vectors[c.index() - 1]` holds channel `c` (AETHER lives in `aether`).
    vectors: [Option<FeatureVector>; CHANNEL_COUNT - 1],
}

/// Fixed-capacity feature vector for a non-AETHER channel.
///
/// Capacity is chosen to comfortably hold the largest non-AETHER channel in
/// the §3.6 schema (the 336-element subcarrier reflection profile, §3.1).
pub const FEATURE_VECTOR_CAP: usize = 336;

/// A bounded, heapless per-channel feature vector.
#[derive(Debug, Clone, Copy)]
pub struct FeatureVector {
    data: [f32; FEATURE_VECTOR_CAP],
    len: usize,
}

impl FeatureVector {
    /// Build a feature vector from a slice.
    ///
    /// # Errors
    /// Returns [`WeightError::NotFinite`] reused as a generic "bad data"
    /// signal if `values` is longer than [`FEATURE_VECTOR_CAP`].
    pub fn from_slice(values: &[f32]) -> Result<Self, FeatureError> {
        if values.len() > FEATURE_VECTOR_CAP {
            return Err(FeatureError::TooLong {
                got: values.len(),
                cap: FEATURE_VECTOR_CAP,
            });
        }
        let mut data = [0.0f32; FEATURE_VECTOR_CAP];
        data[..values.len()].copy_from_slice(values);
        Ok(Self {
            data,
            len: values.len(),
        })
    }

    /// Borrow the populated values.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data[..self.len]
    }

    /// Number of populated elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` if the vector has no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Why a [`FeatureVector`] construction was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FeatureError {
    /// The input slice exceeded [`FEATURE_VECTOR_CAP`].
    #[error("feature vector too long: got {got}, cap {cap}")]
    TooLong {
        /// Length of the supplied slice.
        got: usize,
        /// Maximum capacity.
        cap: usize,
    },
}

impl SoulChannels {
    /// Build an empty signature — every channel `None` (unavailable).
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            aether: None,
            vectors: [const { None }; CHANNEL_COUNT - 1],
        }
    }

    /// Set the AETHER embedding channel (consumes the embedding; I2).
    #[must_use]
    pub fn with_aether(mut self, embedding: IdentityEmbedding) -> Self {
        self.aether = Some(embedding);
        self
    }

    /// Set a non-AETHER channel from a feature vector. Passing
    /// `Channel::AetherEmbedding` is a no-op (use [`Self::with_aether`]).
    #[must_use]
    pub fn with_channel(mut self, channel: Channel, vector: FeatureVector) -> Self {
        if let Some(slot) = self.vector_slot_mut(channel) {
            *slot = Some(vector);
        }
        self
    }

    /// Borrow a non-AETHER channel's vector, if present.
    #[must_use]
    pub fn channel_vector(&self, channel: Channel) -> Option<&FeatureVector> {
        match channel {
            Channel::AetherEmbedding => None,
            other => self.vectors[other.index() - 1].as_ref(),
        }
    }

    /// `true` if `channel` carries a usable (present) vector.
    #[must_use]
    pub fn has_channel(&self, channel: Channel) -> bool {
        match channel {
            Channel::AetherEmbedding => self.aether.is_some(),
            other => self.vectors[other.index() - 1].is_some(),
        }
    }

    /// Borrow channel data as an `f32` slice, regardless of channel kind.
    /// Returns `None` if the channel is unavailable.
    #[must_use]
    pub fn channel_slice(&self, channel: Channel) -> Option<&[f32]> {
        match channel {
            Channel::AetherEmbedding => self.aether.as_ref().map(IdentityEmbedding::as_slice),
            other => self.channel_vector(other).map(FeatureVector::as_slice),
        }
    }

    /// Count of channels currently present (available).
    #[must_use]
    pub fn available_count(&self) -> usize {
        Channel::ALL.iter().filter(|&&c| self.has_channel(c)).count()
    }

    fn vector_slot_mut(&mut self, channel: Channel) -> Option<&mut Option<FeatureVector>> {
        match channel {
            Channel::AetherEmbedding => None,
            other => Some(&mut self.vectors[other.index() - 1]),
        }
    }
}

impl Default for SoulChannels {
    fn default() -> Self {
        Self::empty()
    }
}
