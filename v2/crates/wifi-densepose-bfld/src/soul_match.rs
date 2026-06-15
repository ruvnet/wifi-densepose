//! §3.6 Soul Signature matching algorithm — the **first running implementation**.
//!
//! This module implements, exactly, the per-channel weighted-cosine matcher
//! specified in `docs/research/soul/specification.md` §3.6:
//!
//! ```text
//! match_score = Σ_i ( w_i · cosine_sim(P.channel_i, Q.channel_i) )
//!               / Σ_i ( w_i · availability(P.channel_i, Q.channel_i) )
//! ```
//!
//! where `availability(P_i, Q_i)` is `1.0` iff **both** the profile and the
//! query carry channel `i` (and the data is usable), else `0.0`. The division
//! normalizes the score by the weight mass of the channels that were actually
//! shared, so a probe missing a channel degrades gracefully instead of being
//! penalized for the absence.
//!
//! ## What this module proves — and what it does NOT
//!
//! It **runs**: feed two [`SoulChannels`] and it returns a calibrated, fully
//! transparent [`MatchScore`] (overall score, contributing-channel count, and
//! per-channel cosine contributions). [`EnrolledMatcher`] wires that into the
//! real [`SoulMatchOracle`] the coherence gate already calls — replacing the
//! reliance on [`crate::coherence_gate::NullOracle`], which always returns
//! `NotEnrolled`.
//!
//! It does **NOT** claim working named-person identification. Named-identity
//! locking is gated on the two decisive high-weight channels — the AETHER
//! embedding (0.35), populated from a **real enrollment**, and (in multi-room
//! deployments) the body-resonance / Body-Field-Coupling channel — being fed
//! with real measured data. That has not been done in this repo. On the
//! low-weight cardiac (0.15) + respiratory (0.10) channels **alone**, identity
//! is **not separable above any useful threshold** — heartbeat and breathing
//! rates overlap too much between people. This is not a hypothesis: it is
//! measured by the test
//! `cardiac_alone_cannot_separate_identity_matches_audit` in
//! `tests/soul_match.rs`. The weights themselves are §3.6 **design intent, not
//! validated** (see [`crate::soul_channels::MatchWeights`]).
//!
//! In short: a real matcher that honestly reports where it cannot lock.

use crate::soul_channels::{Channel, MatchWeights, SoulChannels};

/// Result of one §3.6 match evaluation.
///
/// Carries the normalized score **and** the evidence behind it, so a caller
/// (or an auditor) can see exactly which channels contributed and by how much.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchScore {
    /// The normalized §3.6 score, or `None` when the match is **undefined**
    /// because no weighted channel was shared (denominator = 0). A `None`
    /// score is NEVER coerced to a high value — see [`Self::is_defined`].
    score: Option<f32>,
    /// Number of channels with `availability > 0` (shared by both sides) AND
    /// non-zero weight — i.e. channels that actually moved the score.
    contributing_channels: usize,
    /// Per-channel cosine contribution. `None` for channels not shared (or
    /// zero-weight); `Some(cos)` with the raw cosine similarity otherwise.
    /// Index-aligned to [`Channel::ALL`].
    per_channel: [Option<f32>; crate::soul_channels::CHANNEL_COUNT],
}

impl MatchScore {
    /// The normalized score in `[-1, 1]`, or `None` if undefined (no shared
    /// weighted channels). Callers MUST treat `None` as "insufficient
    /// evidence", never as a default-high match.
    #[must_use]
    pub const fn score(&self) -> Option<f32> {
        self.score
    }

    /// `true` iff a score was computable (≥1 shared, weighted channel).
    #[must_use]
    pub const fn is_defined(&self) -> bool {
        self.score.is_some()
    }

    /// Number of channels that contributed to the score (`availability > 0`
    /// and non-zero weight).
    #[must_use]
    pub const fn contributing_channels(&self) -> usize {
        self.contributing_channels
    }

    /// Raw cosine contribution for a specific channel, if it was shared and
    /// weighted. Useful for transparency / dashboards.
    #[must_use]
    pub fn channel_contribution(&self, channel: Channel) -> Option<f32> {
        self.per_channel[channel.index()]
    }

    /// An undefined result — no shared weighted channels.
    const fn insufficient() -> Self {
        Self {
            score: None,
            contributing_channels: 0,
            per_channel: [None; crate::soul_channels::CHANNEL_COUNT],
        }
    }
}

/// Compute the §3.6 match score of `query` against `profile` under `weights`.
///
/// Implements the spec formula verbatim. For each channel `i`:
/// - `availability` is `1.0` iff both `profile` and `query` carry usable data
///   for `i` (a zero-norm or empty channel counts as unavailable — it can
///   never contribute, and must never produce NaN).
/// - `cosine_sim` is the standard cosine similarity in `[-1, 1]`. When the two
///   shared channels have **different lengths**, only the overlapping prefix
///   is compared (channels are expected to be same-length by construction;
///   this is a defensive fallback, never a NaN source).
///
/// If the denominator `Σ w_i · availability_i` is `0` (no shared weighted
/// channel), the score is **undefined** and a typed
/// [`MatchScore::insufficient`] is returned — NOT a default-high score.
#[must_use]
pub fn match_score(
    profile: &SoulChannels,
    query: &SoulChannels,
    weights: &MatchWeights,
) -> MatchScore {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    let mut contributing = 0usize;
    let mut per_channel = [None; crate::soul_channels::CHANNEL_COUNT];

    for channel in Channel::ALL {
        let w = weights.weight(channel);
        if w == 0.0 {
            // Zero-weight channels (e.g. Body-Field-Coupling single-room) can
            // never affect the score; skip them so they do not pollute the
            // contributing-channel count or the denominator.
            continue;
        }

        let availability = availability(profile, query, channel);
        if availability == 0.0 {
            continue;
        }

        // Both sides present and weighted — compute the cosine contribution.
        let (Some(p), Some(q)) = (profile.channel_slice(channel), query.channel_slice(channel))
        else {
            // Unreachable given availability == 1.0, but stay total.
            continue;
        };
        let cos = cosine_sim(p, q);
        numerator += w * cos;
        denominator += w * availability;
        per_channel[channel.index()] = Some(cos);
        contributing += 1;
    }

    if denominator == 0.0 {
        return MatchScore::insufficient();
    }

    MatchScore {
        score: Some(numerator / denominator),
        contributing_channels: contributing,
        per_channel,
    }
}

/// §3.6 `availability(P_i, Q_i)`: `1.0` iff both sides carry usable data for
/// `channel`, else `0.0`. A present-but-zero-norm / empty channel is treated
/// as unavailable (it cannot yield a meaningful cosine and would otherwise
/// risk a NaN).
#[must_use]
fn availability(profile: &SoulChannels, query: &SoulChannels, channel: Channel) -> f32 {
    match (profile.channel_slice(channel), query.channel_slice(channel)) {
        (Some(p), Some(q)) if is_usable(p) && is_usable(q) => 1.0,
        _ => 0.0,
    }
}

/// A channel slice is usable for cosine if it is non-empty and has non-zero
/// L2 norm (so the cosine denominator is positive — never a division by zero).
fn is_usable(v: &[f32]) -> bool {
    !v.is_empty() && v.iter().any(|x| x.is_finite() && *x != 0.0)
}

/// Standard cosine similarity in `[-1, 1]`.
///
/// Guards every NaN/zero-norm path: a zero-norm input (which `availability`
/// already excludes, but we stay total) yields `0.0`, never NaN. When the two
/// vectors differ in length, the overlapping prefix is used.
#[must_use]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        let x = a[i];
        let y = b[i];
        // Treat non-finite components as 0 — never propagate NaN into the score.
        let (x, y) = (if x.is_finite() { x } else { 0.0 }, if y.is_finite() {
            y
        } else {
            0.0
        });
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 || !denom.is_finite() {
        return 0.0;
    }
    let cos = dot / denom;
    // Clamp to [-1, 1] to absorb floating-point overshoot.
    cos.clamp(-1.0, 1.0)
}

// --- EnrolledMatcher: the real SoulMatchOracle -----------------------------

#[cfg(feature = "std")]
pub use self::enrolled::EnrolledMatcher;

#[cfg(feature = "std")]
mod enrolled {
    use core::cell::RefCell;

    use super::{match_score, MatchScore};
    use crate::coherence_gate::{MatchOutcome, SoulMatchOracle};
    use crate::soul_channels::{MatchWeights, SoulChannels};

    /// A real [`SoulMatchOracle`]: holds enrolled `(person_id, SoulChannels)`
    /// profiles and, given a probe, returns the best enrolled match that clears
    /// both a score threshold and a minimum shared-channel count.
    ///
    /// This is the production-honest replacement for relying on
    /// [`crate::coherence_gate::NullOracle`] (which always reports
    /// `NotEnrolled`). `NullOracle` remains the correct default when Soul
    /// Signature is disabled; `EnrolledMatcher` is what runs when it is enabled
    /// **and** real enrolled data is present.
    ///
    /// ## Interior mutability for the `&self` trait method
    ///
    /// [`SoulMatchOracle::matches_enrolled`] takes `&self`, but a match needs a
    /// live probe. The probe is stored behind a [`RefCell`] and set via
    /// [`EnrolledMatcher::set_probe`] before each gate evaluation. With no
    /// probe set, the oracle reports `NotEnrolled` (fail-closed).
    pub struct EnrolledMatcher {
        profiles: Vec<(u64, SoulChannels)>,
        weights: MatchWeights,
        threshold: f32,
        min_channels: usize,
        probe: RefCell<Option<SoulChannels>>,
    }

    impl EnrolledMatcher {
        /// Build a matcher with a score threshold and a minimum
        /// shared-channel requirement.
        ///
        /// `threshold` is the deployment-specific minimum score (§3.6: "a
        /// deployment-specific parameter with a documented FAR/FRR
        /// trade-off"). `min_channels` is the minimum number of channels that
        /// must be shared for a match to be considered at all — set this above
        /// 1 so a single low-weight channel can never lock identity.
        #[must_use]
        pub fn new(weights: MatchWeights, threshold: f32, min_channels: usize) -> Self {
            Self {
                profiles: Vec::new(),
                weights,
                threshold,
                min_channels,
                probe: RefCell::new(None),
            }
        }

        /// Enroll a profile under an opaque `person_id`.
        pub fn enroll(&mut self, person_id: u64, profile: SoulChannels) {
            self.profiles.push((person_id, profile));
        }

        /// Number of enrolled profiles.
        #[must_use]
        pub fn len(&self) -> usize {
            self.profiles.len()
        }

        /// `true` if no profiles are enrolled.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.profiles.is_empty()
        }

        /// Set the live probe to be matched on the next oracle call. Replaces
        /// any previously-set probe.
        pub fn set_probe(&self, probe: SoulChannels) {
            *self.probe.borrow_mut() = Some(probe);
        }

        /// Clear the probe — subsequent oracle calls report `NotEnrolled`.
        pub fn clear_probe(&self) {
            *self.probe.borrow_mut() = None;
        }

        /// Score the current probe against every enrolled profile and return
        /// the best `(person_id, MatchScore)` whose score is **defined**.
        /// Returns `None` if there is no probe, no enrolled profile, or no
        /// defined score. This does NOT apply the threshold — it is the raw
        /// transparency view used by tests and dashboards.
        #[must_use]
        pub fn best_match(&self) -> Option<(u64, MatchScore)> {
            let probe = self.probe.borrow();
            let probe = probe.as_ref()?;
            let mut best: Option<(u64, MatchScore)> = None;
            for (person_id, profile) in &self.profiles {
                let ms = match_score(profile, probe, &self.weights);
                let Some(s) = ms.score() else { continue };
                let better = match best {
                    None => true,
                    Some((_, prev)) => prev.score().map_or(true, |ps| s > ps),
                };
                if better {
                    best = Some((*person_id, ms));
                }
            }
            best
        }
    }

    impl SoulMatchOracle for EnrolledMatcher {
        /// Real §3.6 oracle. Returns [`MatchOutcome::Match`] for the best
        /// enrolled profile whose score is **defined**, clears `threshold`,
        /// **and** shares at least `min_channels` channels. Otherwise
        /// [`MatchOutcome::NotEnrolled`].
        ///
        /// Fail-closed: empty enrolled set, no probe, undefined score,
        /// below-threshold score, or too-few shared channels all yield
        /// `NotEnrolled` — never a false `Match`.
        fn matches_enrolled(&self) -> MatchOutcome {
            match self.best_match() {
                Some((person_id, ms)) => {
                    let score = ms.score().unwrap_or(f32::NEG_INFINITY);
                    if score >= self.threshold
                        && ms.contributing_channels() >= self.min_channels
                    {
                        MatchOutcome::Match { person_id }
                    } else {
                        MatchOutcome::NotEnrolled
                    }
                }
                None => MatchOutcome::NotEnrolled,
            }
        }
    }
}
