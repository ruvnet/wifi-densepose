//! `BfldPipeline` — public entry point. ADR-118 §2.1.
//!
//! Thin facade over [`crate::BfldEmitter`] that adds:
//!
//! - A configuration struct ([`BfldConfig`]) for ergonomic construction.
//! - A `privacy_mode` toggle that flips the active class to
//!   [`PrivacyClass::Restricted`] (and back to the configured baseline)
//!   without rebuilding the underlying emitter state.
//! - A single named consumer call ([`Self::process`]) so callers don't have
//!   to navigate the lower-level emitter API.
//!
//! Future iters add `process_to_frame()` (BfldFrame production) and a `tokio`
//! MQTT loop wrapper on top of this same facade.

#![cfg(feature = "std")]

use crate::coherence_gate::SoulMatchOracle;
use crate::emitter::{BfldEmitter, SensingInputs};
use crate::identity_risk::GateAction;
use crate::signature_hasher::SignatureHasher;
use crate::{BfldEvent, BfldFrame, BfldFrameHeader, BfldPayload, IdentityEmbedding, PrivacyClass};

/// Construction parameters for [`BfldPipeline`]. Matches the ADR-118 default-
/// secure posture: `class = Anonymous`, no zone, no signature hasher.
#[derive(Debug, Clone)]
pub struct BfldConfig {
    /// Node identifier published in every `BfldEvent.node_id`.
    pub node_id: String,
    /// Optional default zone; passed through to every event.
    pub default_zone_id: Option<String>,
    /// Baseline privacy class. `privacy_mode = true` overrides to Restricted.
    pub privacy_class: PrivacyClass,
    /// Optional signature hasher; when present, the pipeline derives
    /// `rf_signature_hash` via [`crate::IdentityFeatures`].
    pub signature_hasher: Option<SignatureHasher>,
}

impl BfldConfig {
    /// Build a minimal config: node_id only, class defaulted to Anonymous.
    #[must_use]
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            default_zone_id: None,
            privacy_class: PrivacyClass::Anonymous,
            signature_hasher: None,
        }
    }

    /// Set the default zone.
    #[must_use]
    pub fn with_zone(mut self, zone_id: impl Into<String>) -> Self {
        self.default_zone_id = Some(zone_id.into());
        self
    }

    /// Override the baseline privacy class.
    #[must_use]
    pub const fn with_privacy_class(mut self, class: PrivacyClass) -> Self {
        self.privacy_class = class;
        self
    }

    /// Install a signature hasher.
    #[must_use]
    pub fn with_signature_hasher(mut self, hasher: SignatureHasher) -> Self {
        self.signature_hasher = Some(hasher);
        self
    }
}

/// Public BFLD entry point. Owns the configured emitter and the
/// `privacy_mode` toggle.
pub struct BfldPipeline {
    /// Baseline class — the class to which `disable_privacy_mode()` returns.
    baseline_class: PrivacyClass,
    privacy_mode: bool,
    emitter: BfldEmitter,
}

impl BfldPipeline {
    /// Build a pipeline from `config`. The underlying emitter is initialized
    /// with the configured class; `privacy_mode` is initially `false`.
    #[must_use]
    pub fn new(config: BfldConfig) -> Self {
        let mut emitter = BfldEmitter::new(config.node_id);
        if let Some(zone) = config.default_zone_id {
            emitter = emitter.with_zone(zone);
        }
        emitter = emitter.with_privacy_class(config.privacy_class);
        if let Some(hasher) = config.signature_hasher {
            emitter = emitter.with_signature_hasher(hasher);
        }
        Self {
            baseline_class: config.privacy_class,
            privacy_mode: false,
            emitter,
        }
    }

    /// Process a single sensing frame. Delegates to the underlying emitter,
    /// then post-processes the resulting event to honor `privacy_mode`. When
    /// privacy mode is engaged the published event is demoted to Restricted
    /// (identity-derived fields stripped) regardless of the configured baseline.
    pub fn process(
        &mut self,
        inputs: SensingInputs,
        embedding: Option<IdentityEmbedding>,
    ) -> Option<BfldEvent> {
        let mut event = self.emitter.emit(inputs, embedding)?;
        if self.privacy_mode {
            event.privacy_class = PrivacyClass::Restricted;
            event.apply_privacy_gating();
        }
        Some(event)
    }

    /// Variant of [`Self::process`] that consults a [`SoulMatchOracle`] before
    /// the coherence gate fires `Recalibrate`. See ADR-121 §2.6 and ADR-118
    /// §1.4. The privacy_mode post-processing still applies; the oracle only
    /// affects whether the gate transitions to Recalibrate at all.
    pub fn process_with_oracle<O: SoulMatchOracle>(
        &mut self,
        inputs: SensingInputs,
        embedding: Option<IdentityEmbedding>,
        oracle: &O,
    ) -> Option<BfldEvent> {
        let mut event = self.emitter.emit_with_oracle(inputs, embedding, oracle)?;
        if self.privacy_mode {
            event.privacy_class = PrivacyClass::Restricted;
            event.apply_privacy_gating();
        }
        Some(event)
    }

    /// Wire-bytes variant of [`Self::process`]: returns a [`BfldFrame`] ready
    /// to serialize via `BfldFrame::to_bytes()`. Caller supplies a
    /// `header_template` carrying AP / STA / session identity fields and a
    /// `payload` typed via [`BfldPayload`]. The pipeline overrides the
    /// template's `timestamp_ns` and `privacy_class` from its own state, then
    /// builds the frame via [`BfldFrame::from_payload`] so the CRC covers the
    /// section-prefixed bytes.
    ///
    /// The emitted frame's payload is forced into compliance with the active
    /// privacy class via [`crate::PrivacyGate::demote`]: at `Anonymous` the
    /// identity-leaky `compressed_angle_matrix` and `csi_delta` sections are
    /// stripped, and at `Restricted` the amplitude/phase proxies are stripped
    /// too. This closes the gap (ADR-141) where a frame stamped with a
    /// restrictive class byte could otherwise carry the full high-information
    /// BFI payload across a [`crate::NetworkSink`]. Research classes (`Raw`,
    /// `Derived`) keep the full payload — `demote` is a no-op there.
    ///
    /// Returns `None` whenever the gate drops the underlying event (Reject or
    /// Recalibrate), so `process_to_frame` is a strict subset of `process`.
    pub fn process_to_frame(
        &mut self,
        inputs: SensingInputs,
        header_template: BfldFrameHeader,
        payload: BfldPayload,
        embedding: Option<IdentityEmbedding>,
    ) -> Option<BfldFrame> {
        let timestamp_ns = inputs.timestamp_ns;
        let active_class = self.current_privacy_class();
        let _gate_signal = self.process(inputs, embedding)?;
        let mut header = header_template;
        header.timestamp_ns = timestamp_ns;
        header.privacy_class = active_class.as_u8();
        let frame = BfldFrame::from_payload(header, &payload);
        // Enforce the payload-content policy for the stamped class. The frame
        // is already at `active_class`, so this is a same-class demotion: it
        // performs no class change but strips the sections that class forbids.
        // demote() only fails on InvalidDemote (target < source), which cannot
        // happen here because source == target, so the expect is unreachable.
        Some(
            crate::PrivacyGate::demote(frame, active_class)
                .expect("same-class demote is always valid"),
        )
    }

    /// `true` if `enable_privacy_mode()` has been called more recently than
    /// `disable_privacy_mode()`.
    #[must_use]
    pub const fn is_privacy_mode_enabled(&self) -> bool {
        self.privacy_mode
    }

    /// Read the currently active class. Returns Restricted if privacy mode is
    /// engaged, otherwise the baseline.
    #[must_use]
    pub const fn current_privacy_class(&self) -> PrivacyClass {
        if self.privacy_mode {
            PrivacyClass::Restricted
        } else {
            self.baseline_class
        }
    }

    /// Read-only access to the current gate action — for diagnostics.
    #[must_use]
    pub const fn current_gate_action(&self) -> GateAction {
        self.emitter.current_action()
    }

    /// Engage privacy mode: future `process()` calls return events demoted
    /// to Restricted (identity_risk_score + rf_signature_hash stripped)
    /// regardless of the configured baseline.
    ///
    /// The override is applied post-emission so the underlying gate / ring /
    /// hasher state remains unchanged and recoverable when privacy mode is
    /// later disabled.
    pub fn enable_privacy_mode(&mut self) {
        self.privacy_mode = true;
    }

    /// Disengage privacy mode: future events return to the configured baseline.
    pub fn disable_privacy_mode(&mut self) {
        self.privacy_mode = false;
    }
}
