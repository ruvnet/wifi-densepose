//! ADR-262 **P4, second modality**: ultrasonic range profiles on the field
//! surface.
//!
//! Until now RuView's `/api/field` and `/ws/field` have carried exactly one
//! modality — WiFi CSI, through [`crate::bridge`]. ADR-262 §4 lists multi-
//! modality as P4 and leaves the choice of the second one open (§8 question 5
//! asks whether it should be `rvcsi`, "making RuField the convergence point for
//! both vendored sensing runtimes").
//!
//! This is a different answer to that question, and the reason is that it costs
//! almost nothing: `rufield-adapters` now ships
//! [`UltrasonicReplayAdapter`](rufield_adapters::UltrasonicReplayAdapter), the
//! first adapter for `Modality::Ultrasonic` — registry code 7, which had sat in
//! the §8 registry since v0.1 with nothing implementing it. It parses, validates
//! and **signs** BatVu recordings upstream. RuView does not have to build any of
//! that; it has to decide what it is willing to put on a wire.
//!
//! # What BatVu is, in one paragraph
//!
//! [BatVu](https://github.com/ruvnet/batvu) is a handheld sonar that runs in a
//! phone browser: a 17.5–20.5 kHz chirp out of the speaker, a matched filter
//! over the microphone, and a **range profile** — echo amplitude against
//! distance along one beam. One microphone, so it measures range and infers
//! bearing only from where the operator was pointing. That asymmetry is why the
//! beam is carried as sensor *pose* rather than as an angle axis.
//!
//! # Why this module exists at all, given the adapter does the work
//!
//! Because RuView's egress rule is **stricter** than RuField's default guard,
//! and the difference has to be structural rather than a runtime refusal.
//!
//! The adapter offers two output modes. [`UltrasonicOutput::RangeProfile`] is
//! the full per-bin frame, classified `P0`, which
//! [`network_egress_allowed`](crate::network_egress_allowed) holds edge-local —
//! correctly, but *silently*, as a dropped event at the end of a pipeline that
//! did all the parsing and signing first.
//! [`UltrasonicOutput::CoarseProfile`] is a 32-bin max-pooled reduction,
//! classified `P1`, which is egress-safe.
//!
//! So this module does not offer the choice. It configures the adapter for the
//! coarse mode and says why, because a consumer cannot un-coarsen a coarse
//! profile and a policy expressed in the *shape of the data* cannot be
//! misconfigured later. The egress gate still runs — belt and braces, and it is
//! asserted by a test — but by then there is nothing left for it to catch.
//!
//! # Honesty (ADR-262 §0 / §6)
//!
//! Same posture as P1, plus one more caveat that is BatVu's rather than ours:
//!
//! 1. **Replay, not live.** A file, not a streaming phone.
//! 2. **Every current BatVu recording is its own simulator's output**, so the
//!    events carry `synthetic: true` and are fusable only under
//!    [`TrustPolicy::simulation`](rufield_provenance::TrustPolicy::simulation).
//!    `captured_replay()` and `production()` reject them outright, as they
//!    should. When a real device recording exists it declares
//!    `device_capture` and this module refuses it unless the caller has asked
//!    for that source explicitly — a recording cannot talk its way up a trust
//!    tier by relabelling itself.
//! 3. **No accuracy is claimed.** The detections are CFAR outputs at a
//!    documented false-alarm rate. That is an operating point, not validated
//!    accuracy against surveyed ground truth.
//! 4. **Nothing here says a person is present.** The adapter deliberately
//!    declines to populate the `presence` feature that would light up the
//!    shipped `person_present` rule: one transducer pair cannot distinguish a
//!    person from a coat over the back of a chair. The visible consequence is
//!    that ultrasonic events currently produce no fused inferences at all, and
//!    `ultrasonic_gates.rs` asserts that rather than papering over it.
//!
//! # What this does NOT do
//!
//! It does not touch the running server. P1 shipped as a library before P3
//! wired it in, and this follows the same staging: a tested translation with
//! its own gates, reviewable on its own, with the surface change as a separate
//! change once the shape is agreed. [`UltrasonicScan::egress_events`] returns
//! exactly what `FieldSurface::emit` would need to broadcast.

use rufield_adapters::{
    UltrasonicConfig, UltrasonicOutput, UltrasonicReplayAdapter, UltrasonicSource,
};
use rufield_core::{FieldEvent, PrivacyClass};

use crate::network_egress_allowed;

/// Which recordings a deployment is willing to accept.
///
/// Mirrors [`UltrasonicSource`] rather than re-exporting it, so a caller does
/// not have to depend on `rufield-adapters` to make the choice — the bridge
/// stays the single coupling point (ADR-262 §5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanSource {
    /// Rendered by BatVu's acoustic simulator. Events are `synthetic: true`.
    /// The safe default: it cannot be mistaken for a measurement.
    Simulated,
    /// Captured from a real phone microphone. Events are `synthetic: false`
    /// and are eligible for captured-replay trust once the sensor key is
    /// enrolled.
    DeviceCapture,
}

impl From<ScanSource> for UltrasonicSource {
    fn from(s: ScanSource) -> Self {
        match s {
            ScanSource::Simulated => UltrasonicSource::Simulated,
            ScanSource::DeviceCapture => UltrasonicSource::DeviceCapture,
        }
    }
}

/// Errors from loading a BatVu recording.
#[derive(Debug, Clone, PartialEq)]
pub enum ScanError {
    /// The recording failed the adapter's parser. Carries its message, which
    /// names the offending line.
    Parse(String),
    /// Calibration could not be established.
    Calibrate(String),
}

impl std::fmt::Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScanError::Parse(m) => write!(f, "ultrasonic recording rejected: {m}"),
            ScanError::Calibrate(m) => write!(f, "ultrasonic calibration failed: {m}"),
        }
    }
}

impl std::error::Error for ScanError {}

/// A loaded BatVu scan, ready to put on the field surface.
#[derive(Debug)]
pub struct UltrasonicScan {
    adapter: UltrasonicReplayAdapter,
    calibration_id: String,
}

impl UltrasonicScan {
    /// Load a `.ultrasonic.jsonl` recording.
    ///
    /// `zone_id` is the room the scan was taken in and lands on every
    /// observation. `accept` declares which source the deployment will take; a
    /// recording declaring anything else is **refused**, in both directions, so
    /// neither is a silent reinterpretation.
    ///
    /// Calibration runs here rather than being optional. The receipt id ends up
    /// on every tensor, so an event that skipped it would be one nobody could
    /// later ask "under what calibration was this measured".
    pub fn load(text: &str, zone_id: &str, accept: ScanSource) -> Result<Self, ScanError> {
        let config = UltrasonicConfig {
            accept: accept.into(),
            zone_id: zone_id.to_string(),
            placement: "handheld".to_string(),
            // Not a caller's choice. See the module docs: the raw per-bin frame
            // is P0 and would be dropped by the egress gate after all the work
            // of parsing and signing it. Coarsening at the source makes the
            // policy a property of the data instead of a property of a check
            // somebody could later reorder.
            output: UltrasonicOutput::CoarseProfile,
        };
        let mut adapter = UltrasonicReplayAdapter::from_jsonl_with(text, config)
            .map_err(|e| ScanError::Parse(e.to_string()))?;
        let receipt = adapter
            .calibrate(zone_id)
            .map_err(|e| ScanError::Calibrate(e.to_string()))?;
        Ok(UltrasonicScan {
            adapter,
            calibration_id: receipt.calibration_id,
        })
    }

    /// Pings in the recording.
    #[must_use]
    pub fn ping_count(&self) -> usize {
        self.adapter.ping_count()
    }

    /// The sensor identity the whole recording belongs to.
    ///
    /// This is the key RuField's replay watermark and trust registry are both
    /// keyed on, so it has to be stable across scans — the adapter refuses a
    /// recording whose device changes midway.
    #[must_use]
    pub fn device_id(&self) -> &str {
        self.adapter.device_id()
    }

    /// The calibration receipt every tensor in this scan cites.
    #[must_use]
    pub fn calibration_id(&self) -> &str {
        &self.calibration_id
    }

    /// Every event in the recording, signed and validated by the adapter.
    ///
    /// Includes events that the egress gate would refuse. Use this for an
    /// edge-local consumer; use [`egress_events`](Self::egress_events) for
    /// anything that reaches a network.
    pub fn events(&mut self) -> Result<Vec<FieldEvent>, ScanError> {
        self.adapter
            .collect_events()
            .map_err(|e| ScanError::Parse(e.to_string()))
    }

    /// Only the events RuView will put on the wire.
    ///
    /// Runs the same [`network_egress_allowed`] gate the CSI path runs, with
    /// `identity_bound: false` — an ultrasonic scan has no identity to bind, and
    /// hard-coding it rather than plumbing a flag means there is no argument a
    /// caller can pass that turns the P4/P5 consent exception on. There is no
    /// consent story for a room scan.
    ///
    /// In the coarse mode configured by [`load`](Self::load) this should drop
    /// nothing, and `coarse_scan_passes_the_egress_gate_intact` asserts it. The
    /// gate stays because a policy that is only enforced where it never fires
    /// is a policy nobody notices removing.
    pub fn egress_events(&mut self) -> Result<Vec<FieldEvent>, ScanError> {
        Ok(self
            .events()?
            .into_iter()
            .filter(|e| network_egress_allowed(e.observation.privacy_class, false))
            .filter(|e| network_egress_allowed(e.tensor.privacy_class, false))
            .collect())
    }
}

/// The privacy class every event from [`UltrasonicScan`] carries.
///
/// `P1` — a derived non-identity feature. Stated as a constant so a test can
/// pin it: if the adapter's default output mode ever changes upstream, that is
/// a submodule bump which silently moves data from edge-local to network-eligible
/// or back, and it should fail a build here rather than change behaviour in a
/// deployment.
pub const ULTRASONIC_EGRESS_CLASS: PrivacyClass = PrivacyClass::P1;
