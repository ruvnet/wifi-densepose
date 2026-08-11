//! ADR-292 — canonical source-provenance state machine.
//!
//! Source state used to be a boolean (`live` vs. not), so any ambiguous
//! condition — an unauthenticated status-endpoint error, a simulator that has
//! not yet produced a frame — collapsed to "live". Two release-path defects
//! (issues #1526, #1557) both trace to that collapse.
//!
//! This module defines one canonical, mutually-exclusive [`SourceState`] and a
//! **pure** transition function of `(last_frame_age, auth_status, source_kind)`.
//! It touches no clock and no socket, so every rule below is unit-testable:
//!
//! - `Unknown` is not a state. An ambiguous condition resolves to
//!   `LiveUnverified`, `Stale`, or `Disconnected` — never `LiveVerified`.
//! - A status-endpoint error resolves to `Disconnected`/`LiveUnverified`,
//!   never live-verified (issue #1526).
//! - A `Synthetic` source can never transition to any `Live*` state without
//!   being reconstructed as a live source *and* presenting a verified frame
//!   (issue #1557).
//! - `Synthetic` is watermarked in every export ([`SourceState::export_watermark`]).

use std::time::Duration;

/// Where a source's data originates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// Generated data — the simulator, or replay of synthetic fixtures.
    Synthetic,
    /// A real external capture source (hardware or a network vendor feed).
    Live,
}

/// Result of authenticating / attesting the source, e.g. from a status-endpoint
/// probe. Deliberately distinguishes "not confirmed yet" from "the probe itself
/// errored" so an authorization failure can never masquerade as verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthStatus {
    /// Source authenticated and attested.
    Verified,
    /// Reachable, but provenance is not yet confirmed.
    Unverified,
    /// The auth / status probe itself failed (401/403, unreachable, malformed).
    Error,
    /// No information available yet.
    Unknown,
}

/// Canonical, mutually-exclusive source state (ADR-292). There is intentionally
/// no `Unknown` / `Live` boolean — every ambiguous input maps to one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceState {
    /// Generated data. Always watermarked; never presented as real.
    Synthetic,
    /// Fresh frames from an authenticated, attested source.
    LiveVerified,
    /// Fresh frames arriving, but provenance not yet confirmed.
    LiveUnverified,
    /// Last frame is older than the freshness window.
    Stale,
    /// No source / no frame.
    Disconnected,
}

impl SourceState {
    /// Pure transition. Resolves the canonical state from the only three inputs
    /// that may influence it — no clock, no socket.
    ///
    /// * `last_frame_age` — age of the most recent frame; `None` when no frame
    ///   has ever been observed.
    /// * `auth` — outcome of the most recent auth/attestation probe.
    /// * `kind` — whether the source is synthetic or a live capture.
    /// * `freshness_window` — maximum age a frame may have and still count as
    ///   fresh.
    pub fn resolve(
        last_frame_age: Option<Duration>,
        auth: AuthStatus,
        kind: SourceKind,
        freshness_window: Duration,
    ) -> SourceState {
        // A synthetic source is *always* synthetic. It can only become live by
        // being reconstructed as `SourceKind::Live` presenting a verified frame,
        // never by a transition here (ADR-292, issue #1557).
        if kind == SourceKind::Synthetic {
            return SourceState::Synthetic;
        }

        match last_frame_age {
            // No frame ever ⇒ no source. An auth error with no frame is
            // Disconnected, never live (issue #1526).
            None => SourceState::Disconnected,
            // A frame exists but is older than the freshness window.
            Some(age) if age > freshness_window => SourceState::Stale,
            // A fresh frame is present. Only an explicitly `Verified` auth
            // status may promote to `LiveVerified`; `Unknown` / `Error` /
            // `Unverified` can never be verified-live (issue #1526).
            Some(_) => match auth {
                AuthStatus::Verified => SourceState::LiveVerified,
                AuthStatus::Unverified | AuthStatus::Error | AuthStatus::Unknown => {
                    SourceState::LiveUnverified
                }
            },
        }
    }

    /// Compatibility accessor for callers migrating off a boolean source flag.
    /// True only for the two states that represent real, arriving frames.
    pub fn is_live(self) -> bool {
        matches!(self, SourceState::LiveVerified | SourceState::LiveUnverified)
    }

    /// True when this state must be watermarked as synthetic in every view and
    /// export (ADR-292).
    pub fn is_synthetic(self) -> bool {
        matches!(self, SourceState::Synthetic)
    }

    /// Short, stable, machine-readable label for wire/JSON surfaces.
    pub fn as_str(self) -> &'static str {
        match self {
            SourceState::Synthetic => "synthetic",
            SourceState::LiveVerified => "live_verified",
            SourceState::LiveUnverified => "live_unverified",
            SourceState::Stale => "stale",
            SourceState::Disconnected => "disconnected",
        }
    }

    /// Watermark that must accompany a synthetic export so generated data can
    /// never be mistaken for a real capture. `None` for non-synthetic states.
    pub fn export_watermark(self) -> Option<&'static str> {
        if self.is_synthetic() {
            Some("SYNTHETIC")
        } else {
            None
        }
    }

    /// Adapter that maps this server's free-form `source` label plus freshness
    /// into a canonical [`SourceState`], so the existing string surface can
    /// report the honest enum instead of a boolean collapse.
    ///
    /// Hardware / vendor feeds resolve to `LiveUnverified` — frames arrive but
    /// provenance is not attested on this path — never `LiveVerified`. An
    /// `":offline"`-suffixed label (issue #1004) resolves to `Disconnected`.
    pub fn from_source_label(
        source: &str,
        last_frame_age: Option<Duration>,
        freshness_window: Duration,
    ) -> SourceState {
        if source.ends_with(":offline") {
            return SourceState::Disconnected;
        }
        let kind = match source {
            "simulated" | "simulate" | "synthetic" | "test" => SourceKind::Synthetic,
            _ => SourceKind::Live,
        };
        // This path carries no attestation, so the best a live feed earns is
        // `Unverified` — the honest label. `resolve` still short-circuits a
        // synthetic kind to `Synthetic` regardless of frame age.
        SourceState::resolve(last_frame_age, AuthStatus::Unverified, kind, freshness_window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Duration = Duration::from_secs(5);
    fn fresh() -> Option<Duration> {
        Some(Duration::from_millis(100))
    }
    fn old() -> Option<Duration> {
        Some(WINDOW + Duration::from_secs(1))
    }

    // ── Core honesty rules ───────────────────────────────────────────────

    #[test]
    fn verified_fresh_live_source_is_live_verified() {
        assert_eq!(
            SourceState::resolve(fresh(), AuthStatus::Verified, SourceKind::Live, WINDOW),
            SourceState::LiveVerified
        );
    }

    #[test]
    fn auth_error_never_resolves_to_live_verified() {
        // Fresh frame but the status probe errored ⇒ unverified, not verified.
        assert_eq!(
            SourceState::resolve(fresh(), AuthStatus::Error, SourceKind::Live, WINDOW),
            SourceState::LiveUnverified
        );
        // No frame and an auth error ⇒ Disconnected, never live (issue #1526).
        assert_eq!(
            SourceState::resolve(None, AuthStatus::Error, SourceKind::Live, WINDOW),
            SourceState::Disconnected
        );
    }

    #[test]
    fn unknown_never_resolves_to_live_verified() {
        let s = SourceState::resolve(fresh(), AuthStatus::Unknown, SourceKind::Live, WINDOW);
        assert_eq!(s, SourceState::LiveUnverified);
        assert_ne!(s, SourceState::LiveVerified);
    }

    #[test]
    fn synthetic_never_becomes_live_even_with_verified_fresh_frame() {
        // Issue #1557: the simulator constructs Synthetic and cannot transition
        // to any Live* state without being reconstructed as a live source.
        assert_eq!(
            SourceState::resolve(fresh(), AuthStatus::Verified, SourceKind::Synthetic, WINDOW),
            SourceState::Synthetic
        );
        assert!(!SourceState::resolve(fresh(), AuthStatus::Verified, SourceKind::Synthetic, WINDOW)
            .is_live());
    }

    #[test]
    fn freshness_expiry_is_stale() {
        assert_eq!(
            SourceState::resolve(old(), AuthStatus::Verified, SourceKind::Live, WINDOW),
            SourceState::Stale
        );
    }

    #[test]
    fn no_frame_is_disconnected() {
        assert_eq!(
            SourceState::resolve(None, AuthStatus::Verified, SourceKind::Live, WINDOW),
            SourceState::Disconnected
        );
    }

    // ── Watermark / compat accessors ─────────────────────────────────────

    #[test]
    fn synthetic_export_is_watermarked() {
        assert_eq!(SourceState::Synthetic.export_watermark(), Some("SYNTHETIC"));
        assert!(SourceState::Synthetic.is_synthetic());
        assert_eq!(SourceState::LiveVerified.export_watermark(), None);
        assert_eq!(SourceState::Disconnected.export_watermark(), None);
    }

    #[test]
    fn is_live_only_for_live_states() {
        assert!(SourceState::LiveVerified.is_live());
        assert!(SourceState::LiveUnverified.is_live());
        assert!(!SourceState::Synthetic.is_live());
        assert!(!SourceState::Stale.is_live());
        assert!(!SourceState::Disconnected.is_live());
    }

    // ── Label adapter ────────────────────────────────────────────────────

    #[test]
    fn simulated_label_is_synthetic_regardless_of_frames() {
        // Even with fresh frames arriving, a simulated source stays synthetic.
        let s = SourceState::from_source_label("simulated", fresh(), WINDOW);
        assert_eq!(s, SourceState::Synthetic);
        assert!(!s.is_live());
        assert_eq!(SourceState::from_source_label("simulate", None, WINDOW), SourceState::Synthetic);
    }

    #[test]
    fn hardware_label_is_live_unverified_never_verified() {
        let s = SourceState::from_source_label("esp32", fresh(), WINDOW);
        assert_eq!(s, SourceState::LiveUnverified);
        assert_ne!(s, SourceState::LiveVerified);
    }

    #[test]
    fn offline_label_is_disconnected() {
        assert_eq!(
            SourceState::from_source_label("esp32:offline", fresh(), WINDOW),
            SourceState::Disconnected
        );
    }

    #[test]
    fn stale_hardware_label_is_stale() {
        assert_eq!(
            SourceState::from_source_label("wifi:home", old(), WINDOW),
            SourceState::Stale
        );
    }

    #[test]
    fn as_str_is_stable() {
        assert_eq!(SourceState::Synthetic.as_str(), "synthetic");
        assert_eq!(SourceState::LiveVerified.as_str(), "live_verified");
        assert_eq!(SourceState::LiveUnverified.as_str(), "live_unverified");
        assert_eq!(SourceState::Stale.as_str(), "stale");
        assert_eq!(SourceState::Disconnected.as_str(), "disconnected");
    }
}
