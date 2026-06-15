//! The ADR-262 §3.3 privacy mapping — the critical correctness item.
//!
//! RuView's effective `PrivacyClass` (4 byte-level classes) is the source of
//! truth; the bridge maps it onto RuField's `PrivacyClass` (P0–P5) **at the
//! egress boundary, by information content, NEVER by byte value**.
//!
//! ## The trap (ADR-262 §3, §6)
//!
//! RuView's `Derived` has byte value `1`, which sorts *below* `Anonymous`
//! (byte `2`). A naive byte-mapping (`Derived = 1 → P1`) would leak
//! identity-bearing features (`identity_embedding`, `identity_risk_score`) as a
//! **low-privacy P1** event. Because `Derived` carries derived *identity*, it
//! must map to the **biometric/identity tier (P4/P5)** — never P1. This is the
//! single most dangerous mapping mistake; it gets a dedicated test
//! (`derived_identity_never_maps_to_low_privacy`).
//!
//! ## Fail-closed
//!
//! [`RuViewPrivacyClass`] is a closed enum, so there is no runtime "unknown"
//! value to receive — but the mapping is written `match`-exhaustively with an
//! explicit, documented arm per class, and the `demoted`/`identity_bound`
//! overlays only ever move the result **toward more privacy**, never less.

use crate::snapshot::RuViewPrivacyClass;
use rufield_core::PrivacyClass;

/// Map a RuView effective `PrivacyClass` onto a RuField `PrivacyClass`
/// (ADR-262 §3.3), by information content.
///
/// | RuView (byte) | → RuField | Rationale |
/// |---|---|---|
/// | `Raw` (0) | `P0` | raw CSI waveform |
/// | `Derived` (1) | `P4` (or `P5` if `identity_bound`) | derived **identity** features ⇒ biometric/identity tier, **not** P1 |
/// | `Anonymous` (2) | `P2` | occupancy / motion only |
/// | `Restricted` (3) | `P2` (raw suppressed) | matches `suppress_raw_outputs` |
///
/// `identity_bound` only promotes `Derived` (already identity-derived) from P4
/// to P5; it can never lower the class.
#[must_use]
pub fn map_privacy(ruview_class: RuViewPrivacyClass, identity_bound: bool) -> PrivacyClass {
    match ruview_class {
        // Raw CSI amplitude → raw waveform tier.
        RuViewPrivacyClass::Raw => PrivacyClass::P0,

        // THE CRITICAL ARM (§3.3 / §6): `Derived` carries identity. Map by
        // information content to the biometric/identity tier P4, and to P5 when
        // the surface is bound to a named identity. NEVER P1.
        RuViewPrivacyClass::Derived => {
            if identity_bound {
                PrivacyClass::P5
            } else {
                PrivacyClass::P4
            }
        }

        // Anonymous occupancy / motion aggregate → P2.
        RuViewPrivacyClass::Anonymous => PrivacyClass::P2,

        // Restricted: occupancy with risk score / hash stripped and raw
        // suppressed. Capped at P2 (occupancy tier), matching
        // `EngineBridge::suppress_raw_outputs` (`engine_bridge.rs:240`).
        RuViewPrivacyClass::Restricted => PrivacyClass::P2,
    }
}

/// The §4 P2 gate (b) monotonicity overlay: a governed-engine **demotion**
/// (`TrustedOutput.demoted == true`) must never let the emitted class fall
/// below P2 (occupancy floor), and raw is suppressed.
///
/// This is applied *after* [`map_privacy`] and can only raise the class
/// (toward more privacy) — it is fail-closed by construction.
#[must_use]
pub fn apply_demotion_floor(class: PrivacyClass, demoted: bool) -> PrivacyClass {
    if demoted && class < PrivacyClass::P2 {
        PrivacyClass::P2
    } else {
        class
    }
}

/// The full egress class for a snapshot: information-content mapping with the
/// demotion floor overlaid. This is what the bridge stamps on the emitted
/// `FieldEvent`.
#[must_use]
pub fn egress_class(
    ruview_class: RuViewPrivacyClass,
    identity_bound: bool,
    demoted: bool,
) -> PrivacyClass {
    apply_demotion_floor(map_privacy(ruview_class, identity_bound), demoted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_maps_to_identity_tier_not_p1() {
        // The single most dangerous mapping mistake: Derived (byte 1) must NOT
        // become P1. It carries identity ⇒ P4, or P5 if identity-bound.
        assert_eq!(map_privacy(RuViewPrivacyClass::Derived, false), PrivacyClass::P4);
        assert_eq!(map_privacy(RuViewPrivacyClass::Derived, true), PrivacyClass::P5);
    }

    #[test]
    fn full_table_matches_adr_262_section_3_3() {
        assert_eq!(map_privacy(RuViewPrivacyClass::Raw, false), PrivacyClass::P0);
        assert_eq!(map_privacy(RuViewPrivacyClass::Derived, false), PrivacyClass::P4);
        assert_eq!(map_privacy(RuViewPrivacyClass::Anonymous, false), PrivacyClass::P2);
        assert_eq!(map_privacy(RuViewPrivacyClass::Restricted, false), PrivacyClass::P2);
    }

    #[test]
    fn mapping_ignores_non_monotonic_byte_value() {
        // Derived's byte (1) is *below* Anonymous's byte (2), but Derived's
        // mapped class must be *above* Anonymous's mapped class — proving the
        // mapping uses information content, not the byte.
        assert!(RuViewPrivacyClass::Derived.raw_byte() < RuViewPrivacyClass::Anonymous.raw_byte());
        assert!(
            map_privacy(RuViewPrivacyClass::Derived, false)
                > map_privacy(RuViewPrivacyClass::Anonymous, false)
        );
    }

    #[test]
    fn demotion_floor_only_raises_privacy() {
        // Raw → P0, but a demoted cycle floors to P2 with raw suppressed.
        assert_eq!(apply_demotion_floor(PrivacyClass::P0, true), PrivacyClass::P2);
        // Already-high classes are never lowered by the floor.
        assert_eq!(apply_demotion_floor(PrivacyClass::P5, true), PrivacyClass::P5);
        // No demotion ⇒ unchanged.
        assert_eq!(apply_demotion_floor(PrivacyClass::P0, false), PrivacyClass::P0);
    }

    #[test]
    fn identity_bound_only_promotes() {
        // identity_bound never lowers privacy; it only promotes Derived P4→P5.
        for c in [
            RuViewPrivacyClass::Raw,
            RuViewPrivacyClass::Derived,
            RuViewPrivacyClass::Anonymous,
            RuViewPrivacyClass::Restricted,
        ] {
            assert!(map_privacy(c, true) >= map_privacy(c, false));
        }
    }
}
