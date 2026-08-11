//! The forward model: expected distribution per link (ADR-315 §1).
//!
//! **SYNTHETIC / L0.** This module is a *simulation* of an observable, not a
//! measurement. It implements a deliberately simple, documented log-distance
//! path-loss model with optional wall attenuation — clearly a didactic model,
//! not real RF. Nothing here is a hardware, `MEASURED`, or accuracy claim.
//!
//! An [`ExpectedDistribution`] is the mean/variance of a modelled observable
//! under the twin. Consistent with ADR-300 rule 1, insufficient information is
//! reported as [`ExpectedDistribution::Unknown`] — a first-class value, never an
//! error or a confident default.

use serde::{Deserialize, Serialize};

use crate::twin::{LinkId, PropagationParams, RfTwin, Wall};

/// The modelled observable a distribution describes. Kept as an enum so the twin
/// can grow phenomena without changing the delta contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Observable {
    /// Modelled received signal strength, dBm (SYNTHETIC).
    Rssi,
}

/// Why a link's expected distribution is unknown. UNKNOWN is a first-class
/// output (ADR-300 rule 1), not an error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownReason {
    /// One or both endpoints are not present in the twin.
    MissingNode,
    /// The two endpoints are effectively coincident, so path loss is undefined.
    ZeroDistance,
    /// The endpoints are the same node.
    SelfLink,
    /// The modelled computation produced a non-finite value.
    NonFinite,
}

/// The predicted distribution of an observable over a link under the twin.
///
/// **SYNTHETIC / L0.** A model-relative statement, never evidence of a physical
/// state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ExpectedDistribution {
    /// A modelled mean and (non-negative) variance for `observable`.
    Known {
        /// Which observable this describes.
        observable: Observable,
        /// Modelled mean, in the observable's units (dBm for RSSI).
        mean: f64,
        /// Modelled variance, in the observable's units squared (dB²).
        variance: f64,
    },
    /// The twin cannot predict this link; carries a first-class reason.
    Unknown {
        /// Why it is unknown.
        reason: UnknownReason,
    },
}

impl ExpectedDistribution {
    /// Borrow `(mean, variance)` when known.
    #[must_use]
    pub fn known(&self) -> Option<(f64, f64)> {
        match self {
            ExpectedDistribution::Known { mean, variance, .. } => Some((*mean, *variance)),
            ExpectedDistribution::Unknown { .. } => None,
        }
    }
}

/// Below this distance (metres) two radios are treated as coincident and the
/// path loss is left [`UnknownReason::ZeroDistance`] rather than diverging.
const MIN_DISTANCE_M: f64 = 1e-6;

/// SYNTHETIC log-distance path loss, decibels:
/// `PL(d) = PL(d0) + 10·n·log10(d / d0) + Σ wall_attenuation`.
///
/// Returns `None` if the result is non-finite. `d` must already be `>=`
/// [`MIN_DISTANCE_M`].
#[must_use]
pub fn path_loss_db(params: &PropagationParams, distance_m: f64, wall_attenuation_db: f64) -> Option<f64> {
    let pl = params.reference_loss_db
        + 10.0 * params.path_loss_exponent * (distance_m / params.reference_distance_m).log10()
        + wall_attenuation_db;
    pl.is_finite().then_some(pl)
}

/// Total attenuation (dB) added by every wall whose floor-plan segment the
/// link's straight path crosses. Deterministic; walls are visited in order.
#[must_use]
pub fn wall_attenuation_db(walls: &[Wall], a_xy: (f64, f64), b_xy: (f64, f64)) -> f64 {
    let mut sum = 0.0;
    for wall in walls {
        if segments_intersect(a_xy, b_xy, wall.a, wall.b) {
            sum += wall.attenuation_db;
        }
    }
    sum
}

/// Predict the expected distribution for one link under the twin.
///
/// **SYNTHETIC / L0.** The modelled transmitter is the link's canonical `a`
/// endpoint (lower id); the mean is `tx_power - PL(d)` and the variance is the
/// base shadowing variance plus any recorded multipath variance.
#[must_use]
pub fn predict_link(twin: &RfTwin, link: &LinkId) -> ExpectedDistribution {
    if link.is_self_link() {
        return ExpectedDistribution::Unknown {
            reason: UnknownReason::SelfLink,
        };
    }
    let (tx, rx) = match (twin.node(&link.a), twin.node(&link.b)) {
        (Some(tx), Some(rx)) => (tx, rx),
        _ => {
            return ExpectedDistribution::Unknown {
                reason: UnknownReason::MissingNode,
            }
        }
    };

    let distance = tx.position.distance_to(&rx.position);
    if distance < MIN_DISTANCE_M {
        return ExpectedDistribution::Unknown {
            reason: UnknownReason::ZeroDistance,
        };
    }

    let wall_att = wall_attenuation_db(&twin.walls, tx.position.xy(), rx.position.xy());
    let pl = match path_loss_db(&twin.params, distance, wall_att) {
        Some(pl) => pl,
        None => {
            return ExpectedDistribution::Unknown {
                reason: UnknownReason::NonFinite,
            }
        }
    };

    let mean = tx.tx_power_dbm - pl;
    let base_var = twin.params.shadowing_sigma_db * twin.params.shadowing_sigma_db;
    let variance = base_var + twin.extra_variance(link);

    if !(mean.is_finite() && variance.is_finite()) {
        return ExpectedDistribution::Unknown {
            reason: UnknownReason::NonFinite,
        };
    }

    ExpectedDistribution::Known {
        observable: Observable::Rssi,
        mean,
        variance,
    }
}

/// Predict every link in the twin, paired with its distribution. Deterministic
/// ordering (matches [`RfTwin::links`](crate::twin::RfTwin::links)).
#[must_use]
pub fn predict_all(twin: &RfTwin) -> Vec<(LinkId, ExpectedDistribution)> {
    twin.links()
        .into_iter()
        .map(|link| {
            let dist = predict_link(twin, &link);
            (link, dist)
        })
        .collect()
}

/// Robust 2D segment-intersection test used for wall crossing. Pure integer-free
/// geometry with an orientation sign; deterministic and allocation-free.
fn segments_intersect(p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), p4: (f64, f64)) -> bool {
    let d1 = orientation(p3, p4, p1);
    let d2 = orientation(p3, p4, p2);
    let d3 = orientation(p1, p2, p3);
    let d4 = orientation(p1, p2, p4);

    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }

    on_segment(p3, p4, p1, d1)
        || on_segment(p3, p4, p2, d2)
        || on_segment(p1, p2, p3, d3)
        || on_segment(p1, p2, p4, d4)
}

/// Signed area (twice) of triangle `(a, b, c)`: `>0` left turn, `<0` right turn.
fn orientation(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// True when collinear point `c` (orientation `d == 0`) lies on segment `a-b`.
fn on_segment(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: f64) -> bool {
    d == 0.0
        && c.0 >= a.0.min(b.0)
        && c.0 <= a.0.max(b.0)
        && c.1 >= a.1.min(b.1)
        && c.1 <= a.1.max(b.1)
}
