//! Cost descriptors and the deployment cost policy (ADR-314 §1).
//!
//! **SYNTHETIC / L0 scaffold (ADR-282).** Every quantity here is a *modelled*
//! resource figure supplied by the caller (in a fielded system, read from the
//! ADR-320 HAL descriptors); nothing in this module measures a device. No
//! `MEASURED` energy/latency/throughput claim is made or implied — a scheduler
//! predicts where budget is best spent, it does not observe hardware.

use serde::{Deserialize, Serialize};

/// Numerical floor for the weighted-cost denominator so a zero-cost (or nearly
/// free) action never produces a non-finite value density. It does not model a
/// physical minimum; it only keeps the division bounded and deterministic.
pub(crate) const MIN_WEIGHTED_COST: f64 = 1e-9;

/// The three scarce edge resources one sensor action is modelled to consume.
///
/// These are the ADR-314 denominator terms — compute, energy, and bandwidth —
/// the three resources ADR-314 names as scarce on the ESP32-class nodes and
/// small gateways RuView targets. Values are unitless modelled magnitudes; the
/// caller supplies them (from ADR-320 HAL descriptors in a fielded system).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Modelled compute cost of running the action (unitless magnitude).
    pub compute: f64,
    /// Modelled energy cost of running the action (unitless magnitude).
    pub energy: f64,
    /// Modelled bandwidth cost of shipping the result (unitless magnitude).
    pub bandwidth: f64,
}

impl Cost {
    /// The zero-cost triple (identity for accumulation).
    pub const ZERO: Cost = Cost {
        compute: 0.0,
        energy: 0.0,
        bandwidth: 0.0,
    };

    /// Construct a cost triple.
    #[must_use]
    pub const fn new(compute: f64, energy: f64, bandwidth: f64) -> Self {
        Self {
            compute,
            energy,
            bandwidth,
        }
    }

    /// True when every component is finite and non-negative. A malformed cost
    /// (NaN/∞/negative) is not silently coerced to a number; the scheduler
    /// defers such a candidate as UNKNOWN-cost rather than guessing (ADR-300
    /// rule 1).
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        [self.compute, self.energy, self.bandwidth]
            .iter()
            .all(|c| c.is_finite() && *c >= 0.0)
    }

    /// Component-wise sum, used to accumulate the spent budget.
    #[must_use]
    pub(crate) fn plus(&self, other: &Cost) -> Cost {
        Cost {
            compute: self.compute + other.compute,
            energy: self.energy + other.energy,
            bandwidth: self.bandwidth + other.bandwidth,
        }
    }
}

/// The deployment cost policy: how the three cost terms are weighted into one
/// scalar denominator (ADR-314 §1).
///
/// The weighting is a *deployment* choice, not a hardcoded constant: a battery
/// node weights energy heavily, a wired gateway weights bandwidth. The policy
/// is configured, never assumed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostPolicy {
    /// Weight applied to the compute term.
    pub w_compute: f64,
    /// Weight applied to the energy term.
    pub w_energy: f64,
    /// Weight applied to the bandwidth term.
    pub w_bandwidth: f64,
}

impl CostPolicy {
    /// Equal weight on all three resources.
    pub const UNIFORM: CostPolicy = CostPolicy {
        w_compute: 1.0,
        w_energy: 1.0,
        w_bandwidth: 1.0,
    };

    /// Construct a policy, clamping any non-finite or negative weight to `0.0`
    /// at the boundary so a malformed weight can never poison the ranking.
    #[must_use]
    pub fn new(w_compute: f64, w_energy: f64, w_bandwidth: f64) -> Self {
        Self {
            w_compute: sanitize_weight(w_compute),
            w_energy: sanitize_weight(w_energy),
            w_bandwidth: sanitize_weight(w_bandwidth),
        }
    }

    /// Collapse a well-formed cost triple into the single scalar denominator
    /// used by the value function, floored at [`MIN_WEIGHTED_COST`] so the
    /// division is always finite. Callers must only pass a
    /// [`Cost::is_well_formed`] triple.
    #[must_use]
    pub fn scalar_cost(&self, cost: &Cost) -> f64 {
        let weighted =
            self.w_compute * cost.compute + self.w_energy * cost.energy + self.w_bandwidth * cost.bandwidth;
        weighted.max(MIN_WEIGHTED_COST)
    }
}

impl Default for CostPolicy {
    fn default() -> Self {
        Self::UNIFORM
    }
}

fn sanitize_weight(w: f64) -> f64 {
    if w.is_finite() && w >= 0.0 {
        w
    } else {
        0.0
    }
}
