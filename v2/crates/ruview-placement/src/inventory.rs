//! Hardware inventory: the radios available to place (ADR-308 §1).
//!
//! **SYNTHETIC / L0.** A coarse description of available hardware — each entry is
//! one physical radio the installer can place, with a modelled transmit power and
//! a capability-envelope label. It bounds the *count* of nodes the optimizer may
//! recommend; the scene bounds their geometry. No measurement claim.

use serde::{Deserialize, Serialize};

use crate::geometry::PlacementError;

/// Upper bound on radios accepted in one inventory. Bounds allocation and the
/// search space on untrusted input.
pub const MAX_INVENTORY: usize = 64;

/// One available radio. The `model` is a coarse capability-envelope label
/// (e.g. `"esp32-s3"`, `"mmwave"`); this crate does not interpret it beyond
/// carrying it through so a recommendation names the hardware it plans for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RadioSpec {
    /// Coarse hardware/capability label (ADR-318/ADR-320 descriptor handle).
    pub model: String,
    /// Modelled transmit power, dBm. A SYNTHETIC parameter of the forward model.
    pub tx_power_dbm: f64,
}

impl RadioSpec {
    /// Construct a radio spec.
    #[must_use]
    pub fn new(model: impl Into<String>, tx_power_dbm: f64) -> Self {
        Self { model: model.into(), tx_power_dbm }
    }
}

/// The set of radios available to place. Its length bounds the recommended node
/// count.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Inventory {
    /// Available radios, one entry per placeable unit.
    pub radios: Vec<RadioSpec>,
}

impl Inventory {
    /// An empty inventory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A homogeneous inventory of `count` radios of one `model`/power. Deterministic.
    #[must_use]
    pub fn homogeneous(model: impl Into<String>, tx_power_dbm: f64, count: usize) -> Self {
        let model = model.into();
        let radios = (0..count)
            .map(|_| RadioSpec::new(model.clone(), tx_power_dbm))
            .collect();
        Self { radios }
    }

    /// Number of placeable radios.
    #[must_use]
    pub fn len(&self) -> usize {
        self.radios.len()
    }

    /// True when there is nothing to place.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.radios.is_empty()
    }

    /// Validate the inventory at the boundary: bounded count, finite powers.
    pub fn validate(&self) -> Result<(), PlacementError> {
        if self.radios.len() > MAX_INVENTORY {
            return Err(PlacementError::TooManyRadios {
                len: self.radios.len(),
                max: MAX_INVENTORY,
            });
        }
        for r in &self.radios {
            if !r.tx_power_dbm.is_finite() {
                return Err(PlacementError::NonFinite { what: "tx_power_dbm" });
            }
        }
        Ok(())
    }
}
