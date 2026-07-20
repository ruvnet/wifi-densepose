//! ADR-270 vendor RF provider contract.
//!
//! This model prevents RSSI, cloud occupancy, or network inventory from being
//! represented as complex CSI. Vendor adapters may emit only declared capabilities.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const MAX_VENDOR_METRICS: usize = 64;
pub const MAX_VENDOR_KEY_LEN: usize = 64;
pub const MAX_VENDOR_TEXT_LEN: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VendorId {
    OriginAi,
    Plume,
    Mist,
    Netgear,
    ElectricImp,
    RfSolutions,
    Linksys,
    Luma,
    GoogleNest,
    Wifigarden,
}

impl VendorId {
    pub const ALL: [Self; 10] = [
        Self::OriginAi,
        Self::Plume,
        Self::Mist,
        Self::Netgear,
        Self::ElectricImp,
        Self::RfSolutions,
        Self::Linksys,
        Self::Luma,
        Self::GoogleNest,
        Self::Wifigarden,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::OriginAi => "origin_ai",
            Self::Plume => "plume",
            Self::Mist => "mist",
            Self::Netgear => "netgear",
            Self::ElectricImp => "electric_imp",
            Self::RfSolutions => "rf_solutions",
            Self::Linksys => "linksys",
            Self::Luma => "luma",
            Self::GoogleNest => "google_nest",
            Self::Wifigarden => "wifigarden",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RfCapability {
    ComplexCsi,
    DerivedSensing,
    RfTelemetry,
    NetworkOnly,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderAvailability {
    Available,
    CredentialsRequired,
    ContractRequired,
    Experimental,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderDescriptor {
    pub vendor: VendorId,
    pub capabilities: Vec<RfCapability>,
    pub availability: ProviderAvailability,
    pub hardware_validated: bool,
    pub reason: String,
}

impl ProviderDescriptor {
    pub fn validate(&self) -> Result<(), VendorEventError> {
        if self.capabilities.is_empty()
            || self.reason.is_empty()
            || self.reason.len() > MAX_VENDOR_TEXT_LEN
        {
            return Err(VendorEventError::InvalidDescriptor);
        }
        if self.hardware_validated && self.availability != ProviderAvailability::Available {
            return Err(VendorEventError::InvalidDescriptor);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VendorRfEvent {
    pub vendor: VendorId,
    pub capability: RfCapability,
    pub sequence: u64,
    pub timestamp_us: u64,
    pub source_id: String,
    pub synthetic: bool,
    pub metrics: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl VendorRfEvent {
    pub fn validate(&self, descriptor: &ProviderDescriptor) -> Result<(), VendorEventError> {
        descriptor.validate()?;
        if self.vendor != descriptor.vendor || !descriptor.capabilities.contains(&self.capability) {
            return Err(VendorEventError::CapabilityMismatch);
        }
        if matches!(
            self.capability,
            RfCapability::ComplexCsi | RfCapability::Unsupported
        ) {
            return Err(VendorEventError::InvalidEventCapability);
        }
        if self.source_id.is_empty()
            || self.source_id.len() > MAX_VENDOR_TEXT_LEN
            || self.metrics.is_empty()
            || self.metrics.len() > MAX_VENDOR_METRICS
            || self
                .metrics
                .iter()
                .any(|(k, v)| k.is_empty() || k.len() > MAX_VENDOR_KEY_LEN || !v.is_finite())
            || self
                .label
                .as_ref()
                .is_some_and(|v| v.len() > MAX_VENDOR_TEXT_LEN)
        {
            return Err(VendorEventError::InvalidPayload);
        }
        Ok(())
    }
}

pub trait VendorRfProvider: Send + Sync {
    fn descriptor(&self) -> ProviderDescriptor;
    fn decode(&self, payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum VendorEventError {
    #[error("invalid provider descriptor")]
    InvalidDescriptor,
    #[error("event capability does not match provider")]
    CapabilityMismatch,
    #[error("complex CSI and unsupported states cannot be represented as scalar vendor events")]
    InvalidEventCapability,
    #[error("invalid or unbounded vendor payload")]
    InvalidPayload,
    #[error("malformed provider payload: {0}")]
    MalformedPayload(String),
    #[error("provider credentials are required")]
    CredentialsRequired,
    #[error("commercial contract or SDK access is required")]
    ContractRequired,
    #[error("provider has no supported sensing interface")]
    Unsupported,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_vendor_has_a_stable_identifier() {
        let names: std::collections::BTreeSet<_> =
            VendorId::ALL.iter().map(|v| v.as_str()).collect();
        assert_eq!(names.len(), VendorId::ALL.len());
    }

    #[test]
    fn scalar_contract_rejects_csi_masquerading() {
        let descriptor = ProviderDescriptor {
            vendor: VendorId::Plume,
            capabilities: vec![RfCapability::RfTelemetry],
            availability: ProviderAvailability::CredentialsRequired,
            hardware_validated: false,
            reason: "OpenSync telemetry".into(),
        };
        let event = VendorRfEvent {
            vendor: VendorId::Plume,
            capability: RfCapability::ComplexCsi,
            sequence: 1,
            timestamp_us: 1,
            source_id: "pod-1".into(),
            synthetic: true,
            metrics: BTreeMap::from([("rssi_dbm".into(), -42.0)]),
            label: None,
        };
        assert_eq!(
            event.validate(&descriptor),
            Err(VendorEventError::CapabilityMismatch)
        );
    }
}
