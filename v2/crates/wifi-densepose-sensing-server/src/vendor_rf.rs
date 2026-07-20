//! ADR-270 provider registry and canonical event helpers.

use serde::Serialize;
use wifi_densepose_hardware::vendor_rf::{
    ProviderDescriptor, VendorEventError, VendorId, VendorRfEvent, VendorRfProvider,
};

use crate::vendor_mist_netgear::{MistProvider, NetgearInsightProvider};
use crate::vendor_origin_plume::{OriginAiProvider, PlumeOpenSyncProvider};
use crate::vendor_remaining::{
    ElectricImpProvider, GoogleNestProvider, LinksysProvider, LumaOpenWrtProvider,
    RfSolutionsProvider, WifigardenProvider,
};

pub fn descriptor_for(vendor: VendorId) -> ProviderDescriptor {
    match vendor {
        VendorId::OriginAi => OriginAiProvider.descriptor(),
        VendorId::Plume => PlumeOpenSyncProvider.descriptor(),
        VendorId::Mist => MistProvider.descriptor(),
        VendorId::Netgear => NetgearInsightProvider.descriptor(),
        VendorId::ElectricImp => ElectricImpProvider.descriptor(),
        VendorId::RfSolutions => RfSolutionsProvider.descriptor(),
        VendorId::Linksys => LinksysProvider.descriptor(),
        VendorId::Luma => LumaOpenWrtProvider.descriptor(),
        VendorId::GoogleNest => GoogleNestProvider.descriptor(),
        VendorId::Wifigarden => WifigardenProvider.descriptor(),
    }
}

pub fn descriptors() -> Vec<ProviderDescriptor> {
    VendorId::ALL.into_iter().map(descriptor_for).collect()
}

pub fn vendor_from_str(value: &str) -> Option<VendorId> {
    VendorId::ALL
        .into_iter()
        .find(|vendor| vendor.as_str() == value)
}

pub fn decode_provider(
    vendor: VendorId,
    payload: &[u8],
) -> Result<Vec<VendorRfEvent>, VendorEventError> {
    match vendor {
        VendorId::OriginAi => OriginAiProvider.decode(payload),
        VendorId::Plume => PlumeOpenSyncProvider.decode(payload),
        VendorId::Mist => MistProvider.decode(payload),
        VendorId::Netgear => NetgearInsightProvider.decode(payload),
        VendorId::ElectricImp => ElectricImpProvider.decode(payload),
        VendorId::RfSolutions => RfSolutionsProvider.decode(payload),
        VendorId::Linksys => LinksysProvider.decode(payload),
        VendorId::Luma => LumaOpenWrtProvider.decode(payload),
        VendorId::GoogleNest => GoogleNestProvider.decode(payload),
        VendorId::Wifigarden => WifigardenProvider.decode(payload),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VendorEventSnapshot {
    pub event_type: &'static str,
    pub source: String,
    #[serde(flatten)]
    pub event: VendorRfEvent,
}

impl VendorEventSnapshot {
    pub fn from_event(event: VendorRfEvent) -> Result<Self, VendorEventError> {
        let descriptor = descriptor_for(event.vendor);
        event.validate(&descriptor)?;
        let provenance = if event.synthetic { "simulated" } else { "live" };
        Ok(Self {
            event_type: "vendor_rf",
            source: format!("vendor:{}:{provenance}", event.vendor.as_str()),
            event,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_exactly_one_valid_descriptor_per_vendor() {
        let values = descriptors();
        assert_eq!(values.len(), VendorId::ALL.len());
        for (vendor, descriptor) in VendorId::ALL.into_iter().zip(values) {
            assert_eq!(descriptor.vendor, vendor);
            descriptor.validate().unwrap();
        }
    }

    #[test]
    fn unsupported_providers_fail_closed() {
        assert_eq!(
            decode_provider(VendorId::Linksys, b"{}"),
            Err(VendorEventError::Unsupported)
        );
        assert_eq!(
            decode_provider(VendorId::Wifigarden, b"{}"),
            Err(VendorEventError::ContractRequired)
        );
    }
}
