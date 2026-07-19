//! ADR-270 providers whose useful integration surface is scalar telemetry,
//! network-only metadata, or an explicit no-go decision.
//!
//! None of these providers emits complex CSI.  The small JSON contract in this
//! module is intended for sidecars and deterministic replay fixtures; it is not
//! a claim that a vendor exposes this exact wire format.

use serde::Deserialize;
use std::collections::BTreeMap;
use wifi_densepose_hardware::vendor_rf::{
    ProviderAvailability, ProviderDescriptor, RfCapability, VendorEventError, VendorId,
    VendorRfEvent, VendorRfProvider,
};

/// Upper bound applied before JSON decoding, limiting parser allocation.
pub const MAX_REMAINING_VENDOR_PAYLOAD_BYTES: usize = 64 * 1024;
/// Upper bound on events accepted in one sidecar envelope.
pub const MAX_REMAINING_VENDOR_EVENTS: usize = 256;

const ELECTRIC_IMP_METRICS: &[MetricRule] = &[
    MetricRule::new("battery_v", 0.0, 100.0, false),
    MetricRule::new("humidity_percent", 0.0, 100.0, false),
    MetricRule::new("rssi_dbm", -127.0, 0.0, false),
    MetricRule::new("temperature_c", -100.0, 200.0, false),
    MetricRule::new("voltage_v", 0.0, 1_000.0, false),
];
const RF_SOLUTIONS_METRICS: &[MetricRule] = &[
    MetricRule::new("battery_v", 0.0, 100.0, false),
    MetricRule::new("humidity_percent", 0.0, 100.0, false),
    MetricRule::new("relay_state", 0.0, 1.0, true),
    MetricRule::new("rssi_dbm", -127.0, 0.0, false),
    MetricRule::new("temperature_c", -100.0, 200.0, false),
];
const LUMA_METRICS: &[MetricRule] = &[
    MetricRule::new("client_count", 0.0, 1_000_000.0, true),
    MetricRule::new("noise_dbm", -127.0, 0.0, false),
    MetricRule::new("rssi_dbm", -127.0, 0.0, false),
    MetricRule::new("rx_bytes", 0.0, 9_007_199_254_740_991.0, true),
    MetricRule::new("tx_bytes", 0.0, 9_007_199_254_740_991.0, true),
];
const GOOGLE_NEST_METRICS: &[MetricRule] = &[
    MetricRule::new("client_count", 0.0, 1_000_000.0, true),
    MetricRule::new("probe_count", 0.0, 1_000_000_000.0, true),
    MetricRule::new("rx_bytes", 0.0, 9_007_199_254_740_991.0, true),
    MetricRule::new("tx_bytes", 0.0, 9_007_199_254_740_991.0, true),
];

#[derive(Debug, Clone, Copy)]
struct MetricRule {
    name: &'static str,
    minimum: f64,
    maximum: f64,
    integer: bool,
}

impl MetricRule {
    const fn new(name: &'static str, minimum: f64, maximum: f64, integer: bool) -> Self {
        Self {
            name,
            minimum,
            maximum,
            integer,
        }
    }

    fn accepts(self, value: f64) -> bool {
        value.is_finite()
            && (self.minimum..=self.maximum).contains(&value)
            && (!self.integer || value.fract() == 0.0)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScalarEnvelope {
    events: Vec<ScalarEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScalarEvent {
    sequence: u64,
    timestamp_us: u64,
    source_id: String,
    synthetic: bool,
    metrics: BTreeMap<String, f64>,
    #[serde(default)]
    label: Option<String>,
}

fn descriptor(
    vendor: VendorId,
    capability: RfCapability,
    availability: ProviderAvailability,
    reason: &str,
) -> ProviderDescriptor {
    ProviderDescriptor {
        vendor,
        capabilities: vec![capability],
        availability,
        hardware_validated: false,
        reason: reason.to_owned(),
    }
}

fn decode_bounded_scalar_events(
    payload: &[u8],
    provider: &ProviderDescriptor,
    capability: RfCapability,
    allowed_metrics: &[MetricRule],
) -> Result<Vec<VendorRfEvent>, VendorEventError> {
    provider.validate()?;
    if !provider.capabilities.contains(&capability)
        || matches!(
            capability,
            RfCapability::ComplexCsi | RfCapability::Unsupported
        )
    {
        return Err(VendorEventError::CapabilityMismatch);
    }
    if payload.is_empty() || payload.len() > MAX_REMAINING_VENDOR_PAYLOAD_BYTES {
        return Err(VendorEventError::InvalidPayload);
    }

    let envelope: ScalarEnvelope = serde_json::from_slice(payload).map_err(|error| {
        VendorEventError::MalformedPayload(error.to_string().chars().take(160).collect())
    })?;
    if envelope.events.is_empty() || envelope.events.len() > MAX_REMAINING_VENDOR_EVENTS {
        return Err(VendorEventError::InvalidPayload);
    }

    envelope
        .events
        .into_iter()
        .map(|event| {
            // An allowlist keeps arbitrary scalar fields from silently changing
            // the meaning of a provider contract (and rejects CSI-shaped data).
            if event.timestamp_us == 0
                || event.source_id.chars().any(char::is_control)
                || event
                    .label
                    .as_deref()
                    .is_some_and(|label| label.chars().any(char::is_control))
            {
                return Err(VendorEventError::InvalidPayload);
            }
            for (key, value) in &event.metrics {
                let rule = allowed_metrics
                    .iter()
                    .find(|rule| rule.name == key)
                    .ok_or(VendorEventError::CapabilityMismatch)?;
                if !rule.accepts(*value) {
                    return Err(VendorEventError::InvalidPayload);
                }
            }
            let event = VendorRfEvent {
                vendor: provider.vendor,
                capability,
                sequence: event.sequence,
                timestamp_us: event.timestamp_us,
                source_id: event.source_id,
                synthetic: event.synthetic,
                metrics: event.metrics,
                label: event.label,
            };
            event.validate(provider)?;
            Ok(event)
        })
        .collect()
}

/// Electric Imp agent/impCentral scalar telemetry bridge for existing fleets.
#[derive(Debug, Clone, Copy, Default)]
pub struct ElectricImpProvider;

impl VendorRfProvider for ElectricImpProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        descriptor(
            VendorId::ElectricImp,
            RfCapability::RfTelemetry,
            ProviderAvailability::CredentialsRequired,
            "Optional authenticated agent/impCentral scalar telemetry bridge; never CSI",
        )
    }

    fn decode(&self, payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        let descriptor = self.descriptor();
        decode_bounded_scalar_events(
            payload,
            &descriptor,
            RfCapability::RfTelemetry,
            ELECTRIC_IMP_METRICS,
        )
    }
}

/// RF Solutions environmental/RIoT scalar telemetry boundary.
#[derive(Debug, Clone, Copy, Default)]
pub struct RfSolutionsProvider;

impl VendorRfProvider for RfSolutionsProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        descriptor(
            VendorId::RfSolutions,
            RfCapability::RfTelemetry,
            ProviderAvailability::Experimental,
            "Optional non-Wi-Fi environmental telemetry fusion; excluded as a CSI source",
        )
    }

    fn decode(&self, payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        let descriptor = self.descriptor();
        decode_bounded_scalar_events(
            payload,
            &descriptor,
            RfCapability::RfTelemetry,
            RF_SOLUTIONS_METRICS,
        )
    }
}

/// Generic OpenWrt telemetry fixture for already-owned discontinued Luma units.
#[derive(Debug, Clone, Copy, Default)]
pub struct LumaOpenWrtProvider;

impl VendorRfProvider for LumaOpenWrtProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        descriptor(
            VendorId::Luma,
            RfCapability::RfTelemetry,
            ProviderAvailability::Experimental,
            "Generic OpenWrt scalar telemetry fixture for already-owned Luma hardware; no Luma CSI claim",
        )
    }

    fn decode(&self, payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        let descriptor = self.descriptor();
        decode_bounded_scalar_events(
            payload,
            &descriptor,
            RfCapability::RfTelemetry,
            LUMA_METRICS,
        )
    }
}

/// Google Nest Wifi may participate as network infrastructure, not a sensor.
#[derive(Debug, Clone, Copy, Default)]
pub struct GoogleNestProvider;

impl VendorRfProvider for GoogleNestProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        descriptor(
            VendorId::GoogleNest,
            RfCapability::NetworkOnly,
            ProviderAvailability::Experimental,
            "Network-only replay events; Device Access exposes no router CSI or RF telemetry",
        )
    }

    fn decode(&self, payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        let descriptor = self.descriptor();
        decode_bounded_scalar_events(
            payload,
            &descriptor,
            RfCapability::NetworkOnly,
            GOOGLE_NEST_METRICS,
        )
    }
}

/// Linksys Aware reached end of support; there is no supported sensing API.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinksysProvider;

impl VendorRfProvider for LinksysProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        descriptor(
            VendorId::Linksys,
            RfCapability::Unsupported,
            ProviderAvailability::Unsupported,
            "Linksys Aware reached end of support in 2024; no supported sensing interface",
        )
    }

    fn decode(&self, _payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        Err(VendorEventError::Unsupported)
    }
}

/// Wifigarden remains gated until its commercial SDK contract is disclosed.
#[derive(Debug, Clone, Copy, Default)]
pub struct WifigardenProvider;

impl VendorRfProvider for WifigardenProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        descriptor(
            VendorId::Wifigarden,
            RfCapability::Unsupported,
            ProviderAvailability::ContractRequired,
            "Commercial SDK, chipset, schema, calibration and data-rights disclosure required",
        )
    }

    fn decode(&self, _payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        Err(VendorEventError::ContractRequired)
    }
}

/// Deterministic synthetic Electric Imp sidecar contract fixture.
pub const ELECTRIC_IMP_CONTRACT_FIXTURE: &[u8] = br#"{"events":[{"sequence":7,"timestamp_us":1700000000000000,"source_id":"imp005-fixture","synthetic":true,"metrics":{"rssi_dbm":-48.0,"temperature_c":21.5},"label":"lab"}]}"#;

/// Deterministic synthetic RF Solutions sidecar contract fixture.
pub const RF_SOLUTIONS_CONTRACT_FIXTURE: &[u8] = br#"{"events":[{"sequence":8,"timestamp_us":1700000000000100,"source_id":"riot-fixture","synthetic":true,"metrics":{"battery_v":3.1,"humidity_percent":44.0},"label":"lab"}]}"#;

/// Deterministic synthetic generic OpenWrt/Luma contract fixture.
pub const LUMA_OPENWRT_CONTRACT_FIXTURE: &[u8] = br#"{"events":[{"sequence":9,"timestamp_us":1700000000000200,"source_id":"luma-openwrt-fixture","synthetic":true,"metrics":{"client_count":3.0,"noise_dbm":-91.0,"rx_bytes":1024.0},"label":"openwrt"}]}"#;

/// Deterministic synthetic Google Nest network-only contract fixture.
pub const GOOGLE_NEST_CONTRACT_FIXTURE: &[u8] = br#"{"events":[{"sequence":10,"timestamp_us":1700000000000300,"source_id":"nest-fixture","synthetic":true,"metrics":{"client_count":4.0,"probe_count":2.0},"label":"network_activity"}]}"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_valid_fixture<P: VendorRfProvider>(
        provider: P,
        fixture: &[u8],
        vendor: VendorId,
        capability: RfCapability,
    ) {
        let descriptor = provider.descriptor();
        descriptor.validate().expect("descriptor must be valid");
        let first = provider
            .decode(fixture)
            .expect("fixture must decode deterministically");
        let second = provider.decode(fixture).expect("fixture must replay");
        assert_eq!(first, second);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].vendor, vendor);
        assert_eq!(first[0].capability, capability);
        assert!(first[0].synthetic);
        first[0]
            .validate(&descriptor)
            .expect("fixture event must satisfy provider contract");
    }

    #[test]
    fn scalar_and_network_fixtures_are_deterministic_and_honest() {
        assert_valid_fixture(
            ElectricImpProvider,
            ELECTRIC_IMP_CONTRACT_FIXTURE,
            VendorId::ElectricImp,
            RfCapability::RfTelemetry,
        );
        assert_valid_fixture(
            RfSolutionsProvider,
            RF_SOLUTIONS_CONTRACT_FIXTURE,
            VendorId::RfSolutions,
            RfCapability::RfTelemetry,
        );
        assert_valid_fixture(
            LumaOpenWrtProvider,
            LUMA_OPENWRT_CONTRACT_FIXTURE,
            VendorId::Luma,
            RfCapability::RfTelemetry,
        );
        assert_valid_fixture(
            GoogleNestProvider,
            GOOGLE_NEST_CONTRACT_FIXTURE,
            VendorId::GoogleNest,
            RfCapability::NetworkOnly,
        );
    }

    #[test]
    fn unavailable_providers_fail_before_interpreting_payloads() {
        let linksys = LinksysProvider;
        assert_eq!(
            linksys.descriptor().availability,
            ProviderAvailability::Unsupported
        );
        assert_eq!(
            linksys.decode(b"not json"),
            Err(VendorEventError::Unsupported)
        );

        let wifigarden = WifigardenProvider;
        assert_eq!(
            wifigarden.descriptor().availability,
            ProviderAvailability::ContractRequired
        );
        assert_eq!(
            wifigarden.decode(ELECTRIC_IMP_CONTRACT_FIXTURE),
            Err(VendorEventError::ContractRequired)
        );
    }

    #[test]
    fn rejects_empty_oversized_and_excess_event_payloads() {
        let provider = ElectricImpProvider;
        assert_eq!(provider.decode(b""), Err(VendorEventError::InvalidPayload));
        assert_eq!(
            provider.decode(&vec![b' '; MAX_REMAINING_VENDOR_PAYLOAD_BYTES + 1]),
            Err(VendorEventError::InvalidPayload)
        );

        let events = (0..=MAX_REMAINING_VENDOR_EVENTS)
            .map(|sequence| {
                format!(
                    r#"{{"sequence":{sequence},"timestamp_us":1,"source_id":"x","synthetic":true,"metrics":{{"rssi_dbm":-40.0}}}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let payload = format!(r#"{{"events":[{events}]}}"#);
        assert_eq!(
            provider.decode(payload.as_bytes()),
            Err(VendorEventError::InvalidPayload)
        );
    }

    #[test]
    fn rejects_csi_or_cross_provider_metric_masquerading() {
        let fake_csi = br#"{"events":[{"sequence":1,"timestamp_us":1,"source_id":"x","synthetic":true,"metrics":{"csi_real":1.0}}]}"#;
        assert_eq!(
            ElectricImpProvider.decode(fake_csi),
            Err(VendorEventError::CapabilityMismatch)
        );

        let rf_metric_in_network_event = br#"{"events":[{"sequence":1,"timestamp_us":1,"source_id":"x","synthetic":true,"metrics":{"rssi_dbm":-40.0}}]}"#;
        assert_eq!(
            GoogleNestProvider.decode(rf_metric_in_network_event),
            Err(VendorEventError::CapabilityMismatch)
        );
    }

    #[test]
    fn rejects_invalid_values_bounds_and_schema_extensions() {
        let non_finite = br#"{"events":[{"sequence":1,"timestamp_us":1,"source_id":"x","synthetic":true,"metrics":{"rssi_dbm":1e999}}]}"#;
        assert!(matches!(
            ElectricImpProvider.decode(non_finite),
            Err(VendorEventError::MalformedPayload(_)) | Err(VendorEventError::InvalidPayload)
        ));

        let empty_metrics = br#"{"events":[{"sequence":1,"timestamp_us":1,"source_id":"x","synthetic":true,"metrics":{}}]}"#;
        assert_eq!(
            ElectricImpProvider.decode(empty_metrics),
            Err(VendorEventError::InvalidPayload)
        );

        let unknown_field = br#"{"events":[{"sequence":1,"timestamp_us":1,"source_id":"x","synthetic":true,"metrics":{"rssi_dbm":-40.0},"csi":[]}]}"#;
        assert!(matches!(
            ElectricImpProvider.decode(unknown_field),
            Err(VendorEventError::MalformedPayload(_))
        ));

        let missing_provenance = br#"{"events":[{"sequence":1,"timestamp_us":1,"source_id":"x","metrics":{"rssi_dbm":-40.0}}]}"#;
        assert!(matches!(
            ElectricImpProvider.decode(missing_provenance),
            Err(VendorEventError::MalformedPayload(_))
        ));
    }

    #[test]
    fn no_remaining_provider_claims_complex_csi_or_hardware_validation() {
        let descriptors = [
            ElectricImpProvider.descriptor(),
            RfSolutionsProvider.descriptor(),
            LumaOpenWrtProvider.descriptor(),
            GoogleNestProvider.descriptor(),
            LinksysProvider.descriptor(),
            WifigardenProvider.descriptor(),
        ];
        for descriptor in descriptors {
            descriptor.validate().expect("descriptor must be valid");
            assert!(!descriptor.hardware_validated);
            assert!(!descriptor.capabilities.contains(&RfCapability::ComplexCsi));
        }
    }
}
