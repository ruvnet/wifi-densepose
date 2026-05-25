//! mDNS advertisement trait and P1 no-op stub.
//!
//! Real mDNS via the `mdns-sd` crate (https://crates.io/crates/mdns-sd)
//! lands in P2 behind the `hap-server` feature flag. P1 ships `NullAdvertiser`
//! so the bridge compiles and tests pass without any mDNS infrastructure.

use async_trait::async_trait;

use crate::error::HapError;

/// Service record advertised over mDNS for HAP discovery.
#[derive(Debug, Clone)]
pub struct HapServiceRecord {
    /// Service instance name shown in Apple Home ("RuView Sense").
    pub instance_name: String,
    /// TCP port the HAP server listens on (default 51826).
    pub port: u16,
    /// HAP pairing setup code (8 digits, formatted as XXX-XX-XXX).
    pub setup_code: String,
    /// Unique device ID (colon-separated MAC-like hex, required by HAP §5.4).
    pub device_id: String,
}

/// Advertise (and retract) a HAP accessory over mDNS (`_hap._tcp`).
///
/// Implementors register the `_hap._tcp` service so HomePod / Apple TV can
/// discover the bridge and initiate pairing. P1 provides only `NullAdvertiser`.
#[async_trait]
pub trait MdnsAdvertiser: Send + Sync {
    /// Begin advertising the service. Idempotent.
    async fn advertise(&self, record: &HapServiceRecord) -> Result<(), HapError>;

    /// Stop advertising. Called on bridge shutdown.
    async fn retract(&self, instance_name: &str) -> Result<(), HapError>;
}

/// No-op advertiser for P1 / test environments.
///
/// All calls succeed without touching the network.
#[derive(Debug, Default, Clone)]
pub struct NullAdvertiser;

#[async_trait]
impl MdnsAdvertiser for NullAdvertiser {
    async fn advertise(&self, record: &HapServiceRecord) -> Result<(), HapError> {
        tracing::debug!(
            instance = %record.instance_name,
            port = record.port,
            "NullAdvertiser: skipping mDNS advertisement (P1 stub)"
        );
        Ok(())
    }

    async fn retract(&self, instance_name: &str) -> Result<(), HapError> {
        tracing::debug!(
            instance = %instance_name,
            "NullAdvertiser: skipping mDNS retract (P1 stub)"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn null_advertiser_is_noop() {
        let adv = NullAdvertiser;
        let rec = HapServiceRecord {
            instance_name: "RuView Sense".into(),
            port: 51826,
            setup_code: "111-22-333".into(),
            device_id: "AA:BB:CC:DD:EE:FF".into(),
        };
        adv.advertise(&rec).await.unwrap();
        adv.retract(&rec.instance_name).await.unwrap();
    }
}
