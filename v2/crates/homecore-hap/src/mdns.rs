//! HAP `_hap._tcp` advertisement.

use async_trait::async_trait;

use crate::error::HapError;

/// HAP service record advertised over mDNS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HapServiceRecord {
    /// Service instance shown in discovery UI.
    pub instance_name: String,
    /// Bound HAP TCP port.
    pub port: u16,
    /// Stable colon-separated accessory device identifier.
    pub device_id: String,
    /// Accessory model (`md` TXT key).
    pub model: String,
    /// Configuration number (`c#`), incremented when accessory layout changes.
    pub configuration_number: u32,
    /// Current state number (`s#`).
    pub state_number: u32,
    /// HAP accessory category identifier. Bridges use `2`.
    pub category: u16,
    /// Whether a controller pairing already exists.
    pub paired: bool,
}

impl HapServiceRecord {
    pub fn bridge(
        instance_name: impl Into<String>,
        port: u16,
        device_id: impl Into<String>,
    ) -> Self {
        Self {
            instance_name: instance_name.into(),
            port,
            device_id: device_id.into(),
            model: "HOMECORE Bridge".into(),
            configuration_number: 1,
            state_number: 1,
            category: 2,
            paired: false,
        }
    }

    fn validate(&self) -> Result<(), HapError> {
        if self.instance_name.is_empty() || self.instance_name.len() > 63 {
            return Err(HapError::MdnsError(
                "instance_name must contain 1..=63 bytes".into(),
            ));
        }
        if self.port == 0 {
            return Err(HapError::MdnsError("advertised port cannot be zero".into()));
        }
        let parts: Vec<&str> = self.device_id.split(':').collect();
        if parts.len() != 6
            || parts
                .iter()
                .any(|part| part.len() != 2 || !part.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(HapError::MdnsError(
                "device_id must be six colon-separated hexadecimal octets".into(),
            ));
        }
        if self.model.is_empty() || self.model.len() > 64 {
            return Err(HapError::MdnsError(
                "model must contain 1..=64 bytes".into(),
            ));
        }
        if self.configuration_number == 0 || self.state_number == 0 {
            return Err(HapError::MdnsError(
                "configuration and state numbers must be non-zero".into(),
            ));
        }
        Ok(())
    }

    /// Standard HAP Bonjour TXT keys. Setup codes and controller data are
    /// intentionally never included because TXT records are plaintext.
    pub fn txt_records(&self) -> Vec<(String, String)> {
        vec![
            ("c#".into(), self.configuration_number.to_string()),
            ("ci".into(), self.category.to_string()),
            ("ff".into(), "0".into()),
            ("id".into(), self.device_id.to_ascii_uppercase()),
            ("md".into(), self.model.clone()),
            ("pv".into(), "1.1".into()),
            ("s#".into(), self.state_number.to_string()),
            ("sf".into(), if self.paired { "0" } else { "1" }.into()),
        ]
    }
}

/// Advertise and retract a HAP service.
#[async_trait]
pub trait MdnsAdvertiser: Send + Sync {
    async fn advertise(&self, record: &HapServiceRecord) -> Result<(), HapError>;
    async fn retract(&self, instance_name: &str) -> Result<(), HapError>;
}

/// Deterministic no-network advertiser for tests and disabled deployments.
#[derive(Debug, Default, Clone)]
pub struct NullAdvertiser;

#[async_trait]
impl MdnsAdvertiser for NullAdvertiser {
    async fn advertise(&self, record: &HapServiceRecord) -> Result<(), HapError> {
        record.validate()?;
        tracing::debug!(
            instance = %record.instance_name,
            port = record.port,
            "HAP mDNS advertisement disabled"
        );
        Ok(())
    }

    async fn retract(&self, instance_name: &str) -> Result<(), HapError> {
        tracing::debug!(instance = %instance_name, "HAP mDNS retraction disabled");
        Ok(())
    }
}

/// Network-backed Bonjour advertiser.
#[cfg(feature = "hap-server")]
pub struct MdnsSdAdvertiser {
    daemon: mdns_sd::ServiceDaemon,
    hostname: String,
    address: String,
    registrations: std::sync::Mutex<std::collections::HashMap<String, String>>,
}

#[cfg(feature = "hap-server")]
impl std::fmt::Debug for MdnsSdAdvertiser {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MdnsSdAdvertiser")
            .field("hostname", &self.hostname)
            .field("address", &self.address)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "hap-server")]
impl MdnsSdAdvertiser {
    /// Bind the mDNS daemon for a LAN-routable host/address.
    pub fn new(hostname: impl Into<String>, address: std::net::IpAddr) -> Result<Self, HapError> {
        let mut hostname = hostname.into();
        if !hostname.ends_with(".local.") {
            hostname = format!("{}.local.", hostname.trim_end_matches('.'));
        }
        let daemon = mdns_sd::ServiceDaemon::new()
            .map_err(|error| HapError::MdnsError(error.to_string()))?;
        Ok(Self {
            daemon,
            hostname,
            address: address.to_string(),
            registrations: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    fn service_info(&self, record: &HapServiceRecord) -> Result<mdns_sd::ServiceInfo, HapError> {
        record.validate()?;
        let properties: std::collections::HashMap<String, String> =
            record.txt_records().into_iter().collect();
        mdns_sd::ServiceInfo::new(
            "_hap._tcp.local.",
            &record.instance_name,
            &self.hostname,
            self.address.as_str(),
            record.port,
            Some(properties),
        )
        .map_err(|error| HapError::MdnsError(error.to_string()))
    }
}

#[cfg(feature = "hap-server")]
#[async_trait]
impl MdnsAdvertiser for MdnsSdAdvertiser {
    async fn advertise(&self, record: &HapServiceRecord) -> Result<(), HapError> {
        let info = self.service_info(record)?;
        let fullname = info.get_fullname().to_owned();
        self.daemon
            .register(info)
            .map_err(|error| HapError::MdnsError(error.to_string()))?;
        self.registrations
            .lock()
            .map_err(|_| HapError::MdnsError("registration lock poisoned".into()))?
            .insert(record.instance_name.clone(), fullname);
        Ok(())
    }

    async fn retract(&self, instance_name: &str) -> Result<(), HapError> {
        let fullname = self
            .registrations
            .lock()
            .map_err(|_| HapError::MdnsError("registration lock poisoned".into()))?
            .remove(instance_name);
        if let Some(fullname) = fullname {
            self.daemon
                .unregister(&fullname)
                .map_err(|error| HapError::MdnsError(error.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn txt_record_is_hap_shaped_and_contains_no_setup_secret() {
        let record = HapServiceRecord::bridge("RuView Sense", 51826, "AA:BB:CC:DD:EE:FF");
        let txt: std::collections::HashMap<_, _> = record.txt_records().into_iter().collect();
        assert_eq!(txt.get("pv").map(String::as_str), Some("1.1"));
        assert_eq!(txt.get("sf").map(String::as_str), Some("1"));
        assert_eq!(txt.get("ci").map(String::as_str), Some("2"));
        assert!(!txt
            .keys()
            .any(|key| key.contains("pin") || key.contains("code")));
    }

    #[tokio::test]
    async fn null_advertiser_validates_without_network_io() {
        let record = HapServiceRecord::bridge("RuView Sense", 51826, "AA:BB:CC:DD:EE:FF");
        NullAdvertiser.advertise(&record).await.unwrap();
        NullAdvertiser.retract(&record.instance_name).await.unwrap();
    }

    #[tokio::test]
    async fn malformed_record_is_rejected() {
        let record = HapServiceRecord::bridge("RuView Sense", 0, "not-an-id");
        assert!(NullAdvertiser.advertise(&record).await.is_err());
    }
}
