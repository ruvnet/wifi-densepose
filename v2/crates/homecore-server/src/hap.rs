//! Optional network HomeKit Accessory Protocol lifecycle.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use anyhow::Result;
use homecore::HomeCore;

#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "hap-server"), allow(dead_code))]
pub(crate) struct HapRuntimeConfig {
    pub bind_addr: Option<SocketAddr>,
    pub device_id: Option<String>,
    pub setup_code: Option<String>,
    pub advertise_addr: Option<IpAddr>,
    pub hostname: String,
    pub instance_name: String,
    pub pairing_store: PathBuf,
}

pub(crate) struct HapRuntime {
    #[cfg(feature = "hap-server")]
    handle: Option<homecore_hap::HapServerHandle>,
    #[cfg(feature = "hap-server")]
    state_task: Option<tokio::task::JoinHandle<()>>,
}

impl HapRuntime {
    pub(crate) async fn shutdown(self) -> Result<()> {
        #[cfg(feature = "hap-server")]
        {
            let mut runtime = self;
            if let Some(task) = runtime.state_task.take() {
                task.abort();
                let _ = task.await;
            }
            if let Some(handle) = runtime.handle.take() {
                handle.shutdown().await?;
            }
        }
        Ok(())
    }
}

pub(crate) async fn start(hc: &HomeCore, config: HapRuntimeConfig) -> Result<HapRuntime> {
    let Some(bind_addr) = config.bind_addr else {
        tracing::info!("HAP network server disabled");
        return Ok(HapRuntime {
            #[cfg(feature = "hap-server")]
            handle: None,
            #[cfg(feature = "hap-server")]
            state_task: None,
        });
    };

    #[cfg(not(feature = "hap-server"))]
    {
        let _ = (hc, bind_addr);
        anyhow::bail!(
            "HAP was requested but this binary was built without the `hap-server` feature"
        );
    }

    #[cfg(feature = "hap-server")]
    {
        use std::sync::Arc;

        use homecore_hap::{
            start_server, HapBridge, HapServerConfig, HapServiceRecord, MdnsSdAdvertiser,
            PairingStore,
        };

        let device_id = config
            .device_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--hap-device-id is required when HAP is enabled"))?;
        let advertise_addr = config.advertise_addr.ok_or_else(|| {
            anyhow::anyhow!("--hap-advertise-addr is required when HAP is enabled")
        })?;
        if let Some(parent) = config.pairing_store.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let advertiser = Arc::new(MdnsSdAdvertiser::new(
            config.hostname.clone(),
            advertise_addr,
        )?);
        let pairings = if config.pairing_store.exists() {
            if config.setup_code.is_some() {
                tracing::warn!(
                    "ignoring --hap-setup-code because the pairing store already exists"
                );
            }
            PairingStore::open(&config.pairing_store)?
        } else {
            let setup_code = config.setup_code.as_deref().ok_or_else(|| {
                anyhow::anyhow!("--hap-setup-code is required when creating a HAP pairing store")
            })?;
            PairingStore::create(
                &config.pairing_store,
                homecore_hap::SetupCode::parse(setup_code)?,
                Some(device_id.to_owned()),
            )?
        };
        let pairings = Arc::new(pairings);
        let record =
            HapServiceRecord::bridge(config.instance_name.clone(), bind_addr.port(), device_id);
        let bridge = HapBridge::new(record);

        let mut state_rx = hc.states().subscribe();
        synchronize(&bridge, hc);
        let sync_bridge = bridge.clone();
        let sync_hc = hc.clone();
        let state_task = tokio::spawn(async move {
            loop {
                match state_rx.recv().await {
                    Ok(change) => match change.new_state {
                        Some(state) => {
                            if sync_bridge
                                .update_accessory(&change.entity_id, &state)
                                .is_err()
                            {
                                let _ = sync_bridge.add_accessory(&change.entity_id, &state);
                            }
                        }
                        None => {
                            let _ = sync_bridge.remove_accessory(&change.entity_id);
                        }
                    },
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(
                            skipped,
                            "HAP state listener lagged; rebuilding accessory snapshot"
                        );
                        synchronize(&sync_bridge, &sync_hc);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        let handle = start_server(
            HapServerConfig {
                bind_addr,
                ..HapServerConfig::default()
            },
            bridge,
            pairings,
            advertiser,
        )
        .await?;
        tracing::info!(
            address = %handle.local_addr(),
            pairing_store = %config.pairing_store.display(),
            "HAP network server and mDNS advertisement started"
        );
        Ok(HapRuntime {
            handle: Some(handle),
            state_task: Some(state_task),
        })
    }
}

#[cfg(feature = "hap-server")]
fn synchronize(bridge: &homecore_hap::HapBridge, hc: &HomeCore) {
    for accessory in bridge.running_accessories() {
        let _ = bridge.remove_accessory(&accessory.entity_id);
    }
    for state in hc.states().all() {
        if let Err(error) = bridge.add_accessory(&state.entity_id, &state) {
            tracing::debug!(
                entity = %state.entity_id,
                %error,
                "entity is not exposed through HAP"
            );
        }
    }
}
