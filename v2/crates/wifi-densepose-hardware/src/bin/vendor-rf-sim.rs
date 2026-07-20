//! Deterministic ADR-270 vendor event simulator.

use clap::{Parser, ValueEnum};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Write},
    net::{SocketAddr, UdpSocket},
    path::PathBuf,
    thread,
    time::Duration,
};
use wifi_densepose_hardware::vendor_rf::{
    ProviderAvailability, ProviderDescriptor, RfCapability, VendorId, VendorRfEvent,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Vendor {
    OriginAi,
    Plume,
    Mist,
    Netgear,
    ElectricImp,
    RfSolutions,
    Luma,
    GoogleNest,
    Linksys,
    Wifigarden,
}

impl Vendor {
    fn id(self) -> VendorId {
        match self {
            Self::OriginAi => VendorId::OriginAi,
            Self::Plume => VendorId::Plume,
            Self::Mist => VendorId::Mist,
            Self::Netgear => VendorId::Netgear,
            Self::ElectricImp => VendorId::ElectricImp,
            Self::RfSolutions => VendorId::RfSolutions,
            Self::Luma => VendorId::Luma,
            Self::GoogleNest => VendorId::GoogleNest,
            Self::Linksys => VendorId::Linksys,
            Self::Wifigarden => VendorId::Wifigarden,
        }
    }
    fn descriptor(self) -> ProviderDescriptor {
        let (capabilities, availability, reason) = match self {
            Self::OriginAi => (
                vec![RfCapability::DerivedSensing],
                ProviderAvailability::ContractRequired,
                "Origin partner API/SDK contract required",
            ),
            Self::Plume => (
                vec![RfCapability::RfTelemetry],
                ProviderAvailability::CredentialsRequired,
                "OpenSync telemetry; Plume Sense is separately gated",
            ),
            Self::Mist => (
                vec![RfCapability::RfTelemetry],
                ProviderAvailability::CredentialsRequired,
                "Mist REST/webhook telemetry",
            ),
            Self::Netgear => (
                vec![RfCapability::RfTelemetry],
                ProviderAvailability::CredentialsRequired,
                "Insight partner API telemetry",
            ),
            Self::ElectricImp => (
                vec![RfCapability::RfTelemetry],
                ProviderAvailability::CredentialsRequired,
                "impCentral scalar telemetry only",
            ),
            Self::RfSolutions => (
                vec![RfCapability::RfTelemetry],
                ProviderAvailability::CredentialsRequired,
                "RIoT environmental telemetry only",
            ),
            Self::Luma => (
                vec![RfCapability::RfTelemetry],
                ProviderAvailability::Experimental,
                "discontinued OpenWrt salvage fixture",
            ),
            Self::GoogleNest => (
                vec![RfCapability::NetworkOnly],
                ProviderAvailability::Experimental,
                "network infrastructure contract fixture only",
            ),
            Self::Linksys => (
                vec![RfCapability::Unsupported],
                ProviderAvailability::Unsupported,
                "Linksys Aware reached end of support",
            ),
            Self::Wifigarden => (
                vec![RfCapability::Unsupported],
                ProviderAvailability::ContractRequired,
                "technical SDK disclosure required",
            ),
        };
        ProviderDescriptor {
            vendor: self.id(),
            capabilities,
            availability,
            hardware_validated: false,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "vendor-rf-sim",
    about = "Emit deterministic ADR-270 vendor RF events"
)]
struct Args {
    #[arg(long, value_enum)]
    vendor: Vendor,
    #[arg(long, default_value_t = 100)]
    frames: u64,
    #[arg(long, default_value_t = 0x5255_5645_4e44_4f52)]
    seed: u64,
    #[arg(long, default_value_t = 100)]
    interval_ms: u64,
    #[arg(long)]
    udp: Option<SocketAddr>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    realtime: bool,
}

fn next_random(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / ((1u64 << 53) as f64)
}

fn event(vendor: Vendor, sequence: u64, timestamp_us: u64, state: &mut u64) -> VendorRfEvent {
    let capability = vendor.descriptor().capabilities[0];
    let wave = (sequence as f64 * 0.17).sin();
    let noise = next_random(state) - 0.5;
    let metrics = match capability {
        RfCapability::DerivedSensing => BTreeMap::from([
            (
                "motion_score".into(),
                (0.5 + 0.35 * wave + 0.05 * noise).clamp(0.0, 1.0),
            ),
            ("occupancy_count".into(), if wave > 0.0 { 2.0 } else { 1.0 }),
            ("confidence".into(), 0.92),
        ]),
        RfCapability::RfTelemetry => BTreeMap::from([
            ("rssi_dbm".into(), -52.0 + 5.0 * wave + noise),
            ("client_count".into(), if wave > 0.0 { 4.0 } else { 3.0 }),
            (
                "channel_utilization".into(),
                (0.31 + 0.08 * wave).clamp(0.0, 1.0),
            ),
        ]),
        RfCapability::NetworkOnly => {
            BTreeMap::from([("reachable".into(), 1.0), ("device_count".into(), 5.0)])
        }
        _ => BTreeMap::new(),
    };
    VendorRfEvent {
        vendor: vendor.id(),
        capability,
        sequence,
        timestamp_us,
        source_id: format!("{}-sim-01", vendor.id().as_str()),
        synthetic: true,
        metrics,
        label: Some("deterministic_contract_fixture".into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.udp.is_none() && args.output.is_none() {
        return Err("select at least one sink with --udp or --output".into());
    }
    let descriptor = args.vendor.descriptor();
    descriptor.validate()?;
    if matches!(
        descriptor.availability,
        ProviderAvailability::Unsupported | ProviderAvailability::ContractRequired
    ) && descriptor.capabilities.contains(&RfCapability::Unsupported)
    {
        return Err(format!(
            "{} has no simulatable event contract: {}",
            descriptor.vendor.as_str(),
            descriptor.reason
        )
        .into());
    }
    let socket = args.udp.map(|_| UdpSocket::bind("0.0.0.0:0")).transpose()?;
    let mut output = args.output.as_ref().map(File::create).transpose()?;
    let mut rng = args.seed;
    let mut bytes = 0usize;
    for sequence in 0..args.frames {
        let value = event(
            args.vendor,
            sequence,
            sequence * args.interval_ms * 1_000,
            &mut rng,
        );
        value.validate(&descriptor)?;
        let wire = serde_json::to_vec(&value)?;
        if let (Some(socket), Some(destination)) = (&socket, args.udp) {
            if socket.send_to(&wire, destination)? != wire.len() {
                return Err(
                    io::Error::new(io::ErrorKind::WriteZero, "partial UDP datagram").into(),
                );
            }
        }
        if let Some(file) = &mut output {
            file.write_all(&wire)?;
            file.write_all(b"\n")?;
        }
        bytes += wire.len();
        if args.realtime {
            thread::sleep(Duration::from_millis(args.interval_ms));
        }
    }
    eprintln!(
        "emitted {} synthetic {} events ({} bytes, seed={:#x})",
        args.frames,
        args.vendor.id().as_str(),
        bytes,
        args.seed
    );
    Ok(())
}
