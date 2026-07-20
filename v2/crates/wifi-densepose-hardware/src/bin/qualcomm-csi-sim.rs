//! Deterministic Qualcomm Atheros MIMO CSI simulator (ADR-268/269).

use clap::{Parser, ValueEnum};
use std::{
    fs::File,
    io::{self, Write},
    net::{SocketAddr, UdpSocket},
    path::PathBuf,
    thread,
    time::Duration,
};
use wifi_densepose_hardware::qualcomm_csi::{
    simulator::{QualcommCsiSimulator, SimulatorConfig},
    ChipsetProfile, CsiFrame,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Profile {
    Qca9300,
    Qcn9074,
    Qcn9274,
}
impl Profile {
    fn chipset(self) -> ChipsetProfile {
        match self {
            Self::Qca9300 => ChipsetProfile::Qca9300,
            Self::Qcn9074 => ChipsetProfile::Qcn9074,
            Self::Qcn9274 => ChipsetProfile::Qcn9274,
        }
    }
    fn default_chains(self) -> u8 {
        match self {
            Self::Qca9300 => 3,
            Self::Qcn9074 | Self::Qcn9274 => 4,
        }
    }
    fn default_bandwidth(self) -> u16 {
        match self {
            Self::Qca9300 => 40,
            Self::Qcn9074 | Self::Qcn9274 => 80,
        }
    }
    fn default_subcarriers(self) -> u16 {
        match self {
            Self::Qca9300 => 114,
            Self::Qcn9074 | Self::Qcn9274 => 256,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "qualcomm-csi-sim",
    about = "Emit synthetic ADR-269 Qualcomm Atheros MIMO CSI frames"
)]
struct Args {
    #[arg(long, value_enum, default_value_t=Profile::Qca9300)]
    profile: Profile,
    #[arg(long, default_value_t = 100)]
    frames: u32,
    #[arg(long, default_value="0x5143414353490001", value_parser=parse_u64)]
    seed: u64,
    #[arg(long)]
    bandwidth: Option<u16>,
    #[arg(long, default_value_t = 2)]
    tx: u8,
    #[arg(long)]
    rx: Option<u8>,
    #[arg(long)]
    subcarriers: Option<u16>,
    #[arg(long, default_value_t = 20)]
    interval_ms: u64,
    #[arg(long)]
    udp: Option<SocketAddr>,
    /// Replay: little-endian u32 length followed by one ADR-269 envelope.
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    realtime: bool,
}
fn parse_u64(v: &str) -> Result<u64, String> {
    if let Some(h) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).map_err(|e| e.to_string())
    } else {
        v.parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())
    }
}
fn emit(
    frame: CsiFrame,
    socket: Option<&UdpSocket>,
    destination: Option<SocketAddr>,
    output: &mut Option<File>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let wire = frame.to_bytes()?;
    if let (Some(s), Some(d)) = (socket, destination) {
        if s.send_to(&wire, d)? != wire.len() {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "partial UDP datagram").into());
        }
    }
    if let Some(f) = output {
        f.write_all(&(wire.len() as u32).to_le_bytes())?;
        f.write_all(&wire)?;
    }
    Ok(wire.len())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    if a.udp.is_none() && a.output.is_none() {
        return Err("select at least one sink with --udp or --output".into());
    }
    let cfg = SimulatorConfig {
        seed: a.seed,
        chipset: a.profile.chipset(),
        bandwidth_mhz: a.bandwidth.unwrap_or_else(|| a.profile.default_bandwidth()),
        tx_count: a.tx,
        rx_count: a.rx.unwrap_or_else(|| a.profile.default_chains()),
        subcarriers: a
            .subcarriers
            .unwrap_or_else(|| a.profile.default_subcarriers()),
        frame_period_us: a.interval_ms * 1000,
        ..Default::default()
    };
    let mut sim = QualcommCsiSimulator::new(cfg)?;
    let socket = a.udp.map(|_| UdpSocket::bind("0.0.0.0:0")).transpose()?;
    let mut output = a.output.as_ref().map(File::create).transpose()?;
    let mut bytes = emit(
        sim.capabilities_frame(),
        socket.as_ref(),
        a.udp,
        &mut output,
    )?;
    for _ in 0..a.frames {
        bytes += emit(sim.next_frame(), socket.as_ref(), a.udp, &mut output)?;
        if a.realtime {
            thread::sleep(Duration::from_millis(a.interval_ms));
        }
    }
    eprintln!(
        "emitted {} synthetic Qualcomm CSI frames ({} bytes, profile={}, seed={:#x})",
        a.frames + 1,
        bytes,
        a.profile.chipset().name(),
        a.seed
    );
    Ok(())
}
