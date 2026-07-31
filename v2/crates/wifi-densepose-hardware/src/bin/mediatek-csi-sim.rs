//! Deterministic MediaTek Filogic MIMO CSI simulator (ADR-266/267).

use clap::{Parser, ValueEnum};
use std::{
    fs::File,
    io::{self, Write},
    net::{SocketAddr, UdpSocket},
    path::PathBuf,
    thread,
    time::Duration,
};
use wifi_densepose_hardware::mediatek_csi::{
    simulator::{MediatekCsiSimulator, SimulatorConfig},
    ChipsetProfile, CsiFrame,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Profile {
    Mt7981,
    Mt7986,
    Mt7996,
}
impl Profile {
    fn chipset(self) -> ChipsetProfile {
        match self {
            Self::Mt7981 => ChipsetProfile::Mt7981Mt7976,
            Self::Mt7986 => ChipsetProfile::Mt7986Mt7975,
            Self::Mt7996 => ChipsetProfile::Mt7988Mt7996,
        }
    }
    fn default_chains(self) -> u8 {
        match self {
            Self::Mt7981 => 3,
            Self::Mt7986 | Self::Mt7996 => 4,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "mediatek-csi-sim",
    about = "Emit synthetic ADR-267 MediaTek Filogic MIMO CSI frames"
)]
struct Args {
    #[arg(long, value_enum, default_value_t=Profile::Mt7981)]
    profile: Profile,
    #[arg(long, default_value_t = 100)]
    frames: u32,
    #[arg(long, default_value="0x4d544b4353490001", value_parser=parse_u64)]
    seed: u64,
    #[arg(long, default_value_t = 80)]
    bandwidth: u16,
    #[arg(long, default_value_t = 2)]
    tx: u8,
    #[arg(long)]
    rx: Option<u8>,
    #[arg(long, default_value_t = 256)]
    subcarriers: u16,
    #[arg(long, default_value_t = 20)]
    interval_ms: u64,
    #[arg(long)]
    udp: Option<SocketAddr>,
    /// Replay: little-endian u32 length followed by one ADR-267 envelope.
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
        bandwidth_mhz: a.bandwidth,
        tx_count: a.tx,
        rx_count: a.rx.unwrap_or_else(|| a.profile.default_chains()),
        subcarriers: a.subcarriers,
        frame_period_us: a.interval_ms * 1000,
        ..Default::default()
    };
    let mut sim = MediatekCsiSimulator::new(cfg)?;
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
        "emitted {} synthetic MediaTek CSI frames ({} bytes, profile={}, seed={:#x})",
        a.frames + 1,
        bytes,
        a.profile.chipset().name(),
        a.seed
    );
    Ok(())
}
