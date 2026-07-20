//! Rust-only RTL8720F radar simulator for pre-hardware integration.

use std::{
    fs::File,
    io::{self, Write},
    net::{SocketAddr, UdpSocket},
    path::PathBuf,
    thread,
    time::Duration,
};

use clap::Parser;
use wifi_densepose_hardware::rtl8720f::{
    simulator::{Rtl8720fSimulator, SimulatorConfig},
    RadarFrame, ReportType,
};

#[derive(Debug, Parser)]
#[command(
    name = "rtl8720f-sim",
    about = "Emit synthetic ADR-264 RTL8720F radar frames"
)]
struct Args {
    #[arg(long, default_value_t = 100)]
    frames: u32,
    #[arg(long, default_value = "0x8720f123456789ab", value_parser = parse_u64)]
    seed: u64,
    #[arg(long, default_value_t = 40)]
    bandwidth: u16,
    #[arg(long, default_value_t = 15)]
    interval_ms: u64,
    /// UDP destination; each frame is one datagram.
    #[arg(long)]
    udp: Option<SocketAddr>,
    /// Replay file; LE u32 length followed by ADR-264 bytes.
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    realtime: bool,
}

fn parse_u64(value: &str) -> Result<u64, String> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|error| error.to_string())
    } else {
        value.parse::<u64>().map_err(|error| error.to_string())
    }
}

fn emit(
    frame: RadarFrame,
    socket: Option<&UdpSocket>,
    destination: Option<SocketAddr>,
    output: &mut Option<File>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let wire = frame.to_bytes()?;
    if let (Some(socket), Some(destination)) = (socket, destination) {
        let sent = socket.send_to(&wire, destination)?;
        if sent != wire.len() {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "partial UDP datagram").into());
        }
    }
    if let Some(file) = output {
        file.write_all(&(wire.len() as u32).to_le_bytes())?;
        file.write_all(&wire)?;
    }
    Ok(wire.len())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.udp.is_none() && args.output.is_none() {
        return Err("select at least one sink with --udp or --output".into());
    }
    let config = SimulatorConfig {
        seed: args.seed,
        bandwidth_mhz: args.bandwidth,
        frame_period_us: args.interval_ms * 1_000,
        ..SimulatorConfig::default()
    };
    let mut simulator = Rtl8720fSimulator::new(config)?;
    let socket = args.udp.map(|_| UdpSocket::bind("0.0.0.0:0")).transpose()?;
    let mut output = args.output.as_ref().map(File::create).transpose()?;
    let mut bytes_emitted = emit(
        simulator.capabilities_frame(),
        socket.as_ref(),
        args.udp,
        &mut output,
    )?;

    for index in 0..args.frames {
        let report_type = match index % 16 {
            15 => ReportType::Interference,
            value if value % 4 == 1 => ReportType::RangeNear,
            value if value % 4 == 3 => ReportType::RangeFar,
            _ => ReportType::Cfr,
        };
        bytes_emitted += emit(
            simulator.next_frame(report_type),
            socket.as_ref(),
            args.udp,
            &mut output,
        )?;
        if args.realtime {
            thread::sleep(Duration::from_millis(args.interval_ms));
        }
    }
    eprintln!(
        "emitted {} synthetic RTL8720F frames ({} bytes, seed={:#x})",
        args.frames + 1,
        bytes_emitted,
        args.seed
    );
    Ok(())
}
