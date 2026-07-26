//! Hardware-in-the-loop test: feeds REAL ADR-018 CSI frames from a live
//! ESP32 node through `WifiCsiAdapter`, not synthetic data.
//!
//! The PR that introduced this crate is explicit that every reported number
//! is generator-produced (L0) and real-data validation (P2) is future work.
//! This example closes part of that gap for the one adapter that matters
//! most: it binds the real ADR-018 UDP port, parses live packets with the
//! already-proven `wifi_densepose_hardware::Esp32CsiParser` (the same parser
//! `aggregator`/`sensing-server` use in production), converts each frame into
//! the exact `wifi_densepose_core::types::CsiFrame` the adapter expects, and
//! runs it through `AdapterRegistry::normalize`.
//!
//! Usage: `cargo run -p ruview-unified --example esp32_live_hardware_test --
//! --bind 0.0.0.0:5005 --frames 8`
//! (point a live ESP32 CSI node's UDP target at this host's IP:5005 first).

use std::net::UdpSocket;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::Array2;
use num_complex::Complex64;
use wifi_densepose_core::types::{AntennaConfig, CsiFrame as CoreCsiFrame, CsiMetadata, DeviceId, FrequencyBand};
use wifi_densepose_hardware::{Esp32CsiParser, ParseError};

use ruview_unified::adapters::{AdapterRegistry, RawCapture};
use ruview_unified::tensor::LinkGeometry;

/// Inverse of `ruview_unified::adapters`'s (now-fixed) channel->frequency
/// map: recovers the 802.11 channel number from the real per-frame
/// `channel_freq_mhz` the hardware parser already computes correctly. Used
/// here only to populate `CsiMetadata::channel` for the adapter under test —
/// exercising the fix end-to-end against a real, independently-computed
/// frequency instead of a value this same test invented.
fn freq_mhz_to_band_and_channel(freq_mhz: u32) -> (FrequencyBand, u8) {
    if freq_mhz == 2484 {
        (FrequencyBand::Band2_4GHz, 14)
    } else if (2412..=2472).contains(&freq_mhz) {
        (FrequencyBand::Band2_4GHz, ((freq_mhz - 2407) / 5) as u8)
    } else if (5000..6000).contains(&freq_mhz) {
        (FrequencyBand::Band5GHz, ((freq_mhz - 5000) / 5) as u8)
    } else {
        (FrequencyBand::Band6GHz, ((freq_mhz.saturating_sub(5950)) / 5) as u8)
    }
}

fn to_core_frame(hw: wifi_densepose_hardware::CsiFrame) -> CoreCsiFrame {
    let (band, channel) = freq_mhz_to_band_and_channel(hw.metadata.channel_freq_mhz);
    let mut meta = CsiMetadata::new(DeviceId::new(format!("esp32-node-{}", hw.metadata.node_id)), band, channel);
    meta.bandwidth_mhz = match hw.metadata.bandwidth {
        wifi_densepose_hardware::Bandwidth::Bw20 => 20,
        wifi_densepose_hardware::Bandwidth::Bw40 => 40,
        wifi_densepose_hardware::Bandwidth::Bw80 => 80,
        wifi_densepose_hardware::Bandwidth::Bw160 => 160,
    };
    meta.antenna_config = AntennaConfig::new(1, hw.metadata.n_antennas.max(1));
    meta.rssi_dbm = hw.metadata.rssi_dbm;
    meta.noise_floor_dbm = hw.metadata.noise_floor_dbm;
    meta.sequence_number = hw.metadata.sequence;

    // Single spatial stream (n_antennas isn't broken out per-antenna in the
    // ADR-018 wire format consumed here) x real subcarrier count.
    let n_bins = hw.subcarriers.len().max(1);
    let data = Array2::from_shape_fn((1, n_bins), |(_, b)| {
        hw.subcarriers.get(b).map_or(Complex64::new(0.0, 0.0), |sc| {
            Complex64::new(f64::from(sc.i), f64::from(sc.q))
        })
    });
    CoreCsiFrame::new(meta, data)
}

fn main() {
    let bind = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--bind")
        .map_or_else(|| "0.0.0.0:5005".to_string(), |w| w[1].clone());
    let want_frames: usize = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--frames")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(8);

    let socket = UdpSocket::bind(&bind).expect("bind UDP socket");
    socket.set_read_timeout(Some(std::time::Duration::from_secs(30))).unwrap();
    eprintln!("Listening on {bind} for real ESP32 ADR-018 CSI frames (need {want_frames})...");

    let mut buf = [0u8; 2048];
    let mut core_frames: Vec<CoreCsiFrame> = Vec::new();
    let mut real_channel_freq_hz: Option<f64> = None;

    while core_frames.len() < want_frames {
        let (n, _src) = socket.recv_from(&mut buf).expect("recv (timed out — is a live node targeting this host?)");
        match Esp32CsiParser::parse_frame(&buf[..n]) {
            Ok((hw_frame, _consumed)) => {
                // Lock onto the first node/shape seen so all frames in the
                // window share (n_links, n_bins), as the adapter requires.
                if let Some(first) = core_frames.first() {
                    let n_bins = hw_frame.subcarriers.len();
                    if n_bins != first.num_subcarriers() {
                        eprintln!("  [skip: shape changed mid-window ({n_bins} vs {})]", first.num_subcarriers());
                        continue;
                    }
                }
                if real_channel_freq_hz.is_none() {
                    real_channel_freq_hz = Some(f64::from(hw_frame.metadata.channel_freq_mhz) * 1e6);
                }
                eprintln!(
                    "  [captured frame {}: sc={} rssi={} node={}]",
                    core_frames.len() + 1,
                    hw_frame.subcarriers.len(),
                    hw_frame.metadata.rssi_dbm,
                    hw_frame.metadata.node_id,
                );
                core_frames.push(to_core_frame(hw_frame));
            }
            Err(ParseError::NonCsiPacket { .. }) => {}
            Err(e) => eprintln!("  [parse error: {e}]"),
        }
    }

    let raw = RawCapture::WifiCsi {
        frames: core_frames,
        links: vec![LinkGeometry { tx_pos: [0.0, 0.0, 1.0], rx_pos: [3.0, 0.0, 1.0] }],
        age_s: 0.05,
        clock_quality: 0.5, // free-running ESP32 crystal, not disciplined — honest, not 1.0
    };

    let registry = AdapterRegistry::with_reference_adapters();
    let now_ns = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
    match registry.normalize("esp32s3-csi", &raw) {
        Ok(tensor) => {
            let real_hz = real_channel_freq_hz.unwrap();
            println!("=== RfTensor from REAL ESP32 hardware (not synthetic) ===");
            println!("dims (links, bins, snapshots): {:?}", tensor.data.dim());
            println!("adapter-computed center_freq_hz : {:.0}", tensor.center_freq_hz);
            println!("real per-frame channel_freq_hz   : {:.0} (from hardware parser)", real_hz);
            println!(
                "match within 1 MHz: {} (this is the ADR-018 channel-aware fix — the pre-fix \
                 code always reported the fixed-band constant, 2437000000 Hz for 2.4 GHz, \
                 regardless of the real channel)",
                (tensor.center_freq_hz - real_hz).abs() < 1e6
            );
            println!("bandwidth_hz: {:.0}", tensor.bandwidth_hz);
            println!("uncertainty (from real SNR): {:.3}", tensor.uncertainty);
            println!("device_id: {}", tensor.device_id);
            println!("timestamp_ns (now, for reference): {now_ns}");
            println!("No panic on real hardware data, including subcarrier counts != CANONICAL_BINS=56.");
        }
        Err(e) => {
            println!("ADAPTER REJECTED real hardware data: {e}");
            std::process::exit(1);
        }
    }
}
