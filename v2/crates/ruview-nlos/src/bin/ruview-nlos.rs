//! RuView NLOS research CLI. Hardware evidence requires the external witness
//! protocol; this binary never promotes its synthetic benchmark.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use ruview_nlos::calibration::{Calibration, CalibrationConfig};
use ruview_nlos::server::NlosHub;
use ruview_nlos::{
    CanonicalObject, FrameSource, MotionApertureTracker, SyntheticScene, TrackEnvelope,
    TrackerConfig, TransientFrame, TransientKind, Vec3, MAX_WIRE_BYTES,
};
#[cfg(feature = "hardware")]
use ruview_nlos::{EvidenceLevel, Provenance, SensorPose, StAsciiDecoder, StDecoderConfig};

#[derive(Parser)]
#[command(
    name = "ruview-nlos",
    version,
    about = "Consumer transient NLOS research runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the deterministic L0 performance and fusion benchmark.
    Benchmark {
        /// Number of frames.
        #[arg(long, default_value_t = 300)]
        frames: usize,
        /// Particle count.
        #[arg(long, default_value_t = 1_000)]
        particles: usize,
    },
    /// Validate every bounded track JSONL record.
    ValidateTrack {
        /// JSONL file.
        input: PathBuf,
    },
    /// Process transient JSONL; the initial frames form empty-room calibration.
    TrackJsonl {
        /// Transient JSONL file.
        input: PathBuf,
        /// Empty-room frames at the beginning of the file.
        #[arg(long, default_value_t = 60)]
        background_frames: usize,
    },
    /// Capture the public VL53L8CH STM32 stream directly over USB serial.
    #[cfg(feature = "hardware")]
    CaptureSt {
        /// Explicit serial device path; auto-discovery is intentionally avoided.
        #[arg(long)]
        port: String,
        /// Capture session identifier.
        #[arg(long)]
        session: String,
        /// Stable enrolled sensor identifier; never inferred from a port path.
        #[arg(long)]
        sensor_id: String,
        /// Exact sensor model.
        #[arg(long, default_value = "VL53L8CH")]
        sensor_model: String,
        /// Exact flashed firmware revision or digest label.
        #[arg(long)]
        firmware_version: String,
        /// One externally estimated sensor pose per output frame as JSONL.
        #[arg(long, conflicts_with = "fixed_sensor")]
        pose_jsonl: Option<PathBuf>,
        /// Explicitly declare a fixed identity pose. This does not synthesize
        /// camera motion; aperture diversity must then come from target motion.
        #[arg(long, conflicts_with = "pose_jsonl")]
        fixed_sensor: bool,
        /// Frames to emit as transient JSONL.
        #[arg(long, default_value_t = 300)]
        frames: usize,
        /// Grid height.
        #[arg(long, default_value_t = 4)]
        height: u16,
        /// Grid width.
        #[arg(long, default_value_t = 4)]
        width: u16,
        /// Histogram bins.
        #[arg(long, default_value_t = 48)]
        bins: u16,
        /// Firmware start bin.
        #[arg(long, default_value_t = 30)]
        start_bin: u16,
        /// Requested ranging rate.
        #[arg(long, default_value_t = 30)]
        frequency_hz: u16,
    },
    /// Serve authenticated native and browser clients.
    Serve {
        /// Bind address. Non-loopback requires an explicit reverse-proxy flag.
        #[arg(long, default_value = "127.0.0.1:8787")]
        bind: SocketAddr,
        /// Environment variable holding a bearer token of at least 32 characters.
        #[arg(long, default_value = "RUVIEW_NLOS_TOKEN")]
        token_env: String,
        /// Publish deterministic L0 frames at 30 Hz for UI validation.
        #[arg(long)]
        synthetic: bool,
        /// Confirm that a TLS reverse proxy and network policy protect non-loopback bind.
        #[arg(long)]
        behind_tls_proxy: bool,
        /// One exact HTTPS browser origin, or HTTP loopback origin for development.
        #[arg(long)]
        allowed_origin: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_target(false).init();
    match Cli::parse().command {
        Command::Benchmark { frames, particles } => {
            if !(1..=10_000).contains(&frames) || !(64..=20_000).contains(&particles) {
                return Err("frames or particles outside bounded range".into());
            }
            let report = SyntheticScene::benchmark(frames, particles);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::ValidateTrack { input } => {
            let count = read_jsonl::<TrackEnvelope>(&input, |frame| Ok(frame.validate()?))?;
            println!("validated {count} bounded track frames");
        }
        Command::TrackJsonl {
            input,
            background_frames,
        } => {
            track_jsonl(&input, background_frames, std::io::stdout().lock())?;
        }
        #[cfg(feature = "hardware")]
        Command::CaptureSt {
            port,
            session,
            sensor_id,
            sensor_model,
            firmware_version,
            pose_jsonl,
            fixed_sensor,
            frames,
            height,
            width,
            bins,
            start_bin,
            frequency_hz,
        } => capture_st(
            &port,
            &session,
            &sensor_id,
            &sensor_model,
            &firmware_version,
            pose_jsonl.as_ref(),
            fixed_sensor,
            frames,
            height,
            width,
            bins,
            start_bin,
            frequency_hz,
        )?,
        Command::Serve {
            bind,
            token_env,
            synthetic,
            behind_tls_proxy,
            allowed_origin,
        } => {
            if !is_loopback(bind.ip()) && !behind_tls_proxy {
                return Err("non-loopback bind requires --behind-tls-proxy".into());
            }
            let token = std::env::var(&token_env).map_err(|_| {
                format!("required token environment variable {token_env} is absent")
            })?;
            let server_session = if synthetic {
                "synthetic-nlos-1"
            } else {
                "ruview-nlos-server"
            };
            let mut hub = NlosHub::new(&token, server_session)?;
            if let Some(origin) = allowed_origin {
                hub = hub.with_allowed_origin(&origin)?;
            }
            if synthetic {
                spawn_synthetic(hub.clone());
            }
            let listener = tokio::net::TcpListener::bind(bind).await?;
            tracing::info!(%bind, synthetic, "RuView NLOS server listening");
            axum::serve(listener, hub.router()).await?;
        }
    }
    Ok(())
}

#[cfg(feature = "hardware")]
#[allow(clippy::too_many_arguments)]
fn capture_st(
    port: &str,
    session: &str,
    sensor_id: &str,
    sensor_model: &str,
    firmware_version: &str,
    pose_jsonl: Option<&PathBuf>,
    fixed_sensor: bool,
    frame_limit: usize,
    height: u16,
    width: u16,
    bins: u16,
    start_bin: u16,
    frequency_hz: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    if !(1..=1_000_000).contains(&frame_limit) || !(1..=30).contains(&frequency_hz) {
        return Err("capture frame or frequency bound exceeded".into());
    }
    if pose_jsonl.is_some() == fixed_sensor {
        return Err("choose exactly one of --pose-jsonl or --fixed-sensor".into());
    }
    for value in [session, sensor_id, sensor_model, firmware_version] {
        if !valid_cli_label(value) {
            return Err("session and provenance labels must use 1..64 safe characters".into());
        }
    }
    let poses = if let Some(path) = pose_jsonl {
        let poses = load_sensor_poses(path)?;
        if poses.len() != frame_limit {
            return Err("pose JSONL must contain exactly one pose per captured frame".into());
        }
        Some(poses)
    } else {
        None
    };
    let mut decoder = StAsciiDecoder::new(StDecoderConfig {
        session_id: session.into(),
        height,
        width,
        num_bins: bins,
        start_bin,
        bin_width_ps: 250.0,
        fov_x_degrees: 45.0,
        fov_y_degrees: 45.0,
        add_back_ambient: false,
        require_frame_marker: true,
        source: FrameSource::Live,
        evidence_level: EvidenceLevel::L1Measured,
        calibration_hash: "0".repeat(64),
        provenance: Provenance {
            sensor_id: sensor_id.into(),
            sensor_model: sensor_model.into(),
            firmware_version: firmware_version.into(),
            transient_kind: TransientKind::CompactNormalizedHistogram,
            histogram_preserved: true,
            transport: "usb_serial".into(),
        },
    })?;
    let mut serial = serialport::new(port, 2_250_000)
        .timeout(Duration::from_secs(1))
        .open()?;
    std::thread::sleep(Duration::from_secs(1));
    serial.clear(serialport::ClearBuffer::Input)?;
    serial.write_all(&decoder.firmware_config_bytes(1, frequency_hz, 10, 1))?;
    serial.flush()?;
    std::thread::sleep(Duration::from_secs(1));
    serial.clear(serialport::ClearBuffer::Input)?;
    let started = Instant::now();
    let mut reader = BufReader::new(serial);
    let mut emitted = 0_usize;
    while emitted < frame_limit {
        let line = match read_bounded_line(&mut reader, 4_096) {
            Ok(None) => continue,
            Ok(Some(line)) => line,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let line = std::str::from_utf8(&line)?;
        let captured_at_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
        if let Some(frame) = decoder.push_line(
            line,
            captured_at_unix_ms,
            started.elapsed().as_nanos() as u64,
            poses
                .as_ref()
                .map_or_else(SensorPose::default, |values| values[emitted]),
        )? {
            println!("{}", serde_json::to_string(&frame)?);
            emitted += 1;
        }
    }
    Ok(())
}

fn is_loopback(ip: IpAddr) -> bool {
    ip.is_loopback()
}

#[cfg(feature = "hardware")]
fn valid_cli_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

#[cfg(feature = "hardware")]
fn load_sensor_poses(path: &PathBuf) -> Result<Vec<SensorPose>, Box<dyn std::error::Error>> {
    let mut poses = Vec::new();
    read_jsonl::<SensorPose>(path, |pose| {
        pose.validate()?;
        poses.push(*pose);
        Ok(())
    })?;
    Ok(poses)
}

fn spawn_synthetic(hub: NlosHub) {
    tokio::spawn(async move {
        let mut scene = SyntheticScene::default();
        let Ok(calibration) = Calibration::from_background(
            &scene.background_frames(60),
            CalibrationConfig::default(),
        ) else {
            return;
        };
        let Ok(mut tracker) = MotionApertureTracker::new(
            calibration,
            CanonicalObject::point(),
            TrackerConfig::default(),
        ) else {
            return;
        };
        let mut sequence = 100_u64;
        let mut interval = tokio::time::interval(Duration::from_millis(33));
        loop {
            interval.tick().await;
            let t = sequence as f32 * 0.02;
            let target = Vec3::new(0.25 * t.sin(), 0.15 * (t * 0.7).cos(), 1.0);
            let mut frame = scene.frame(Some(target), 1.0, sequence);
            frame.captured_at_unix_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_millis() as u64);
            if let Ok(output) = tracker.update(&frame, None) {
                let _ = hub.publish(output).await;
            }
            sequence = sequence.saturating_add(1);
        }
    });
}

fn track_jsonl(
    path: &PathBuf,
    background_frames: usize,
    mut output_writer: impl Write,
) -> Result<usize, Box<dyn std::error::Error>> {
    let calibration_config = CalibrationConfig::default();
    if !(2..=calibration_config.max_background_frames).contains(&background_frames) {
        return Err("background-frames must be between 2 and 256".into());
    }

    let mut background = Vec::with_capacity(background_frames);
    let mut tracker = None;
    let mut tracked_frames = 0_usize;
    read_jsonl::<TransientFrame>(path, |frame| {
        let frame = as_offline_replay(frame.clone())?;
        if background.len() < background_frames {
            background.push(frame);
            return Ok(());
        }
        if tracker.is_none() {
            let calibration_frames = std::mem::take(&mut background);
            let calibration =
                Calibration::from_background(&calibration_frames, calibration_config)?;
            tracker = Some(MotionApertureTracker::new(
                calibration,
                CanonicalObject::point(),
                TrackerConfig::default(),
            )?);
        }
        let tracked = tracker
            .as_mut()
            .ok_or("tracker initialization failed")?
            .update(&frame, None)?;
        serde_json::to_writer(&mut output_writer, &tracked)?;
        output_writer.write_all(b"\n")?;
        tracked_frames = tracked_frames.saturating_add(1);
        Ok(())
    })?;
    if tracked_frames == 0 {
        return Err("background-frames must leave at least one tracking frame".into());
    }
    Ok(tracked_frames)
}

fn as_offline_replay(
    mut frame: TransientFrame,
) -> Result<TransientFrame, ruview_nlos::protocol::ContractError> {
    frame.validate()?;
    if frame.source == FrameSource::Live {
        frame.source = FrameSource::Replay;
        frame.provenance.transient_kind = TransientKind::Replay;
        frame.provenance.transport = "replay".into();
        frame.provenance.histogram_preserved = true;
    }
    frame.validate()?;
    Ok(frame)
}

fn read_jsonl<T>(
    path: &PathBuf,
    mut validate: impl FnMut(&T) -> Result<(), Box<dyn std::error::Error>>,
) -> Result<usize, Box<dyn std::error::Error>>
where
    T: serde::de::DeserializeOwned,
{
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > 512 * 1024 * 1024 {
        return Err("JSONL input exceeds 512 MiB bound".into());
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut count = 0_usize;
    while let Some(line) = read_bounded_line(&mut reader, MAX_WIRE_BYTES)? {
        let line = std::str::from_utf8(&line)?;
        if line.trim().is_empty() {
            continue;
        }
        let value: T = serde_json::from_str(line)?;
        validate(&value)?;
        count += 1;
    }
    Ok(count)
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    maximum_bytes: usize,
) -> std::io::Result<Option<Vec<u8>>> {
    let mut line = Vec::with_capacity(maximum_bytes.min(8 * 1024));
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok((!line.is_empty()).then_some(line));
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let payload = newline.unwrap_or(available.len());
        if line.len().saturating_add(payload) > maximum_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "line exceeds bounded parser limit",
            ));
        }
        line.extend_from_slice(&available[..payload]);
        let consumed = payload + usize::from(newline.is_some());
        reader.consume(consumed);
        if newline.is_some() {
            return Ok(Some(line));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{as_offline_replay, read_bounded_line, track_jsonl};
    use ruview_nlos::{EvidenceLevel, FrameSource, SyntheticScene, TransientKind, Vec3};
    use std::fs::{remove_file, File};
    use std::io::{Cursor, Write};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn bounded_reader_rejects_before_growing_past_limit() {
        let mut reader = Cursor::new(vec![b'x'; 65]);
        let error = read_bounded_line(&mut reader, 64).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn bounded_reader_accepts_exact_limit_and_multiple_lines() {
        let mut reader = Cursor::new(b"1234\nok\n".to_vec());
        assert_eq!(read_bounded_line(&mut reader, 4).unwrap().unwrap(), b"1234");
        assert_eq!(read_bounded_line(&mut reader, 4).unwrap().unwrap(), b"ok");
        assert!(read_bounded_line(&mut reader, 4).unwrap().is_none());
    }

    #[test]
    fn offline_live_capture_is_relabelled_as_replay() {
        let mut scene = SyntheticScene::default();
        let mut frame = scene.frame(Some(Vec3::new(0.0, 0.0, 1.0)), 1.0, 7);
        frame.source = FrameSource::Live;
        frame.evidence_level = EvidenceLevel::L1Measured;
        frame.provenance.transient_kind = TransientKind::CompactNormalizedHistogram;
        frame.provenance.transport = "usb_serial".into();
        let replay = as_offline_replay(frame).unwrap();
        assert_eq!(replay.source, FrameSource::Replay);
        assert_eq!(replay.provenance.transient_kind, TransientKind::Replay);
        assert_eq!(replay.provenance.transport, "replay");
    }

    #[test]
    fn track_jsonl_streams_after_bounded_background_window() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ruview-nlos-track-{}-{nonce}.jsonl",
            std::process::id()
        ));
        let mut scene = SyntheticScene::default();
        let mut frames = scene.background_frames(2);
        frames.push(scene.frame(Some(Vec3::new(0.1, 0.0, 1.0)), 1.0, 2));
        let mut input = File::create(&path).unwrap();
        for frame in frames {
            serde_json::to_writer(&mut input, &frame).unwrap();
            input.write_all(b"\n").unwrap();
        }
        drop(input);

        let mut output = Vec::new();
        let tracked = track_jsonl(&path, 2, &mut output).unwrap();
        remove_file(&path).unwrap();

        assert_eq!(tracked, 1);
        assert_eq!(output.iter().filter(|byte| **byte == b'\n').count(), 1);
        let envelope: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(envelope["source"], "synthetic");
    }

    #[test]
    fn track_jsonl_rejects_unbounded_calibration_window_before_io() {
        let error =
            track_jsonl(&PathBuf::from("does-not-exist.jsonl"), 257, Vec::new()).unwrap_err();
        assert!(error.to_string().contains("between 2 and 256"));
    }
}
