//! `enroll` / `train-room` / `room-status` / `room-watch` — ADR-151 Stages 2–5 CLI.
//!
//! Drives the `wifi-densepose-calibration` pipeline against a live ESP32 CSI
//! stream (requires `edge_tier=0` raw CSI). `enroll` walks the guided anchors and
//! writes labelled features; `train-room` fits the specialist bank; `room-watch`
//! runs the mixture runtime and prints live room state.

use anyhow::{bail, Result};
use clap::Args;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::UdpSocket;
use wifi_densepose_calibration::{
    Anchor, AnchorLabel, AnchorQualityGate, AnchorRecorder, EnrollmentEvent, EnrollmentSession,
    MixtureOfSpecialists, MultiNodeMixture, NodeGeometry, SpecialistBank,
};
use wifi_densepose_calibration::extract::{AnchorFeature, Features};
use wifi_densepose_core::types::CsiFrame;
use wifi_densepose_signal::BaselineCalibration;

use crate::calibrate::parse_csi_packet;

const RECV_BUF: usize = 2048;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Per-frame scalar: mean amplitude across all subcarriers/streams.
///
/// Carries presence/motion energy plus the breathing amplitude modulation.
/// (Validated live on the ESP32 — picks up breathing where a max-variance
/// subcarrier instead locks onto motion artifacts. A phase-based carrier on a
/// *stable* subcarrier is the proper higher-SNR refinement — ADR-151 §4.)
fn frame_scalar(frame: &CsiFrame) -> f32 {
    let a = &frame.amplitude;
    if a.is_empty() {
        return 0.0;
    }
    (a.sum() / a.len() as f64) as f32
}

fn load_baseline(path: &str) -> Result<BaselineCalibration> {
    let bytes = std::fs::read(path)
        .map_err(|e| anyhow::anyhow!("cannot read baseline {path}: {e} — run `calibrate` first"))?;
    BaselineCalibration::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("invalid baseline {path}: {e}"))
}

/// Persisted enrollment output (labelled features + audit log).
#[derive(serde::Serialize, serde::Deserialize)]
struct EnrollmentData {
    room_id: String,
    baseline_id: String,
    fs_hz: f32,
    anchors: Vec<AnchorFeature>,
    session: EnrollmentSession,
}

// ---------------------------------------------------------------------------
// enroll
// ---------------------------------------------------------------------------

/// Arguments for `enroll`.
#[derive(Args, Debug, Clone)]
pub struct EnrollArgs {
    /// UDP port for ESP32 CSI frames (raw CSI; provision with `--edge-tier 0`).
    #[arg(long, default_value_t = 5005)]
    pub udp_port: u16,
    /// Bind address for the UDP socket.
    #[arg(long, default_value = "0.0.0.0")]
    pub bind: String,
    /// Path to the empty-room baseline produced by `calibrate`.
    #[arg(long, default_value = "./baseline.bin")]
    pub baseline: String,
    /// PHY tier (ht20 / ht40 / he20 / he40).
    #[arg(long, default_value = "ht20")]
    pub tier: String,
    /// Room label.
    #[arg(long, default_value = "default")]
    pub room_id: String,
    /// Output enrollment file.
    #[arg(long, default_value = "./enrollment.json")]
    pub output: String,
    /// CSI sample rate (Hz) used for periodicity extraction.
    #[arg(long, default_value_t = 15.0)]
    pub fs_hz: f32,
    /// Max attempts per anchor before moving on.
    #[arg(long, default_value_t = 2)]
    pub attempts: u32,
}

/// Capture one anchor: returns (accepted feature?, anchor verdict, reason).
async fn capture_anchor(
    socket: &UdpSocket,
    baseline: &BaselineCalibration,
    gate: &AnchorQualityGate,
    label: AnchorLabel,
    tier: &str,
    fs_hz: f32,
    room_id: &str,
) -> Result<(Option<AnchorFeature>, Anchor, Option<String>)> {
    eprintln!("\n[enroll] {} — {}", label.as_str(), label.prompt());
    for c in (1..=3).rev() {
        eprintln!("[enroll]   starting in {c}…");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    eprintln!("[enroll]   capturing {} s…", label.duration_s());

    let mut recorder = AnchorRecorder::new(label);
    let mut series: Vec<f32> = Vec::new();
    let mut buf = vec![0u8; RECV_BUF];
    let deadline = Instant::now() + Duration::from_secs(label.duration_s() as u64);

    while Instant::now() < deadline {
        let timeout = Duration::from_millis(500);
        if let Ok(Ok(n)) = tokio::time::timeout(timeout, socket.recv(&mut buf)).await {
            if let Some(frame) = parse_csi_packet(&buf[..n], tier) {
                recorder.record_frame(baseline, &frame);
                series.push(frame_scalar(&frame));
            }
        }
    }

    let (anchor, reason) = recorder.finalize(gate, now_unix());
    let feature = if anchor.quality.accepted {
        Some(AnchorFeature::from_series(room_id, label, &series, fs_hz))
    } else {
        None
    };
    Ok((feature, anchor, reason))
}

/// Execute `enroll`.
pub async fn enroll(args: EnrollArgs) -> Result<()> {
    let baseline = load_baseline(&args.baseline)?;
    let baseline_id = baseline.calibration_uuid().to_string();
    let gate = AnchorQualityGate::default();

    let addr = format!("{}:{}", args.bind, args.udp_port);
    let socket = UdpSocket::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("cannot bind {addr}: {e}"))?;
    eprintln!("[enroll] room='{}' baseline={} on udp://{addr}", args.room_id, &baseline_id[..8]);
    eprintln!("[enroll] follow each prompt; bad captures are re-prompted.");

    let mut session = EnrollmentSession::new(&args.room_id, &baseline_id, now_unix());
    let mut features: Vec<AnchorFeature> = Vec::new();

    for label in AnchorLabel::SEQUENCE {
        let mut accepted = false;
        for attempt in 1..=args.attempts {
            let (feat, anchor, reason) =
                capture_anchor(&socket, &baseline, &gate, label, &args.tier, args.fs_hz, &args.room_id)
                    .await?;
            if anchor.quality.accepted {
                eprintln!(
                    "[enroll]   ✓ accepted (presence_z={:.2} motion={:.0}% frames={})",
                    anchor.quality.presence_z,
                    anchor.quality.motion_rate * 100.0,
                    anchor.quality.frames
                );
                if let Some(f) = feat {
                    features.push(f);
                }
                session.apply(EnrollmentEvent::AnchorAccepted { anchor });
                accepted = true;
                break;
            } else {
                let why = reason.unwrap_or_default();
                eprintln!("[enroll]   ✗ rejected: {why}");
                session.apply(EnrollmentEvent::AnchorRejected {
                    label,
                    reason: why,
                    at: now_unix(),
                });
                if attempt < args.attempts {
                    eprintln!("[enroll]   retrying ({}/{})…", attempt + 1, args.attempts);
                }
            }
        }
        if !accepted {
            eprintln!("[enroll]   moving on without '{}'", label.as_str());
        }
    }

    if session.is_complete() {
        session.apply(EnrollmentEvent::Completed { at: now_unix() });
    }
    let (got, total) = session.progress();
    let data = EnrollmentData {
        room_id: args.room_id.clone(),
        baseline_id,
        fs_hz: args.fs_hz,
        anchors: features,
        session,
    };
    std::fs::write(
        &args.output,
        serde_json::to_string_pretty(&data).map_err(|e| anyhow::anyhow!("serialize: {e}"))?,
    )
    .map_err(|e| anyhow::anyhow!("cannot write {}: {e}", args.output))?;
    eprintln!(
        "\n[enroll] done: {got}/{total} anchors accepted → {} (next: `train-room`)",
        args.output
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// train-room
// ---------------------------------------------------------------------------

/// Arguments for `train-room`.
#[derive(Args, Debug, Clone)]
pub struct TrainRoomArgs {
    /// Enrollment file from `enroll`.
    #[arg(long, default_value = "./enrollment.json")]
    pub enrollment: String,
    /// Output specialist-bank file.
    #[arg(long, default_value = "./room-bank.json")]
    pub output: String,
    /// Optional transceiver-geometry file: a JSON array of `NodeGeometry`
    /// records (ADR-152 §2.1.1). Recorded into the enrollment session before
    /// training so the bank carries the layout it was trained under.
    #[arg(long)]
    pub geometry: Option<String>,
}

/// Execute `train-room`.
///
/// If the enrollment session carries a transceiver-geometry snapshot (recorded
/// at enroll time or supplied here via `--geometry`), it is threaded into the
/// bank (ADR-152 §2.1.1); a geometry-free enrollment still trains a valid bank.
pub async fn train_room(args: TrainRoomArgs) -> Result<()> {
    let raw = std::fs::read_to_string(&args.enrollment)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e} — run `enroll` first", args.enrollment))?;
    let mut data: EnrollmentData =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("invalid enrollment: {e}"))?;
    if data.anchors.is_empty() {
        bail!("no accepted anchors in {} — re-run enroll", args.enrollment);
    }

    if let Some(path) = &args.geometry {
        let graw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read geometry {path}: {e}"))?;
        let geometry: Vec<NodeGeometry> = serde_json::from_str(&graw).map_err(|e| {
            anyhow::anyhow!("invalid geometry {path}: {e} (expected a JSON array of NodeGeometry records)")
        })?;
        data.session.record_geometry(geometry, now_unix());
    }

    let mut bank = SpecialistBank::train(&data.room_id, &data.baseline_id, &data.anchors, now_unix())
        .map_err(|e| anyhow::anyhow!("training failed: {e}"))?;
    match data.session.geometry() {
        Some(g) if !g.is_empty() => {
            bank = bank.with_geometry(g.to_vec());
            eprintln!(
                "[train-room] geometry: {} node(s) snapshotted into the bank (ADR-152 §2.1.1)",
                bank.geometry.len()
            );
        }
        _ => eprintln!(
            "[train-room] no transceiver geometry recorded — bank will not support geometry conditioning (ADR-152 §2.1.2)"
        ),
    }
    std::fs::write(&args.output, bank.to_json().map_err(|e| anyhow::anyhow!("{e}"))?)
        .map_err(|e| anyhow::anyhow!("cannot write {}: {e}", args.output))?;

    eprintln!(
        "[train-room] room='{}' trained {} specialists from {} anchors → {}",
        bank.room_id,
        bank.trained_kinds().len(),
        bank.anchor_count,
        args.output
    );
    for k in bank.trained_kinds() {
        eprintln!("[train-room]   • {k:?}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// room-status
// ---------------------------------------------------------------------------

/// Arguments for `room-status`.
#[derive(Args, Debug, Clone)]
pub struct RoomStatusArgs {
    /// Specialist-bank file.
    #[arg(long, default_value = "./room-bank.json")]
    pub bank: String,
}

/// Execute `room-status`.
pub async fn room_status(args: RoomStatusArgs) -> Result<()> {
    let raw = std::fs::read_to_string(&args.bank)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", args.bank))?;
    let bank = SpecialistBank::from_json(&raw).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("room:        {}", bank.room_id);
    println!("baseline:    {}", bank.baseline_id);
    println!("trained_at:  {}", bank.trained_at_unix_s);
    println!("anchors:     {}", bank.anchor_count);
    println!("specialists: {:?}", bank.trained_kinds());
    Ok(())
}

// ---------------------------------------------------------------------------
// room-watch
// ---------------------------------------------------------------------------

/// Arguments for `room-watch`.
#[derive(Args, Debug, Clone)]
pub struct RoomWatchArgs {
    /// Specialist-bank file (single-node mode).
    #[arg(long, default_value = "./room-bank.json")]
    pub bank: String,
    /// Multistatic mode: map a node id to its bank as `N:path` (repeatable).
    /// When supplied, frames are grouped by node id and fused (ADR-029/151).
    #[arg(long = "node-bank", value_name = "N:PATH")]
    pub node_bank: Vec<String>,
    /// UDP port for ESP32 CSI frames (raw CSI).
    #[arg(long, default_value_t = 5005)]
    pub udp_port: u16,
    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    pub bind: String,
    /// PHY tier.
    #[arg(long, default_value = "ht20")]
    pub tier: String,
    /// CSI sample rate (Hz).
    #[arg(long, default_value_t = 15.0)]
    pub fs_hz: f32,
    /// Rolling window length (frames) for each inference.
    #[arg(long, default_value_t = 200)]
    pub window: usize,
    /// Seconds to run (0 = until Ctrl-C).
    #[arg(long, default_value_t = 0)]
    pub seconds: u32,
}

/// Execute `room-watch` — live (multistatic) mixture-of-specialists readout.
pub async fn room_watch(args: RoomWatchArgs) -> Result<()> {
    if !args.node_bank.is_empty() {
        return room_watch_multi(args).await;
    }
    let raw = std::fs::read_to_string(&args.bank)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", args.bank))?;
    let bank = SpecialistBank::from_json(&raw).map_err(|e| anyhow::anyhow!("{e}"))?;
    let baseline_id = bank.baseline_id.clone();
    let mix = MixtureOfSpecialists::new(bank);

    let addr = format!("{}:{}", args.bind, args.udp_port);
    let socket = UdpSocket::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("cannot bind {addr}: {e}"))?;
    eprintln!("[room-watch] inferring on udp://{addr} (window={} frames)", args.window);

    let mut buf = vec![0u8; RECV_BUF];
    let mut win: std::collections::VecDeque<f32> = std::collections::VecDeque::new();
    let start = Instant::now();
    let mut last_print = Instant::now();

    loop {
        if args.seconds > 0 && start.elapsed() >= Duration::from_secs(args.seconds as u64) {
            break;
        }
        if let Ok(Ok(n)) = tokio::time::timeout(Duration::from_millis(500), socket.recv(&mut buf)).await {
            if let Some(frame) = parse_csi_packet(&buf[..n], &args.tier) {
                win.push_back(frame_scalar(&frame));
                while win.len() > args.window {
                    win.pop_front();
                }
            }
        }
        if last_print.elapsed() >= Duration::from_secs(1) && win.len() >= 32 {
            let series: Vec<f32> = win.iter().copied().collect();
            let f = Features::from_series(&series, args.fs_hz);
            let s = mix.infer(&f, &baseline_id);
            let pres = s.presence.as_ref().map(|r| r.label.clone().unwrap_or_default()).unwrap_or("-".into());
            let post = s.posture.as_ref().and_then(|r| r.label.clone()).unwrap_or("-".into());
            let br = s.breathing.as_ref().map(|r| format!("{:.1}bpm", r.value)).unwrap_or("-".into());
            let hr = s.heartbeat.as_ref().map(|r| format!("{:.0}bpm", r.value)).unwrap_or("-".into());
            let rest = s.restlessness.as_ref().map(|r| format!("{:.2}", r.value)).unwrap_or("-".into());
            let flags = format!(
                "{}{}",
                if s.vetoed { " VETO" } else { "" },
                if s.stale { " STALE" } else { "" }
            );
            println!(
                "presence={pres:<7} posture={post:<8} breathing={br:<8} heart={hr:<7} restless={rest}{flags}"
            );
            last_print = Instant::now();
        }
    }
    Ok(())
}

/// Multistatic `room-watch`: fuse several co-located nodes (ADR-029/151).
async fn room_watch_multi(args: RoomWatchArgs) -> Result<()> {
    use std::collections::{BTreeMap, VecDeque};

    let mut mix = MultiNodeMixture::new();
    let mut node_ids: Vec<u8> = Vec::new();
    for spec in &args.node_bank {
        let (id_s, path) = spec
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("--node-bank must be N:path (got {spec:?})"))?;
        let id: u8 = id_s
            .parse()
            .map_err(|_| anyhow::anyhow!("bad node id in {spec:?}"))?;
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
        let bank = SpecialistBank::from_json(&raw).map_err(|e| anyhow::anyhow!("{e}"))?;
        let baseline = bank.baseline_id.clone();
        mix.add_node(id, bank, baseline);
        node_ids.push(id);
    }
    eprintln!("[room-watch] multistatic over nodes {node_ids:?}");

    let addr = format!("{}:{}", args.bind, args.udp_port);
    let socket = UdpSocket::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("cannot bind {addr}: {e}"))?;
    eprintln!("[room-watch] fusing on udp://{addr} (window={} frames)", args.window);

    let mut buf = vec![0u8; RECV_BUF];
    let mut wins: BTreeMap<u8, VecDeque<f32>> = BTreeMap::new();
    let start = Instant::now();
    let mut last_print = Instant::now();

    loop {
        if args.seconds > 0 && start.elapsed() >= Duration::from_secs(args.seconds as u64) {
            break;
        }
        if let Ok(Ok(n)) =
            tokio::time::timeout(Duration::from_millis(500), socket.recv(&mut buf)).await
        {
            if n < 5 {
                continue;
            }
            let node_id = buf[4];
            if !node_ids.contains(&node_id) {
                continue;
            }
            if let Some(frame) = parse_csi_packet(&buf[..n], &args.tier) {
                let w = wins.entry(node_id).or_default();
                w.push_back(frame_scalar(&frame));
                while w.len() > args.window {
                    w.pop_front();
                }
            }
        }
        if last_print.elapsed() >= Duration::from_secs(1) {
            let per_node: BTreeMap<u8, Features> = wins
                .iter()
                .filter(|(_, w)| w.len() >= 32)
                .map(|(id, w)| {
                    let series: Vec<f32> = w.iter().copied().collect();
                    (*id, Features::from_series(&series, args.fs_hz))
                })
                .collect();
            if !per_node.is_empty() {
                let active: Vec<u8> = per_node.keys().copied().collect();
                let s = mix.infer(&per_node);
                let pres = s.presence.as_ref().and_then(|r| r.label.clone()).unwrap_or("-".into());
                let post = s.posture.as_ref().and_then(|r| r.label.clone()).unwrap_or("-".into());
                let br = s.breathing.as_ref().map(|r| format!("{:.1}bpm", r.value)).unwrap_or("-".into());
                let flags = format!(
                    "{}{}",
                    if s.vetoed { " VETO" } else { "" },
                    if s.stale { " STALE" } else { "" }
                );
                println!(
                    "nodes={active:?} presence={pres:<7} posture={post:<8} breathing={br:<8}{flags}"
                );
            }
            last_print = Instant::now();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature(label: AnchorLabel, variance: f32, motion: f32) -> AnchorFeature {
        AnchorFeature {
            room_id: "t".into(),
            label,
            features: Features {
                mean: 1.0,
                variance,
                motion,
                breathing_score: 0.0,
                breathing_hz: 0.0,
                heart_score: 0.0,
                heart_hz: 0.0,
            },
        }
    }

    /// Write a minimal valid enrollment file (two anchors, no geometry event).
    fn write_enrollment(dir: &std::path::Path) -> String {
        let data = EnrollmentData {
            room_id: "t".into(),
            baseline_id: "base-1".into(),
            fs_hz: 15.0,
            anchors: vec![
                feature(AnchorLabel::Empty, 1.0, 0.1),
                feature(AnchorLabel::StandStill, 10.0, 0.2),
            ],
            session: EnrollmentSession::new("t", "base-1", 1000),
        };
        let path = dir.join("enrollment.json");
        std::fs::write(&path, serde_json::to_string(&data).unwrap()).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn trained_bank(out: &std::path::Path) -> SpecialistBank {
        SpecialistBank::from_json(&std::fs::read_to_string(out).unwrap()).unwrap()
    }

    /// ADR-152 §2.1.1: `--geometry` records into the session and the bank
    /// snapshots it — enrollment geometry reaches the trained bank.
    #[tokio::test]
    async fn train_room_threads_geometry_when_provided() {
        let dir = tempfile::tempdir().unwrap();
        let enrollment = write_enrollment(dir.path());
        let geometry = vec![
            NodeGeometry::new(1, "tape-measure").with_position(0.0, 0.0, 1.0),
            NodeGeometry::unknown(2),
        ];
        let gpath = dir.path().join("geometry.json");
        std::fs::write(&gpath, serde_json::to_string(&geometry).unwrap()).unwrap();
        let out = dir.path().join("bank.json");

        train_room(TrainRoomArgs {
            enrollment,
            output: out.to_string_lossy().into_owned(),
            geometry: Some(gpath.to_string_lossy().into_owned()),
        })
        .await
        .unwrap();

        assert_eq!(trained_bank(&out).geometry, geometry);
    }

    /// A geometry-free enrollment still trains a valid bank (optional by
    /// design) — it just carries no snapshot.
    #[tokio::test]
    async fn train_room_without_geometry_yields_geometry_free_bank() {
        let dir = tempfile::tempdir().unwrap();
        let enrollment = write_enrollment(dir.path());
        let out = dir.path().join("bank.json");

        train_room(TrainRoomArgs {
            enrollment,
            output: out.to_string_lossy().into_owned(),
            geometry: None,
        })
        .await
        .unwrap();

        let bank = trained_bank(&out);
        assert!(bank.geometry.is_empty());
        assert!(bank.presence.is_some(), "bank still trains without geometry");
    }

    /// Geometry recorded at enroll time (in the session event log) is picked up
    /// without the `--geometry` flag.
    #[tokio::test]
    async fn train_room_uses_session_geometry() {
        let dir = tempfile::tempdir().unwrap();
        let geometry = vec![NodeGeometry::new(3, "floor-plan").with_position(1.0, 2.0, 1.5)];
        let mut session = EnrollmentSession::new("t", "base-1", 1000);
        session.record_geometry(geometry.clone(), 1000);
        let data = EnrollmentData {
            room_id: "t".into(),
            baseline_id: "base-1".into(),
            fs_hz: 15.0,
            anchors: vec![
                feature(AnchorLabel::Empty, 1.0, 0.1),
                feature(AnchorLabel::StandStill, 10.0, 0.2),
            ],
            session,
        };
        let epath = dir.path().join("enrollment.json");
        std::fs::write(&epath, serde_json::to_string(&data).unwrap()).unwrap();
        let out = dir.path().join("bank.json");

        train_room(TrainRoomArgs {
            enrollment: epath.to_string_lossy().into_owned(),
            output: out.to_string_lossy().into_owned(),
            geometry: None,
        })
        .await
        .unwrap();

        assert_eq!(trained_bank(&out).geometry, geometry);
    }

    #[tokio::test]
    async fn train_room_rejects_invalid_geometry_file() {
        let dir = tempfile::tempdir().unwrap();
        let enrollment = write_enrollment(dir.path());
        let gpath = dir.path().join("geometry.json");
        std::fs::write(&gpath, r#"{"not":"an array"}"#).unwrap();

        let err = train_room(TrainRoomArgs {
            enrollment,
            output: dir.path().join("bank.json").to_string_lossy().into_owned(),
            geometry: Some(gpath.to_string_lossy().into_owned()),
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("invalid geometry"), "{err}");
    }
}
