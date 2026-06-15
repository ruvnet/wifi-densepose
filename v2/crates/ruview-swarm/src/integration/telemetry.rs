//! JSONL telemetry recorder for the swarm training/sim visualizer.
//!
//! Emits newline-delimited JSON records consumed by `viz/swarm_viz.html`:
//!   - one `meta` record (mission profile, area, ground-truth victims)
//!   - many `step` records (per-tick drone positions, coverage, detections)
//!   - optional `episode` records (per-episode training metrics)
//!
//! Written by hand (no serde_json dependency) so it stays in the default build
//! and never affects the test/CI surface. The schema is flat and the only
//! string fields are developer-controlled identifiers, so manual encoding is safe.

use crate::types::{DroneState, Position3D};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Records swarm telemetry to a JSONL file for offline visualization.
pub struct TelemetryRecorder {
    writer: BufWriter<File>,
}

/// One drone's per-step visual state.
pub struct DroneFrame {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub heading_rad: f64,
    pub battery_pct: f32,
    pub detected: bool,
}

impl DroneFrame {
    pub fn from_state(state: &DroneState, detected: bool) -> Self {
        Self {
            id: state.id.0,
            x: state.position.x,
            y: state.position.y,
            heading_rad: state.heading_rad,
            battery_pct: state.battery_pct,
            detected,
        }
    }
}

impl TelemetryRecorder {
    /// Open a telemetry file for writing.
    pub fn create<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self { writer: BufWriter::new(file) })
    }

    /// Write the one-time mission metadata header.
    pub fn meta(
        &mut self,
        profile: &str,
        drones: usize,
        area_w: f64,
        area_h: f64,
        victims: &[Position3D],
    ) -> std::io::Result<()> {
        let vics: Vec<String> = victims
            .iter()
            .map(|v| format!("[{:.2},{:.2}]", v.x, v.y))
            .collect();
        writeln!(
            self.writer,
            r#"{{"type":"meta","profile":"{}","drones":{},"area_w":{:.2},"area_h":{:.2},"victims":[{}]}}"#,
            sanitize(profile),
            drones,
            area_w,
            area_h,
            vics.join(",")
        )
    }

    /// Write one simulation step (all drones at this tick).
    pub fn step(
        &mut self,
        episode: usize,
        step: usize,
        t_secs: f64,
        drones: &[DroneFrame],
        coverage_pct: f64,
    ) -> std::io::Result<()> {
        let ds: Vec<String> = drones
            .iter()
            .map(|d| {
                format!(
                    r#"{{"id":{},"x":{:.2},"y":{:.2},"hdg":{:.3},"batt":{:.1},"det":{}}}"#,
                    d.id, d.x, d.y, d.heading_rad, d.battery_pct, d.detected
                )
            })
            .collect();
        writeln!(
            self.writer,
            r#"{{"type":"step","ep":{},"step":{},"t":{:.2},"coverage":{:.4},"drones":[{}]}}"#,
            episode,
            step,
            t_secs,
            coverage_pct,
            ds.join(",")
        )
    }

    /// Write one episode's training metrics.
    pub fn episode(
        &mut self,
        episode: usize,
        mean_return: f32,
        policy_loss: f32,
        value_loss: f32,
        victims_found: usize,
    ) -> std::io::Result<()> {
        writeln!(
            self.writer,
            r#"{{"type":"episode","ep":{},"mean_return":{:.4},"policy_loss":{:.4},"value_loss":{:.4},"victims_found":{}}}"#,
            episode, mean_return, policy_loss, value_loss, victims_found
        )
    }

    /// Flush buffered records to disk.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

/// Strip characters that would break the flat JSON string field.
fn sanitize(s: &str) -> String {
    s.chars().filter(|c| *c != '"' && *c != '\\' && *c != '\n').collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeId, Velocity3D};

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    #[test]
    fn test_records_valid_jsonl() {
        let path = tmp_path("ruview_telemetry_test.jsonl");
        {
            let mut rec = TelemetryRecorder::create(&path).unwrap();
            rec.meta("sar", 2, 400.0, 400.0, &[Position3D { x: 80.0, y: 120.0, z: 0.0 }])
                .unwrap();
            let state = DroneState {
                id: NodeId(0),
                position: Position3D { x: 10.5, y: 20.25, z: -30.0 },
                velocity: Velocity3D::default(),
                heading_rad: 1.57,
                altitude_agl_m: 30.0,
                battery_pct: 88.0,
                link_quality: 0.9,
                timestamp_ms: 0,
            };
            rec.step(0, 0, 0.0, &[DroneFrame::from_state(&state, true)], 0.05)
                .unwrap();
            rec.episode(0, 103.7, -61.2, 12643.3, 1).unwrap();
            rec.flush().unwrap();
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3, "meta + step + episode = 3 records");
        assert!(lines[0].contains(r#""type":"meta""#));
        assert!(lines[1].contains(r#""type":"step""#));
        assert!(lines[1].contains(r#""det":true"#));
        assert!(lines[2].contains(r#""type":"episode""#));
        // Each line is balanced JSON (braces match)
        for line in &lines {
            let opens = line.matches('{').count();
            let closes = line.matches('}').count();
            assert_eq!(opens, closes, "balanced braces in: {line}");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_sanitize_strips_quotes() {
        assert_eq!(sanitize("sa\"r\n"), "sar");
    }
}
