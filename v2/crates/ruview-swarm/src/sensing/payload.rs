use crate::types::{NodeId, Position3D, CsiDetection};

/// Configuration for the onboard CSI sensing payload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PayloadConfig {
    pub scan_freq_hz: f64,         // 10.0 nominal, 20.0 during Phase 3 convergence
    pub detection_range_m: f64,    // ~28.0 m (Wi2SAR validated)
    pub confidence_threshold: f32, // minimum confidence to report detection (0.6)
    pub esp32_baud_rate: u32,      // 921600
}

impl Default for PayloadConfig {
    fn default() -> Self {
        Self {
            scan_freq_hz: 10.0,
            detection_range_m: 28.0,
            confidence_threshold: 0.6,
            esp32_baud_rate: 921600,
        }
    }
}

/// Represents the CSI sensing payload pipeline running on the drone's companion compute.
/// In production: reads from ESP32-S3 via serial TDM; runs CIR (ADR-134) -> RF encoder (ADR-146).
/// In demo/sim mode: generates synthetic detections.
pub struct CsiPayloadPipeline {
    pub node_id: NodeId,
    pub config: PayloadConfig,
    mode: PipelineMode,
}

// Fields in Live and Replay variants are unused until the serial/file backends are wired up.
#[allow(dead_code)]
enum PipelineMode {
    /// Live pipeline: reads from serial port.
    Live { port_path: String },
    /// Demo/simulation mode: synthetic CSI generation.
    Synthetic {
        victim_positions: Vec<Position3D>,
        noise_std: f64,
        rng_seed: u64,
    },
    /// Replay mode: reads from recorded CSI file.
    Replay { file_path: String, loop_replay: bool },
}

impl CsiPayloadPipeline {
    pub fn new_live(node_id: NodeId, config: PayloadConfig, port: &str) -> Self {
        Self { node_id, config, mode: PipelineMode::Live { port_path: port.to_string() } }
    }

    pub fn new_synthetic(
        node_id: NodeId,
        config: PayloadConfig,
        victims: Vec<Position3D>,
        noise_std: f64,
        seed: u64,
    ) -> Self {
        Self {
            node_id,
            config,
            mode: PipelineMode::Synthetic {
                victim_positions: victims,
                noise_std,
                rng_seed: seed,
            },
        }
    }

    pub fn new_replay(node_id: NodeId, config: PayloadConfig, path: &str, loop_replay: bool) -> Self {
        Self {
            node_id,
            config,
            mode: PipelineMode::Replay {
                file_path: path.to_string(),
                loop_replay,
            },
        }
    }

    /// Scan the current position and return a detection report (if any).
    pub async fn scan(&self, drone_pos: &Position3D) -> Option<CsiDetection> {
        match &self.mode {
            PipelineMode::Synthetic { victim_positions, noise_std, rng_seed } => {
                self.synthetic_scan(drone_pos, victim_positions, *noise_std, *rng_seed)
            }
            PipelineMode::Live { .. } => {
                // Production: would read from serial port, run CIR+RF encoder pipeline
                // For now: return None (requires hardware)
                None
            }
            PipelineMode::Replay { .. } => {
                // Production: would read from recorded file
                None
            }
        }
    }

    fn synthetic_scan(
        &self,
        drone_pos: &Position3D,
        victims: &[Position3D],
        noise_std: f64,
        _seed: u64,
    ) -> Option<CsiDetection> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for victim in victims {
            let dist = drone_pos.distance_to(victim);
            if dist < self.config.detection_range_m {
                let base_confidence = (-dist / self.config.detection_range_m).exp();
                let noise: f64 = rng.gen_range(-noise_std..noise_std);
                let confidence = (base_confidence + noise).clamp(0.0, 1.0) as f32;

                if confidence >= self.config.confidence_threshold {
                    let pos_noise_x: f64 = rng.gen_range(-noise_std * 5.0..noise_std * 5.0);
                    let pos_noise_y: f64 = rng.gen_range(-noise_std * 5.0..noise_std * 5.0);
                    return Some(CsiDetection {
                        drone_id: self.node_id,
                        confidence,
                        victim_position: Some(Position3D {
                            x: victim.x + pos_noise_x,
                            y: victim.y + pos_noise_y,
                            z: victim.z,
                        }),
                        timestamp_ms: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as u64)
                            .unwrap_or(0),
                    });
                }
            }
        }
        None
    }
}
