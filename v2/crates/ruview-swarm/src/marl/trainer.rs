use serde::{Deserialize, Serialize};

/// Which environment the MARL training loop runs against.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum TrainingMode {
    /// Pure Rust simulation — no real hardware or external simulator.
    Simulation,
    /// Gazebo + PX4 SITL (requires Gazebo running on localhost).
    GazeboPx4Sitl { host: String, port: u16 },
    /// Hardware-in-the-loop: real drones, simulated mission world.
    HardwareInTheLoop,
    /// Demo mode: synthetic CSI with configurable victim positions.
    #[default]
    Demo,
}

/// Full MAPPO training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub mode: TrainingMode,
    pub num_drones: usize,
    pub num_episodes: usize,
    pub max_steps_per_episode: usize,
    /// PPO clip epsilon.
    pub clip_epsilon: f32,
    /// Generalised Advantage Estimation lambda.
    pub gae_lambda: f32,
    /// Adam learning rate.
    pub lr: f32,
    /// Entropy coefficient (encourages exploration).
    pub entropy_coeff: f32,
    /// Number of transitions per PPO update batch.
    pub batch_size: usize,
    /// PPO epochs per update step.
    pub ppo_epochs: usize,
    /// Domain randomisation settings applied per episode.
    pub domain_rand: DomainRandomizationConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::Demo,
            num_drones: 4,
            num_episodes: 1000,
            max_steps_per_episode: 2000,
            clip_epsilon: 0.2,
            gae_lambda: 0.95,
            lr: 3e-4,
            entropy_coeff: 0.01,
            batch_size: 2048,
            ppo_epochs: 10,
            domain_rand: DomainRandomizationConfig::default(),
        }
    }
}

/// Per-episode domain randomisation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRandomizationConfig {
    /// Maximum wind speed (Dryden turbulence model), m/s.
    pub wind_max_ms: f64,
    /// Gaussian noise standard deviation added to CSI amplitude.
    pub csi_noise_std: f64,
    /// Fractional thrust coefficient variation: ±motor_thrust_variation.
    pub motor_thrust_variation: f64,
    /// Mean packet loss percentage [0–100].
    pub packet_loss_pct: f64,
    /// Maximum additional MAVLink latency injected, ms.
    pub extra_latency_max_ms: u64,
}

impl Default for DomainRandomizationConfig {
    fn default() -> Self {
        Self {
            wind_max_ms: 6.0,
            csi_noise_std: 0.05,
            motor_thrust_variation: 0.10,
            packet_loss_pct: 15.0,
            extra_latency_max_ms: 100,
        }
    }
}

impl TrainingConfig {
    /// Quick 10-episode demo run — suitable for CI smoke tests.
    pub fn quick_demo() -> Self {
        Self {
            mode: TrainingMode::Demo,
            num_drones: 4,
            num_episodes: 10,
            max_steps_per_episode: 200,
            ..Default::default()
        }
    }

    /// Full training preset with aggressive domain randomisation.
    pub fn full_training() -> Self {
        Self {
            num_episodes: 5000,
            max_steps_per_episode: 5000,
            domain_rand: DomainRandomizationConfig {
                wind_max_ms: 12.0,
                csi_noise_std: 0.1,
                motor_thrust_variation: 0.15,
                packet_loss_pct: 30.0,
                extra_latency_max_ms: 200,
            },
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_demo_has_fewer_episodes() {
        let quick = TrainingConfig::quick_demo();
        let full = TrainingConfig::full_training();
        assert!(quick.num_episodes < full.num_episodes);
        assert_eq!(quick.mode, TrainingMode::Demo);
    }

    #[test]
    fn full_training_has_larger_domain_rand() {
        let full = TrainingConfig::full_training();
        let def = DomainRandomizationConfig::default();
        assert!(full.domain_rand.wind_max_ms > def.wind_max_ms);
        assert!(full.domain_rand.packet_loss_pct > def.packet_loss_pct);
    }
}
