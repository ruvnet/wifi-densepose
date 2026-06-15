use super::observation::LocalObservation;

/// Action output from the MAPPO actor.
#[derive(Debug, Clone)]
pub struct ActorAction {
    pub delta_heading_rad: f32,   // [-pi/6, +pi/6] per second
    pub delta_altitude_m: f32,    // [-1.0, +1.0] m per second
    pub speed_ms: f32,            // [0.0, 8.0] m/s
    pub trigger_csi_scan: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActorConfig {
    /// Hidden layer dimensions; default [128, 64].
    pub hidden_dims: Vec<usize>,
    pub max_speed_ms: f32,
    pub max_heading_delta_rad: f32,
    pub max_altitude_delta_m: f32,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            hidden_dims: vec![128, 64],
            max_speed_ms: 8.0,
            max_heading_delta_rad: std::f32::consts::PI / 6.0,
            max_altitude_delta_m: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// MLP helper functions
// ---------------------------------------------------------------------------

#[inline]
fn relu(x: f32) -> f32 { x.max(0.0) }

#[inline]
fn tanh_f32(x: f32) -> f32 { x.tanh() }

#[inline]
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn matmul_vec(weights: &[Vec<f32>], input: &[f32], bias: &[f32]) -> Vec<f32> {
    weights
        .iter()
        .zip(bias.iter())
        .map(|(row, b)| row.iter().zip(input.iter()).map(|(w, x)| w * x).sum::<f32>() + b)
        .collect()
}

// ---------------------------------------------------------------------------
// MAPPO actor
// ---------------------------------------------------------------------------

/// Simple 3-layer MLP actor (pure Rust, no ONNX).
///
/// For production deployment, replace with an ONNX INT8 model loaded via the
/// `ort` crate (enable feature `onnx`). The interface — `forward(&obs) -> ActorAction`
/// — remains identical.
pub struct MappoActor {
    pub config: ActorConfig,
    /// Layer 1: obs_dim × hidden1
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    /// Layer 2: hidden1 × hidden2
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    /// Output layer: hidden2 × 4
    w_out: Vec<Vec<f32>>,
    b_out: Vec<f32>,
}

impl MappoActor {
    /// Create an actor with random weights using the standard observation dimension.
    ///
    /// Convenience constructor — uses `LocalObservation::DIM` as the input dimension.
    pub fn random_init(config: ActorConfig) -> Self {
        Self::random_init_with_dim(LocalObservation::DIM, config)
    }

    /// Create an actor with random (untrained) weights — for testing only.
    pub fn random_init_with_dim(obs_dim: usize, config: ActorConfig) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let h1 = config.hidden_dims[0];
        let h2 = config.hidden_dims.get(1).copied().unwrap_or(64);

        let w1 = (0..h1)
            .map(|_| (0..obs_dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let b1 = vec![0.0f32; h1];
        let w2 = (0..h2)
            .map(|_| (0..h1).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let b2 = vec![0.0f32; h2];
        let w_out = (0..4)
            .map(|_| (0..h2).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let b_out = vec![0.0f32; 4];

        Self { config, w1, b1, w2, b2, w_out, b_out }
    }

    /// Forward pass: observation -> action.
    pub fn forward(&self, obs: &LocalObservation) -> ActorAction {
        let input = obs.to_vec();
        let h1: Vec<f32> = matmul_vec(&self.w1, &input, &self.b1)
            .into_iter().map(relu).collect();
        let h2: Vec<f32> = matmul_vec(&self.w2, &h1, &self.b2)
            .into_iter().map(relu).collect();
        let out = matmul_vec(&self.w_out, &h2, &self.b_out);

        ActorAction {
            delta_heading_rad: tanh_f32(out[0]) * self.config.max_heading_delta_rad,
            delta_altitude_m:  tanh_f32(out[1]) * self.config.max_altitude_delta_m,
            speed_ms:          sigmoid(out[2]) * self.config.max_speed_ms,
            trigger_csi_scan:  sigmoid(out[3]) > 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_obs() -> LocalObservation {
        LocalObservation {
            own_state: [0.5; 9],
            neighbor_relative_pos: [0.0; 18],
            grid_tile: [0.1; 25],
            csi_reading: [0.0; 5],
            task_encoding: [0.0; 7],
        }
    }

    #[test]
    fn forward_action_bounds() {
        let config = ActorConfig::default();
        let actor = MappoActor::random_init_with_dim(LocalObservation::DIM, config.clone());
        let action = actor.forward(&dummy_obs());

        assert!(action.delta_heading_rad.abs() <= config.max_heading_delta_rad + 1e-5);
        assert!(action.delta_altitude_m.abs() <= config.max_altitude_delta_m + 1e-5);
        assert!(action.speed_ms >= 0.0 && action.speed_ms <= config.max_speed_ms + 1e-5);
    }

    #[test]
    fn forward_deterministic_with_zero_weights() {
        // Manually craft an actor with zero weights so output is deterministic.
        let config = ActorConfig::default();
        let h1 = config.hidden_dims[0];
        let h2 = config.hidden_dims[1];

        let actor = MappoActor {
            w1: vec![vec![0.0; LocalObservation::DIM]; h1],
            b1: vec![0.0; h1],
            w2: vec![vec![0.0; h1]; h2],
            b2: vec![0.0; h2],
            w_out: vec![vec![0.0; h2]; 4],
            b_out: vec![0.0; 4],
            config,
        };
        let action = actor.forward(&dummy_obs());
        // tanh(0) = 0, sigmoid(0) = 0.5
        assert!((action.delta_heading_rad).abs() < 1e-6);
        assert!((action.delta_altitude_m).abs() < 1e-6);
        assert!((action.speed_ms - 4.0).abs() < 1e-4); // sigmoid(0) * 8 = 4
    }

    #[test]
    fn test_actor_action_bounds() {
        let cfg = ActorConfig::default();
        let actor = MappoActor::random_init(cfg.clone());
        let obs = LocalObservation::zeros();
        let action = actor.forward(&obs);
        assert!(action.delta_heading_rad.abs() <= cfg.max_heading_delta_rad * 1.001);
        assert!(action.delta_altitude_m.abs() <= cfg.max_altitude_delta_m * 1.001);
        assert!(action.speed_ms >= 0.0 && action.speed_ms <= cfg.max_speed_ms * 1.001);
    }

    #[test]
    fn test_actor_inference_speed() {
        let actor = MappoActor::random_init(ActorConfig::default());
        let obs = LocalObservation::zeros();
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = actor.forward(&obs);
        }
        let elapsed = start.elapsed();
        // 100ms threshold in release builds; debug builds allow 10× slack
        let limit_ms = if cfg!(debug_assertions) { 1000 } else { 100 };
        assert!(elapsed.as_millis() < limit_ms, "1000 inferences took {}ms, limit {}ms", elapsed.as_millis(), limit_ms);
    }
}
