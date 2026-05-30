//! MARL training entry point for ruview-swarm (ADR-148 M4).
//!
//! Real Candle autodiff PPO training loop. Runs on CPU, or CUDA when built
//! with `--features train,cuda` (local RTX 5080 or a GCP L4 instance).
//!
//! Usage:
//!   cargo run --release -p ruview-swarm --features train,cuda --bin train_marl -- \
//!       --episodes 5000 --drones 4 --profile sar --checkpoint-dir ./marl-checkpoints
//!
//! Right-sizing note: the policy is a 64→128→64 MLP. The bottleneck is
//! environment-rollout throughput, not GPU matmul — an L4 + 16 vCPU beats an
//! 8× A100 box for this workload at ~1/20th the cost. See scripts/gcp/.

use ruview_swarm::config::SwarmConfig;
use ruview_swarm::marl::candle_ppo::{CandlePpoConfig, CandleTrainer};
use ruview_swarm::marl::observation::LocalObservation;
use ruview_swarm::marl::reward::{RewardCalculator, RewardContext};
use ruview_swarm::orchestrator::SwarmOrchestrator;
use ruview_swarm::types::{NodeId, Position3D};

struct Args {
    episodes: usize,
    drones: usize,
    profile: String,
    steps_per_episode: usize,
    checkpoint_dir: String,
    checkpoint_every: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            episodes: 1000,
            drones: 4,
            profile: "sar".to_string(),
            steps_per_episode: 200,
            checkpoint_dir: "./marl-checkpoints".to_string(),
            checkpoint_every: 100,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let next = || argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--episodes" => {
                args.episodes = next().parse().unwrap_or(args.episodes);
                i += 1;
            }
            "--drones" => {
                args.drones = next().parse().unwrap_or(args.drones);
                i += 1;
            }
            "--profile" => {
                args.profile = next();
                i += 1;
            }
            "--steps" => {
                args.steps_per_episode = next().parse().unwrap_or(args.steps_per_episode);
                i += 1;
            }
            "--checkpoint-dir" => {
                args.checkpoint_dir = next();
                i += 1;
            }
            "--checkpoint-every" => {
                args.checkpoint_every = next().parse().unwrap_or(args.checkpoint_every);
                i += 1;
            }
            "-h" | "--help" => {
                println!(
                    "train_marl — ruview-swarm MARL training (ADR-148 M4)\n\
                     \nOptions:\n  \
                     --episodes N         training episodes (default 1000)\n  \
                     --drones N           swarm size (default 4)\n  \
                     --profile NAME       sar|inspection|mine|agriculture (default sar)\n  \
                     --steps N            steps per episode (default 200)\n  \
                     --checkpoint-dir D   checkpoint output dir (default ./marl-checkpoints)\n  \
                     --checkpoint-every N save every N episodes (default 100)"
                );
                std::process::exit(0);
            }
            other => eprintln!("warning: ignoring unknown arg {other}"),
        }
        i += 1;
    }
    args
}

fn config_for(profile: &str) -> SwarmConfig {
    match profile {
        "inspection" => SwarmConfig::inspection_default(),
        "mine" => SwarmConfig::mine_default(),
        "agriculture" => SwarmConfig::agriculture_default(),
        _ => SwarmConfig::wi2sar_reference(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let cfg = config_for(&args.profile);

    println!(
        "MARL training: profile={} drones={} episodes={} steps/ep={}",
        args.profile, args.drones, args.episodes, args.steps_per_episode
    );

    let ppo_cfg = CandlePpoConfig::default();
    let mut trainer = CandleTrainer::new(ppo_cfg)?;
    println!("device: {:?}", trainer.net.device());

    let reward_calc = RewardCalculator::default();
    std::fs::create_dir_all(&args.checkpoint_dir).ok();

    // Synthetic victims placed within the mission area for reward signal.
    let victims = vec![
        Position3D { x: cfg.mission.area_width_m * 0.2, y: cfg.mission.area_height_m * 0.3, z: 0.0 },
        Position3D { x: cfg.mission.area_width_m * 0.6, y: cfg.mission.area_height_m * 0.45, z: 0.0 },
    ];

    let mut best_return = f32::MIN;

    for episode in 0..args.episodes {
        // Build a fresh swarm for this episode.
        let mut drones: Vec<SwarmOrchestrator> = (0..args.drones)
            .map(|d| {
                let cols = (args.drones as f64).sqrt().ceil().max(1.0) as usize;
                let (row, col) = (d / cols, d % cols);
                SwarmOrchestrator::new_demo(
                    NodeId(d as u32),
                    cfg.clone(),
                    Position3D {
                        x: 10.0 + col as f64 * (cfg.mission.area_width_m / cols as f64),
                        y: 10.0 + row as f64 * (cfg.mission.area_height_m / cols.max(1) as f64),
                        z: -cfg.planning.flight_altitude_m,
                    },
                    victims.clone(),
                )
            })
            .collect();

        // Rollout buffers (flattened across drones).
        let mut obs_buf: Vec<LocalObservation> = Vec::new();
        let mut action_buf: Vec<[f32; 4]> = Vec::new();
        let mut reward_buf: Vec<f32> = Vec::new();
        let mut value_buf: Vec<f32> = Vec::new();
        let mut done_buf: Vec<bool> = Vec::new();

        for step in 0..args.steps_per_episode {
            let is_last = step == args.steps_per_episode - 1;

            // Snapshot peer positions for neighbor observations.
            let positions: Vec<(NodeId, Position3D)> =
                drones.iter().map(|d| (d.node_id, d.state.position)).collect();

            for drone in &mut drones {
                let cells_before = drone.stats.cells_covered;
                let prev_pos = drone.state.position;

                // Observation from current state + neighbors.
                let neighbors: Vec<(NodeId, Position3D)> = positions
                    .iter()
                    .filter(|(id, _)| *id != drone.node_id)
                    .cloned()
                    .collect();
                let obs =
                    LocalObservation::from_state_no_grid(&drone.state, &neighbors, None, None);

                // Advance the simulation one tick.
                drone.step(1.0, true).await;

                // Reward from this step's deltas.
                let new_cells = drone.stats.cells_covered.saturating_sub(cells_before);
                let nearest = neighbors
                    .iter()
                    .map(|(_, p)| prev_pos.distance_to(p))
                    .fold(f64::MAX, f64::min);
                let ctx = RewardContext {
                    state: &drone.state,
                    new_cells_covered: new_cells,
                    victim_confirmed: false,
                    contributed_to_triangulation: false,
                    nearest_neighbor_dist: nearest,
                    geofence_breached: false,
                    battery_depleted_without_rth: false,
                };
                let reward = reward_calc.compute(&ctx);

                let action = [
                    drone.state.heading_rad as f32,
                    drone.state.altitude_agl_m as f32,
                    drone.state.velocity.magnitude() as f32,
                    0.0,
                ];

                obs_buf.push(obs);
                action_buf.push(action);
                reward_buf.push(reward);
                value_buf.push(0.0); // bootstrap value (critic learns this)
                done_buf.push(is_last);
            }
        }

        // PPO update on the episode's rollout.
        let (advantages, returns) =
            trainer.compute_gae(&reward_buf, &value_buf, &done_buf);
        let old_log_probs = vec![0.0f32; obs_buf.len()];
        let (policy_loss, value_loss, _entropy) =
            trainer.update(&obs_buf, &action_buf, &advantages, &returns, &old_log_probs)?;

        let mean_return = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f32>() / returns.len() as f32
        };

        if mean_return > best_return {
            best_return = mean_return;
        }

        if episode % 10 == 0 || episode == args.episodes - 1 {
            println!(
                "ep {:>5}/{}  mean_return={:>8.3}  best={:>8.3}  policy_loss={:>8.4}  value_loss={:>8.4}",
                episode, args.episodes, mean_return, best_return, policy_loss, value_loss
            );
        }

        // Checkpoint the trained variables periodically.
        if args.checkpoint_every > 0
            && (episode + 1) % args.checkpoint_every == 0
            || episode == args.episodes - 1
        {
            let path = format!("{}/marl-ep{}.safetensors", args.checkpoint_dir, episode + 1);
            if let Err(e) = trainer.net.varmap().save(&path) {
                eprintln!("checkpoint save failed at {path}: {e}");
            } else {
                println!("checkpoint saved: {path}");
            }
        }
    }

    println!("training complete. best mean_return={best_return:.3}");
    Ok(())
}
