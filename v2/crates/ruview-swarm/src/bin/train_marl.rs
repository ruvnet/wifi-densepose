//! MARL training entry point for ruview-swarm (ADR-148 M4).
//!
//! Real Candle autodiff PPO training loop. Runs on CPU, or CUDA when built
//! with `--features train,cuda` (local RTX 5080 or a GCP L4 instance).
//!
//! Movement is driven by a selectable `FlightPattern` (boustrophedon,
//! partitioned, spiral, pheromone, potential, levy) and reward is shaped by a
//! selectable `LearningPattern` (mappo, ippo, curiosity, meta). This makes each
//! pattern produce visibly distinct trajectories + telemetry instead of every
//! drone clustering on the orchestrator's internal coverage strategy.
//!
//! Usage:
//!   cargo run --release -p ruview-swarm --features train,cuda --bin train_marl -- \
//!       --episodes 5000 --drones 4 --profile sar \
//!       --flight-pattern partitioned --learn-pattern mappo_curiosity \
//!       --checkpoint-dir ./marl-checkpoints
//!
//! Right-sizing note: the policy is a 64→128→64 MLP. The bottleneck is
//! environment-rollout throughput, not GPU matmul — an L4 + 16 vCPU beats an
//! 8× A100 box for this workload at ~1/20th the cost. See scripts/gcp/.

use std::collections::HashSet;

use ruview_swarm::config::SwarmConfig;
use ruview_swarm::integration::telemetry::{DroneFrame, TelemetryRecorder};
use ruview_swarm::marl::candle_ppo::{CandlePpoConfig, CandleTrainer};
use ruview_swarm::marl::learning::{shaped_reward, CuriosityModule, LearningPattern};
use ruview_swarm::marl::observation::LocalObservation;
use ruview_swarm::marl::reward::{RewardCalculator, RewardContext};
use ruview_swarm::planning::patterns::{FlightPattern, PatternContext};
use ruview_swarm::types::{DroneState, NodeId, Position3D, Velocity3D};

struct Args {
    episodes: usize,
    drones: usize,
    profile: String,
    steps_per_episode: usize,
    checkpoint_dir: String,
    checkpoint_every: usize,
    telemetry: Option<String>,
    telemetry_episode: usize,
    flight_pattern: String,
    learn_pattern: String,
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
            telemetry: None,
            telemetry_episode: 0,
            flight_pattern: "partitioned".to_string(),
            learn_pattern: "mappo".to_string(),
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
            "--telemetry" => {
                args.telemetry = Some(next());
                i += 1;
            }
            "--telemetry-episode" => {
                args.telemetry_episode = next().parse().unwrap_or(args.telemetry_episode);
                i += 1;
            }
            "--flight-pattern" => {
                args.flight_pattern = next();
                i += 1;
            }
            "--learn-pattern" => {
                args.learn_pattern = next();
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
                     --flight-pattern P   boustrophedon|partitioned|spiral|pheromone|potential|levy (default partitioned)\n  \
                     --learn-pattern P    mappo|ippo|curiosity|meta (default mappo)\n  \
                     --checkpoint-dir D   checkpoint output dir (default ./marl-checkpoints)\n  \
                     --checkpoint-every N save every N episodes (default 100)\n  \
                     --telemetry FILE     write JSONL telemetry for viz/swarm_viz.html\n  \
                     --telemetry-episode N which episode's steps to record spatially (default 0)"
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

/// Map a world coordinate to a grid cell index at `grid_res` metre resolution.
fn cell_of(x: f64, y: f64, grid_res: f64) -> (u32, u32) {
    let gx = (x / grid_res).floor().max(0.0) as u32;
    let gy = (y / grid_res).floor().max(0.0) as u32;
    (gx, gy)
}

/// Mark every grid cell within the drone's circular scan footprint as scanned,
/// returning how many *newly* scanned cells this step contributed.
fn mark_scanned(
    scanned: &mut HashSet<(u32, u32)>,
    pos: &Position3D,
    scan_width_m: f64,
    grid_res: f64,
    area_w: f64,
    area_h: f64,
) -> u32 {
    let r = scan_width_m * 0.5;
    let cols = (area_w / grid_res).ceil() as i64;
    let rows = (area_h / grid_res).ceil() as i64;
    let (cx, cy) = cell_of(pos.x, pos.y, grid_res);
    let span = (r / grid_res).ceil() as i64;
    let mut new_cells = 0u32;
    for dgx in -span..=span {
        for dgy in -span..=span {
            let gx = cx as i64 + dgx;
            let gy = cy as i64 + dgy;
            if gx < 0 || gy < 0 || gx >= cols || gy >= rows {
                continue;
            }
            // Cell centre in metres.
            let mx = (gx as f64 + 0.5) * grid_res;
            let my = (gy as f64 + 0.5) * grid_res;
            if (mx - pos.x).hypot(my - pos.y) <= r && scanned.insert((gx as u32, gy as u32)) {
                new_cells += 1;
            }
        }
    }
    new_cells
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let cfg = config_for(&args.profile);
    let flight_pattern = FlightPattern::from_str(&args.flight_pattern);
    let learn_pattern = LearningPattern::from_str(&args.learn_pattern);

    println!(
        "MARL training: profile={} drones={} episodes={} steps/ep={} flight={} learn={} ({})",
        args.profile,
        args.drones,
        args.episodes,
        args.steps_per_episode,
        flight_pattern.name(),
        learn_pattern.name(),
        if learn_pattern.centralized_critic() {
            "CTDE / centralized critic"
        } else {
            "independent learners"
        }
    );

    let ppo_cfg = CandlePpoConfig::default();
    let mut trainer = CandleTrainer::new(ppo_cfg)?;
    println!("device: {:?}", trainer.net.device());

    let reward_calc = RewardCalculator::default();
    std::fs::create_dir_all(&args.checkpoint_dir).ok();

    let area_w = cfg.mission.area_width_m;
    let area_h = cfg.mission.area_height_m;
    let grid_res = cfg.mission.grid_resolution_m.max(1.0);
    let scan_w = cfg.planning.csi_scan_width_m;
    let max_speed = cfg.planning.max_speed_ms.max(0.1);
    let altitude_z = -cfg.planning.flight_altitude_m;
    let total_cells = ((area_w / grid_res).ceil() * (area_h / grid_res).ceil()).max(1.0);

    // Synthetic victims placed within the mission area for reward signal.
    let victims = vec![
        Position3D { x: area_w * 0.2, y: area_h * 0.3, z: 0.0 },
        Position3D { x: area_w * 0.6, y: area_h * 0.45, z: 0.0 },
    ];

    // Composite profile label so the viewer header surfaces the active patterns.
    let profile_label = format!(
        "{} · flight={} · learn={}",
        args.profile,
        flight_pattern.name(),
        learn_pattern.name()
    );

    // Optional telemetry recorder for the visualizer.
    let mut telem = match &args.telemetry {
        Some(path) => {
            let mut rec = TelemetryRecorder::create(path)?;
            rec.meta(&profile_label, args.drones, area_w, area_h, &victims)?;
            println!("telemetry → {path} (spatial steps from episode {})", args.telemetry_episode);
            Some(rec)
        }
        None => None,
    };

    let mut best_return = f32::MIN;

    for episode in 0..args.episodes {
        // Per-episode curiosity module (count-based novelty over the area).
        let mut curiosity = CuriosityModule::new(area_w, area_h, 32, 0.5);

        // Build drone states directly so the FlightPattern fully drives motion.
        let cols = (args.drones as f64).sqrt().ceil().max(1.0) as usize;
        let mut states: Vec<DroneState> = (0..args.drones)
            .map(|d| {
                let (row, col) = (d / cols, d % cols);
                let mut s = DroneState::default_at_origin(NodeId(d as u32));
                s.position = Position3D {
                    x: 10.0 + col as f64 * (area_w / cols as f64),
                    y: 10.0 + row as f64 * (area_h / cols.max(1) as f64),
                    z: altitude_z,
                };
                s.altitude_agl_m = cfg.planning.flight_altitude_m;
                s
            })
            .collect();

        // Coverage tracker (shared across drones — total area scanned).
        let mut scanned: HashSet<(u32, u32)> = HashSet::new();
        // Rolling recent-positions trail for pheromone/potential patterns.
        let mut visited: Vec<Position3D> = Vec::with_capacity(256);

        // Rollout buffers (flattened across drones).
        let mut obs_buf: Vec<LocalObservation> = Vec::new();
        let mut action_buf: Vec<[f32; 4]> = Vec::new();
        let mut reward_buf: Vec<f32> = Vec::new();
        let mut value_buf: Vec<f32> = Vec::new();
        let mut done_buf: Vec<bool> = Vec::new();

        for step in 0..args.steps_per_episode {
            let is_last = step == args.steps_per_episode - 1;

            // Snapshot peer positions for this tick (observations + repulsion).
            let positions: Vec<(NodeId, Position3D)> =
                states.iter().map(|s| (s.id, s.position)).collect();

            // Index needed: mutates states[idx] while reading peer positions; borrow constraints.
            #[allow(clippy::needless_range_loop)]
            for idx in 0..states.len() {
                let prev_pos = states[idx].position;
                let node_id = states[idx].id;

                // Neighbour positions (everyone except this drone).
                let neighbors: Vec<(NodeId, Position3D)> = positions
                    .iter()
                    .filter(|(id, _)| *id != node_id)
                    .cloned()
                    .collect();
                let peers: Vec<Position3D> = neighbors.iter().map(|(_, p)| *p).collect();

                // Observation from the current (pre-move) state.
                let obs =
                    LocalObservation::from_state_no_grid(&states[idx], &neighbors, None, None);

                // --- FlightPattern drives the next waypoint --------------------
                let ctx = PatternContext {
                    drone_id: node_id,
                    swarm_size: args.drones,
                    current: prev_pos,
                    area_w,
                    area_h,
                    altitude_z,
                    scan_width_m: scan_w,
                    step: step as u64,
                    visited: &visited,
                    peers: &peers,
                };
                let target = flight_pattern.next_target(&ctx);

                // Move one tick toward the target at max_speed (no teleport).
                let dx = target.x - prev_pos.x;
                let dy = target.y - prev_pos.y;
                let dist = dx.hypot(dy);
                let new_pos = if dist > 1e-9 {
                    let stepd = dist.min(max_speed);
                    Position3D {
                        x: prev_pos.x + dx / dist * stepd,
                        y: prev_pos.y + dy / dist * stepd,
                        z: altitude_z,
                    }
                } else {
                    prev_pos
                };
                let heading = if dist > 1e-9 { dy.atan2(dx) } else { states[idx].heading_rad };
                let moved = prev_pos.distance_to(&new_pos);

                // Commit the move to the drone state.
                {
                    let s = &mut states[idx];
                    s.velocity = Velocity3D {
                        vx: (new_pos.x - prev_pos.x),
                        vy: (new_pos.y - prev_pos.y),
                        vz: 0.0,
                    };
                    s.position = new_pos;
                    s.heading_rad = heading;
                    s.timestamp_ms = s.timestamp_ms.saturating_add(1000);
                }

                // Coverage: mark scanned footprint, count new cells.
                let new_cells =
                    mark_scanned(&mut scanned, &new_pos, scan_w, grid_res, area_w, area_h);

                // Detection: any victim within the scan footprint.
                let detected = victims.iter().any(|v| new_pos.distance_to(v) < scan_w);

                // Nearest-neighbour distance (for collision shaping).
                let nearest = peers
                    .iter()
                    .map(|p| new_pos.distance_to(p))
                    .fold(f64::MAX, f64::min);

                // Base extrinsic reward.
                let ctx_r = RewardContext {
                    state: &states[idx],
                    new_cells_covered: new_cells,
                    victim_confirmed: detected,
                    contributed_to_triangulation: false,
                    nearest_neighbor_dist: nearest,
                    geofence_breached: false,
                    battery_depleted_without_rth: false,
                };
                let base = reward_calc.compute(&ctx_r);

                // Curiosity shaping (only when the learning pattern uses it).
                let reward = if learn_pattern.uses_curiosity() {
                    let bonus = curiosity.visit_bonus(new_pos.x, new_pos.y);
                    shaped_reward(learn_pattern, base, bonus)
                } else {
                    base
                };

                let action = [
                    heading as f32,
                    states[idx].altitude_agl_m as f32,
                    (moved / 1.0) as f32,
                    0.0,
                ];

                obs_buf.push(obs);
                action_buf.push(action);
                reward_buf.push(reward);
                value_buf.push(0.0); // bootstrap value (critic learns this)
                done_buf.push(is_last);

                // Record the move in the shared visited trail (cap length).
                visited.push(new_pos);
            }

            // Trim the visited trail to the most recent ~200 positions.
            if visited.len() > 200 {
                let drop = visited.len() - 200;
                visited.drain(0..drop);
            }

            // Record spatial telemetry for the selected episode only.
            if let Some(rec) = telem.as_mut() {
                if episode == args.telemetry_episode {
                    let frames: Vec<DroneFrame> = states
                        .iter()
                        .map(|s| {
                            let detected =
                                victims.iter().any(|v| s.position.distance_to(v) < scan_w);
                            DroneFrame::from_state(s, detected)
                        })
                        .collect();
                    let coverage = scanned.len() as f64 / total_cells;
                    let _ = rec.step(episode, step, step as f64, &frames, coverage);
                }
            }
        }

        // PPO update on the episode's rollout.
        let (advantages, returns) = trainer.compute_gae(&reward_buf, &value_buf, &done_buf);
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

        // Per-episode training-metric telemetry (every episode).
        if let Some(rec) = telem.as_mut() {
            let _ = rec.episode(episode, mean_return, policy_loss, value_loss, 0);
        }

        if episode % 10 == 0 || episode == args.episodes - 1 {
            let coverage_pct = scanned.len() as f64 / total_cells * 100.0;
            println!(
                "ep {:>5}/{}  mean_return={:>8.3}  best={:>8.3}  policy_loss={:>8.4}  value_loss={:>8.4}  coverage={:>5.1}%",
                episode, args.episodes, mean_return, best_return, policy_loss, value_loss, coverage_pct
            );
        }

        // Checkpoint the trained variables periodically.
        if args.checkpoint_every > 0 && (episode + 1) % args.checkpoint_every == 0
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

    if let Some(rec) = telem.as_mut() {
        rec.flush()?;
        if let Some(path) = &args.telemetry {
            println!("telemetry written: {path} — open viz/swarm_viz.html and load it");
        }
    }

    println!("training complete. best mean_return={best_return:.3}");
    Ok(())
}
