use criterion::{criterion_group, criterion_main, Criterion};
use ruview_swarm::marl::{MappoActor, ActorConfig};
use ruview_swarm::marl::LocalObservation;
use ruview_swarm::sensing::MultiViewFusion;
use ruview_swarm::planning::RrtApfPlanner;
use ruview_swarm::demo::{DemoScenario};
use ruview_swarm::types::{CsiDetection, NodeId, Position3D};

fn bench_marl_inference(c: &mut Criterion) {
    let actor = MappoActor::random_init(ActorConfig::default());
    let obs = LocalObservation::zeros();
    c.bench_function("marl_actor_inference", |b| b.iter(|| actor.forward(&obs)));
}

fn bench_rrt_apf_plan(c: &mut Criterion) {
    let planner = RrtApfPlanner::new(3.0);
    let start = Position3D { x: 0.0, y: 0.0, z: -30.0 };
    let goal  = Position3D { x: 50.0, y: 50.0, z: -30.0 };
    c.bench_function("rrt_apf_100iter", |b| b.iter(|| {
        let mut rng = rand::thread_rng();
        planner.plan(start, goal, 100, &mut rng)
    }));
}

fn bench_multiview_fusion(c: &mut Criterion) {
    let fusion = MultiViewFusion::default();
    let detections = vec![
        CsiDetection { drone_id: NodeId(0), confidence: 0.85, victim_position: Some(Position3D { x: 51.0, y: 49.0, z: 0.0 }), timestamp_ms: 0 },
        CsiDetection { drone_id: NodeId(1), confidence: 0.78, victim_position: Some(Position3D { x: 49.0, y: 51.0, z: 0.0 }), timestamp_ms: 0 },
        CsiDetection { drone_id: NodeId(2), confidence: 0.92, victim_position: Some(Position3D { x: 50.0, y: 50.0, z: 0.0 }), timestamp_ms: 0 },
    ];
    let positions = vec![
        (NodeId(0), Position3D { x: 0.0,   y: 0.0,  z: -30.0 }),
        (NodeId(1), Position3D { x: 100.0, y: 0.0,  z: -30.0 }),
        (NodeId(2), Position3D { x: 50.0,  y: 86.6, z: -30.0 }),
    ];
    c.bench_function("multiview_fusion_3drones", |b| b.iter(|| fusion.fuse(&detections, &positions)));
}

fn bench_demo_coverage_estimate(c: &mut Criterion) {
    let scenario = DemoScenario::sar_rubble_field(4);
    c.bench_function("demo_coverage_estimate", |b| b.iter(|| scenario.estimate_coverage_time_secs()));
}

fn bench_ppo_update(c: &mut Criterion) {
    use ruview_swarm::marl::{MappoActor, ActorConfig, LocalObservation};
    use ruview_swarm::marl::training_loop::{ReplayBuffer, Transition, PpoConfig, ppo_update};
    use ruview_swarm::marl::actor::ActorAction;

    let mut buf = ReplayBuffer::new(64);
    for i in 0..64 {
        buf.push(Transition {
            obs: LocalObservation::zeros(),
            action: ActorAction { delta_heading_rad: 0.1, delta_altitude_m: 0.0, speed_ms: 5.0, trigger_csi_scan: true },
            reward: if i % 2 == 0 { 10.0 } else { -2.0 },
            next_obs: LocalObservation::zeros(),
            done: i == 63,
        });
    }
    let cfg = PpoConfig::default();
    c.bench_function("ppo_update_64transitions", |b| {
        b.iter(|| {
            let mut actor = MappoActor::random_init(ActorConfig::default());
            ppo_update(&mut actor, &buf, &cfg)
        })
    });
}

criterion_group!(benches, bench_marl_inference, bench_rrt_apf_plan, bench_multiview_fusion, bench_demo_coverage_estimate, bench_ppo_update);
criterion_main!(benches);
