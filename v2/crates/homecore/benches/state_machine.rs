//! Criterion benchmarks for the HOMECORE state-machine hot paths.
//!
//! Run with:
//!
//!     cargo bench -p homecore --bench state_machine
//!
//! Hot paths covered:
//! - `set` first-time-write (cold path: insert + allocate + broadcast)
//! - `set` repeat-write (warm path: same entity, fires broadcast)
//! - `set` no-op (suppress path: same state + same attrs, no broadcast)
//! - `get` (zero-copy Arc<State> clone)
//! - `all` snapshot (allocates Vec; REST GET /api/states path)
//! - `all_by_domain` filter
//! - Broadcast fan-out: 1 sender + N subscribers

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tokio::runtime::Runtime;

use homecore::{Context, EntityId, StateMachine};

fn bench_set_first_write(c: &mut Criterion) {
    let mut g = c.benchmark_group("set");
    g.throughput(Throughput::Elements(1));
    g.bench_function("first_write", |b| {
        b.iter_with_setup(
            || (StateMachine::new(), EntityId::parse("light.benchmark").unwrap()),
            |(sm, id)| {
                sm.set(
                    id,
                    black_box("on"),
                    black_box(serde_json::json!({"brightness": 200})),
                    Context::new(),
                )
            },
        )
    });
    g.finish();
}

fn bench_set_warm_write(c: &mut Criterion) {
    let sm = StateMachine::new();
    let id = EntityId::parse("light.benchmark").unwrap();
    // Prime the entry
    sm.set(id.clone(), "off", serde_json::json!({}), Context::new());

    let mut g = c.benchmark_group("set");
    g.throughput(Throughput::Elements(1));
    g.bench_function("warm_write_state_change", |b| {
        let mut toggle = false;
        b.iter(|| {
            toggle = !toggle;
            let v = if toggle { "on" } else { "off" };
            sm.set(
                id.clone(),
                black_box(v),
                black_box(serde_json::json!({"toggle": toggle})),
                Context::new(),
            )
        });
    });
    g.finish();
}

fn bench_set_noop(c: &mut Criterion) {
    let sm = StateMachine::new();
    let id = EntityId::parse("light.benchmark").unwrap();
    sm.set(id.clone(), "on", serde_json::json!({"brightness": 200}), Context::new());

    let mut g = c.benchmark_group("set");
    g.throughput(Throughput::Elements(1));
    g.bench_function("noop_suppressed", |b| {
        b.iter(|| {
            sm.set(
                id.clone(),
                black_box("on"),
                black_box(serde_json::json!({"brightness": 200})),
                Context::new(),
            )
        });
    });
    g.finish();
}

fn bench_get(c: &mut Criterion) {
    let sm = StateMachine::new();
    let id = EntityId::parse("sensor.temperature").unwrap();
    sm.set(id.clone(), "20.5", serde_json::json!({"unit": "C"}), Context::new());

    let mut g = c.benchmark_group("get");
    g.throughput(Throughput::Elements(1));
    g.bench_function("hit", |b| {
        b.iter(|| {
            let _ = black_box(sm.get(&id));
        });
    });
    g.bench_function("miss", |b| {
        let missing = EntityId::parse("sensor.missing").unwrap();
        b.iter(|| {
            let _ = black_box(sm.get(&missing));
        });
    });
    g.finish();
}

fn bench_all_snapshot(c: &mut Criterion) {
    let mut g = c.benchmark_group("all_snapshot");
    for n_entities in [10, 100, 1000].iter() {
        let sm = StateMachine::new();
        for i in 0..*n_entities {
            let id = EntityId::parse(format!("sensor.entity_{}", i)).unwrap();
            sm.set(id, "on", serde_json::json!({"i": i}), Context::new());
        }
        g.throughput(Throughput::Elements(*n_entities as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(n_entities),
            n_entities,
            |b, _| {
                b.iter(|| black_box(sm.all()));
            },
        );
    }
    g.finish();
}

fn bench_all_by_domain(c: &mut Criterion) {
    let sm = StateMachine::new();
    // 100 entities split across 5 domains
    for i in 0..100 {
        let domain = match i % 5 {
            0 => "light",
            1 => "sensor",
            2 => "switch",
            3 => "binary_sensor",
            _ => "automation",
        };
        let id = EntityId::parse(format!("{}.e_{}", domain, i)).unwrap();
        sm.set(id, "on", serde_json::json!({}), Context::new());
    }

    c.bench_function("all_by_domain_light_20_of_100", |b| {
        b.iter(|| black_box(sm.all_by_domain("light")));
    });
}

fn bench_broadcast_fan_out(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut g = c.benchmark_group("broadcast_fan_out");
    for n_subscribers in [1, 4, 16, 64].iter() {
        g.throughput(Throughput::Elements(*n_subscribers as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(n_subscribers),
            n_subscribers,
            |b, &n| {
                b.iter_custom(|iters| {
                    rt.block_on(async {
                        let sm = StateMachine::new();
                        let id = Arc::new(EntityId::parse("light.fanout").unwrap());

                        // Spawn N subscribers
                        let mut handles = Vec::new();
                        for _ in 0..n {
                            let mut rx = sm.subscribe();
                            handles.push(tokio::spawn(async move {
                                for _ in 0..iters {
                                    let _ = rx.recv().await;
                                }
                            }));
                        }

                        let start = std::time::Instant::now();
                        for i in 0..iters {
                            let v = if i % 2 == 0 { "on" } else { "off" };
                            sm.set(
                                (*id).clone(),
                                v,
                                serde_json::json!({"i": i}),
                                Context::new(),
                            );
                        }
                        for h in handles {
                            let _ = h.await;
                        }
                        start.elapsed()
                    })
                });
            },
        );
    }
    g.finish();
}

criterion_group! {
    name = state_machine;
    config = Criterion::default().sample_size(20);
    targets = bench_set_first_write,
              bench_set_warm_write,
              bench_set_noop,
              bench_get,
              bench_all_snapshot,
              bench_all_by_domain,
              bench_broadcast_fan_out
}
criterion_main!(state_machine);
