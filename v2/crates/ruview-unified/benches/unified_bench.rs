//! Criterion benchmarks for the unified RF spatial world model hot paths
//! (ADR-273 §7): encoder inference (the edge-latency budget), tokenization
//! (with vs without precomputed DFT twiddles), Gaussian map queries (spatial
//! hash vs linear-scan baseline), channel-gain evaluation, and synthetic
//! window generation.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use ruview_unified::encoder::{EncoderConfig, RfEncoder};
use ruview_unified::gaussian::map::GaussianMap;
use ruview_unified::gaussian::primitive::{Provenance, RfGaussian};
use ruview_unified::gaussian::{channel_gain, observe_link};
use ruview_unified::math::{dft_magnitudes, DftPlan};
use ruview_unified::synth::{SynthConfig, SynthGenerator};
use ruview_unified::tokenizer::RfTokenizer;

fn synth_corpus() -> Vec<ruview_unified::synth::LabeledWindow> {
    SynthGenerator::new(SynthConfig {
        seed: 11,
        n_rooms: 2,
        windows_per_room: 4,
        links: 3,
        snapshot_dt_s: 0.05,
    })
    .generate()
}

fn bench_encoder_forward(c: &mut Criterion) {
    let corpus = synth_corpus();
    let tokenizer = RfTokenizer::new();
    let window = tokenizer.tokenize(&corpus[0].tensor);
    let encoder = RfEncoder::new(EncoderConfig::default(), 42);
    c.bench_function("encoder_encode_window_21tok_d128", |b| {
        b.iter(|| std::hint::black_box(encoder.encode(&window)));
    });
}

fn bench_tokenizer(c: &mut Criterion) {
    let corpus = synth_corpus();
    let tokenizer = RfTokenizer::new();
    c.bench_function("tokenize_3link_56bin_8snap", |b| {
        b.iter(|| std::hint::black_box(tokenizer.tokenize(&corpus[0].tensor)));
    });
}

fn bench_dft_plan_vs_naive(c: &mut Criterion) {
    use num_complex::Complex64;
    let x: Vec<Complex64> =
        (0..8).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect();
    let plan = DftPlan::new(8, 4);
    c.bench_function("dft8x4_naive", |b| {
        b.iter(|| std::hint::black_box(dft_magnitudes(&x, 4)));
    });
    c.bench_function("dft8x4_planned", |b| {
        b.iter(|| std::hint::black_box(plan.magnitudes(&x)));
    });
}

fn populated_map(n_side: usize) -> GaussianMap {
    let mut map = GaussianMap::new(1.0);
    for x in 0..n_side {
        for y in 0..n_side {
            let g = RfGaussian::new(
                [x as f64 * 1.5, y as f64 * 1.5, 1.0],
                [0.3, 0.3, 0.3],
                [1.0, 0.0, 0.0, 0.0],
                0.4,
                0.9,
                0,
                600.0,
                Provenance { device_id: "bench".into(), model_version: 1, synthetic: true },
            )
            .expect("valid");
            map.insert(g);
        }
    }
    map
}

fn bench_gaussian_map(c: &mut Criterion) {
    // Two sizes to show the hash/linear crossover honestly: at 1,024
    // Gaussians a brute-force scan is competitive for small-radius queries;
    // at 16,384 the hash wins decisively.
    for (label, side) in [("1k", 32usize), ("16k", 128)] {
        let map = populated_map(side);
        let centre = [side as f64 * 0.75, side as f64 * 0.75, 1.0];
        c.bench_function(&format!("map{label}_query_radius_hash"), |b| {
            b.iter(|| std::hint::black_box(map.query_radius(centre, 3.0)));
        });
        c.bench_function(&format!("map{label}_query_radius_linear"), |b| {
            b.iter(|| std::hint::black_box(map.query_radius_linear(centre, 3.0)));
        });
        c.bench_function(&format!("map{label}_segment_corridor_hash"), |b| {
            b.iter(|| {
                std::hint::black_box(map.query_near_segment(
                    [0.0, 0.0, 1.0],
                    [10.0, 10.0, 1.0],
                    3.0,
                ))
            });
        });
        c.bench_function(&format!("map{label}_segment_corridor_linear"), |b| {
            b.iter(|| {
                std::hint::black_box(map.query_near_segment_linear(
                    [0.0, 0.0, 1.0],
                    [10.0, 10.0, 1.0],
                    3.0,
                ))
            });
        });
        c.bench_function(&format!("map{label}_channel_gain"), |b| {
            b.iter(|| {
                std::hint::black_box(channel_gain(
                    &map,
                    [0.0, 0.0, 1.0],
                    [10.0, 10.0, 1.0],
                    2.437e9,
                ))
            });
        });
    }
    c.bench_function("map_observe_link_inverse_update", |b| {
        b.iter_batched(
            || populated_map(8),
            |mut m| {
                std::hint::black_box(observe_link(
                    &mut m,
                    [0.0, 0.0, 1.0],
                    [10.0, 10.0, 1.0],
                    2.437e9,
                    1e-4,
                    0.7,
                    1,
                ))
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_increment2_paths(c: &mut Criterion) {
    use num_complex::Complex64;
    use ruview_unified::adapters::{ble_cs_range, BleCsFrame};
    use ruview_unified::control::{
        ActiveSensingPlanner, CoherentSensorGroup, MemberSyncState, SpatialStateFreshness,
        SpatialZone,
    };
    use ruview_unified::frame::{
        AntennaElement, CalibrationState, EvidenceLevel, FieldAxis, FrameProvenance, PhaseState,
        Pose3, ProvenanceClass, RfFrameV2, SignalQuality,
    };
    use ruview_unified::heads::FactorizedPoseHead;
    use ruview_unified::tensor::{LinkGeometry, RfModality};

    // Delay-Doppler: separable vs direct.
    let corpus = synth_corpus();
    c.bench_function("delay_doppler_separable_56x8", |b| {
        b.iter(|| std::hint::black_box(corpus[0].tensor.delay_doppler_map(0).unwrap()));
    });
    c.bench_function("delay_doppler_direct_56x8", |b| {
        b.iter(|| std::hint::black_box(corpus[0].tensor.delay_doppler_map_direct(0).unwrap()));
    });

    // Native frame → canonical derived view (114 subcarriers, 12 snapshots).
    let n = 114 * 12;
    let frame = RfFrameV2::new(
        1,
        0,
        RfModality::WifiCsi,
        vec![FieldAxis::Antenna, FieldAxis::Frequency, FieldAxis::Time],
        2.437e9,
        20e6,
        100.0,
        vec![1, 114, 12],
        (0..n).map(|i| Complex64::new(1.0 + 0.01 * (i % 13) as f64, 0.001 * (i % 7) as f64)).collect(),
        vec![true; n],
        Some(Pose3 { position_m: [0.0, 0.0, 2.0], orientation: [1.0, 0.0, 0.0, 0.0] }),
        Some(Pose3 { position_m: [4.0, 0.0, 2.0], orientation: [1.0, 0.0, 0.0, 0.0] }),
        vec![AntennaElement { position_m: [0.0; 3], gain_dbi: 2.0 }],
        5_000_000,
        CalibrationState {
            phase_state: PhaseState::Raw,
            gain_calibrated: false,
            clock_ppm: 12.0,
            baseline_id: None,
            confidence: 0.8,
        },
        SignalQuality { rssi_dbm: -45.0, noise_floor_dbm: -92.0, packet_loss: 0.02, interference: 0.05 },
        FrameProvenance {
            class: ProvenanceClass::Measured,
            evidence: EvidenceLevel::L2Lab,
            device_id: "bench".into(),
            firmware: "fw".into(),
            receipt_id: 1,
        },
    )
    .expect("valid frame");
    let geo = vec![LinkGeometry { tx_pos: [0.0, 0.0, 2.0], rx_pos: [4.0, 0.0, 2.0] }];
    c.bench_function("rfframe_to_canonical_114x12", |b| {
        b.iter(|| std::hint::black_box(frame.to_canonical(geo.clone()).unwrap()));
    });

    // BLE CS ranging (40 steps).
    let cs = {
        let c_light = 299_792_458.0;
        let steps: Vec<f64> = (0..40).map(|k| 2.402e9 + 1e6 * k as f64).collect();
        let phases: Vec<f64> = steps
            .iter()
            .map(|f| (-4.0 * std::f64::consts::PI * f * 5.0 / c_light)
                .rem_euclid(2.0 * std::f64::consts::PI))
            .collect();
        BleCsFrame { frequency_steps_hz: steps, phase_samples_rad: phases, round_trip_time_ns: Some(33.36) }
    };
    c.bench_function("ble_cs_range_40steps", |b| {
        b.iter(|| std::hint::black_box(ble_cs_range(&cs).unwrap()));
    });

    // AoI planner over 200 regions.
    let mut planner = ActiveSensingPlanner::new(0.01);
    for i in 0..200 {
        planner.upsert_region(SpatialStateFreshness {
            region_id: format!("r{i}"),
            region: SpatialZone { id: format!("r{i}"), min_m: [0.0; 3], max_m: [5.0; 3] },
            last_observed_ns: (i as u64) * 1_000_000,
            expected_change_rate: 0.01 + (i % 7) as f64 * 0.05,
            uncertainty_growth_rate: 0.1,
            business_criticality: 1.0 + (i % 3) as f64,
            sensing_cost: 1.0,
        });
    }
    c.bench_function("aoi_planner_next_action_200regions", |b| {
        b.iter(|| {
            std::hint::black_box(planner.next_action(60_000_000_000, RfModality::WifiCsi))
        });
    });

    // Coherent fusion gate, 32 members.
    let group = CoherentSensorGroup {
        group_id: "g".into(),
        members: (0..32).map(|i| format!("ap-{i}")).collect(),
        maximum_time_error_ns: 50.0,
        maximum_phase_error_rad: 0.2,
        baseline_geometry_hash: 7,
    };
    let states: Vec<MemberSyncState> = (0..32)
        .map(|i| MemberSyncState {
            member_id: format!("ap-{i}"),
            time_error_ns: 10.0,
            phase_error_rad: 0.05,
            geometry_hash: 7,
        })
        .collect();
    c.bench_function("coherent_group_can_fuse_32members", |b| {
        b.iter(|| std::hint::black_box(group.can_fuse(&states).is_ok()));
    });

    // Factorized pose predict at deployment dims.
    let head = FactorizedPoseHead::new(152, 128, 2);
    let content = vec![0.1f64; 152];
    let full = vec![0.1f64; 128];
    c.bench_function("factorized_pose_predict_d152_d128", |b| {
        b.iter(|| std::hint::black_box(head.predict(&content, &full)));
    });
}

fn bench_synth_generation(c: &mut Criterion) {
    c.bench_function("synth_generate_1room_4windows_3links", |b| {
        b.iter(|| {
            std::hint::black_box(
                SynthGenerator::new(SynthConfig {
                    seed: 3,
                    n_rooms: 1,
                    windows_per_room: 4,
                    links: 3,
                    snapshot_dt_s: 0.05,
                })
                .generate(),
            )
        });
    });
}

criterion_group!(
    benches,
    bench_encoder_forward,
    bench_tokenizer,
    bench_dft_plan_vs_naive,
    bench_gaussian_map,
    bench_increment2_paths,
    bench_synth_generation
);
criterion_main!(benches);
