//! End-to-end acceptance pipeline for the unified RF spatial world model
//! (ADR-273 §6), on SYNTHETIC data (ADR-276 generator — every number below
//! is a synthetic-world result until validated on measured captures).
//!
//! The pipeline exercised is the deployment pipeline, not a shortcut:
//! physics-simulated CSI → canonical tensor → tokenizer → self-supervised
//! masked-reconstruction pretraining (labels never touch the encoder) →
//! frozen encoder → ≤1 %-budget presence adapter → strict anti-leakage
//! evaluation → policy-wrapped export.
//!
//! Acceptance gates checked here (synthetic analogues of ADR-273 §6):
//! 1. presence F1 ≥ 0.90 on completely held-out rooms;
//! 2. the same gate under a held-out *chipset* split;
//! 3. known→unknown relative degradation < 20 %;
//! 4. adapters < 1 % of backbone parameters;
//! 5. p95 tokenize+encode latency < 50 ms;
//! 6. every exported output carries uncertainty, provenance, model version,
//!    and purpose, and leaves only through the policy trust boundary.

use std::time::Instant;

use ruview_unified::encoder::{EncoderConfig, RfEncoder};
use ruview_unified::eval::{
    expected_calibration_error, f1_score, relative_degradation, selective_metrics, PartitionDim,
    StrictSplit,
};
use ruview_unified::gaussian::primitive::Provenance;
use ruview_unified::heads::{within_adapter_budget, PresenceHead};
use ruview_unified::policy::{
    BoundedEvent, EventValue, PolicyEngine, PrivacyZone, SensingPurpose, TrustBoundary,
};
use ruview_unified::pretrain::{pretrain, PretrainConfig};
use ruview_unified::synth::{LabeledWindow, SynthConfig, SynthGenerator};
use ruview_unified::tokenizer::{RfTokenizer, TokenizedWindow};

/// Shared fixture: corpus, tokenized windows, encoder config.
struct Fixture {
    corpus: Vec<LabeledWindow>,
    windows: Vec<TokenizedWindow>,
    cfg: EncoderConfig,
}

fn build_fixture() -> Fixture {
    let corpus = SynthGenerator::new(SynthConfig {
        seed: 273_273,
        n_rooms: 8,
        windows_per_room: 20,
        links: 3,
        snapshot_dt_s: 0.05,
    })
    .generate();
    let tokenizer = RfTokenizer::new();
    let windows: Vec<TokenizedWindow> =
        corpus.iter().map(|w| tokenizer.tokenize(&w.tensor)).collect();
    // d_model = 64 keeps the debug-profile test fast; the ≤1 % adapter
    // budget is checked against THIS backbone, not a larger one.
    let cfg = EncoderConfig { d_model: 64, ..EncoderConfig::default() };
    Fixture { corpus, windows, cfg }
}

/// Pretrains on the training side only, trains a presence head on frozen
/// representations, and returns (known-condition F1, held-out F1, held-out
/// probabilities and labels) for a given strict split.
fn run_split(fx: &Fixture, split: &StrictSplit) -> (f64, f64, Vec<f64>, Vec<bool>) {
    // Self-supervised pretraining: training windows only, no labels.
    let train_windows: Vec<TokenizedWindow> =
        split.train.iter().map(|&i| fx.windows[i].clone()).collect();
    let mut encoder = RfEncoder::new(fx.cfg, 7);
    let report = pretrain(
        &mut encoder,
        &train_windows,
        &PretrainConfig { mask_fraction: 0.25, lr: 0.03, epochs: 8, seed: 0x5EED },
    );
    assert!(
        report.final_loss < report.initial_loss,
        "pretraining must reduce masked loss: {report:?}"
    );

    // Frozen encoder → environment-invariant content representation for
    // every window (presence is a cross-room task; the geometry-conditioned
    // `encode()` view is for localization-style heads).
    let zs: Vec<Vec<f64>> = fx.windows.iter().map(|w| encoder.encode_content(w)).collect();

    // ≤1 % adapter on the frozen backbone.
    let mut head = PresenceHead::new(encoder.content_dim());
    assert!(
        within_adapter_budget(encoder.param_count(), head.param_count()),
        "presence adapter {} params vs backbone {}",
        head.param_count(),
        encoder.param_count()
    );
    let train_z: Vec<Vec<f64>> = split.train.iter().map(|&i| zs[i].clone()).collect();
    let train_y: Vec<bool> = split.train.iter().map(|&i| fx.corpus[i].presence).collect();
    head.train(&train_z, &train_y, 1.0, 800);

    let eval = |idx: &[usize]| -> (Vec<f64>, Vec<bool>) {
        let probs: Vec<f64> = idx.iter().map(|&i| head.predict_prob(&zs[i])).collect();
        let labels: Vec<bool> = idx.iter().map(|&i| fx.corpus[i].presence).collect();
        (probs, labels)
    };
    let (train_p, train_l) = eval(&split.train);
    let (test_p, test_l) = eval(&split.test);
    let known_f1 = f1_score(&train_p.iter().map(|p| *p > 0.5).collect::<Vec<_>>(), &train_l);
    let held_f1 = f1_score(&test_p.iter().map(|p| *p > 0.5).collect::<Vec<_>>(), &test_l);
    (known_f1, held_f1, test_p, test_l)
}

#[test]
fn acceptance_pipeline_on_synthetic_worlds() {
    let fx = build_fixture();
    let keys: Vec<_> = fx.corpus.iter().map(|w| w.key.clone()).collect();

    // ---- Gate 1: held-out ROOMS (rooms 6 and 7 never seen in any stage).
    let room_split = StrictSplit::holdout(&keys, PartitionDim::Room, &["room-6", "room-7"]);
    assert!(room_split.verify(&keys), "room split must be leak-free");
    assert!(room_split.test.len() >= 30, "held-out set must be substantial");
    let (known_f1, room_f1, test_p, test_l) = run_split(&fx, &room_split);
    println!("SYNTHETIC room-holdout: known F1 {known_f1:.4}, held-out F1 {room_f1:.4}");
    assert!(
        room_f1 >= 0.90,
        "ADR-273 gate: presence F1 on unseen rooms must be >= 0.90, got {room_f1:.4} (SYNTHETIC)"
    );

    // ---- Gate 3: known → unknown degradation < 20 %.
    let degradation = relative_degradation(known_f1, room_f1);
    println!("SYNTHETIC degradation known→unknown: {degradation:.4}");
    assert!(
        degradation < 0.20,
        "cross-environment degradation must stay under 20 %, got {degradation:.4}"
    );

    // ---- Calibration + abstention on the held-out side: raising the
    // confidence threshold must never raise selective risk, and full-
    // coverage risk is bounded by the F1 gate above.
    let ece = expected_calibration_error(&test_p, &test_l, 10);
    println!("SYNTHETIC held-out ECE: {ece:.4}");
    let full = selective_metrics(&test_p, &test_l, 0.5);
    let strict = selective_metrics(&test_p, &test_l, 0.9);
    println!(
        "SYNTHETIC abstention: coverage {:.2}→{:.2}, risk {:.4}→{:.4}",
        full.coverage, strict.coverage, full.selective_risk, strict.selective_risk
    );
    assert!(strict.selective_risk <= full.selective_risk + 1e-12);

    // ---- Gate 2: held-out CHIPSET (every chip-2 room excluded from
    // training; per-room gain/phase/CFO/noise profiles are random, so the
    // held-out hardware profile is genuinely unseen).
    let chip_split = StrictSplit::holdout(&keys, PartitionDim::Chipset, &["chip-2"]);
    assert!(chip_split.verify(&keys), "chipset split must be leak-free");
    let (chip_known, chip_f1, _, _) = run_split(&fx, &chip_split);
    println!("SYNTHETIC chipset-holdout: known F1 {chip_known:.4}, held-out F1 {chip_f1:.4}");
    assert!(
        chip_f1 >= 0.90,
        "presence F1 on unseen chipset must be >= 0.90, got {chip_f1:.4} (SYNTHETIC)"
    );
}

#[test]
fn edge_latency_budget_tokenize_plus_encode() {
    let fx = build_fixture();
    let tokenizer = RfTokenizer::new();
    let encoder = RfEncoder::new(fx.cfg, 7);

    // Warm up, then measure 100 full tokenize+encode passes.
    let mut latencies_us: Vec<u128> = Vec::with_capacity(100);
    for i in 0..110 {
        let idx = i % fx.corpus.len();
        let start = Instant::now();
        let w = tokenizer.tokenize(&fx.corpus[idx].tensor);
        let z = encoder.encode(&w);
        std::hint::black_box(z);
        if i >= 10 {
            latencies_us.push(start.elapsed().as_micros());
        }
    }
    latencies_us.sort_unstable();
    let p50 = latencies_us[latencies_us.len() / 2];
    let p95 = latencies_us[latencies_us.len() * 95 / 100];
    println!("edge latency tokenize+encode: p50 {p50} µs, p95 {p95} µs (debug profile)");
    // ADR-273 gate is 50 ms at p95 on edge hardware; even the unoptimized
    // debug profile must clear it with margin on a dev machine.
    assert!(p95 < 50_000, "p95 latency {p95} µs exceeds the 50 ms budget");
}

#[test]
fn outputs_leave_only_through_the_policy_boundary_fully_attributed() {
    let fx = build_fixture();
    let tokenizer = RfTokenizer::new();
    let encoder = RfEncoder::new(fx.cfg, 7);
    let mut head = PresenceHead::new(encoder.content_dim());
    // Small supervised fit so probabilities are meaningful.
    let zs: Vec<Vec<f64>> = fx
        .corpus
        .iter()
        .take(60)
        .map(|w| encoder.encode_content(&tokenizer.tokenize(&w.tensor)))
        .collect();
    let ys: Vec<bool> = fx.corpus.iter().take(60).map(|w| w.presence).collect();
    head.train(&zs, &ys, 0.5, 100);

    let mut engine = PolicyEngine::new();
    engine.upsert_zone(PrivacyZone {
        id: "room-0".into(),
        allowed_purposes: [SensingPurpose::Presence].into_iter().collect(),
        retention_s: 600,
        identity_explicitly_enabled: false,
    });
    let boundary = TrustBoundary::new(engine);

    let p = head.predict_prob(&zs[0]);
    let event = BoundedEvent::new(
        SensingPurpose::Presence,
        EventValue::Presence(p > 0.5),
        1.0 - (2.0 * p - 1.0).abs(), // confidence → uncertainty
        Provenance {
            device_id: fx.corpus[0].tensor.device_id.clone(),
            model_version: 1,
            synthetic: true, // honest labeling: synthetic evidence
        },
        1,
        fx.corpus[0].tensor.timestamp_ns,
        "room-0",
    )
    .expect("attributed event");

    // Compliant export passes and the payload is a typed verdict — the
    // BoundedEvent type has no variant that can carry RF samples, so raw
    // CSI export is unrepresentable, not merely forbidden.
    let exported =
        boundary.export(event.clone(), fx.corpus[0].tensor.timestamp_ns).expect("export ok");
    assert!(exported.uncertainty >= 0.0 && exported.uncertainty <= 1.0);
    assert!(exported.provenance.synthetic, "synthetic provenance must survive export");

    // An ungranted purpose in the same zone is denied (fail-closed).
    let denied = BoundedEvent { purpose: SensingPurpose::IdentityRecognition, ..event };
    assert!(boundary.export(denied, fx.corpus[0].tensor.timestamp_ns).is_err());
}
