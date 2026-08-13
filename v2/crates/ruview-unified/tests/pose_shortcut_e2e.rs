//! End-to-end room-shortcut leakage measurement for the factorized pose
//! head (ADR-281 §4, the RePos lesson), on SYNTHETIC data — issue #1438
//! item 1.
//!
//! The unit test in `heads.rs`
//! (`factorized_pose_head_algebra_resists_hand_built_room_feature`) checks
//! the head's *algebra* on hand-built 2/3-dim representations where the
//! content/geometry separation is baked in by the test author. This test
//! makes the end-to-end claim instead: nothing about the representation is
//! hand-built. Body scale exists only as a radar cross-section in the
//! physics simulation, the room exists only as the link-geometry metadata a
//! deployment would report, and the representations are the real
//! `RfEncoder::encode_content` / `encode` outputs after masked-
//! reconstruction pretraining on the training rooms.
//!
//! The leakage trap mirrors the RePos experiment:
//! - training room A contains only small people, training room B only large
//!   ones (scale ↔ room correlation);
//! - the held-out room C (a strict ADR-273 room holdout, disjointness
//!   certified) breaks the correlation with every scale;
//! - a monolithic absolute-pose regressor on the geometry-conditioned `z`
//!   can exploit the room feature as a scale shortcut and must degrade in
//!   room C, while the factorized head's skeleton branch reads the
//!   environment-invariant content representation — which carries scale
//!   only through the physics (scattered-path amplitude ∝ √RCS) — and must
//!   generalize.

use ruview_unified::encoder::{EncoderConfig, RfEncoder};
use ruview_unified::eval::{mpjpe, PartitionDim, PartitionKey, SplitManifest, StrictSplit};
use ruview_unified::heads::{FactorizedPoseHead, LowRankLinear, NUM_JOINTS};
use ruview_unified::pretrain::{pretrain, PretrainConfig};
use ruview_unified::synth::{synthesize_csi, Material, PersonSpec, RoomSpec, SynthGenerator};
use ruview_unified::tensor::{
    CalibrationMeta, LinkGeometry, RfModality, RfTensor, CANONICAL_BINS, CANONICAL_SNAPSHOTS,
};
use ruview_unified::tokenizer::{RfTokenizer, TokenizedWindow};

use ndarray::Array3;
use num_complex::Complex64;

const SNAPSHOT_DT_S: f64 = 0.05;

/// Room-local base layout, identical in every room so the *only* room
/// signal available anywhere is the global link-geometry offset — exactly
/// the feature the factorization claims to keep out of the content path.
const ROOM_SIZE: [f64; 3] = [5.0, 4.0, 2.8];
const LOCAL_LINKS: [([f64; 3], [f64; 3]); 2] = [
    ([0.4, 0.5, 1.6], [4.6, 1.0, 1.4]),
    ([0.5, 3.4, 1.5], [4.5, 3.0, 1.7]),
];

/// Deterministic xorshift noise source (integration tests cannot reach the
/// crate's seeded ChaCha; determinism is what matters here, not quality).
struct XorShift(u64);

impl XorShift {
    fn next_unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// The same deterministic 17-joint skeleton family as the unit test: joint
/// offsets scale linearly with body scale `s ∈ [-1, 1]`.
fn skeleton(scale: f64) -> [[f64; 3]; NUM_JOINTS] {
    let mut joints = [[0.0; 3]; NUM_JOINTS];
    for (j, joint) in joints.iter_mut().enumerate() {
        let a = j as f64 * 0.37;
        let dir = [a.cos() * 0.3, a.sin() * 0.3, 0.1 * ((j % 5) as f64 - 2.0)];
        for (k, v) in joint.iter_mut().enumerate() {
            *v = dir[k] * (1.0 + 0.5 * scale);
        }
    }
    joints
}

/// Body scale enters the physics ONLY here: torso radar cross-section, with
/// √RCS (the scattered-path amplitude factor) linear in `s`.
fn rcs_for_scale(s: f64) -> f64 {
    0.55 * (1.0 + 0.5 * s).powi(2)
}

struct PoseSample {
    window: TokenizedWindow,
    key: PartitionKey,
    rel_joints: [[f64; 3]; NUM_JOINTS],
    root: [f64; 3],
}

/// Per-room chipset-style nuisances (the tokenizer's hardware-invariance
/// steps must absorb these — they carry no usable room signal downstream).
struct RoomNuisance {
    gain: f64,
    phase: f64,
    cfo_rad_per_snap: f64,
}

/// Synthesizes one room's windows through the full deployment path:
/// image-method CSI physics (room-local coordinates — propagation is
/// translation-invariant) → per-room hardware nuisances + thermal noise →
/// canonical `RfTensor` whose link geometry carries the *global* room
/// placement, as a site map would → real tokenizer.
fn build_room(
    room_name: &str,
    room_idx: u64,
    offset_x: f64,
    scales: &[f64],
    hw: &RoomNuisance,
    noise: &mut XorShift,
) -> Vec<PoseSample> {
    let tokenizer = RfTokenizer::new();
    let freqs = SynthGenerator::subcarrier_freqs();
    let mid_t = 0.5 * CANONICAL_SNAPSHOTS as f64 * SNAPSHOT_DT_S;

    scales
        .iter()
        .enumerate()
        .map(|(w, &s)| {
            let person = PersonSpec {
                start: [2.6, 2.0, 1.2],
                velocity: [0.3, 0.2, 0.0],
                rcs_m2: rcs_for_scale(s),
            };
            let room = RoomSpec::new(ROOM_SIZE, Material::drywall(), vec![person])
                .expect("static room spec is valid");

            let mut data =
                Array3::zeros((LOCAL_LINKS.len(), CANONICAL_BINS, CANONICAL_SNAPSHOTS));
            for (l, (tx, rx)) in LOCAL_LINKS.iter().enumerate() {
                for snap in 0..CANONICAL_SNAPSHOTS {
                    let t = snap as f64 * SNAPSHOT_DT_S;
                    let h = synthesize_csi(&room, *tx, *rx, &freqs, t);
                    let rot = Complex64::from_polar(
                        hw.gain,
                        hw.phase + hw.cfo_rad_per_snap * snap as f64,
                    );
                    for (b, hv) in h.iter().enumerate() {
                        let n = Complex64::new(noise.next_unit(), noise.next_unit()) * 2.0e-5;
                        data[[l, b, snap]] = hv * rot + n;
                    }
                }
            }

            let links: Vec<LinkGeometry> = LOCAL_LINKS
                .iter()
                .map(|(tx, rx)| LinkGeometry {
                    tx_pos: [tx[0] + offset_x, tx[1], tx[2]],
                    rx_pos: [rx[0] + offset_x, rx[1], rx[2]],
                })
                .collect();
            let tensor = RfTensor::new(
                RfModality::Synthetic,
                2.437e9,
                20e6,
                data,
                links,
                0.05,
                (room_idx << 32) | w as u64,
                format!("synth-{room_name}"),
                0.8,
                0.1,
                CalibrationMeta::default(),
            )
            .expect("synthesized tensor is finite and in range");

            let p = person.position_at(mid_t);
            PoseSample {
                window: tokenizer.tokenize(&tensor),
                key: PartitionKey {
                    room: room_name.into(),
                    day: "day-0".into(),
                    person: "p0".into(),
                    chipset: format!("chip-{room_name}"),
                    firmware: "fw-0".into(),
                    layout: "layout-0".into(),
                    session: format!("{room_name}-w{w}"),
                },
                rel_joints: skeleton(s),
                root: [p[0] + offset_x, p[1], p[2]],
            }
        })
        .collect()
}

#[test]
fn factorized_pose_resists_room_shortcut_through_the_real_pipeline() {
    // --- Corpus: the leakage trap. Training rooms tie scale to room;
    // the held-out room breaks the tie with the full scale range.
    let scales_a: Vec<f64> = (0..24).map(|i| -1.0 + i as f64 / 24.0).collect(); // [-1, 0)
    let scales_b: Vec<f64> = (0..24).map(|i| i as f64 / 24.0).collect(); // [0, 1)
    let scales_c: Vec<f64> = (0..16).map(|i| -1.0 + i as f64 / 8.0).collect(); // [-1, 1)

    let mut noise = XorShift(0x1438_5EED);
    let mut samples = Vec::new();
    samples.extend(build_room(
        "room-a",
        0,
        0.0,
        &scales_a,
        &RoomNuisance { gain: 0.8, phase: 0.7, cfo_rad_per_snap: 0.12 },
        &mut noise,
    ));
    samples.extend(build_room(
        "room-b",
        1,
        5.0,
        &scales_b,
        &RoomNuisance { gain: 1.4, phase: -1.3, cfo_rad_per_snap: -0.2 },
        &mut noise,
    ));
    samples.extend(build_room(
        "room-c",
        2,
        10.0,
        &scales_c,
        &RoomNuisance { gain: 1.1, phase: 2.1, cfo_rad_per_snap: 0.05 },
        &mut noise,
    ));

    // --- Strict ADR-273 room holdout with an independently verified
    // disjointness certificate (room C is unseen by every training stage,
    // pretraining included).
    let keys: Vec<PartitionKey> = samples.iter().map(|s| s.key.clone()).collect();
    let split = StrictSplit::holdout(&keys, PartitionDim::Room, &["room-c"]);
    assert!(split.verify(&keys), "room split must be leak-free");
    let manifest = SplitManifest::build(&keys, &split.train, &split.test);
    assert!(manifest.is_disjoint(PartitionDim::Room), "manifest must certify the room holdout");
    assert_eq!(split.test.len(), scales_c.len());

    // --- Self-supervised pretraining on the training rooms only.
    let train_windows: Vec<TokenizedWindow> =
        split.train.iter().map(|&i| samples[i].window.clone()).collect();
    let cfg = EncoderConfig { d_model: 64, ..EncoderConfig::default() };
    let mut encoder = RfEncoder::new(cfg, 7);
    let report = pretrain(
        &mut encoder,
        &train_windows,
        &PretrainConfig { mask_fraction: 0.25, lr: 0.03, epochs: 8, seed: 0x5EED },
    );
    assert!(
        report.final_loss < report.initial_loss,
        "pretraining must reduce masked loss: {report:?}"
    );

    // --- Frozen encoder → the two representation views of ADR-273 §5.
    let content_zs: Vec<Vec<f64>> =
        samples.iter().map(|s| encoder.encode_content(&s.window)).collect();
    let full_zs: Vec<Vec<f64>> = samples.iter().map(|s| encoder.encode(&s.window)).collect();

    let gather = |idx: &[usize], from: &[Vec<f64>]| -> Vec<Vec<f64>> {
        idx.iter().map(|&i| from[i].clone()).collect()
    };
    let train_content = gather(&split.train, &content_zs);
    let train_full = gather(&split.train, &full_zs);
    let train_rel: Vec<[[f64; 3]; NUM_JOINTS]> =
        split.train.iter().map(|&i| samples[i].rel_joints).collect();
    let train_root: Vec<[f64; 3]> = split.train.iter().map(|&i| samples[i].root).collect();

    // --- Factorized head on the real representations.
    let mut head = FactorizedPoseHead::new(encoder.content_dim(), cfg.d_model, 4);
    let (rel_mse, root_mse) =
        head.train(&train_content, &train_full, &train_rel, &train_root, 0.05, 3000);
    println!("SYNTHETIC e2e fit: relative mse {rel_mse:.6} m², root mse {root_mse:.6} m²");

    // --- Monolithic baseline: absolute joints from the geometry-conditioned
    // z, same rank and training budget as the factorized skeleton branch.
    let abs_flat = |rel: &[[f64; 3]; NUM_JOINTS], root: &[f64; 3]| -> Vec<f64> {
        rel.iter().flat_map(|j| (0..3).map(move |k| j[k] + root[k])).collect()
    };
    let train_abs: Vec<Vec<f64>> = split
        .train
        .iter()
        .map(|&i| abs_flat(&samples[i].rel_joints, &samples[i].root))
        .collect();
    let mut monolithic = LowRankLinear::new(cfg.d_model, NUM_JOINTS * 3, 4);
    let mono_mse = monolithic.train(&train_full, &train_abs, 0.05, 3000);
    println!("SYNTHETIC e2e fit: monolithic mse {mono_mse:.6} m²");

    // --- Evaluate MPJPE on both sides of the certified split.
    let eval = |idx: &[usize]| -> (f64, f64) {
        let mut fact = 0.0;
        let mut mono = 0.0;
        for &i in idx {
            let truth: Vec<[f64; 3]> = {
                let flat = abs_flat(&samples[i].rel_joints, &samples[i].root);
                (0..NUM_JOINTS).map(|j| [flat[j * 3], flat[j * 3 + 1], flat[j * 3 + 2]]).collect()
            };

            let pose = head.predict(&content_zs[i], &full_zs[i]);
            assert!(pose.root_uncertainty_m.is_finite());
            assert!(pose.joint_uncertainty_m.iter().all(|u| u.is_finite() && *u >= 0.0));
            fact += mpjpe(&pose.absolute_joints_m(), &truth);

            let m = monolithic.predict(&full_zs[i]);
            let mono_abs: Vec<[f64; 3]> =
                (0..NUM_JOINTS).map(|j| [m[j * 3], m[j * 3 + 1], m[j * 3 + 2]]).collect();
            mono += mpjpe(&mono_abs, &truth);
        }
        (fact / idx.len() as f64, mono / idx.len() as f64)
    };
    let (fact_train, mono_train) = eval(&split.train);
    let (fact_test, mono_test) = eval(&split.test);
    println!(
        "SYNTHETIC e2e room-holdout MPJPE: factorized {fact_train:.4} m → {fact_test:.4} m, \
         monolithic {mono_train:.4} m → {mono_test:.4} m (train → held-out room)"
    );

    // Both heads must actually fit the training rooms — a baseline that
    // never learned anything would make the held-out comparison vacuous.
    assert!(fact_train < 0.10, "factorized head must fit training rooms, MPJPE {fact_train}");
    assert!(mono_train < 0.10, "monolithic head must fit training rooms, MPJPE {mono_train}");

    // The end-to-end anti-leakage claims, measured on real representations:
    // the factorized head generalizes to the unseen room, the monolithic
    // head pays for the room→scale shortcut when the correlation breaks.
    assert!(
        fact_test < 0.15,
        "factorized head must generalize to the held-out room, MPJPE {fact_test}"
    );
    assert!(
        mono_test > 1.5 * fact_test,
        "monolithic head must show the shortcut collapse: {mono_test} vs {fact_test}"
    );
}
