//! AetherArena ("AA") Score Runner + Witness Chain (ADR-149).
//!
//! Benchmark-first scorer for the official Spatial-Intelligence Benchmark. It runs
//! the **real** `wifi-densepose-train::ruview_metrics` pose-acceptance harness and
//! emits a **witness record** for proof + repeatability analysis:
//!
//!   witness = { inputs_sha256, harness_version, metrics, tier, proof_sha256 }
//!
//! The `proof_sha256` is a cross-platform-stable hash of the quantised score; the
//! `inputs_sha256` binds the witness to the exact inputs it scored. Together with
//! the append-only hash-chained ledger (`aether-arena/ledger`), every published
//! rank traces back to a reproducible witness — the witness chain.
//!
//! Modes:
//!   # 1. Determinism self-test on the committed fixture (CI gate default):
//!   cargo run -p wifi-densepose-train --bin aa_score_runner --no-default-features
//!
//!   # 2. Repeatability analysis — run K times, confirm identical proof hash:
//!   cargo run ... --bin aa_score_runner --no-default-features -- --repeat 8
//!
//!   # 3. Real model scoring — score predictions against an eval split:
//!   cargo run ... --bin aa_score_runner --no-default-features -- \
//!       --split eval.json --pred predictions.json --json
//!
//!   # 4. Regenerate the fixture's expected hash (after an intentional change):
//!   cargo run ... --bin aa_score_runner --no-default-features -- --generate-hash \
//!       > ../aether-arena/fixtures/expected_score.sha256
//!
//! Input JSON (split = private ground truth; pred = the submitted model's output):
//!   split.json : {"frames":[{"gt":[[x,y]*17],"vis":[v*17],"scale":1.0}, ...]}
//!   pred.json  : {"frames":[{"pred":[[x,y]*17]}, ...]}   (index-aligned with split)
//!
//! Determinism discipline (lesson from calibration_proof_runner.rs): PCK/OKS use
//! libm `sqrt` which differs ~1e-7 across glibc/MSVC/Apple — so we hash only the
//! quantised metrics (1e-3 / 1e-4), never raw f32. No sort, no truncation.

use std::env;
use std::process::ExitCode;

use ndarray::{Array1, Array2};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use wifi_densepose_train::ruview_metrics::{
    evaluate_joint_error, JointErrorResult, JointErrorThresholds,
};

/// Bump on a purposeful fixture/canonical-form change. Pinned into every witness
/// so a `harness_version` change forces a re-score (ADR-149 §2.4).
const AA_HARNESS_VERSION: u32 = 2;

const N_FRAMES: usize = 120;
const N_KPTS: usize = 17;

// ── input schema ────────────────────────────────────────────────────────────
#[derive(Deserialize)]
struct SplitFile {
    frames: Vec<SplitFrame>,
}
#[derive(Deserialize)]
struct SplitFrame {
    gt: Vec<[f32; 2]>,
    vis: Vec<f32>,
    #[serde(default = "one")]
    scale: f32,
}
#[derive(Deserialize)]
struct PredFile {
    frames: Vec<PredFrame>,
}
#[derive(Deserialize)]
struct PredFrame {
    pred: Vec<[f32; 2]>,
}
fn one() -> f32 {
    1.0
}

// ── deterministic fixture (libm-free LCG) ─────────────────────────────────────
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() % 1_000_000) as f32 / 1_000_000.0
    }
}

fn build_fixture() -> (Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array1<f32>>, Vec<f32>) {
    let mut rng = Lcg(42);
    let (mut pred, mut gt, mut vis, mut scale) = (vec![], vec![], vec![], vec![]);
    for _ in 0..N_FRAMES {
        let mut g = Array2::<f32>::zeros((N_KPTS, 2));
        let mut p = Array2::<f32>::zeros((N_KPTS, 2));
        let mut v = Array1::<f32>::ones(N_KPTS);
        for k in 0..N_KPTS {
            let gx = 0.2 + 0.6 * rng.unit();
            let gy = 0.2 + 0.6 * rng.unit();
            let ox = (rng.unit() - 0.5) * 0.06;
            let oy = (rng.unit() - 0.5) * 0.06;
            g[[k, 0]] = gx;
            g[[k, 1]] = gy;
            p[[k, 0]] = (gx + ox).clamp(0.0, 1.0);
            p[[k, 1]] = (gy + oy).clamp(0.0, 1.0);
            if rng.next_u32() % 10 == 0 {
                v[k] = 0.0;
            }
        }
        gt.push(g);
        pred.push(p);
        vis.push(v);
        scale.push(1.0);
    }
    (pred, gt, vis, scale)
}

/// Load (pred, gt, vis, scale) from index-aligned split + prediction files.
fn load_inputs(
    split_path: &str,
    pred_path: &str,
) -> Result<(Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array1<f32>>, Vec<f32>), String> {
    let split: SplitFile = serde_json::from_str(
        &std::fs::read_to_string(split_path).map_err(|e| format!("read split: {e}"))?,
    )
    .map_err(|e| format!("parse split: {e}"))?;
    let pred: PredFile = serde_json::from_str(
        &std::fs::read_to_string(pred_path).map_err(|e| format!("read pred: {e}"))?,
    )
    .map_err(|e| format!("parse pred: {e}"))?;
    if split.frames.len() != pred.frames.len() {
        return Err(format!(
            "frame count mismatch: split={} pred={}",
            split.frames.len(),
            pred.frames.len()
        ));
    }
    let (mut gt, mut pr, mut vis, mut scale) = (vec![], vec![], vec![], vec![]);
    for (i, (s, p)) in split.frames.iter().zip(pred.frames.iter()).enumerate() {
        let to_arr = |kps: &[[f32; 2]]| -> Result<Array2<f32>, String> {
            if kps.len() != N_KPTS {
                return Err(format!("frame {i}: expected {N_KPTS} keypoints, got {}", kps.len()));
            }
            let mut a = Array2::<f32>::zeros((N_KPTS, 2));
            for (k, xy) in kps.iter().enumerate() {
                a[[k, 0]] = xy[0];
                a[[k, 1]] = xy[1];
            }
            Ok(a)
        };
        gt.push(to_arr(&s.gt)?);
        pr.push(to_arr(&p.pred)?);
        vis.push(Array1::from(s.vis.clone()));
        scale.push(s.scale);
    }
    Ok((pr, gt, vis, scale))
}

/// Canonical, libm-stable byte form of the score for the proof hash.
fn canonical_bytes(r: &JointErrorResult) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"AA-SCORE-v0");
    b.extend_from_slice(&AA_HARNESS_VERSION.to_le_bytes());
    let q = |x: f32, s: f32| -> u32 { (x.max(0.0) * s).round() as u32 };
    b.extend_from_slice(&q(r.pck_all, 1e3).to_le_bytes());
    b.extend_from_slice(&q(r.pck_torso, 1e3).to_le_bytes());
    b.extend_from_slice(&q(r.oks, 1e3).to_le_bytes());
    b.extend_from_slice(&q(r.jitter_rms_m, 1e4).to_le_bytes());
    b.extend_from_slice(&q(r.max_error_p95_m, 1e4).to_le_bytes());
    b.push(r.passes as u8);
    b
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

/// Bind the witness to its exact inputs: hash the quantised gt+pred+vis bytes.
fn inputs_hash(
    pred: &[Array2<f32>],
    gt: &[Array2<f32>],
    vis: &[Array1<f32>],
) -> String {
    let mut h = Sha256::new();
    h.update(b"AA-INPUTS-v0");
    h.update((pred.len() as u32).to_le_bytes());
    let q = |x: f32| -> i32 { (x * 1e4).round() as i32 };
    for f in 0..gt.len() {
        for k in 0..N_KPTS {
            h.update(q(gt[f][[k, 0]]).to_le_bytes());
            h.update(q(gt[f][[k, 1]]).to_le_bytes());
            h.update(q(pred[f][[k, 0]]).to_le_bytes());
            h.update(q(pred[f][[k, 1]]).to_le_bytes());
            h.update([(vis[f][k] >= 0.5) as u8]);
        }
    }
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

struct Witness {
    inputs_sha256: String,
    proof_sha256: String,
    result: JointErrorResult,
}

fn score(
    pred: &[Array2<f32>],
    gt: &[Array2<f32>],
    vis: &[Array1<f32>],
    scale: &[f32],
) -> Witness {
    let result = evaluate_joint_error(pred, gt, vis, scale, &JointErrorThresholds::default());
    Witness {
        inputs_sha256: inputs_hash(pred, gt, vis),
        proof_sha256: sha256_hex(&canonical_bytes(&result)),
        result,
    }
}

fn witness_json(w: &Witness) -> String {
    format!(
        "{{\"category\":\"pose\",\"harness_version\":{},\"inputs_sha256\":\"{}\",\"proof_sha256\":\"{}\",\"pck_all\":{:.4},\"pck_torso\":{:.4},\"oks\":{:.4},\"jitter_rms_m\":{:.5},\"max_error_p95_m\":{:.5},\"pose_passes\":{}}}",
        AA_HARNESS_VERSION, w.inputs_sha256, w.proof_sha256,
        w.result.pck_all, w.result.pck_torso, w.result.oks,
        w.result.jitter_rms_m, w.result.max_error_p95_m, w.result.passes
    )
}

fn arg_val<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mode_json = args.iter().any(|a| a == "--json");
    let mode_gen = args.iter().any(|a| a == "--generate-hash");
    let repeat: usize = arg_val(&args, "--repeat").and_then(|v| v.parse().ok()).unwrap_or(0);

    // Inputs: real split+pred if provided, else the deterministic fixture.
    let (pred, gt, vis, scale) = match (arg_val(&args, "--split"), arg_val(&args, "--pred")) {
        (Some(s), Some(p)) => match load_inputs(s, p) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("input error: {e}");
                return ExitCode::FAILURE;
            }
        },
        _ => build_fixture(),
    };

    let w = score(&pred, &gt, &vis, &scale);

    // ── Repeatability analysis: run K times, confirm an identical proof hash ──
    if repeat > 0 {
        let mut hashes = std::collections::BTreeSet::new();
        for _ in 0..repeat {
            let wi = score(&pred, &gt, &vis, &scale);
            hashes.insert(wi.proof_sha256);
        }
        let repeatable = hashes.len() == 1;
        println!(
            "{{\"repeatability\":{{\"runs\":{},\"unique_proof_hashes\":{},\"repeatable\":{},\"proof_sha256\":\"{}\"}}}}",
            repeat, hashes.len(), repeatable, w.proof_sha256
        );
        return if repeatable { ExitCode::SUCCESS } else {
            eprintln!("REPEATABILITY FAIL: {} distinct hashes across {} runs (nondeterminism)", hashes.len(), repeat);
            ExitCode::FAILURE
        };
    }

    if mode_gen {
        println!("{}", w.proof_sha256);
        return ExitCode::SUCCESS;
    }
    if mode_json {
        println!("{}", witness_json(&w));
        return ExitCode::SUCCESS;
    }

    // Default: determinism gate against the committed expected hash (CI).
    println!(
        "AA pose witness: PCK_all={:.4} PCK_torso={:.4} OKS={:.4} jitter={:.5}m p95={:.5}m passes={}",
        w.result.pck_all, w.result.pck_torso, w.result.oks,
        w.result.jitter_rms_m, w.result.max_error_p95_m, w.result.passes
    );
    println!("AA inputs_sha256: {}", w.inputs_sha256);
    println!("AA proof_sha256:  {}", w.proof_sha256);

    let expected_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../aether-arena/fixtures/expected_score.sha256");
    match std::fs::read_to_string(expected_path).ok().map(|s| s.trim().to_string()) {
        Some(exp) if exp == w.proof_sha256 => {
            println!("VERDICT: PASS (determinism hash matches expected)");
            ExitCode::SUCCESS
        }
        Some(exp) => {
            eprintln!("VERDICT: FAIL — scorer drift.\n  expected: {exp}\n  actual:   {}", w.proof_sha256);
            eprintln!("If intentional, regenerate with --generate-hash and review the diff.");
            ExitCode::FAILURE
        }
        None => {
            eprintln!("VERDICT: NO-EXPECTED-HASH — {expected_path} missing. Generate with --generate-hash.");
            ExitCode::FAILURE
        }
    }
}
