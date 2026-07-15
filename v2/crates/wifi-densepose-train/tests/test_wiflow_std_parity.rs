//! Numerical parity between the Rust WiFlow-STD port and the retrained
//! PyTorch checkpoint (ADR-152 §2.2).
//!
//! The fixtures are produced by `benchmarks/wiflow-std/export_to_safetensors.py`
//! (gitignored — they derive from the retrained checkpoint, which is itself
//! gitignored):
//!
//! - `results/retrained_wiflow_std.safetensors` — the epoch-36 checkpoint
//!   (val PCK@20 96.99%) remapped to tch `VarStore` variable names
//! - `results/parity_fixture.json` — a deterministic input (seed 42, shape
//!   `(2, 540, 20)`, uniform `[0, 1]`) and the upstream `WiFlowPoseModel`'s
//!   eval-mode output on it
//!
//! Run explicitly (needs LibTorch, e.g. `LIBTORCH_USE_PYTORCH=1` with the
//! torch DLL directory on `PATH`):
//!
//! ```text
//! cargo test -p wifi-densepose-train --features tch-backend \
//!     --test test_wiflow_std_parity -- --ignored --nocapture
//! ```

#![cfg(feature = "tch-backend")]

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use tch::{Device, Tensor};
use wifi_densepose_train::{WiFlowStdConfig, WiFlowStdModel};

#[derive(serde::Deserialize)]
struct ParityFixture {
    input_shape: Vec<i64>,
    input: Vec<f32>,
    output_shape: Vec<i64>,
    output: Vec<f32>,
}

fn results_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("benchmarks")
        .join("wiflow-std")
        .join("results")
}

/// Loads the retrained checkpoint into the Rust model and asserts the forward
/// pass matches PyTorch to within 1e-4 max absolute difference.
///
/// `#[ignore]`d by default: it needs the gitignored fixtures above plus a
/// working LibTorch environment, neither of which exist in CI.
#[test]
#[ignore = "needs gitignored fixtures (run export_to_safetensors.py) + LibTorch env; run with --ignored"]
fn retrained_checkpoint_matches_pytorch_forward() {
    let dir = results_dir();
    let weights = dir.join("retrained_wiflow_std.safetensors");
    let fixture_path = dir.join("parity_fixture.json");
    for p in [&weights, &fixture_path] {
        assert!(
            p.exists(),
            "missing fixture {} — run benchmarks/wiflow-std/export_to_safetensors.py first",
            p.display()
        );
    }

    let fixture: ParityFixture = serde_json::from_reader(BufReader::new(
        File::open(&fixture_path).expect("open parity_fixture.json"),
    ))
    .expect("parse parity_fixture.json");
    assert_eq!(fixture.input_shape, vec![2, 540, 20]);
    assert_eq!(fixture.output_shape, vec![2, 15, 2]);

    let cfg = WiFlowStdConfig::default();
    let mut model = WiFlowStdModel::new(&cfg, Device::Cpu).expect("build default model");
    model
        .load(&weights)
        .expect("safetensors load: every VarStore variable must match by name and shape");

    let input = Tensor::from_slice(&fixture.input).reshape(&fixture.input_shape[..]);
    let expected = Tensor::from_slice(&fixture.output).reshape(&fixture.output_shape[..]);

    let output = model.forward_inference(&input);
    assert_eq!(output.size(), fixture.output_shape);

    let max_diff = (&output - &expected).abs().max().double_value(&[]);
    println!("max |rust - python| = {max_diff:.3e}");
    assert!(
        max_diff < 1e-4,
        "Rust forward pass diverges from PyTorch: max abs diff {max_diff:.3e} >= 1e-4"
    );
}
