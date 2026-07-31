//! ADR-185 §4.1 — MAT bit-for-bit parity: native-Rust reference half.
//!
//! Drives the canonical `wifi-densepose-mat` `DisasterResponse` pipeline
//! DIRECTLY (no PyO3) over the committed `tests/golden/mat_input.json` CSI
//! stream and locks the SHA-256 of a canonical result string
//! (`count=<K>;triage_priorities=<sorted>`) into
//! `tests/golden/mat_result.sha256`.
//!
//! Only the survivor **count** and **triage classes** are hashed — survivor
//! UUIDs and event timestamps are non-deterministic and deliberately
//! excluded, so the hash captures exactly the "identical triage +
//! survivor count for a fixed CSI stream" invariant of ADR-185 §4.1.
//!
//! Honesty note: the fixture is a synthetic breathing-modulated stream, so
//! this proves the Python binding drives the real pipeline byte-identically
//! to native Rust — it is NOT a detection-accuracy claim on real rubble.
//!
//! Regenerate (only on an intentional Rust change): delete the .sha256 and
//! re-run `cargo test --features mat --test mat_parity`.
#![cfg(feature = "mat")]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use sha2::{Digest, Sha256};
use wifi_densepose_mat::{DisasterConfig, DisasterResponse, DisasterType, ScanZone, ZoneBounds};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

fn fixture() -> Value {
    let raw = fs::read_to_string(golden_dir().join("mat_input.json"))
        .expect("read mat_input.json fixture");
    serde_json::from_str(&raw).expect("parse mat_input.json")
}

/// Build the response identically to the Python binding's default
/// construction and run one scan over the fixture CSI stream. Returns the
/// canonical `count=<K>;triage_priorities=<sorted>` string.
fn mat_canonical_result(fx: &Value) -> String {
    let config = DisasterConfig::builder()
        .disaster_type(DisasterType::Earthquake)
        .sensitivity(0.9)
        .confidence_threshold(0.1)
        .max_depth(5.0)
        .continuous_monitoring(false)
        .build();
    let mut resp = DisasterResponse::new(config);
    resp.initialize_event(geo::Point::new(0.0, 0.0), "parity-fixture")
        .expect("initialize_event");
    resp.add_zone(ScanZone::new(
        "Zone A",
        ZoneBounds::rectangle(0.0, 0.0, 50.0, 30.0),
    ))
    .expect("add_zone");

    for frame in fx["stream"].as_array().unwrap() {
        let amp: Vec<f64> = frame["amplitude"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let ph: Vec<f64> = frame["phase"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        resp.push_csi_data(&amp, &ph).expect("push_csi_data");
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    rt.block_on(resp.start_scanning()).expect("scan");

    let survivors = resp.survivors();
    let mut priorities: Vec<u8> = survivors
        .iter()
        .map(|s| s.triage_status().priority())
        .collect();
    priorities.sort_unstable();
    format!("count={};triage_priorities={:?}", survivors.len(), priorities)
}

fn sha256_hex(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn native_mat_result_is_deterministic() {
    let fx = fixture();
    // Two independent runs must agree (survivor count + triage classes are
    // deterministic; UUIDs/timestamps are excluded from the canonical form).
    assert_eq!(mat_canonical_result(&fx), mat_canonical_result(&fx));
}

#[test]
fn native_mat_matches_committed_golden() {
    let canon = mat_canonical_result(&fixture());
    let got = sha256_hex(&canon);
    let path = golden_dir().join("mat_result.sha256");
    match fs::read_to_string(&path) {
        Ok(expected) => assert_eq!(
            got,
            expected.trim(),
            "native MAT result drifted from committed golden (canonical form: {canon})"
        ),
        Err(_) => {
            fs::write(&path, &got).expect("write golden sha256");
            panic!("no committed golden found; wrote {got} for [{canon}]. Re-run to verify.");
        }
    }
}
