#![cfg(all(feature = "cpu", feature = "cli"))]

use std::io::Write as _;
use std::process::Command;

use ruview_forecast_core::SeriesKey;
use ruview_forecast_model::ForecastModelConfig;
use ruview_forecast_train::corpus::JsonlWindow;

/// Deterministic synthetic values, distinct from the `smoke` command's own
/// generator so the held-out windows are not simply repeats of training data.
fn synthetic_value(row: usize, variate: usize, salt: u64) -> f32 {
    let phase = salt as f32 * 1.3;
    ((row as f32 * 0.05 + phase + variate as f32 * 0.7).sin()
        + (row as f32 * 0.11 + phase * 0.4).cos() * 0.3)
        * (1.0 + 0.1 * variate as f32)
}

fn build_window(config: &ForecastModelConfig, index: u64, salt: u64) -> JsonlWindow {
    let variates = 3_usize;
    let context_len = config.context_len;
    let horizon = config.horizon;
    let mut values = Vec::with_capacity(context_len * variates);
    for row in 0..context_len {
        for variate in 0..variates {
            values.push(synthetic_value(row, variate, salt));
        }
    }
    let mut targets = Vec::with_capacity(variates * horizon);
    for variate in 0..variates {
        for step in 0..horizon {
            targets.push(synthetic_value(context_len + step, variate, salt));
        }
    }
    JsonlWindow {
        version: 1,
        series_key: SeriesKey::new("evaluate-smoke", "device-a", format!("held-out-{index}"))
            .expect("series key"),
        context_start_ms: 1_000,
        variates: u16::try_from(variates).expect("small variate count"),
        values,
        observed_mask: vec![1; context_len * variates],
        targets,
        target_mask: vec![1; variates * horizon],
    }
}

#[test]
fn evaluate_scores_a_smoke_trained_candidate_against_baselines() {
    let output = tempfile::tempdir().expect("temporary artifact root");
    let train = Command::new(env!("CARGO_BIN_EXE_ruforecast"))
        .args([
            "smoke",
            "--job-id",
            "evaluate-smoke",
            "--windows",
            "4",
            "--output",
        ])
        .arg(output.path())
        .status()
        .expect("run ruforecast smoke");
    assert!(train.success());

    let candidate_path = output.path().join("evaluate-smoke").join("model.mpk");
    assert!(candidate_path.is_file());

    let config = ForecastModelConfig::tiny_ci();
    let mut test_jsonl = tempfile::NamedTempFile::new().expect("test jsonl file");
    for index in 0..3_u64 {
        let window = build_window(&config, index, 29 + index);
        let line = serde_json::to_string(&window).expect("serialize window");
        writeln!(test_jsonl, "{line}").expect("write window line");
    }
    test_jsonl.flush().expect("flush test jsonl");

    let run = Command::new(env!("CARGO_BIN_EXE_ruforecast"))
        .args(["evaluate", "--candidate"])
        .arg(&candidate_path)
        .args(["--test-jsonl"])
        .arg(test_jsonl.path())
        .args(["--seasonal-period", "6"])
        .output()
        .expect("run ruforecast evaluate");
    assert!(
        run.status.success(),
        "evaluate failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );

    let report: serde_json::Value =
        serde_json::from_slice(&run.stdout).expect("evaluate prints one JSON report");
    assert_eq!(report["n_test_windows"], 3);
    assert_eq!(report["variates"], 3);
    assert_eq!(report["horizon"], config.horizon as u64);
    assert_eq!(report["seasonal_period"], 6);

    for forecaster in ["model", "last_value_baseline", "seasonal_naive_baseline"] {
        let wql = report[forecaster]["weighted_quantile_loss"]
            .as_f64()
            .unwrap_or_else(|| panic!("{forecaster} weighted_quantile_loss must be a number"));
        assert!(wql.is_finite() && wql >= 0.0, "{forecaster} WQL was {wql}");
        let by_horizon = report[forecaster]["weighted_quantile_loss_by_horizon"]
            .as_array()
            .unwrap_or_else(|| panic!("{forecaster} weighted_quantile_loss_by_horizon must be an array"));
        assert_eq!(by_horizon.len(), config.horizon);
    }

    let not_implemented = report["not_implemented"]
        .as_array()
        .expect("not_implemented must list unimplemented protocol fields");
    assert!(not_implemented
        .iter()
        .any(|value| value == "ruvector_retrieval_ablation"));
}

#[test]
fn evaluate_rejects_an_empty_test_file() {
    let output = tempfile::tempdir().expect("temporary artifact root");
    let train = Command::new(env!("CARGO_BIN_EXE_ruforecast"))
        .args([
            "smoke",
            "--job-id",
            "evaluate-empty",
            "--windows",
            "2",
            "--output",
        ])
        .arg(output.path())
        .status()
        .expect("run ruforecast smoke");
    assert!(train.success());

    let candidate_path = output.path().join("evaluate-empty").join("model.mpk");
    let empty_jsonl = tempfile::NamedTempFile::new().expect("empty test jsonl");

    let run = Command::new(env!("CARGO_BIN_EXE_ruforecast"))
        .args(["evaluate", "--candidate"])
        .arg(&candidate_path)
        .args(["--test-jsonl"])
        .arg(empty_jsonl.path())
        .status()
        .expect("run ruforecast evaluate");
    assert!(!run.success(), "evaluate must reject an empty test file");
}
