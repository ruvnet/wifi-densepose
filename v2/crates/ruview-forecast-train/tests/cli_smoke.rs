#![cfg(all(feature = "cpu", feature = "cli"))]

use std::process::Command;

#[test]
fn cli_prepares_a_loadable_local_example_without_overwrite() {
    let parent = tempfile::tempdir().expect("temporary parent");
    let example = parent.path().join("local-example");
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_ruforecast"))
            .args(["prepare-local-example", "--directory"])
            .arg(&example)
            .status()
            .expect("run local example generator")
    };
    assert!(run().success());
    assert!(example.join("train.jsonl").is_file());
    let request_path = example.join("train-local.toml");
    assert!(request_path.is_file());
    ruview_forecast_train::config::load_request(&request_path).expect("load generated request");
    assert!(!run().success(), "existing example must not be overwritten");
}

#[test]
fn cli_smoke_trains_and_writes_the_complete_candidate_set() {
    let output = tempfile::tempdir().expect("temporary artifact root");
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_ruforecast"))
            .args([
                "smoke",
                "--job-id",
                "cli-smoke",
                "--windows",
                "2",
                "--output",
            ])
            .arg(output.path())
            .status()
            .expect("run ruforecast smoke")
    };
    assert!(run().success());

    let job = output.path().join("cli-smoke");
    for filename in [
        "model.mpk",
        "checkpoint.mpk",
        "artifact-manifest.json",
        "training-receipt.json",
    ] {
        let path = job.join(filename);
        let metadata = path.metadata().unwrap_or_else(|error| {
            panic!("missing {filename}: {error}");
        });
        assert!(metadata.is_file());
        assert!(metadata.len() > 0);
    }

    let receipt = std::fs::read(job.join("training-receipt.json")).expect("first receipt");
    assert!(run().success());
    assert_eq!(
        std::fs::read(job.join("training-receipt.json")).expect("recovered receipt"),
        receipt,
        "an identical job must recover without rewriting its receipt",
    );

    std::fs::write(job.join("artifact-manifest.json"), b"{}").expect("tamper manifest fixture");
    assert!(
        !run().success(),
        "recovery must reject a sidecar that no longer matches the embedded manifest",
    );
}
