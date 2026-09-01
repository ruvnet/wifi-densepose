#![cfg(feature = "cpu")]

use std::io::Write;

use ruview_forecast_core::{
    CanonicalDigest, DataPolicy, HoldoutKey, NormalizationPolicy, PrivacyClass, QuantileSet,
    SeriesKey, SplitMember, SplitStrategy, TemporalSplitPlan, TimeRange, TrainSpec,
};
use ruview_forecast_model::ForecastModelConfig;
use ruview_forecast_train::{
    artifact::ArtifactStore,
    cancel::NeverCancel,
    config::{
        DatasetInput, DatasetSource, JobId, ModelProfile, OptimizerSpec, RelativeDataPath,
        Sha256Digest, TrainingBudget, TrainingDevice, TrainingRequest,
    },
    corpus::JsonlWindow,
    runner::{unix_time_millis, BurnTrainer},
};

#[test]
fn local_hash_addressed_jsonl_executes_one_real_optimizer_step() {
    let config = ForecastModelConfig::tiny_ci();
    let train_key = SeriesKey::new("local-train", "device-a", "session-a").expect("train key");
    let test_key = SeriesKey::new("local-test", "device-b", "session-b").expect("test key");
    let split = TemporalSplitPlan::new(
        SplitStrategy::EntityHoldout(HoldoutKey::Strict),
        vec![SplitMember::new(
            train_key.clone(),
            TimeRange::new(1, 100_000).expect("train range"),
        )],
        vec![],
        vec![SplitMember::new(
            test_key,
            TimeRange::new(1, 100_000).expect("test range"),
        )],
        config.horizon,
        1_000,
        0,
    )
    .expect("strict split");

    let variates = 3_usize;
    let window = JsonlWindow {
        version: 1,
        series_key: train_key,
        context_start_ms: 1_000,
        variates: variates as u16,
        values: (0..config.context_len * variates)
            .map(|index| (index as f32 * 0.01).sin())
            .collect(),
        observed_mask: vec![1; config.context_len * variates],
        targets: (0..config.horizon * variates)
            .map(|index| (index as f32 * 0.02).cos())
            .collect(),
        target_mask: vec![1; config.horizon * variates],
    };
    let mut shard = serde_json::to_vec(&window).expect("window JSON");
    shard.push(b'\n');

    let data_root = tempfile::tempdir().expect("data root");
    let mut file = std::fs::File::create(data_root.path().join("train.jsonl")).expect("shard");
    file.write_all(&shard).expect("write shard");
    file.sync_all().expect("sync shard");

    let input = DatasetInput {
        path: RelativeDataPath::new("train.jsonl").expect("relative path"),
        size_bytes: shard.len() as u64,
        sha256: Sha256Digest::of_bytes(&shard),
        window_count: 1,
        variates: variates as u16,
        feature_schema_digest: CanonicalDigest::of_bytes(
            b"local-feature-schema-v1",
            b"amplitude-0,amplitude-1,amplitude-2",
        ),
    };
    let dataset_digest =
        CanonicalDigest::of_bytes(b"ruview-jsonl-window-shard-v1", input.sha256.as_bytes());
    let policy = DataPolicy::new(
        PrivacyClass::P3,
        "local-test-tenant",
        "local-test-account",
        "local-test-workspace",
        "forecast-foundation-pretraining",
        CanonicalDigest::of_bytes(b"test-policy-v1", b"approved"),
        None,
        None,
        None,
        unix_time_millis().saturating_add(86_400_000),
        true,
    )
    .expect("policy");
    let train = TrainSpec::new_local(
        config.context_len,
        config.horizon,
        1_000,
        QuantileSet::new(config.quantiles.to_vec()).expect("quantiles"),
        split,
        NormalizationPolicy::None,
        dataset_digest,
        policy,
    )
    .expect("train spec");
    let request = TrainingRequest::new_local(
        JobId::new("local-jsonl-smoke").expect("job"),
        train,
        DatasetSource::Manifest(input),
        ModelProfile::TinyCi,
        TrainingDevice::Cpu,
        OptimizerSpec {
            epochs: 1,
            batch_size: 1,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            gradient_clip_norm: 1.0,
            checkpoint_every_epochs: 1,
            seed: 11,
        },
        TrainingBudget {
            max_optimizer_steps: 1,
            max_wall_time_seconds: 60,
            max_memory_bytes: 1024 * 1024 * 1024,
            max_artifact_bytes: 64 * 1024 * 1024,
            max_checkpoints: 1,
        },
    )
    .and_then(TrainingRequest::into_validated)
    .expect("request");

    let output = tempfile::tempdir().expect("output");
    let outcome = BurnTrainer::with_dataset_root(
        ArtifactStore::new(output.path()).expect("artifact store"),
        data_root.path().to_path_buf(),
    )
    .train(&request, &NeverCancel)
    .expect("local training");

    assert_eq!(outcome.optimizer_steps, 1);
    assert_eq!(outcome.epochs_completed, 1);
    assert!(!outcome.is_production_signed());
    assert!(output.path().join("local-jsonl-smoke/model.mpk").is_file());
}
