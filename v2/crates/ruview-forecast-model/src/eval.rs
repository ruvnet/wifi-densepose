//! Local, evaluation-only model activation and inference-input construction.
//!
//! Nothing here is a production trust path. [`activate_for_evaluation`] signs
//! an unsigned candidate with a fixed, publicly-known, non-secret key so a
//! CLI operator can run inference against their own just-trained candidate
//! without a real release signature. The resulting [`CpuForecastRuntime`]
//! must never be treated as release-authorized, and the fixed key must never
//! be reused for a real signer identity.

use burn_core::tensor::{Tensor, TensorData};
use ed25519_dalek::{Signer, SigningKey};

use crate::artifact::{
    ArtifactActivationPolicy, ModelArtifact, ModelError, SignedModelArtifact, TrustedSignerSet,
};
use crate::config::ForecastModelConfig;
use crate::network::ModelInput;
use crate::runtime::{CpuBackend, CpuDevice, CpuForecastRuntime};

/// Fixed, deliberately public 32-byte seed for the evaluation-only signing
/// key. Anyone can derive the same key from this constant, which is the
/// point: a signature from it means "decoded and shape-checked locally by
/// the `ruforecast` CLI," never "released."
const EVAL_ONLY_SEED: [u8; 32] = *b"ruforecast-eval-only-not-a-real!";

/// Self-signs `candidate_bytes` with the fixed evaluation-only key and
/// activates it on the CPU runtime for local scoring.
///
/// `expected_feature_schema_digest` is passed straight to
/// [`ArtifactActivationPolicy`] as-is: callers that want the schema check to
/// mean something should pass an independently-derived digest (e.g. from
/// their own held-out dataset's declared feature names), not the candidate's
/// own declared digest, which would make the check trivially pass. Never use
/// the result to make a release/production claim; see module docs.
pub fn activate_for_evaluation(
    candidate_bytes: &[u8],
    now_unix_ms: u64,
    expected_feature_schema_digest: [u8; 32],
) -> Result<CpuForecastRuntime, ModelError> {
    let candidate = ModelArtifact::decode(candidate_bytes)?;
    let signing_key = SigningKey::from_bytes(&EVAL_ONLY_SEED);
    let signature = signing_key
        .sign(&candidate.signing_message()?)
        .to_bytes();
    let public_key = signing_key.verifying_key().to_bytes();
    let encoded = SignedModelArtifact::new(&candidate, public_key, signature)?.encode()?;
    let trusted = TrustedSignerSet::new(vec![public_key])?;
    let policy = ArtifactActivationPolicy::new(
        candidate.manifest().release_epoch,
        candidate.manifest().minimum_runtime_version,
        now_unix_ms,
        expected_feature_schema_digest,
    )?;
    CpuForecastRuntime::activate(&encoded, &trusted, &policy)
}

/// Deterministic periodic UTC time features, matching the training encoder
/// in `ruview-forecast-train`'s batch builder.
fn time_features(step_seconds: u64, width: usize) -> Vec<f32> {
    let periods = [60.0_f64, 3_600.0, 86_400.0, 604_800.0];
    let mut values = Vec::with_capacity(width.max(periods.len() * 2));
    for period in periods {
        let angle = std::f64::consts::TAU * (step_seconds as f64 % period) / period;
        values.push(angle.sin() as f32);
        values.push(angle.cos() as f32);
    }
    values.resize(width, 0.0);
    values
}

/// Builds a single-window [`ModelInput`] for CPU inference from raw,
/// context-major values/mask arrays (the same layout as a training
/// `JsonlWindow`'s `values`/`observed_mask` fields: `row * variates +
/// variate`). Ages are zeroed — single-window evaluation has no prior
/// window to track staleness against.
#[allow(clippy::too_many_arguments)]
pub fn build_eval_input(
    config: &ForecastModelConfig,
    device: &CpuDevice,
    context_start_ms: u64,
    step_ms: u64,
    variates: usize,
    values: &[f32],
    observed_mask: &[u8],
) -> Result<ModelInput<CpuBackend>, ModelError> {
    let context_len = config.context_len;
    let horizon = config.horizon;
    if variates == 0
        || values.len() != context_len * variates
        || observed_mask.len() != context_len * variates
    {
        return Err(ModelError::Shape("eval input shape mismatch".into()));
    }
    let observed_f: Vec<f32> = observed_mask.iter().map(|mask| f32::from(*mask)).collect();
    let ages = vec![0.0_f32; context_len * variates];
    let mut context_time = Vec::with_capacity(context_len * config.time_width);
    for row in 0..context_len {
        let timestamp = context_start_ms.saturating_add((row as u64).saturating_mul(step_ms));
        context_time.extend(time_features(timestamp / 1_000, config.time_width));
    }
    let mut future_time = Vec::with_capacity(horizon * config.time_width);
    for step in 0..horizon {
        let row = context_len + step;
        let timestamp = context_start_ms.saturating_add((row as u64).saturating_mul(step_ms));
        future_time.extend(time_features(timestamp / 1_000, config.time_width));
    }
    let mut descriptors = Vec::with_capacity(variates * config.descriptor_width);
    for variate in 0..variates {
        let mut descriptor = vec![0.0_f32; config.descriptor_width];
        descriptor[variate % 32] = 1.0;
        descriptor[32 + variate % 8] = 1.0;
        descriptors.extend(descriptor);
    }
    ModelInput::new(
        config,
        Tensor::from_data(
            TensorData::new(values.to_vec(), [1, context_len, variates]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(observed_f, [1, context_len, variates]),
            device,
        ),
        Tensor::from_data(TensorData::new(ages, [1, context_len, variates]), device),
        Tensor::from_data(
            TensorData::new(context_time, [1, context_len, config.time_width]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(future_time, [1, horizon, config.time_width]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(descriptors, [1, variates, config.descriptor_width]),
            device,
        ),
        Tensor::from_data(TensorData::new(vec![1.0_f32; variates], [1, variates]), device),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArtifactManifest, ModelArtifact, TINY_PARAMETER_COUNT};

    fn unsigned_candidate() -> Vec<u8> {
        let weights = b"eval-module-test-record".to_vec();
        let manifest = ArtifactManifest {
            schema_version: 1,
            architecture: "ruview-factorized-forecast-mixer-v1".to_owned(),
            parameter_count: TINY_PARAMETER_COUNT,
            config: ForecastModelConfig::tiny_ci(),
            feature_schema_digest: [7; 32],
            training_manifest_digest: [2; 32],
            weights_digest: *blake3::hash(&weights).as_bytes(),
            seed: 7,
            release_epoch: 1,
            minimum_runtime_version: 1,
            maximum_runtime_version: 1,
            expires_at_unix_ms: None,
            build_id: "eval-module-test".into(),
            teacher_outputs_used: false,
            independently_implemented: true,
        };
        ModelArtifact::new(manifest, weights).unwrap().encode().unwrap()
    }

    #[test]
    fn activation_rejects_a_mismatched_expected_digest() {
        // The schema-digest check is real even on the eval-only path: only
        // signing is relaxed, not this comparison. A caller that wants the
        // check to mean something must pass the digest they actually expect.
        let candidate_bytes = unsigned_candidate();
        let result = activate_for_evaluation(&candidate_bytes, 5_000_000, [9; 32]);
        assert!(matches!(result, Err(ModelError::ActivationPolicy(_))));
    }

    #[test]
    fn activation_with_a_matching_digest_gets_past_the_policy_gate() {
        // This fixture's weights aren't a real trained Burn record, so full
        // activation still fails -- but on record decoding, not the policy
        // gate, which is what this test is checking. A real trained
        // candidate's successful activation is covered by the crate's
        // integration tests (`evaluate` CLI smoke test), which build one via
        // real training rather than a hand-built fixture.
        let candidate_bytes = unsigned_candidate();
        let result = activate_for_evaluation(&candidate_bytes, 5_000_000, [7; 32]);
        assert!(!matches!(result, Err(ModelError::ActivationPolicy(_))));
    }

    #[test]
    fn activation_rejects_a_malformed_candidate() {
        let bytes = b"not-a-model-artifact".to_vec();
        assert!(activate_for_evaluation(&bytes, 5_000_000, [0; 32]).is_err());
    }

    #[test]
    fn build_eval_input_rejects_shape_mismatch() {
        let config = ForecastModelConfig::tiny_ci();
        let device = CpuDevice::default();
        let too_few = vec![0.0_f32; 4];
        let mask = vec![1_u8; 4];
        let result = build_eval_input(&config, &device, 0, 1_000, 3, &too_few, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn build_eval_input_accepts_the_declared_shape() {
        let config = ForecastModelConfig::tiny_ci();
        let device = CpuDevice::default();
        let variates = 3;
        let values = vec![0.1_f32; config.context_len * variates];
        let mask = vec![1_u8; config.context_len * variates];
        build_eval_input(&config, &device, 1_000, 1_000, variates, &values, &mask)
            .expect("well-shaped eval input");
    }

    #[test]
    fn time_features_are_bounded_unit_circle_values() {
        let features = time_features(123_456, 8);
        assert_eq!(features.len(), 8);
        assert!(features.iter().all(|value| value.abs() <= 1.0));
    }
}
