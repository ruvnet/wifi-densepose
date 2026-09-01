// SPDX-License-Identifier: MIT
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import {
  evaluateResearchEvidence,
  NLOS_EVIDENCE_SCHEMA,
  NLOS_REFERENCE_SENSOR_MODEL,
  NLOS_TRACK_SCHEMA,
  NLOS_UPSTREAM_COMMIT,
} from '../src/nlos.js';
import { runTool } from '../src/tools.js';
import { run as cliRun } from '../bin/cli.js';

const DIGEST = 'a'.repeat(64);

function arm(updateHz, positionErrorM, lostTrackRate) {
  const evaluableDurationS = 100;
  const acceptedUpdateCount = Math.round(updateHz * evaluableDurationS);
  return {
    sequence_count: 100,
    accepted_sensor_frame_count: 3_000,
    accepted_update_count: acceptedUpdateCount,
    evaluable_duration_s: evaluableDurationS,
    update_hz: acceptedUpdateCount / evaluableDurationS,
    position_error_sum_m: positionErrorM * 100,
    position_error_sample_count: 100,
    position_error_m: positionErrorM,
    lost_track_duration_s: lostTrackRate * evaluableDurationS,
    evaluable_track_duration_s: evaluableDurationS,
    lost_track_rate: lostTrackRate,
  };
}

function csiArm(updateHz, lostTrackRate) {
  const value = arm(updateHz, 0, lostTrackRate);
  delete value.position_error_sum_m;
  delete value.position_error_sample_count;
  delete value.position_error_m;
  return value;
}

function passingEvidence(overrides = {}) {
  return {
    schema: NLOS_EVIDENCE_SCHEMA,
    source: 'LIVE_HARDWARE',
    claim_tag: 'MEASURED',
    evidence_level: 'L2',
    ground_truth: 'EXTERNAL',
    protocol_frozen_before_capture: true,
    witness_reviewed: true,
    privacy_review_passed: true,
    security_review_passed: true,
    los_exclusion_verified: true,
    independent_csi_verified: true,
    sensor_model: NLOS_REFERENCE_SENSOR_MODEL,
    transient_kind: 'COMPACT_NORMALIZED_HISTOGRAM',
    sensor_configured_max_hz: 30,
    upstream_commit: NLOS_UPSTREAM_COMMIT,
    protocol_sha256: DIGEST,
    capture_manifest_sha256: DIGEST,
    sensor_identity_sha256: DIGEST,
    calibration_sha256: DIGEST,
    firmware_sha256: DIGEST,
    analysis_sha256: DIGEST,
    endpoint_pairing_sha256: DIGEST,
    sensor_configuration_sha256: DIGEST,
    witness_report_sha256: DIGEST,
    privacy_review_sha256: DIGEST,
    security_review_sha256: DIGEST,
    guardrail_report_sha256: DIGEST,
    csi_capture_manifest_sha256: DIGEST,
    csi_sensor_identity_sha256: DIGEST,
    csi_calibration_sha256: DIGEST,
    synthetic_frames: 0,
    replay_frames: 0,
    paired_sequences: 100,
    csi_source_count: 1,
    offered_optical_frame_count: 3_000,
    lidar_only: arm(29.4, 0.12, 0.20),
    csi_only: csiArm(15, 0.40),
    fused: arm(28.9, 0.089, 0.19),
    confidence: {
      bootstrap_resamples: 10_000,
      bootstrap_seed_list_sha256: DIGEST,
      familywise_confidence_level: 0.975,
      position_error_reduction_lower: 0.01,
      position_error_reduction_upper: 0.40,
      lost_track_reduction_lower: -0.10,
      lost_track_reduction_upper: 0.20,
    },
    guardrails: {
      empty_false_track_rate: 0,
      empty_false_track_rate_max: 0.01,
      fused_p95_latency_ms: 80,
      fused_p95_latency_max_ms: 100,
      fused_frame_loss_rate: 0,
      fused_frame_loss_rate_max: 0.05,
      exclusion_fraction: 0.02,
      exclusion_fraction_max: 0.05,
      fused_update_rate_ratio_min: 0.25,
      nonwinning_position_error_regression_max: 0.10,
      nonwinning_lost_track_regression_max: 0.10,
    },
    ...overrides,
  };
}

test('nlos plan is advisory, staged, and names the canonical track schema', async () => {
  const result = await runTool('ruview_nlos_plan', {});
  assert.equal(result.ok, true);
  assert.equal(result.advisory, true);
  assert.equal(result.phases.length, 4);
  assert.match(JSON.stringify(result), new RegExp(NLOS_TRACK_SCHEMA.replaceAll('.', '\\.')));
});

test('research gate accepts live hardware at roughly 30 Hz and either 25% fusion gain', () => {
  const result = evaluateResearchEvidence(passingEvidence());
  assert.equal(result.pass, true, JSON.stringify(result));
  assert.equal(result.reproduction.pass, true);
  assert.ok(result.fusion.position_error_reduction >= 0.25);

  const lostTrackWin = passingEvidence({
    lidar_only: arm(27, 0.10, 0.20),
    fused: arm(19, 0.10, 0.14),
    confidence: {
      bootstrap_resamples: 10_000,
      bootstrap_seed_list_sha256: DIGEST,
      familywise_confidence_level: 0.975,
      position_error_reduction_lower: -0.10,
      position_error_reduction_upper: 0.10,
      lost_track_reduction_lower: 0.01,
      lost_track_reduction_upper: 0.50,
    },
  });
  assert.equal(evaluateResearchEvidence(lostTrackWin).pass, true);
});

test('research gate never accepts synthetic or replay evidence', () => {
  for (const evidence of [
    passingEvidence({ source: 'SYNTHETIC' }),
    passingEvidence({ synthetic_frames: 1 }),
    passingEvidence({ replay_frames: 1 }),
    passingEvidence({ claim_tag: 'SYNTHETIC', evidence_level: 'L0' }),
  ]) {
    const result = evaluateResearchEvidence(evidence);
    assert.equal(result.pass, false, JSON.stringify(result));
    assert.equal(result.status, 'FAIL');
  }
});

test('research gate fails sub-rate reproduction and sub-threshold fusion', () => {
  const slow = passingEvidence({
    lidar_only: arm(26.9, 0.12, 0.20),
  });
  assert.equal(evaluateResearchEvidence(slow).reproduction.pass, false);

  const noGain = passingEvidence({
    fused: arm(29, 0.091, 0.16),
  });
  assert.equal(evaluateResearchEvidence(noGain).pass, false);

  const slowFusion = passingEvidence({ fused: arm(12, 0.08, 0.19) });
  assert.equal(evaluateResearchEvidence(slowFusion).pass, true, 'fused rate is reported but is not the user hard gate');
});

test('research evidence rejects unknown and incomplete fields', () => {
  const extraTopLevel = evaluateResearchEvidence(passingEvidence({ bearer_token: 'must-not-be-here' }));
  assert.equal(extraTopLevel.pass, false);
  assert.ok(extraTopLevel.findings.includes('bearer_token:unknown-field'));

  const incompleteArm = passingEvidence();
  delete incompleteArm.fused.sequence_count;
  const missing = evaluateResearchEvidence(incompleteArm);
  assert.equal(missing.pass, false);
  assert.ok(missing.findings.includes('fused.sequence_count:missing'));

  const impossible = passingEvidence({ sensor_identity_sha256: '0'.repeat(64) });
  impossible.lidar_only.update_hz = 1e300;
  const bounded = evaluateResearchEvidence(impossible);
  assert.equal(bounded.pass, false);
  assert.ok(bounded.findings.includes('sensor_identity_sha256:invalid-or-zero'));
  assert.ok(bounded.findings.includes('lidar_only.update_hz:above-maximum'));

  for (const sensorModel of ['VL53L8CX', 'iPhone15Pro']) {
    const wrongSensor = evaluateResearchEvidence(passingEvidence({ sensor_model: sensorModel }));
    assert.equal(wrongSensor.pass, false);
    assert.ok(wrongSensor.findings.includes('sensor_model:v1-requires-VL53L8CH'));
  }

  const zeroCommit = evaluateResearchEvidence(passingEvidence({ upstream_commit: '0'.repeat(40) }));
  assert.equal(zeroCommit.pass, false);
  assert.ok(zeroCommit.findings.includes('upstream_commit:expected-nonzero-full-sha1'));

  const wrongCommit = evaluateResearchEvidence(passingEvidence({
    upstream_commit: '0123456789abcdef0123456789abcdef01234567',
  }));
  assert.equal(wrongCommit.pass, false);
  assert.ok(wrongCommit.findings.includes('upstream_commit:must-match-pinned-baseline'));
});

test('research evidence recomputes loss and requires powered paired endpoint coverage', () => {
  const falseLoss = passingEvidence();
  falseLoss.fused.accepted_sensor_frame_count = 2_700;
  falseLoss.guardrails.fused_frame_loss_rate = 0;
  const lossResult = evaluateResearchEvidence(falseLoss);
  assert.equal(lossResult.pass, false);
  assert.ok(lossResult.findings.includes('guardrails.fused_frame_loss_rate:arithmetic-mismatch'));

  const impossibleOfferRate = passingEvidence({ offered_optical_frame_count: 3_001 });
  impossibleOfferRate.lidar_only.accepted_sensor_frame_count = 3_001;
  impossibleOfferRate.fused.accepted_sensor_frame_count = 3_001;
  const offerResult = evaluateResearchEvidence(impossibleOfferRate);
  assert.equal(offerResult.pass, false);
  assert.ok(offerResult.findings.includes('offered_optical_frame_rate:above-configured-maximum'));

  const oneOfOneHundred = passingEvidence();
  oneOfOneHundred.lidar_only.position_error_sample_count = 1;
  oneOfOneHundred.lidar_only.position_error_sum_m = 0.12;
  oneOfOneHundred.fused.position_error_sample_count = 1;
  oneOfOneHundred.fused.position_error_sum_m = 0.089;
  const coverageResult = evaluateResearchEvidence(oneOfOneHundred);
  assert.equal(coverageResult.pass, false);
  assert.ok(coverageResult.findings.includes(
    'lidar_only.position_error_sample_count:must-equal-paired-sequences',
  ));

  const mismatchedTrackDenominator = passingEvidence();
  mismatchedTrackDenominator.fused.evaluable_track_duration_s = 10;
  mismatchedTrackDenominator.fused.lost_track_duration_s = 1.5;
  mismatchedTrackDenominator.fused.lost_track_rate = 0.15;
  const pairingResult = evaluateResearchEvidence(mismatchedTrackDenominator);
  assert.equal(pairingResult.pass, false);
  assert.ok(pairingResult.findings.includes('lost_track_endpoint_pairing:evaluable-duration-mismatch'));
});

test('nlos verify degrades cleanly when all optional feature surfaces are absent', async () => {
  const repo = mkdtempSync(join(tmpdir(), 'ruview-nlos-absent-'));
  try {
    mkdirSync(join(repo, 'v2'), { recursive: true });
    writeFileSync(join(repo, 'v2', 'Cargo.toml'), '[workspace]\n');
    const result = await runTool('ruview_nlos_verify', { repo });
    assert.equal(result.ok, true, JSON.stringify(result));
    assert.ok(result.surfaces.every((surface) => surface.status === 'ABSENT'));
    assert.equal(result.research_gate.status, 'NOT_EVALUATED');
    assert.equal(result.meta_harness_required_by_runtime, false);
  } finally { rmSync(repo, { recursive: true, force: true }); }
});

test('nlos verify fails a malformed present surface', async () => {
  const repo = mkdtempSync(join(tmpdir(), 'ruview-nlos-malformed-'));
  try {
    mkdirSync(join(repo, 'v2', 'crates', 'ruview-nlos'), { recursive: true });
    writeFileSync(join(repo, 'v2', 'Cargo.toml'), '[workspace]\n');
    writeFileSync(join(repo, 'v2', 'crates', 'ruview-nlos', 'Cargo.toml'), '[package]\nname="wrong"\n');
    const result = await runTool('ruview_nlos_verify', { repo });
    assert.equal(result.ok, false);
    assert.equal(result.surfaces.find((surface) => surface.id === 'rust-core').status, 'MALFORMED');
  } finally { rmSync(repo, { recursive: true, force: true }); }
});

test('nlos evidence is confined, bounded, and any supplied failing record fails closed', async () => {
  const repo = mkdtempSync(join(tmpdir(), 'ruview-nlos-evidence-'));
  const outside = join(tmpdir(), `ruview-nlos-outside-${process.pid}.json`);
  try {
    mkdirSync(join(repo, 'v2'), { recursive: true });
    writeFileSync(join(repo, 'v2', 'Cargo.toml'), '[workspace]\n');
    writeFileSync(join(repo, 'evidence.json'), JSON.stringify(passingEvidence()));
    writeFileSync(join(repo, 'failing.json'), JSON.stringify(passingEvidence({ source: 'SYNTHETIC' })));
    writeFileSync(join(repo, 'malformed.json'), '{"schema":');
    writeFileSync(join(repo, 'oversize.json'), ' '.repeat(1024 * 1024 + 1));
    writeFileSync(outside, JSON.stringify(passingEvidence()));

    const pass = await runTool('ruview_nlos_verify', {
      repo, evidence_file: 'evidence.json', require_research_pass: true,
    });
    assert.equal(pass.ok, true, JSON.stringify(pass));
    assert.equal(pass.research_gate.status, 'PASS');

    const failing = await runTool('ruview_nlos_verify', { repo, evidence_file: 'failing.json' });
    assert.equal(failing.ok, false);
    assert.equal(failing.research_gate.status, 'FAIL');

    const malformed = await runTool('ruview_nlos_verify', { repo, evidence_file: 'malformed.json' });
    assert.equal(malformed.ok, false);
    assert.equal(malformed.reason, 'evidence_file_malformed_json');

    const oversized = await runTool('ruview_nlos_verify', { repo, evidence_file: 'oversize.json' });
    assert.equal(oversized.ok, false);
    assert.equal(oversized.reason, 'evidence_file_too_large');

    const escaped = await runTool('ruview_nlos_verify', { repo, evidence_file: outside });
    assert.equal(escaped.ok, false);
    assert.equal(escaped.reason, 'evidence_file_outside_repo');
  } finally {
    rmSync(repo, { recursive: true, force: true });
    rmSync(outside, { force: true });
  }
});

test('MCP NLOS verification cannot select a repository or execute build tools', async () => {
  const selected = await runTool('ruview_nlos_verify', { repo: '/tmp' }, { source: 'mcp' });
  assert.equal(selected.ok, false);
  assert.equal(selected.reason, 'cli_only_repo_or_builds');
  const builds = await runTool('ruview_nlos_verify', { run_builds: true }, { source: 'mcp' });
  assert.equal(builds.ok, false);
  assert.equal(builds.reason, 'cli_only_repo_or_builds');
});

test('nlos CLI nested commands reject unknown actions', async () => {
  assert.equal(await cliRun(['nlos', 'plan']), 0);
  assert.equal(await cliRun(['nlos', 'definitely-unknown']), 2);
});
