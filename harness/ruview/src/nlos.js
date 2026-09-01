// SPDX-License-Identifier: MIT
// Advisory consumer-NLOS planning and verification helpers.
//
// This module never captures a sensor, changes runtime state, or treats a build
// as sensing evidence. It is intentionally dependency-free so RuView remains
// fully usable when MetaHarness, Xcode, hardware, or optional UI surfaces are
// absent (ADR-331).

import { existsSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import { scrubEnvironment } from './process-runner.js';
import { redact } from './redact.js';

export const NLOS_EVIDENCE_SCHEMA = 'ruview.nlos.acceptance.v1';
export const NLOS_TRACK_SCHEMA = 'ruview.nlos.track.v1';
export const NLOS_REFERENCE_SENSOR_MODEL = 'VL53L8CH';
export const NLOS_UPSTREAM_COMMIT = '15314de422a765a2d1b72ea7037dfafb2f908d7c';

const MAX_EVIDENCE_BYTES = 1024 * 1024;
const MAX_SURFACE_FILE_BYTES = 4 * 1024 * 1024;
const SHA256 = /^[a-f0-9]{64}$/i;
const ZERO_SHA256 = /^0{64}$/;
const GIT_SHA1 = /^[a-f0-9]{40}$/i;
const ZERO_GIT_SHA1 = /^0{40}$/;
const SAFE_LABEL = /^[A-Za-z0-9._:+-]{1,128}$/;
const EVIDENCE_KEYS = new Set([
  'schema',
  'source',
  'claim_tag',
  'evidence_level',
  'ground_truth',
  'protocol_frozen_before_capture',
  'witness_reviewed',
  'privacy_review_passed',
  'security_review_passed',
  'los_exclusion_verified',
  'independent_csi_verified',
  'sensor_model',
  'transient_kind',
  'sensor_configured_max_hz',
  'upstream_commit',
  'protocol_sha256',
  'capture_manifest_sha256',
  'sensor_identity_sha256',
  'calibration_sha256',
  'firmware_sha256',
  'analysis_sha256',
  'endpoint_pairing_sha256',
  'sensor_configuration_sha256',
  'witness_report_sha256',
  'privacy_review_sha256',
  'security_review_sha256',
  'guardrail_report_sha256',
  'csi_capture_manifest_sha256',
  'csi_sensor_identity_sha256',
  'csi_calibration_sha256',
  'synthetic_frames',
  'replay_frames',
  'paired_sequences',
  'csi_source_count',
  'offered_optical_frame_count',
  'lidar_only',
  'csi_only',
  'fused',
  'confidence',
  'guardrails',
]);
const ARM_KEYS = new Set([
  'sequence_count',
  'accepted_sensor_frame_count',
  'accepted_update_count',
  'evaluable_duration_s',
  'update_hz',
  'position_error_sum_m',
  'position_error_sample_count',
  'position_error_m',
  'lost_track_duration_s',
  'evaluable_track_duration_s',
  'lost_track_rate',
]);
const CSI_ARM_KEYS = new Set([
  'sequence_count',
  'accepted_sensor_frame_count',
  'accepted_update_count',
  'evaluable_duration_s',
  'update_hz',
  'lost_track_duration_s',
  'evaluable_track_duration_s',
  'lost_track_rate',
]);
const CONFIDENCE_KEYS = new Set([
  'bootstrap_resamples',
  'bootstrap_seed_list_sha256',
  'familywise_confidence_level',
  'position_error_reduction_lower',
  'position_error_reduction_upper',
  'lost_track_reduction_lower',
  'lost_track_reduction_upper',
]);
const GUARDRAIL_KEYS = new Set([
  'empty_false_track_rate',
  'empty_false_track_rate_max',
  'fused_p95_latency_ms',
  'fused_p95_latency_max_ms',
  'fused_frame_loss_rate',
  'fused_frame_loss_rate_max',
  'exclusion_fraction',
  'exclusion_fraction_max',
  'fused_update_rate_ratio_min',
  'nonwinning_position_error_regression_max',
  'nonwinning_lost_track_regression_max',
]);

const SURFACES = Object.freeze([
  {
    id: 'rust-core',
    base: 'v2/crates/ruview-nlos',
    marker: 'v2/crates/ruview-nlos/Cargo.toml',
    required: [
      'v2/crates/ruview-nlos/Cargo.toml',
      'v2/crates/ruview-nlos/src/lib.rs',
    ],
  },
  {
    id: 'native-ios',
    base: 'ui/ios-nlos',
    marker: 'ui/ios-nlos/Package.swift',
    required: [
      'ui/ios-nlos/Package.swift',
      'ui/ios-nlos/RuViewNLOS.xcodeproj/project.pbxproj',
    ],
  },
  {
    id: 'web-ios',
    base: 'ui/mobile',
    marker: 'ui/mobile/src/types/nlos.ts',
    required: [
      'ui/mobile/src/types/nlos.ts',
      'ui/mobile/src/services/nlos.service.ts',
      'ui/mobile/src/stores/nlosStore.ts',
      'ui/mobile/src/hooks/useNlosStream.ts',
      'ui/mobile/src/screens/NLOSScreen/index.tsx',
    ],
  },
]);

function fileContains(path, needle) {
  try { return readFileSync(path, 'utf8').includes(needle); } catch { return false; }
}

function inspectRepoFile(repo, path) {
  const candidate = join(repo, path);
  if (!existsSync(candidate)) return { ok: false, reason: 'missing', candidate };
  try {
    const canonicalRepo = realpathSync(repo);
    const canonicalFile = realpathSync(candidate);
    const rel = relative(canonicalRepo, canonicalFile);
    if (rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel)) {
      return { ok: false, reason: 'outside-repo', candidate };
    }
    const stat = statSync(canonicalFile);
    if (!stat.isFile()) return { ok: false, reason: 'not-regular', candidate };
    if (stat.size > MAX_SURFACE_FILE_BYTES) return { ok: false, reason: 'oversize', candidate };
    return { ok: true, candidate: canonicalFile };
  } catch {
    return { ok: false, reason: 'unreadable', candidate };
  }
}

function inspectSurface(repo, surface) {
  const marker = inspectRepoFile(repo, surface.marker);
  if (!marker.ok && marker.reason === 'missing') {
    return { id: surface.id, status: 'ABSENT', base: surface.base, findings: [] };
  }
  const states = new Map(surface.required.map((path) => [path, inspectRepoFile(repo, path)]));
  const findings = [];
  if (!marker.ok) findings.push(`${surface.marker}:${marker.reason}`);
  for (const [path, state] of states) {
    if (!state.ok && path !== surface.marker) findings.push(`${path}:${state.reason}`);
  }

  if (surface.id === 'rust-core' && !findings.length) {
    const cargo = states.get('v2/crates/ruview-nlos/Cargo.toml').candidate;
    const lib = states.get('v2/crates/ruview-nlos/src/lib.rs').candidate;
    if (!fileContains(cargo, 'name = "ruview-nlos"')) findings.push('rust-core:crate-name-mismatch');
    if (!fileContains(lib, NLOS_TRACK_SCHEMA)) findings.push(`rust-core:${NLOS_TRACK_SCHEMA}:missing`);
  }
  if (surface.id === 'native-ios' && !findings.length) {
    const packageFile = states.get('ui/ios-nlos/Package.swift').candidate;
    if (!fileContains(packageFile, 'RuViewNLOSCore')) findings.push('native-ios:RuViewNLOSCore-product-missing');
    if (!fileContains(packageFile, 'RuViewNLOSApple')) findings.push('native-ios:RuViewNLOSApple-product-missing');
  }
  if (surface.id === 'web-ios' && !findings.length) {
    const typeFile = states.get('ui/mobile/src/types/nlos.ts').candidate;
    if (!fileContains(typeFile, NLOS_TRACK_SCHEMA)) findings.push(`web-ios:${NLOS_TRACK_SCHEMA}:missing`);
  }

  return {
    id: surface.id,
    status: findings.length ? 'MALFORMED' : 'PRESENT',
    base: surface.base,
    findings,
  };
}

function buildResult(id, command, result) {
  return {
    id,
    command: command.join(' '),
    status: result.ok ? 'PASS' : 'FAIL',
    exit: result.status,
    error: redact(result.error, { env: process.env }),
    stdout_tail: redact(result.stdout.slice(-1200), { env: process.env }),
    stderr_tail: redact(result.stderr.slice(-800), { env: process.env }),
  };
}

async function runBuilds(repo, surfaces, { runCommand, whichCommand }) {
  const results = [];
  const buildEnv = scrubEnvironment(process.env);
  const present = new Set(surfaces.filter((item) => item.status === 'PRESENT').map((item) => item.id));

  if (present.has('rust-core')) {
    if (!whichCommand('cargo')) results.push({ id: 'rust-core', status: 'SKIPPED', reason: 'cargo_unavailable' });
    else {
      const args = ['test', '--locked', '--offline', '-p', 'ruview-nlos'];
      results.push(buildResult('rust-core', ['cargo', ...args], await runCommand('cargo', args, {
        cwd: join(repo, 'v2'), timeout: 600000, env: buildEnv,
      })));
    }
  }

  if (present.has('native-ios')) {
    if (!whichCommand('swift')) results.push({ id: 'native-swift', status: 'SKIPPED', reason: 'swift_unavailable' });
    else {
      const args = ['test'];
      results.push(buildResult('native-swift', ['swift', ...args], await runCommand('swift', args, {
        cwd: join(repo, 'ui/ios-nlos'), timeout: 600000, env: buildEnv,
      })));
    }
    if (process.platform !== 'darwin' || !whichCommand('xcodebuild')) {
      results.push({ id: 'native-xcode', status: 'SKIPPED', reason: 'xcodebuild_requires_macos' });
    } else {
      const args = [
        '-project', 'RuViewNLOS.xcodeproj', '-scheme', 'RuViewNLOS', '-sdk', 'iphonesimulator',
        '-destination', 'generic/platform=iOS Simulator', 'CODE_SIGNING_ALLOWED=NO', 'build',
      ];
      results.push(buildResult('native-xcode', ['xcodebuild', ...args], await runCommand('xcodebuild', args, {
        cwd: join(repo, 'ui/ios-nlos'), timeout: 900000, env: buildEnv,
      })));
    }
  }

  if (present.has('web-ios')) {
    const mobile = join(repo, 'ui/mobile');
    if (!existsSync(join(mobile, 'node_modules'))) {
      results.push({ id: 'web-ios', status: 'SKIPPED', reason: 'node_modules_absent_run_npm_ci' });
    } else {
      const commands = [
        [process.execPath, [join(mobile, 'node_modules/jest/bin/jest.js'), '--runInBand'], 600000, 'web-test'],
        [process.execPath, [join(mobile, 'node_modules/typescript/bin/tsc'), '--noEmit'], 600000, 'web-types'],
        [process.execPath, [join(mobile, 'node_modules/eslint/bin/eslint.js'), '.'], 600000, 'web-lint'],
        [process.execPath, [join(mobile, 'node_modules/expo/bin/cli'), 'export', '--platform', 'web'], 900000, 'web-export'],
      ];
      for (const [cmd, args, timeout, id] of commands) {
        if (!existsSync(args[0])) results.push({ id, status: 'SKIPPED', reason: 'local_tool_unavailable' });
        else {
          const env = id === 'web-export'
            ? { ...buildEnv, CI: '1', EXPO_NO_TELEMETRY: '1', EXPO_OFFLINE: '1' }
            : buildEnv;
          results.push(buildResult(id, ['node', ...args], await runCommand(cmd, args, {
            cwd: mobile, timeout, env,
          })));
        }
      }
    }
  }
  return results;
}

function number(value, name, { min = 0, max = Infinity, positive = false } = {}) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return `${name}:must-be-finite-number`;
  if (positive ? value <= min : value < min) return `${name}:below-minimum`;
  if (value > max) return `${name}:above-maximum`;
  return null;
}

function nearlyEqual(actual, expected) {
  return Math.abs(actual - expected) <= Math.max(1e-9, Math.abs(expected) * 1e-6);
}

function isNonzeroSha256(value) {
  const text = String(value || '');
  return SHA256.test(text) && !ZERO_SHA256.test(text);
}

function validateArm(arm, name, pairedSequences, { position = true } = {}) {
  const findings = [];
  if (!arm || typeof arm !== 'object' || Array.isArray(arm)) return [`${name}:missing`];
  const allowedKeys = position ? ARM_KEYS : CSI_ARM_KEYS;
  for (const key of Object.keys(arm)) {
    if (!allowedKeys.has(key)) findings.push(`${name}.${key}:unknown-field`);
  }
  for (const key of allowedKeys) {
    if (!(key in arm)) findings.push(`${name}.${key}:missing`);
  }
  const numericFindings = [
    number(arm.update_hz, `${name}.update_hz`, { positive: true, max: 240 }),
    number(arm.lost_track_rate, `${name}.lost_track_rate`, { max: 1 }),
    number(arm.evaluable_duration_s, `${name}.evaluable_duration_s`, { positive: true, max: 1e9 }),
    number(arm.lost_track_duration_s, `${name}.lost_track_duration_s`, { max: 1e9 }),
    number(arm.evaluable_track_duration_s, `${name}.evaluable_track_duration_s`, { positive: true, max: 1e9 }),
  ];
  if (position) numericFindings.push(
    number(arm.position_error_m, `${name}.position_error_m`, { max: 1000 }),
    number(arm.position_error_sum_m, `${name}.position_error_sum_m`, { max: 1e12 }),
  );
  for (const finding of numericFindings) if (finding) findings.push(finding);
  for (const [key, min, max] of [
    ['sequence_count', 1, 1e6],
    ['accepted_sensor_frame_count', 1, 1e9],
    ['accepted_update_count', 1, 1e9],
    ...(position ? [['position_error_sample_count', 1, 1e6]] : []),
  ]) {
    if (!Number.isSafeInteger(arm[key]) || arm[key] < min || arm[key] > max) {
      findings.push(`${name}.${key}:invalid-safe-integer`);
    }
  }
  if (Number.isSafeInteger(pairedSequences) && arm.sequence_count !== pairedSequences) {
    findings.push(`${name}.sequence_count:must-equal-paired-sequences`);
  }
  if (Number.isSafeInteger(arm.accepted_update_count)
    && Number.isSafeInteger(arm.accepted_sensor_frame_count)
    && arm.accepted_update_count > arm.accepted_sensor_frame_count) {
    findings.push(`${name}.accepted_update_count:above-sensor-frame-count`);
  }
  if (position && Number.isSafeInteger(arm.position_error_sample_count)
    && Number.isSafeInteger(arm.accepted_update_count)
    && arm.position_error_sample_count > arm.accepted_update_count) {
    findings.push(`${name}.position_error_sample_count:above-update-count`);
  }
  if (position && Number.isSafeInteger(pairedSequences)
    && Number.isSafeInteger(arm.position_error_sample_count)
    && arm.position_error_sample_count !== pairedSequences) {
    findings.push(`${name}.position_error_sample_count:must-equal-paired-sequences`);
  }
  if (Number.isFinite(arm.lost_track_duration_s)
    && Number.isFinite(arm.evaluable_track_duration_s)
    && arm.lost_track_duration_s > arm.evaluable_track_duration_s) {
    findings.push(`${name}.lost_track_duration_s:above-evaluable-duration`);
  }
  if (Number.isSafeInteger(arm.accepted_update_count) && Number.isFinite(arm.evaluable_duration_s)
    && arm.evaluable_duration_s > 0 && Number.isFinite(arm.update_hz)
    && !nearlyEqual(arm.update_hz, arm.accepted_update_count / arm.evaluable_duration_s)) {
    findings.push(`${name}.update_hz:arithmetic-mismatch`);
  }
  if (position && Number.isSafeInteger(arm.position_error_sample_count) && arm.position_error_sample_count > 0
    && Number.isFinite(arm.position_error_sum_m) && Number.isFinite(arm.position_error_m)
    && !nearlyEqual(arm.position_error_m, arm.position_error_sum_m / arm.position_error_sample_count)) {
    findings.push(`${name}.position_error_m:arithmetic-mismatch`);
  }
  if (Number.isFinite(arm.lost_track_duration_s) && Number.isFinite(arm.evaluable_track_duration_s)
    && arm.evaluable_track_duration_s > 0 && Number.isFinite(arm.lost_track_rate)
    && !nearlyEqual(arm.lost_track_rate, arm.lost_track_duration_s / arm.evaluable_track_duration_s)) {
    findings.push(`${name}.lost_track_rate:arithmetic-mismatch`);
  }
  return findings;
}

function validateConfidence(confidence) {
  const findings = [];
  if (!confidence || typeof confidence !== 'object' || Array.isArray(confidence)) return ['confidence:missing'];
  for (const key of Object.keys(confidence)) {
    if (!CONFIDENCE_KEYS.has(key)) findings.push(`confidence.${key}:unknown-field`);
  }
  for (const key of CONFIDENCE_KEYS) {
    if (!(key in confidence)) findings.push(`confidence.${key}:missing`);
  }
  if (!Number.isSafeInteger(confidence.bootstrap_resamples)
    || confidence.bootstrap_resamples < 10_000 || confidence.bootstrap_resamples > 10_000_000) {
    findings.push('confidence.bootstrap_resamples:out-of-range');
  }
  if (!isNonzeroSha256(confidence.bootstrap_seed_list_sha256)) {
    findings.push('confidence.bootstrap_seed_list_sha256:invalid');
  }
  for (const finding of [
    number(confidence.familywise_confidence_level, 'confidence.familywise_confidence_level', { min: 0.975, max: 1 }),
    number(confidence.position_error_reduction_lower, 'confidence.position_error_reduction_lower', { min: -100, max: 1 }),
    number(confidence.position_error_reduction_upper, 'confidence.position_error_reduction_upper', { min: -100, max: 1 }),
    number(confidence.lost_track_reduction_lower, 'confidence.lost_track_reduction_lower', { min: -100, max: 1 }),
    number(confidence.lost_track_reduction_upper, 'confidence.lost_track_reduction_upper', { min: -100, max: 1 }),
  ]) if (finding) findings.push(finding);
  if (Number.isFinite(confidence.position_error_reduction_lower)
    && Number.isFinite(confidence.position_error_reduction_upper)
    && confidence.position_error_reduction_lower > confidence.position_error_reduction_upper) {
    findings.push('confidence.position_error_reduction:interval-inverted');
  }
  if (Number.isFinite(confidence.lost_track_reduction_lower)
    && Number.isFinite(confidence.lost_track_reduction_upper)
    && confidence.lost_track_reduction_lower > confidence.lost_track_reduction_upper) {
    findings.push('confidence.lost_track_reduction:interval-inverted');
  }
  return findings;
}

function validateGuardrails(guardrails) {
  const findings = [];
  if (!guardrails || typeof guardrails !== 'object' || Array.isArray(guardrails)) return ['guardrails:missing'];
  for (const key of Object.keys(guardrails)) {
    if (!GUARDRAIL_KEYS.has(key)) findings.push(`guardrails.${key}:unknown-field`);
  }
  for (const key of GUARDRAIL_KEYS) {
    if (!(key in guardrails)) findings.push(`guardrails.${key}:missing`);
  }
  for (const finding of [
    number(guardrails.empty_false_track_rate, 'guardrails.empty_false_track_rate', { max: 1 }),
    number(guardrails.empty_false_track_rate_max, 'guardrails.empty_false_track_rate_max', { max: 1 }),
    number(guardrails.fused_p95_latency_ms, 'guardrails.fused_p95_latency_ms', { max: 60_000 }),
    number(guardrails.fused_p95_latency_max_ms, 'guardrails.fused_p95_latency_max_ms', { positive: true, max: 60_000 }),
    number(guardrails.fused_frame_loss_rate, 'guardrails.fused_frame_loss_rate', { max: 1 }),
    number(guardrails.fused_frame_loss_rate_max, 'guardrails.fused_frame_loss_rate_max', { max: 1 }),
    number(guardrails.exclusion_fraction, 'guardrails.exclusion_fraction', { max: 1 }),
    number(guardrails.exclusion_fraction_max, 'guardrails.exclusion_fraction_max', { max: 1 }),
    number(guardrails.fused_update_rate_ratio_min, 'guardrails.fused_update_rate_ratio_min', { max: 1 }),
    number(guardrails.nonwinning_position_error_regression_max, 'guardrails.nonwinning_position_error_regression_max', { max: 10 }),
    number(guardrails.nonwinning_lost_track_regression_max, 'guardrails.nonwinning_lost_track_regression_max', { max: 10 }),
  ]) if (finding) findings.push(finding);
  for (const [value, limit, label] of [
    ['empty_false_track_rate', 'empty_false_track_rate_max', 'empty_false_track_rate'],
    ['fused_p95_latency_ms', 'fused_p95_latency_max_ms', 'fused_p95_latency_ms'],
    ['fused_frame_loss_rate', 'fused_frame_loss_rate_max', 'fused_frame_loss_rate'],
    ['exclusion_fraction', 'exclusion_fraction_max', 'exclusion_fraction'],
  ]) {
    if (Number.isFinite(guardrails[value]) && Number.isFinite(guardrails[limit])
      && guardrails[value] > guardrails[limit]) findings.push(`guardrails.${label}:above-frozen-limit`);
  }
  return findings;
}

/** Validate a preregistered, real-hardware comparison. Never accepts replay-only evidence. */
export function evaluateResearchEvidence(evidence) {
  if (evidence === null || evidence === undefined) {
    return {
      status: 'NOT_EVALUATED', pass: false,
      reason: 'no_evidence_file',
      note: 'Software or replay validation is not real-hardware NLOS evidence.',
    };
  }
  const findings = [];
  if (!evidence || typeof evidence !== 'object' || Array.isArray(evidence)) findings.push('evidence:must-be-object');
  if (findings.length) return { status: 'FAIL', pass: false, findings };

  for (const key of Object.keys(evidence)) {
    if (!EVIDENCE_KEYS.has(key)) findings.push(`${key}:unknown-field`);
  }
  for (const key of EVIDENCE_KEYS) {
    if (!(key in evidence)) findings.push(`${key}:missing`);
  }

  if (evidence.schema !== NLOS_EVIDENCE_SCHEMA) findings.push('schema:mismatch');
  if (evidence.source !== 'LIVE_HARDWARE') findings.push('source:must-be-LIVE_HARDWARE');
  if (evidence.claim_tag !== 'MEASURED') findings.push('claim_tag:must-be-MEASURED');
  if (evidence.evidence_level !== 'L2') findings.push('evidence_level:must-be-L2');
  if (evidence.ground_truth !== 'EXTERNAL') findings.push('ground_truth:must-be-EXTERNAL');
  if (evidence.protocol_frozen_before_capture !== true) findings.push('protocol:not-preregistered');
  for (const field of [
    'witness_reviewed', 'privacy_review_passed', 'security_review_passed',
    'los_exclusion_verified', 'independent_csi_verified',
  ]) if (evidence[field] !== true) findings.push(`${field}:must-be-true`);
  if (!SAFE_LABEL.test(String(evidence.sensor_model || ''))) findings.push('sensor_model:invalid');
  if (evidence.sensor_model !== NLOS_REFERENCE_SENSOR_MODEL) {
    findings.push(`sensor_model:v1-requires-${NLOS_REFERENCE_SENSOR_MODEL}`);
  }
  if (evidence.transient_kind !== 'COMPACT_NORMALIZED_HISTOGRAM'
    && evidence.transient_kind !== 'RAW_PHOTON_HISTOGRAM') {
    findings.push('transient_kind:unsupported');
  }
  const sensorRateFinding = number(
    evidence.sensor_configured_max_hz, 'sensor_configured_max_hz', { positive: true, max: 60 },
  );
  if (sensorRateFinding) findings.push(sensorRateFinding);
  if (!GIT_SHA1.test(String(evidence.upstream_commit || ''))
    || ZERO_GIT_SHA1.test(String(evidence.upstream_commit || ''))) {
    findings.push('upstream_commit:expected-nonzero-full-sha1');
  } else if (evidence.upstream_commit !== NLOS_UPSTREAM_COMMIT) {
    findings.push('upstream_commit:must-match-pinned-baseline');
  }
  for (const field of [
    'protocol_sha256', 'capture_manifest_sha256', 'sensor_identity_sha256',
    'calibration_sha256', 'firmware_sha256', 'analysis_sha256',
    'endpoint_pairing_sha256',
    'sensor_configuration_sha256', 'witness_report_sha256', 'privacy_review_sha256',
    'security_review_sha256', 'guardrail_report_sha256',
    'csi_capture_manifest_sha256', 'csi_sensor_identity_sha256', 'csi_calibration_sha256',
  ]) if (!isNonzeroSha256(evidence[field])) findings.push(`${field}:invalid-or-zero`);
  if (evidence.synthetic_frames !== 0) findings.push('synthetic_frames:must-be-zero');
  if (evidence.replay_frames !== 0) findings.push('replay_frames:must-be-zero');
  if (!Number.isSafeInteger(evidence.paired_sequences)
    || evidence.paired_sequences < 100 || evidence.paired_sequences > 1_000_000) {
    findings.push('paired_sequences:out-of-range');
  }
  if (!Number.isSafeInteger(evidence.csi_source_count)
    || evidence.csi_source_count < 1 || evidence.csi_source_count > 256) {
    findings.push('csi_source_count:out-of-range');
  }
  if (!Number.isSafeInteger(evidence.offered_optical_frame_count)
    || evidence.offered_optical_frame_count < 1 || evidence.offered_optical_frame_count > 1_000_000_000) {
    findings.push('offered_optical_frame_count:out-of-range');
  }
  findings.push(...validateArm(evidence.lidar_only, 'lidar_only', evidence.paired_sequences));
  findings.push(...validateArm(evidence.csi_only, 'csi_only', evidence.paired_sequences, { position: false }));
  findings.push(...validateArm(evidence.fused, 'fused', evidence.paired_sequences));
  findings.push(...validateConfidence(evidence.confidence));
  findings.push(...validateGuardrails(evidence.guardrails));
  for (const name of ['lidar_only', 'fused']) {
    const arm = evidence[name];
    if (arm && Number.isFinite(arm.evaluable_duration_s) && arm.evaluable_duration_s > 0
      && Number.isSafeInteger(arm.accepted_sensor_frame_count)
      && Number.isFinite(evidence.sensor_configured_max_hz)
      && arm.accepted_sensor_frame_count / arm.evaluable_duration_s > evidence.sensor_configured_max_hz + 1e-6) {
      findings.push(`${name}.accepted_sensor_frame_rate:above-configured-maximum`);
    }
    if (arm && Number.isSafeInteger(arm.accepted_sensor_frame_count)
      && Number.isSafeInteger(evidence.offered_optical_frame_count)
      && arm.accepted_sensor_frame_count > evidence.offered_optical_frame_count) {
      findings.push(`${name}.accepted_sensor_frame_count:above-offered-count`);
    }
  }
  if (evidence.lidar_only && Number.isSafeInteger(evidence.offered_optical_frame_count)
    && Number.isFinite(evidence.lidar_only.evaluable_duration_s)
    && evidence.lidar_only.evaluable_duration_s > 0
    && Number.isFinite(evidence.sensor_configured_max_hz)
    && evidence.offered_optical_frame_count / evidence.lidar_only.evaluable_duration_s
      > evidence.sensor_configured_max_hz + 1e-6) {
    findings.push('offered_optical_frame_rate:above-configured-maximum');
  }
  if (evidence.fused && evidence.guardrails
    && Number.isSafeInteger(evidence.offered_optical_frame_count)
    && evidence.offered_optical_frame_count > 0
    && Number.isSafeInteger(evidence.fused.accepted_sensor_frame_count)
    && Number.isFinite(evidence.guardrails.fused_frame_loss_rate)) {
    const measuredLoss = (evidence.offered_optical_frame_count
      - evidence.fused.accepted_sensor_frame_count) / evidence.offered_optical_frame_count;
    if (!nearlyEqual(evidence.guardrails.fused_frame_loss_rate, measuredLoss)) {
      findings.push('guardrails.fused_frame_loss_rate:arithmetic-mismatch');
    }
  }
  if (evidence.lidar_only && evidence.fused
    && !nearlyEqual(evidence.lidar_only.evaluable_duration_s, evidence.fused.evaluable_duration_s)) {
    findings.push('paired_optical_evaluable_duration:mismatch');
  }
  if (evidence.lidar_only && evidence.fused
    && !nearlyEqual(evidence.lidar_only.evaluable_track_duration_s, evidence.fused.evaluable_track_duration_s)) {
    findings.push('lost_track_endpoint_pairing:evaluable-duration-mismatch');
  }
  if (evidence.lidar_only && evidence.fused
    && evidence.lidar_only.position_error_sample_count !== evidence.fused.position_error_sample_count) {
    findings.push('position_endpoint_pairing:sample-count-mismatch');
  }
  if (evidence.lidar_only && evidence.fused && evidence.guardrails) {
    const updateRatio = evidence.lidar_only.update_hz > 0
      ? evidence.fused.update_hz / evidence.lidar_only.update_hz : 0;
    const positionRegression = evidence.lidar_only.position_error_m > 0
      ? Math.max(0, (evidence.fused.position_error_m - evidence.lidar_only.position_error_m)
        / evidence.lidar_only.position_error_m)
      : (evidence.fused.position_error_m > 0 ? Infinity : 0);
    const lostRegression = evidence.lidar_only.lost_track_rate > 0
      ? Math.max(0, (evidence.fused.lost_track_rate - evidence.lidar_only.lost_track_rate)
        / evidence.lidar_only.lost_track_rate)
      : (evidence.fused.lost_track_rate > 0 ? Infinity : 0);
    if (Number.isFinite(evidence.guardrails.fused_update_rate_ratio_min)
      && updateRatio < evidence.guardrails.fused_update_rate_ratio_min) {
      findings.push('guardrails.fused_update_rate_ratio:below-frozen-limit');
    }
    if (Number.isFinite(evidence.guardrails.nonwinning_position_error_regression_max)
      && positionRegression > evidence.guardrails.nonwinning_position_error_regression_max) {
      findings.push('guardrails.position_error_regression:above-frozen-limit');
    }
    if (Number.isFinite(evidence.guardrails.nonwinning_lost_track_regression_max)
      && lostRegression > evidence.guardrails.nonwinning_lost_track_regression_max) {
      findings.push('guardrails.lost_track_regression:above-frozen-limit');
    }
  }
  if (findings.length) return { status: 'FAIL', pass: false, findings };

  const baseError = evidence.lidar_only.position_error_m;
  const fusedError = evidence.fused.position_error_m;
  const baseLost = evidence.lidar_only.lost_track_rate;
  const fusedLost = evidence.fused.lost_track_rate;
  const positionErrorReduction = baseError > 0 ? (baseError - fusedError) / baseError : 0;
  const lostTrackReduction = baseLost > 0 ? (baseLost - fusedLost) / baseLost : 0;
  const reproductionPass = evidence.lidar_only.update_hz >= 27;
  const positionEndpointPass = positionErrorReduction >= 0.25
    && evidence.confidence.position_error_reduction_lower > 0
    && positionErrorReduction >= evidence.confidence.position_error_reduction_lower
    && positionErrorReduction <= evidence.confidence.position_error_reduction_upper;
  const lostTrackEndpointPass = lostTrackReduction >= 0.25
    && evidence.confidence.lost_track_reduction_lower > 0
    && lostTrackReduction >= evidence.confidence.lost_track_reduction_lower
    && lostTrackReduction <= evidence.confidence.lost_track_reduction_upper;
  const fusionPass = positionEndpointPass || lostTrackEndpointPass;
  const pass = reproductionPass && fusionPass;
  return {
    status: pass ? 'PASS' : 'FAIL',
    pass,
    reproduction: {
      pass: reproductionPass,
      threshold_hz: 27,
      measured_hz: evidence.lidar_only.update_hz,
      label: 'MEASURED',
    },
    fusion: {
      pass: fusionPass,
      measured_hz: evidence.fused.update_hz,
      threshold_fraction: 0.25,
      position_error_reduction: positionErrorReduction,
      lost_track_reduction: lostTrackReduction,
      position_endpoint_pass: positionEndpointPass,
      lost_track_endpoint_pass: lostTrackEndpointPass,
      adjusted_confidence_level: evidence.confidence.familywise_confidence_level,
      rule: 'position_error_reduction OR lost_track_reduction',
      label: 'MEASURED',
    },
    paired_sequences: evidence.paired_sequences,
    capture_manifest_sha256: evidence.capture_manifest_sha256,
    claim_tag: evidence.claim_tag,
    evidence_level: evidence.evidence_level,
  };
}

function readEvidence(repo, evidenceFile) {
  if (!evidenceFile) return { evidence: null, path: null };
  const candidate = resolve(repo, String(evidenceFile));
  if (!existsSync(candidate)) throw new Error('evidence_file_missing');
  const canonicalRepo = realpathSync(repo);
  const canonicalFile = realpathSync(candidate);
  const rel = relative(canonicalRepo, canonicalFile);
  if (rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel)) throw new Error('evidence_file_outside_repo');
  const stat = statSync(canonicalFile);
  if (!stat.isFile()) throw new Error('evidence_file_not_regular');
  if (stat.size > MAX_EVIDENCE_BYTES) throw new Error('evidence_file_too_large');
  let evidence;
  try { evidence = JSON.parse(readFileSync(canonicalFile, 'utf8')); }
  catch { throw new Error('evidence_file_malformed_json'); }
  return { evidence, path: rel.replaceAll('\\', '/') };
}

export function makeNlosPlan({ repoRoot = null } = {}) {
  return {
    ok: true,
    advisory: true,
    title: 'RuView consumer NLOS implementation plan',
    invariant: 'Preserve per-zone photon timing histograms; processed depth maps cannot substitute for transient samples.',
    phases: [
      {
        id: 1, name: 'Upstream reproduction',
        input: `Pinned ${NLOS_REFERENCE_SENSOR_MODEL} baseline at ${NLOS_UPSTREAM_COMMIT} with verified compact normalized histogram access and recorded silicon identity`,
        exit: 'LIVE_HARDWARE hidden-target tracking at >=27 Hz with frozen capture provenance',
      },
      {
        id: 2, name: 'RuView temporal state',
        input: 'Calibrated transient frames, pose, wall geometry, and canonical target response',
        exit: 'ruview.nlos.track.v1 hypotheses with covariance, freshness, calibration, and evidence labels',
      },
      {
        id: 3, name: 'CSI fusion',
        input: 'Synchronized optical likelihood and CSI likelihood in one world frame',
        exit: 'Paired preregistered comparison; >=25% lower position error OR lost-track rate',
      },
      {
        id: 4, name: 'Native and web iOS adapters',
        input: 'Authenticated versioned RuView tracks; ARKit scene depth only as pose/context unless raw transients become public',
        exit: 'Native build and authenticated web transport/replay pass without an iPhone-NLOS claim',
      },
    ],
    surfaces: SURFACES.map(({ id, base, marker }) => ({ id, base, marker })),
    verifier: {
      static: 'ruview nlos verify --repo <root>',
      builds: 'ruview nlos verify --repo <root> --run-builds',
      research: `ruview nlos verify --repo <root> --evidence-file <${NLOS_EVIDENCE_SCHEMA}.json> --require-research-pass`,
    },
    repo_detected: Boolean(repoRoot),
    repo_root: repoRoot,
  };
}

export async function verifyNlos(args = {}, {
  repoRoot = null,
  source = 'internal',
  runCommand = null,
  whichCommand = null,
} = {}) {
  if (source === 'mcp' && (args.repo !== undefined || args.run_builds === true)) {
    return { ok: false, reason: 'cli_only_repo_or_builds', advisory: true };
  }
  const repo = args.repo ? resolve(args.repo) : repoRoot;
  if (!repo) return { ok: false, reason: 'not_in_ruview_repo' };
  const isRepo = existsSync(join(repo, 'v2/Cargo.toml'))
    || existsSync(join(repo, 'archive/v1/data/proof/verify.py'));
  if (!isRepo) return { ok: false, reason: 'not_in_ruview_repo' };

  const surfaces = SURFACES.map((surface) => inspectSurface(repo, surface));
  const malformed = surfaces.filter((surface) => surface.status === 'MALFORMED');
  let loaded;
  try { loaded = readEvidence(repo, args.evidence_file); }
  catch (error) {
    return { ok: false, reason: String(error.message || error), surfaces, advisory: true };
  }
  const research = evaluateResearchEvidence(loaded.evidence);
  if (args.run_builds === true && (!runCommand || !whichCommand)) {
    return { ok: false, reason: 'build_executor_unavailable', surfaces, advisory: true };
  }
  const builds = args.run_builds === true
    ? await runBuilds(repo, surfaces, { runCommand, whichCommand })
    : [];
  const buildFailures = builds.filter((result) => result.status === 'FAIL');
  const requireResearch = args.require_research_pass === true;
  const evidenceSupplied = loaded.path !== null;
  const researchMustPass = requireResearch || evidenceSupplied;
  const ok = malformed.length === 0 && buildFailures.length === 0
    && (!researchMustPass || research.pass);
  const completedBuilds = builds.filter((result) => result.status !== 'SKIPPED');
  let softwareStatus = 'STATIC_DISCOVERY_ONLY';
  if (malformed.length || buildFailures.length) softwareStatus = 'FAIL';
  else if (args.run_builds && builds.length === 0) softwareStatus = 'NO_PRESENT_BUILD_SURFACES';
  else if (args.run_builds && completedBuilds.length === 0) softwareStatus = 'NO_BUILD_TOOLCHAINS_AVAILABLE';
  else if (args.run_builds && builds.some((result) => result.status === 'SKIPPED')) {
    softwareStatus = 'AVAILABLE_CHECKS_PASS_WITH_SKIPS';
  } else if (args.run_builds) softwareStatus = 'AVAILABLE_CHECKS_PASS';

  return {
    ok,
    advisory: true,
    meta_harness_required_by_runtime: false,
    track_schema: NLOS_TRACK_SCHEMA,
    surfaces,
    software_gate: {
      status: softwareStatus,
      malformed: malformed.map((surface) => surface.id),
      build_failures: buildFailures.map((result) => result.id),
      builds,
    },
    research_gate: { ...research, evidence_file: loaded.path },
    note: 'Static discovery and available-build results are advisory subsets of Gate A. They are never hardware or release validation.',
    build_authority: args.run_builds ? 'OPT_IN_LOCAL_CLI_TRUSTED_CHECKOUT' : 'NONE',
  };
}
