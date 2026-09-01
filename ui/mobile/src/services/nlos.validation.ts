import {
  NLOS_MAX_EXPIRY_WINDOW_MS,
  NLOS_MAX_MESSAGE_BYTES,
  NLOS_MAX_TRACKS,
  NLOS_TRACK_SCHEMA,
  type NlosEvidenceLevel,
  type NlosProvenance,
  type NlosRejectReason,
  type NlosTrack,
  type NlosTrackFrame,
  type NlosValidationResult,
  type NlosVector3,
} from '@/types/nlos';

const SAFE_ID = /^[A-Za-z0-9._:-]+$/;
const CALIBRATION_HASH = /^[0-9a-f]{64}$/;
const ZERO_CALIBRATION_HASH = '0'.repeat(64);
const EVIDENCE_LEVELS: ReadonlyArray<NlosEvidenceLevel> = [
  'l0_synthetic',
  'l1_measured',
  'l2_calibrated',
  'l3_corroborated',
];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const hasExactKeys = (value: Record<string, unknown>, keys: readonly string[]): boolean => {
  const actual = Object.keys(value);
  return actual.length === keys.length && actual.every((key) => keys.includes(key));
};

const isSafeId = (value: unknown, maxLength = 64): value is string =>
  typeof value === 'string' && value.length >= 1 && value.length <= maxLength && SAFE_ID.test(value);

const isSafeUint = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;

const isFiniteInRange = (value: unknown, min: number, max: number): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;

const isVector = (value: unknown, absoluteBound: number, nonNegative = false): value is NlosVector3 => {
  if (!isRecord(value) || !hasExactKeys(value, ['x', 'y', 'z'])) return false;
  const min = nonNegative ? 0 : -absoluteBound;
  return (
    isFiniteInRange(value.x, min, absoluteBound) &&
    isFiniteInRange(value.y, min, absoluteBound) &&
    isFiniteInRange(value.z, min, absoluteBound)
  );
};

const isProvenance = (value: unknown): value is NlosProvenance => {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'sensorId',
      'sensorModel',
      'firmwareVersion',
      'transientKind',
      'histogramPreserved',
      'transport',
    ])
  ) {
    return false;
  }

  return (
    isSafeId(value.sensorId, 64) &&
    isSafeId(value.sensorModel, 64) &&
    isSafeId(value.firmwareVersion, 64) &&
    (value.transientKind === 'raw_histogram' ||
      value.transientKind === 'compact_normalized_histogram' ||
      value.transientKind === 'depth_only' ||
      value.transientKind === 'replay') &&
    typeof value.histogramPreserved === 'boolean' &&
    (value.transport === 'usb_serial' || value.transport === 'ruview_server' || value.transport === 'replay')
  );
};

const isTrack = (value: unknown): value is NlosTrack => {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'trackId',
      'state',
      'positionM',
      'velocityMps',
      'covarianceDiagonalM2',
      'confidence',
      'posteriorEntropy',
      'signalQuality',
      'modalityContributions',
    ])
  ) {
    return false;
  }

  if (!isRecord(value.modalityContributions) || !hasExactKeys(value.modalityContributions, ['lidar', 'csi'])) {
    return false;
  }

  const lidar = value.modalityContributions.lidar;
  const csi = value.modalityContributions.csi;
  return (
    isSafeId(value.trackId) &&
    (value.state === 'tracking' || value.state === 'degraded' || value.state === 'unknown') &&
    isVector(value.positionM, 100) &&
    isVector(value.velocityMps, 20) &&
    isVector(value.covarianceDiagonalM2, 10, true) &&
    isFiniteInRange(value.confidence, 0, 1) &&
    typeof value.posteriorEntropy === 'number' &&
    Number.isFinite(value.posteriorEntropy) &&
    value.posteriorEntropy >= 0 &&
    isFiniteInRange(value.signalQuality, 0, 1) &&
    isFiniteInRange(lidar, 0, 1) &&
    isFiniteInRange(csi, 0, 1) &&
    lidar + csi >= 0.999 &&
    lidar + csi <= 1.001
  );
};

const classifyShapeFailure = (value: Record<string, unknown>): NlosRejectReason => {
  if (value.schema !== NLOS_TRACK_SCHEMA) return 'invalid_schema';
  if (
    !isSafeUint(value.sequence) ||
    !isSafeUint(value.capturedAtUnixMs) ||
    !isSafeUint(value.expiresAtUnixMs) ||
    !Array.isArray(value.tracks) ||
    value.tracks.length > NLOS_MAX_TRACKS
  ) {
    return 'invalid_bounds';
  }
  return 'invalid_shape';
};

export const utf8ByteLength = (value: string): number => {
  let bytes = 0;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code < 0x80) bytes += 1;
    else if (code < 0x800) bytes += 2;
    else if (code >= 0xd800 && code <= 0xdbff && index + 1 < value.length) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4;
        index += 1;
      } else {
        bytes += 3;
      }
    } else bytes += 3;
  }
  return bytes;
};

export const validateNlosTrackFrame = (value: unknown): NlosValidationResult => {
  if (!isRecord(value)) return { ok: false, reason: 'invalid_shape' };
  if (
    !hasExactKeys(value, [
      'schema',
      'sessionId',
      'sequence',
      'capturedAtUnixMs',
      'expiresAtUnixMs',
      'source',
      'evidenceLevel',
      'algorithmVersion',
      'calibrationHash',
      'provenance',
      'tracks',
    ]) ||
    value.schema !== NLOS_TRACK_SCHEMA ||
    !isSafeId(value.sessionId) ||
    !isSafeUint(value.sequence) ||
    !isSafeUint(value.capturedAtUnixMs) ||
    !isSafeUint(value.expiresAtUnixMs) ||
    (value.source !== 'live' && value.source !== 'replay' && value.source !== 'synthetic') ||
    !EVIDENCE_LEVELS.includes(value.evidenceLevel as NlosEvidenceLevel) ||
    !isSafeId(value.algorithmVersion) ||
    typeof value.calibrationHash !== 'string' ||
    !CALIBRATION_HASH.test(value.calibrationHash) ||
    !isProvenance(value.provenance) ||
    !Array.isArray(value.tracks) ||
    value.tracks.length > NLOS_MAX_TRACKS ||
    !value.tracks.every(isTrack)
  ) {
    return { ok: false, reason: classifyShapeFailure(value) };
  }

  const frame = value as unknown as NlosTrackFrame;
  if (frame.evidenceLevel === 'l3_corroborated') {
    return { ok: false, reason: 'invalid_provenance' };
  }
  const expiryWindow = frame.expiresAtUnixMs - frame.capturedAtUnixMs;
  if (expiryWindow <= 0 || expiryWindow > NLOS_MAX_EXPIRY_WINDOW_MS) {
    return { ok: false, reason: 'invalid_bounds' };
  }

  const evidenceIndex = EVIDENCE_LEVELS.indexOf(frame.evidenceLevel);
  const liveProvenanceValid =
    frame.source !== 'live' ||
    (evidenceIndex >= EVIDENCE_LEVELS.indexOf('l1_measured') &&
      frame.provenance.histogramPreserved &&
      frame.provenance.transientKind !== 'depth_only' &&
      frame.provenance.transientKind !== 'replay' &&
      frame.provenance.transport !== 'replay');
  const syntheticProvenanceValid =
    frame.source !== 'synthetic' ||
    (frame.evidenceLevel === 'l0_synthetic' &&
      frame.calibrationHash === ZERO_CALIBRATION_HASH &&
      frame.provenance.transport === 'replay' &&
      frame.provenance.transientKind === 'replay');
  const replayProvenanceValid =
    frame.source !== 'replay' ||
    (frame.provenance.transport === 'replay' &&
      frame.provenance.transientKind === 'replay' &&
      frame.provenance.histogramPreserved);
  const depthOnlyNotLive = frame.provenance.transientKind !== 'depth_only' || frame.source !== 'live';
  const calibratedHashValid =
    evidenceIndex < EVIDENCE_LEVELS.indexOf('l2_calibrated') ||
    frame.calibrationHash !== ZERO_CALIBRATION_HASH;
  const trackIds = new Set(frame.tracks.map((track) => track.trackId));
  const trackIdsUnique = trackIds.size === frame.tracks.length;

  if (
    !liveProvenanceValid ||
    !syntheticProvenanceValid ||
    !replayProvenanceValid ||
    !depthOnlyNotLive ||
    !calibratedHashValid ||
    !trackIdsUnique
  ) {
    return { ok: false, reason: 'invalid_provenance' };
  }

  return { ok: true, value: frame };
};

export const parseNlosTrackFrame = (raw: string): NlosValidationResult => {
  if (utf8ByteLength(raw) > NLOS_MAX_MESSAGE_BYTES) {
    return { ok: false, reason: 'message_too_large' };
  }

  try {
    return validateNlosTrackFrame(JSON.parse(raw) as unknown);
  } catch {
    return { ok: false, reason: 'malformed_json' };
  }
};
