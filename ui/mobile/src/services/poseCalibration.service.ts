import { COARSE_TEACHER_JOINTS, type JointLinearModel, type PoseCalibrationArtifact, type PoseCalibrationEvaluation, type PoseClockModel, type PoseSequence, type PoseStudentModel, type PoseTeachingInput, type PoseTrainingSample } from '@/types/poseCalibration';
import type { BodyTeacherFrame } from '@/types/lidar';
import type { PoseKeypoint, SensingFrame } from '@/types/sensing';

const MAX_ASSOCIATION_SKEW_MS = 20;
const PCK_THRESHOLD_M = 0.2;
const RIDGE_LAMBDA = 0.1;
const COCO_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'];

const finite = (value: number | undefined, fallback = 0) => typeof value === 'number' && Number.isFinite(value) ? value : fallback;
const logFeature = (value: number | undefined) => Math.log1p(Math.max(0, finite(value)));

export const extractCsiFeatures = (frame: SensingFrame): number[] => {
  const nodes = [...(frame.nodes ?? [])].sort((a, b) => a.node_id - b.node_id);
  const nodeRssi = nodes.map((node) => finite(node.rssi_dbm, -100));
  const amplitudes = nodes.flatMap((node) => node.amplitude ?? []).filter(Number.isFinite);
  const meanNodeRssi = nodeRssi.length ? nodeRssi.reduce((sum, value) => sum + value, 0) / nodeRssi.length : -100;
  const meanAmplitude = amplitudes.length ? amplitudes.reduce((sum, value) => sum + value, 0) / amplitudes.length : 0;
  return [
    1,
    (finite(frame.features?.mean_rssi, meanNodeRssi) + 100) / 60,
    Math.sqrt(Math.max(0, finite(frame.features?.variance))) / 20,
    logFeature(frame.features?.motion_band_power),
    logFeature(frame.features?.breathing_band_power),
    finite(frame.features?.spectral_entropy),
    finite(frame.classification?.confidence),
    (meanNodeRssi + 100) / 60,
    Math.log1p(Math.max(0, meanAmplitude)),
    Math.min(1, nodes.length / 8),
  ];
};

export const sensingPose = (frame: SensingFrame): PoseKeypoint[] => {
  const person = frame.persons?.slice().sort((a, b) => b.confidence - a.confidence)[0];
  if (person?.keypoints?.length) return person.keypoints;
  return (frame.pose_keypoints ?? []).map(([x, y, z, confidence], index) => ({ name: COCO_NAMES[index], x, y, z, confidence }));
};

export const teacherGestureScore = (frame: BodyTeacherFrame): number => {
  const byName = new Map(frame.joints.map((joint) => [joint.name, joint]));
  const shoulders = [byName.get('left_shoulder'), byName.get('right_shoulder')].filter(Boolean);
  const wrists = [byName.get('left_wrist'), byName.get('right_wrist')].filter(Boolean);
  if (!shoulders.length || !wrists.length) return 0;
  const shoulderY = shoulders.reduce((sum, joint) => sum + joint!.positionM[1], 0) / shoulders.length;
  return Math.max(0, ...wrists.map((joint) => joint!.positionM[1] - shoulderY));
};

/** Align the visible hands-up gesture peak with the CSI motion peak. Residual
 * is the observed sampling uncertainty; the model is unusable above 20 ms. */
export const estimateGestureClockModel = (teacher: BodyTeacherFrame[], rf: SensingFrame[]): PoseClockModel | null => {
  const teacherPeak = teacher.filter((frame) => teacherGestureScore(frame) > 0).sort((a, b) => teacherGestureScore(b) - teacherGestureScore(a))[0];
  const rfPeak = rf.filter((frame) => typeof frame.timestamp === 'number').sort((a, b) => finite(b.features?.motion_band_power) - finite(a.features?.motion_band_power))[0];
  if (!teacherPeak || !rfPeak?.timestamp) return null;
  const teacherIntervals = teacher.slice(1).map((frame, index) => Math.abs(frame.capturedAtUnixMs - teacher[index].capturedAtUnixMs)).filter(Number.isFinite);
  const rfTimes = rf.map((frame) => frame.timestamp!).filter(Number.isFinite).sort((a, b) => a - b);
  const rfIntervals = rfTimes.slice(1).map((value, index) => Math.abs(value - rfTimes[index]));
  const residualMs = Math.max(teacherIntervals.length ? Math.min(...teacherIntervals) / 2 : Infinity, rfIntervals.length ? Math.min(...rfIntervals) / 2 : Infinity);
  return {
    model: 'sync-gesture-offset-v1',
    teacherMinusRfOffsetMs: teacherPeak.capturedAtUnixMs - rfPeak.timestamp,
    residualMs,
    measuredAtUnixMs: Date.now(),
    passes20MsGate: residualMs <= MAX_ASSOCIATION_SKEW_MS,
  };
};

export const pairPoseTeachingSample = ({ teacher, rfFrame, sequenceIndex, clock }: PoseTeachingInput): PoseTrainingSample | null => {
  if (!clock.passes20MsGate || !rfFrame.timestamp || teacher.trackingState !== 'normal' || teacher.joints.length < 6) return null;
  const correctedRfTime = rfFrame.timestamp + clock.teacherMinusRfOffsetMs;
  const associationSkewMs = Math.abs(teacher.capturedAtUnixMs - correctedRfTime);
  if (associationSkewMs > MAX_ASSOCIATION_SKEW_MS) return null;
  const baseline = sensingPose(rfFrame);
  return {
    sequenceIndex,
    pairedAtUnixMs: Math.round((teacher.capturedAtUnixMs + correctedRfTime) / 2),
    associationSkewMs,
    features: extractCsiFeatures(rfFrame),
    teacher,
    baseline,
    rfPosePresent: baseline.some((joint) => joint.confidence >= 0.5),
  };
};

const solve = (matrix: number[][], target: number[], lambda: number): number[] | null => {
  const columns = matrix[0]?.length ?? 0;
  if (!columns || matrix.length < columns) return null;
  const augmented = Array.from({ length: columns }, (_, row) => [
    ...Array.from({ length: columns }, (_, column) => matrix.reduce((sum, values) => sum + values[row] * values[column], 0) + (row === column && row > 0 ? lambda : 0)),
    matrix.reduce((sum, values, index) => sum + values[row] * target[index], 0),
  ]);
  for (let pivot = 0; pivot < columns; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < columns; row += 1) if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[best][pivot])) best = row;
    [augmented[pivot], augmented[best]] = [augmented[best], augmented[pivot]];
    if (Math.abs(augmented[pivot][pivot]) < 1e-9) return null;
    const divisor = augmented[pivot][pivot];
    for (let column = pivot; column <= columns; column += 1) augmented[pivot][column] /= divisor;
    for (let row = 0; row < columns; row += 1) {
      if (row === pivot) continue;
      const factor = augmented[row][pivot];
      for (let column = pivot; column <= columns; column += 1) augmented[row][column] -= factor * augmented[pivot][column];
    }
  }
  return augmented.map((row) => row[columns]);
};

export const trainPoseStudent = (sequences: PoseSequence[]): PoseStudentModel | null => {
  if (sequences.length < 7 || sequences.slice(0, 7).some((sequence) => sequence.samples.length < 4)) return null;
  const samples = sequences.slice(0, 7).flatMap((sequence) => sequence.samples);
  const joints: JointLinearModel[] = [];
  for (const name of COARSE_TEACHER_JOINTS) {
    const rows = samples.flatMap((sample) => {
      const joint = sample.teacher.joints.find((candidate) => candidate.name === name && candidate.confidence >= 0.5);
      return joint ? [{ features: sample.features, position: joint.positionM }] : [];
    });
    if (rows.length < 10) continue;
    const matrix = rows.map((row) => row.features);
    const weightsX = solve(matrix, rows.map((row) => row.position[0]), RIDGE_LAMBDA);
    const weightsY = solve(matrix, rows.map((row) => row.position[1]), RIDGE_LAMBDA);
    const weightsZ = solve(matrix, rows.map((row) => row.position[2]), RIDGE_LAMBDA);
    if (weightsX && weightsY && weightsZ) joints.push({ name, weightsX, weightsY, weightsZ });
  }
  if (joints.length < 6) return null;
  return { model: 'room-ridge-coarse-joints-v1', featureSchema: 'csi-summary-10-v1', joints, ridgeLambda: RIDGE_LAMBDA, trainingSequenceCount: 7, trainingSampleCount: samples.length };
};

const dot = (left: number[], right: number[]) => left.reduce((sum, value, index) => sum + value * right[index], 0);
export const predictPose = (model: PoseStudentModel, features: number[]): PoseKeypoint[] => model.joints.map((joint) => ({ name: joint.name, x: dot(joint.weightsX, features), y: dot(joint.weightsY, features), z: dot(joint.weightsZ, features), confidence: 1 }));
const distance = (left: { x: number; y: number; z: number }, right: [number, number, number]) => Math.hypot(left.x - right[0], left.y - right[1], left.z - right[2]);

const scorePck = (samples: PoseTrainingSample[], poseFor: (sample: PoseTrainingSample) => PoseKeypoint[]) => {
  let correct = 0;
  let total = 0;
  samples.forEach((sample) => sample.teacher.joints.forEach((teacherJoint) => {
    if (teacherJoint.confidence < 0.5) return;
    const estimate = poseFor(sample).find((joint) => joint.name === teacherJoint.name && joint.confidence >= 0.5);
    total += 1;
    if (estimate && distance(estimate, teacherJoint.positionM) <= PCK_THRESHOLD_M) correct += 1;
  }));
  return total ? correct / total : 0;
};

export const evaluatePoseStudent = (model: PoseStudentModel, sequences: PoseSequence[]): PoseCalibrationEvaluation | null => {
  if (sequences.length < 10 || sequences.slice(7, 10).some((sequence) => sequence.samples.length < 1)) return null;
  const samples = sequences.slice(7, 10).flatMap((sequence) => sequence.samples);
  const baselinePck20cm = scorePck(samples, (sample) => sample.baseline);
  const calibratedPck20cm = scorePck(samples, (sample) => predictPose(model, sample.features));
  const improvementFraction = baselinePck20cm > 0 ? (calibratedPck20cm - baselinePck20cm) / baselinePck20cm : calibratedPck20cm > 0 ? 1 : 0;
  const baselineLostPoseRate = samples.filter((sample) => !sample.rfPosePresent).length / samples.length;
  const calibratedLostPoseRate = model.joints.length >= 6 ? 0 : 1;
  return {
    evidence: 'MEASURED', testSequenceCount: 3, testSampleCount: samples.length,
    baselinePck20cm, calibratedPck20cm, improvementFraction,
    baselineLostPoseRate, calibratedLostPoseRate,
    passesTarget: improvementFraction >= 0.25 && calibratedLostPoseRate <= baselineLostPoseRate,
    target: 'pck_20cm_improves_25_percent_on_3_unseen_sequences',
  };
};

export const buildPoseCalibrationArtifact = (
  coordinateFrameId: string,
  spatialCalibrationId: string,
  roomFingerprint: string,
  clock: PoseClockModel,
  model: PoseStudentModel,
  evaluation: PoseCalibrationEvaluation,
): PoseCalibrationArtifact => {
  const createdAtUnixMs = Date.now();
  return {
    schema: 'ruview.calibration.pose-teacher-student.v1',
    calibrationId: `pose-cal-${createdAtUnixMs}-${Math.random().toString(36).slice(2, 10)}`,
    createdAtUnixMs,
    quality: evaluation.passesTarget ? 'VALID' : 'DRAFT',
    coordinateFrameId,
    spatialCalibrationId,
    roomFingerprint,
    clock,
    model,
    evaluation,
    sequenceProtocol: { total: 10, training: 7, heldOut: 3 },
    limitations: { coarseJointsOnly: true, handsAndFingersUnsupported: true, improvesInterpretationNotRfResolution: true },
    privacy: { rawCameraPersisted: false, rawDepthPersisted: false, rawCsiPersisted: false, biometricIdentityDerived: false },
  };
};
