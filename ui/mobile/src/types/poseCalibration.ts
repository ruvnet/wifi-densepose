import type { BodyTeacherFrame } from '@/types/lidar';
import type { PoseKeypoint, SensingFrame } from '@/types/sensing';

export const COARSE_TEACHER_JOINTS = [
  'nose', 'neck', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'pelvis', 'left_hip', 'right_hip', 'left_knee',
  'right_knee', 'left_ankle', 'right_ankle',
] as const;

export type CoarseTeacherJoint = typeof COARSE_TEACHER_JOINTS[number];

export interface PoseClockModel {
  model: 'sync-gesture-offset-v1';
  teacherMinusRfOffsetMs: number;
  residualMs: number;
  measuredAtUnixMs: number;
  passes20MsGate: boolean;
}

export interface PoseTrainingSample {
  sequenceIndex: number;
  pairedAtUnixMs: number;
  associationSkewMs: number;
  features: number[];
  teacher: BodyTeacherFrame;
  baseline: PoseKeypoint[];
  rfPosePresent: boolean;
}

export interface PoseSequence {
  index: number;
  samples: PoseTrainingSample[];
  attemptedPairs: number;
}

export interface JointLinearModel {
  name: CoarseTeacherJoint;
  weightsX: number[];
  weightsY: number[];
  weightsZ: number[];
}

export interface PoseStudentModel {
  model: 'room-ridge-coarse-joints-v1';
  featureSchema: 'csi-summary-10-v1';
  joints: JointLinearModel[];
  ridgeLambda: number;
  trainingSequenceCount: 7;
  trainingSampleCount: number;
}

export interface PoseCalibrationEvaluation {
  evidence: 'MEASURED';
  testSequenceCount: 3;
  testSampleCount: number;
  baselinePck20cm: number;
  calibratedPck20cm: number;
  improvementFraction: number;
  baselineLostPoseRate: number;
  calibratedLostPoseRate: number;
  passesTarget: boolean;
  target: 'pck_20cm_improves_25_percent_on_3_unseen_sequences';
}

export interface PoseCalibrationArtifact {
  schema: 'ruview.calibration.pose-teacher-student.v1';
  calibrationId: string;
  createdAtUnixMs: number;
  quality: 'DRAFT' | 'VALID';
  coordinateFrameId: string;
  spatialCalibrationId: string;
  roomFingerprint: string;
  clock: PoseClockModel;
  model: PoseStudentModel;
  evaluation: PoseCalibrationEvaluation;
  sequenceProtocol: { total: 10; training: 7; heldOut: 3 };
  limitations: {
    coarseJointsOnly: true;
    handsAndFingersUnsupported: true;
    improvesInterpretationNotRfResolution: true;
  };
  privacy: { rawCameraPersisted: false; rawDepthPersisted: false; rawCsiPersisted: false; biometricIdentityDerived: false };
}

export interface PoseTeachingInput {
  teacher: BodyTeacherFrame;
  rfFrame: SensingFrame;
  sequenceIndex: number;
  clock: PoseClockModel;
}
