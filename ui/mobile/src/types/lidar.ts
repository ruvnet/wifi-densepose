export type {
  NativeCameraPreviewFrame as CameraPreviewFrame,
  NativeBodyTeacherFrame as BodyTeacherFrame,
  NativeLidarCapabilities as LidarCapabilities,
  NativeLidarCaptureOptions as LidarCaptureOptions,
  NativeLidarPointFrame as LidarPointFrame,
  NativeLidarStatus as LidarStatus,
  NativeRoomCapture as RoomCapture,
  NativeRoomObject as RoomObject,
  NativeRoomSurface as RoomSurface,
  NativeSpatialPose as SpatialPose,
  NativeVisibleDepthDiagnostic as VisibleDepthDiagnostic,
  NativeVisibleDepthMetrics as VisibleDepthMetrics,
} from '../../modules/ruview-lidar/src';

import type {
  NativeRoomCapture,
  NativeSpatialPose,
} from '../../modules/ruview-lidar/src';

export type CalibrationQuality = 'DRAFT' | 'VALID';

export interface CalibratedNode {
  nodeId: string;
  role: 'esp32_csi_receiver' | 'wifi_transmitter' | 'reference';
  capturedAtUnixMs: number;
  coordinateFrameId: string;
  positionM: [number, number, number];
  transform: number[];
  trackingState: string;
}

export interface SpatialCalibrationArtifact {
  schema: 'ruview.calibration.spatial.v1';
  calibrationId: string;
  createdAtUnixMs: number;
  roomId: string;
  coordinateFrameId: string;
  quality: CalibrationQuality;
  room: NativeRoomCapture;
  nodes: CalibratedNode[];
  referenceWalk?: ReferenceWalk;
  correctionModel?: SpatialCorrectionModel;
  validation?: CalibrationValidation;
  staleness: CalibrationStaleness;
  privacy: {
    rawCameraPersisted: false;
    rawDepthPersisted: false;
    boundedGeometryOnly: true;
  };
  digestAlgorithm: 'SHA-256';
  digestSha256: string;
}

export interface ReferenceWalkSample {
  capturedAtUnixMs: number;
  coordinateFrameId: string;
  referencePositionM: [number, number, number];
  rfPositionM: [number, number, number];
  rfTrackId: string;
  rfConfidence: number;
  rfCapturedAtUnixMs: number;
  poseAssociationSkewMs: number;
}

export interface ReferenceWalk {
  referenceKind: 'visible_device_path_proxy';
  evidence: 'MEASURED';
  samples: ReferenceWalkSample[];
  attemptedSamples: number;
  lostTrackRate: number;
}

export interface SpatialCorrectionModel {
  model: 'similarity_2d_plus_y_offset';
  scale: number;
  yawRadians: number;
  translationM: [number, number, number];
  transform: number[];
  baselineMedianErrorM: number;
  fittedMedianErrorM: number;
  fittedImprovementFraction: number;
  sampleCount: number;
  residualCovarianceM2: number[];
}

export interface CalibrationValidation {
  evidence: 'MEASURED';
  sampleCount: number;
  attemptedSamples: number;
  rawMedianErrorM: number;
  calibratedMedianErrorM: number;
  improvementFraction: number;
  baselineLostTrackRate: number;
  calibratedLostTrackRate: number;
  passesTarget: boolean;
  target: 'median_error_improves_25_percent_without_more_lost_tracks';
}

export interface CalibrationStaleness {
  state: 'CURRENT' | 'STALE';
  roomFingerprint: string;
  maximumResidualM: number;
  reason: string | null;
}

export const poseCanCalibrate = (
  pose: NativeSpatialPose | null,
  room: NativeRoomCapture | null,
) => Boolean(
  pose
  && room
  && pose.coordinateFrameId === room.coordinateFrameId
  && pose.trackingState === 'normal',
);

export const validateLidarPointFrame = (value: unknown): value is import('../../modules/ruview-lidar/src').NativeLidarPointFrame => {
  if (!value || typeof value !== 'object') return false;
  const frame = value as Record<string, unknown>;
  const pointCount = frame.pointCount;
  return frame.schema === 'ruview.lidar.points.v1'
    && typeof frame.sessionId === 'string'
    && typeof frame.coordinateFrameId === 'string'
    && Number.isInteger(frame.sequence)
    && typeof pointCount === 'number'
    && Number.isInteger(pointCount)
    && pointCount >= 0
    && pointCount <= 4096
    && Array.isArray(frame.points)
    && frame.points.length === pointCount * 3
    && frame.points.every(Number.isFinite)
    && Array.isArray(frame.confidences)
    && frame.confidences.length === pointCount
    && frame.confidences.every((entry) => Number.isInteger(entry) && entry >= 0 && entry <= 2)
    && Array.isArray(frame.cameraTransform)
    && frame.cameraTransform.length === 16
    && Array.isArray(frame.cameraIntrinsics)
    && frame.cameraIntrinsics.length === 9
    && frame.rawDepthPersisted === false
    && frame.capturedImagePersisted === false;
};
