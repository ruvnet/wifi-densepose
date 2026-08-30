jest.mock('expo-crypto', () => ({
  CryptoDigestAlgorithm: { SHA256: 'SHA-256' },
  digestStringAsync: jest.fn(async () => 'a'.repeat(64)),
}));

import { buildSpatialCalibration, fitSpatialCorrection, validateSpatialCorrection } from '@/services/lidar.service';
import { validateLidarPointFrame } from '@/types/lidar';
import type { CalibratedNode, RoomCapture } from '@/types/lidar';

const frame = {
  schema: 'ruview.lidar.points.v1',
  sessionId: 'session-1',
  coordinateFrameId: 'frame-1',
  sequence: 1,
  capturedAtUnixMs: 1,
  monotonicTimestampSeconds: 1,
  points: [0, 0, -1],
  confidences: [2],
  pointCount: 1,
  cameraTransform: Array(16).fill(0),
  cameraIntrinsics: Array(9).fill(0),
  depthWidth: 1,
  depthHeight: 1,
  smoothed: true,
  trackingState: 'normal',
  rawDepthPersisted: false,
  capturedImagePersisted: false,
} as const;

const room: RoomCapture = {
  schema: 'ruview.roomplan.geometry.v1',
  roomId: 'room-1',
  coordinateFrameId: 'frame-1',
  capturedAtUnixMs: 1,
  surfaces: [],
  objects: [],
  surfaceCount: 4,
  objectCount: 0,
  rawCameraPersisted: false,
  rawDepthPersisted: false,
};

const node: CalibratedNode = {
  nodeId: 'esp32-s3-01',
  role: 'esp32_csi_receiver',
  coordinateFrameId: 'frame-1',
  capturedAtUnixMs: 1,
  positionM: [0, 0, 0],
  transform: Array(16).fill(0),
  trackingState: 'normal',
};

describe('LiDAR contracts', () => {
  it('accepts a bounded geometry-only point frame', () => {
    expect(validateLidarPointFrame(frame)).toBe(true);
  });

  it('rejects mismatched point buffers', () => {
    expect(validateLidarPointFrame({ ...frame, pointCount: 2 })).toBe(false);
  });

  it('keeps geometry-only calibration draft until held-out validation passes', async () => {
    const artifact = await buildSpatialCalibration(room, [node]);
    expect(artifact.quality).toBe('DRAFT');
    expect(artifact.digestSha256).toHaveLength(64);
    expect(artifact.privacy).toEqual({
      rawCameraPersisted: false,
      rawDepthPersisted: false,
      boundedGeometryOnly: true,
    });

    const mismatched = await buildSpatialCalibration(room, [{ ...node, coordinateFrameId: 'other' }]);
    expect(mismatched.quality).toBe('DRAFT');
  });

  it('fits and validates a room correction before promoting calibration', async () => {
    const samples = Array.from({ length: 8 }, (_, index) => ({
      capturedAtUnixMs: index,
      coordinateFrameId: 'frame-1',
      rfPositionM: [index * 0.25, 0, index % 2] as [number, number, number],
      referencePositionM: [index * 0.25 + 1, 0.1, index % 2 - 0.5] as [number, number, number],
      rfTrackId: 'track-1',
      rfConfidence: 0.9,
      rfCapturedAtUnixMs: index,
      poseAssociationSkewMs: 0,
    }));
    const baseline = { referenceKind: 'visible_device_path_proxy' as const, evidence: 'MEASURED' as const, samples, attemptedSamples: 8, lostTrackRate: 0 };
    const model = fitSpatialCorrection(samples);
    expect(model).not.toBeNull();
    const validation = validateSpatialCorrection(baseline, baseline, model!);
    expect(validation.passesTarget).toBe(true);
    const nodes = [node, { ...node, nodeId: 'esp32-s3-02' }, { ...node, nodeId: 'esp32-s3-03' }];
    const artifact = await buildSpatialCalibration(room, nodes, baseline, model!, validation);
    expect(artifact.quality).toBe('VALID');
    expect(artifact.staleness.state).toBe('CURRENT');
  });
});
