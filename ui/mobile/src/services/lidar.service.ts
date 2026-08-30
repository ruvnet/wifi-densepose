import * as Crypto from 'expo-crypto';
import {
  nativeRuViewLidar,
  nativeRuViewLidarEvents,
  type NativeCameraPreviewFrame,
  type NativeLidarCaptureOptions,
  type NativeLidarCapabilities,
  type NativeLidarDepthPacket,
  type NativeLidarPointFrame,
  type NativeLidarStatus,
  type NativeRoomCapture,
  type NativeSpatialPose,
  type NativeVisibleDepthDiagnostic,
  type NativeVisibleDepthMetrics,
} from '../../modules/ruview-lidar/src';
import type {
  CalibratedNode,
  CalibrationValidation,
  ReferenceWalk,
  ReferenceWalkSample,
  SpatialCalibrationArtifact,
  SpatialCorrectionModel,
} from '@/types/lidar';

export const unavailableLidarCapabilities: NativeLidarCapabilities = {
  platform: 'ios',
  lidarSupported: false,
  sceneDepthSupported: false,
  smoothedSceneDepthSupported: false,
  sceneReconstructionSupported: false,
  roomPlanSupported: false,
  rawTransientTimingAvailable: false,
};

const stableJson = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => `${JSON.stringify(key)}:${stableJson(entry)}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
};

const newId = (prefix: string) => `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

export const lidarService = {
  available: Boolean(nativeRuViewLidar),
  events: nativeRuViewLidarEvents,
  getCapabilities: () => nativeRuViewLidar?.getCapabilities() ?? Promise.resolve(unavailableLidarCapabilities),
  startDepthCapture: (options: NativeLidarCaptureOptions = {}) => {
    if (!nativeRuViewLidar) return Promise.reject(new Error('Native LiDAR capture requires an iOS development build.'));
    return nativeRuViewLidar.startDepthCapture(options);
  },
  stopCapture: (): Promise<NativeLidarStatus> => nativeRuViewLidar?.stopCapture() ?? Promise.resolve({ state: 'idle' }),
  startRoomCapture: () => {
    if (!nativeRuViewLidar) return Promise.reject(new Error('RoomPlan requires an iOS development build.'));
    return nativeRuViewLidar.startRoomCapture();
  },
  stopRoomCapture: (): Promise<NativeLidarStatus> => nativeRuViewLidar?.stopRoomCapture() ?? Promise.resolve({ state: 'idle' }),
  getLatestRoom: (): Promise<NativeRoomCapture | null> => nativeRuViewLidar?.getLatestRoom() ?? Promise.resolve(null),
  getCurrentPose: (): Promise<NativeSpatialPose | null> => nativeRuViewLidar?.getCurrentPose() ?? Promise.resolve(null),
  startVisibleDepthValidation: () => {
    if (!nativeRuViewLidar) return Promise.reject(new Error('Visible-depth validation requires an iOS development build.'));
    return nativeRuViewLidar.startVisibleDepthValidation();
  },
  cancelVisibleDepthValidation: (): Promise<NativeLidarStatus> => nativeRuViewLidar?.cancelVisibleDepthValidation() ?? Promise.resolve({ state: 'idle' }),
};

class LidarRelayClient {
  private socket: WebSocket | null = null;

  connect(endpoint: string, onState: (state: 'disconnected' | 'connecting' | 'connected' | 'error') => void) {
    const url = new URL(endpoint);
    if (url.protocol !== 'ws:' && url.protocol !== 'wss:') throw new Error('LiDAR relay endpoint must use ws:// or wss://.');
    this.disconnect();
    onState('connecting');
    const socket = new WebSocket(url.toString());
    socket.onopen = () => onState('connected');
    socket.onerror = () => onState('error');
    socket.onclose = () => onState('disconnected');
    this.socket = socket;
  }

  send(packet: NativeLidarDepthPacket) {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(packet));
  }

  disconnect() {
    this.socket?.close();
    this.socket = null;
  }
}

export const lidarRelay = new LidarRelayClient();

const distance = (a: [number, number, number], b: [number, number, number]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return Number.POSITIVE_INFINITY;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

export const applySpatialCorrection = (
  position: [number, number, number],
  model: SpatialCorrectionModel,
): [number, number, number] => {
  const cosine = Math.cos(model.yawRadians);
  const sine = Math.sin(model.yawRadians);
  return [
    model.scale * (cosine * position[0] - sine * position[2]) + model.translationM[0],
    position[1] + model.translationM[1],
    model.scale * (sine * position[0] + cosine * position[2]) + model.translationM[2],
  ];
};

/** Fit a bounded 2D similarity transform plus vertical offset. This corrects
 * room scale, yaw, and systematic translation without pretending to solve a
 * full RF multipath model from a short installation walk. */
export const fitSpatialCorrection = (samples: ReferenceWalkSample[]): SpatialCorrectionModel | null => {
  if (samples.length < 5) return null;
  const rfMean = samples.reduce((sum, sample) => [sum[0] + sample.rfPositionM[0], sum[1] + sample.rfPositionM[1], sum[2] + sample.rfPositionM[2]] as [number, number, number], [0, 0, 0]).map((value) => value / samples.length) as [number, number, number];
  const refMean = samples.reduce((sum, sample) => [sum[0] + sample.referencePositionM[0], sum[1] + sample.referencePositionM[1], sum[2] + sample.referencePositionM[2]] as [number, number, number], [0, 0, 0]).map((value) => value / samples.length) as [number, number, number];
  let dot = 0;
  let cross = 0;
  let energy = 0;
  samples.forEach((sample) => {
    const rx = sample.rfPositionM[0] - rfMean[0];
    const rz = sample.rfPositionM[2] - rfMean[2];
    const gx = sample.referencePositionM[0] - refMean[0];
    const gz = sample.referencePositionM[2] - refMean[2];
    dot += rx * gx + rz * gz;
    cross += rx * gz - rz * gx;
    energy += rx * rx + rz * rz;
  });
  if (!Number.isFinite(energy) || energy < 1e-6) return null;
  const yawRadians = Math.atan2(cross, dot);
  const scale = Math.max(0.5, Math.min(2, Math.hypot(dot, cross) / energy));
  const cosine = Math.cos(yawRadians);
  const sine = Math.sin(yawRadians);
  const translationM: [number, number, number] = [
    refMean[0] - scale * (cosine * rfMean[0] - sine * rfMean[2]),
    refMean[1] - rfMean[1],
    refMean[2] - scale * (sine * rfMean[0] + cosine * rfMean[2]),
  ];
  const provisional: SpatialCorrectionModel = {
    model: 'similarity_2d_plus_y_offset', scale, yawRadians, translationM,
    transform: [scale * cosine, 0, scale * sine, 0, 0, 1, 0, 0, -scale * sine, 0, scale * cosine, 0, translationM[0], translationM[1], translationM[2], 1],
    baselineMedianErrorM: 0, fittedMedianErrorM: 0, fittedImprovementFraction: 0, sampleCount: samples.length,
    residualCovarianceM2: Array(9).fill(0),
  };
  const rawMedian = median(samples.map((sample) => distance(sample.rfPositionM, sample.referencePositionM)));
  const fittedMedian = median(samples.map((sample) => distance(applySpatialCorrection(sample.rfPositionM, provisional), sample.referencePositionM)));
  const residuals = samples.map((sample) => {
    const corrected = applySpatialCorrection(sample.rfPositionM, provisional);
    return corrected.map((value, index) => value - sample.referencePositionM[index]) as [number, number, number];
  });
  const residualMean = [0, 1, 2].map((axis) => residuals.reduce((sum, value) => sum + value[axis], 0) / residuals.length);
  const residualCovarianceM2 = Array.from({ length: 9 }, (_, index) => {
    const row = Math.floor(index / 3);
    const column = index % 3;
    return residuals.reduce((sum, value) => sum + (value[row] - residualMean[row]) * (value[column] - residualMean[column]), 0) / Math.max(1, residuals.length - 1);
  });
  return { ...provisional, residualCovarianceM2, baselineMedianErrorM: rawMedian, fittedMedianErrorM: fittedMedian, fittedImprovementFraction: rawMedian > 0 ? Math.max(-1, Math.min(1, 1 - fittedMedian / rawMedian)) : 0 };
};

export const validateSpatialCorrection = (
  baseline: ReferenceWalk,
  validation: ReferenceWalk,
  model: SpatialCorrectionModel,
): CalibrationValidation => {
  const rawMedianErrorM = median(validation.samples.map((sample) => distance(sample.rfPositionM, sample.referencePositionM)));
  const calibratedMedianErrorM = median(validation.samples.map((sample) => distance(applySpatialCorrection(sample.rfPositionM, model), sample.referencePositionM)));
  const improvementFraction = rawMedianErrorM > 0 ? 1 - calibratedMedianErrorM / rawMedianErrorM : 0;
  const passesTarget = validation.samples.length >= 5
    && improvementFraction >= 0.25
    && validation.lostTrackRate <= baseline.lostTrackRate
    && validation.samples.every((sample) => sample.poseAssociationSkewMs <= 100);
  return {
    evidence: 'MEASURED', sampleCount: validation.samples.length, attemptedSamples: validation.attemptedSamples,
    rawMedianErrorM, calibratedMedianErrorM, improvementFraction,
    baselineLostTrackRate: baseline.lostTrackRate, calibratedLostTrackRate: validation.lostTrackRate,
    passesTarget, target: 'median_error_improves_25_percent_without_more_lost_tracks',
  };
};

export const buildSpatialCalibration = async (
  room: NativeRoomCapture,
  nodes: CalibratedNode[],
  referenceWalk?: ReferenceWalk,
  correctionModel?: SpatialCorrectionModel,
  validation?: CalibrationValidation,
): Promise<SpatialCalibrationArtifact> => {
  const createdAtUnixMs = Date.now();
  const quality: SpatialCalibrationArtifact['quality'] = room.surfaceCount >= 4
    && nodes.length >= 3
    && nodes.every((node) => node.coordinateFrameId === room.coordinateFrameId && node.trackingState === 'normal')
    && Boolean(validation?.passesTarget)
    ? 'VALID'
    : 'DRAFT';
  const unsigned = {
    schema: 'ruview.calibration.spatial.v1' as const,
    calibrationId: newId('cal'),
    createdAtUnixMs,
    roomId: room.roomId,
    coordinateFrameId: room.coordinateFrameId,
    quality,
    room,
    nodes,
    referenceWalk,
    correctionModel,
    validation,
    staleness: {
      state: 'CURRENT' as const,
      roomFingerprint: await Crypto.digestStringAsync(Crypto.CryptoDigestAlgorithm.SHA256, stableJson({ room: room.surfaces, nodes })),
      maximumResidualM: Math.max(0.25, 2 * (validation?.calibratedMedianErrorM ?? correctionModel?.fittedMedianErrorM ?? 0.5)),
      reason: null,
    },
    privacy: {
      rawCameraPersisted: false as const,
      rawDepthPersisted: false as const,
      boundedGeometryOnly: true as const,
    },
    digestAlgorithm: 'SHA-256' as const,
  };
  const digestSha256 = await Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    stableJson(unsigned),
  );
  return { ...unsigned, digestSha256 };
};

export type LidarEventMap = {
  onCameraPreview: NativeCameraPreviewFrame;
  onLidarFrame: NativeLidarPointFrame;
  onLidarDepthPacket: NativeLidarDepthPacket;
  onLidarStatus: NativeLidarStatus;
  onRoomUpdate: Partial<NativeRoomCapture> & NativeLidarStatus;
  onRoomComplete: NativeRoomCapture;
  onLidarError: { code: string; message: string };
  onVisibleDepthMetrics: NativeVisibleDepthMetrics;
  onVisibleDepthDiagnostic: NativeVisibleDepthDiagnostic;
};
