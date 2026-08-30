import { requireOptionalNativeModule, type NativeModule } from 'expo-modules-core';

export type NativeLidarCapabilities = {
  platform: 'ios';
  lidarSupported: boolean;
  sceneDepthSupported: boolean;
  smoothedSceneDepthSupported: boolean;
  sceneReconstructionSupported: boolean;
  roomPlanSupported: boolean;
  rawTransientTimingAvailable: false;
};

export type NativeLidarCaptureOptions = {
  maxPoints?: number;
  maxFramesPerSecond?: number;
  minimumConfidence?: 0 | 1 | 2;
  maximumDepthMeters?: number;
  useSmoothedDepth?: boolean;
  includeCameraPreview?: boolean;
  includeBodyTeacher?: boolean;
  maxBodyFramesPerSecond?: number;
  preserveRoomCoordinateFrame?: boolean;
};

export type NativeBodyTeacherJoint = {
  name: 'nose' | 'neck' | 'left_shoulder' | 'right_shoulder' | 'left_elbow' | 'right_elbow' | 'left_wrist' | 'right_wrist' | 'pelvis' | 'left_hip' | 'right_hip' | 'left_knee' | 'right_knee' | 'left_ankle' | 'right_ankle';
  positionM: [number, number, number];
  confidence: number;
  depthMeters: number;
};

export type NativeBodyTeacherFrame = {
  schema: 'ruview.teacher.body.v1';
  sessionId: string;
  coordinateFrameId: string;
  capturedAtUnixMs: number;
  monotonicTimestampSeconds: number;
  clockModelId: 'arkit-monotonic+session-wall-offset-v1';
  trackingState: 'normal';
  source: 'vision-2d+same-frame-scene-depth';
  evidence: 'MEASURED';
  visible: true;
  joints: NativeBodyTeacherJoint[];
  rawCameraPersisted: false;
  rawDepthPersisted: false;
  biometricIdentityDerived: false;
};

export type NativeCameraPreviewFrame = {
  schema: 'ruview.camera.preview.v1';
  sessionId: string;
  coordinateFrameId: string;
  capturedAtUnixMs: number;
  width: number;
  height: number;
  jpegBase64: string;
  rawPersisted: false;
};

export type NativeLidarPointFrame = {
  schema: 'ruview.lidar.points.v1';
  sessionId: string;
  coordinateFrameId: string;
  sequence: number;
  capturedAtUnixMs: number;
  monotonicTimestampSeconds: number;
  points: number[];
  confidences: number[];
  pointCount: number;
  cameraTransform: number[];
  cameraIntrinsics: number[];
  depthWidth: number;
  depthHeight: number;
  smoothed: boolean;
  trackingState: string;
  rawDepthPersisted: false;
  capturedImagePersisted: false;
};

export type NativeLidarDepthPacket = {
  type: 'ruview.lidar.depth.v1';
  intrinsics: { fx: number; fy: number; cx: number; cy: number; imageWidth: number; imageHeight: number };
  pose: { matrix: number[] };
  depth: {
    width: number;
    height: number;
    encoding: 'u16le-mm+u8-confidence';
    millimetersBase64: string;
    confidenceBase64: string;
  };
  provenance: {
    sensor: 'apple-arkit-scene-depth';
    sessionId: string;
    coordinateFrameId: string;
    source: 'live';
    privacyClass: 'geometry-only';
    sequence: number;
    timestampNs: number;
    captureTimeNs: number;
    clockModelId: 'arkit-monotonic+session-wall-offset-v1';
    calibrationId: string;
    trackingState: string;
    evidence: 'MEASURED';
    schema: 'ruview.lidar.depth.v1';
  };
};

export type NativeRoomSurface = {
  id: string;
  kind: 'wall' | 'door' | 'window' | 'opening' | 'floor';
  category: string;
  confidence: string;
  dimensionsM: [number, number, number];
  transform: number[];
};

export type NativeRoomObject = {
  id: string;
  category: string;
  confidence: string;
  dimensionsM: [number, number, number];
  transform: number[];
};

export type NativeRoomCapture = {
  schema: 'ruview.roomplan.geometry.v1';
  roomId: string;
  capturedAtUnixMs: number;
  coordinateFrameId: string;
  surfaces: NativeRoomSurface[];
  objects: NativeRoomObject[];
  surfaceCount: number;
  objectCount: number;
  rawCameraPersisted: false;
  rawDepthPersisted: false;
};

export type NativeLidarStatus = {
  state: 'idle' | 'requesting_permission' | 'capturing_depth' | 'capturing_room' | 'processing_room' | 'validating_calibration' | 'validating_wall_scan' | 'validation_complete' | 'unsupported' | 'error';
  message?: string;
  instruction?: string;
};

export type NativeVisibleDepthMetrics = {
  phase: 'calibration' | 'wall_scan';
  fps: number;
  depthCoverage: number;
  trackingState: string;
  movementMetersPerSecond: number;
  thermalState: string;
  phaseSecondsRemaining: number;
};

export type NativeVisibleDepthDiagnostic = {
  schema: 'ruview.ios.visible-depth-diagnostic.v1';
  sessionId: string;
  createdAt: string;
  deviceModelFamily: string;
  osVersion: string;
  appVersion: string;
  capabilities: {
    worldTracking: boolean;
    sceneDepth: boolean;
    smoothedSceneDepth: boolean;
    sceneMesh: boolean;
    rawPhotonHistograms: false;
  };
  phases: Array<{
    phase: 'calibration' | 'wall_scan';
    plannedDurationSeconds: number;
    observedDurationSeconds: number;
    frameCount: number;
    averageFPS: number;
    averageDepthCoverage: number;
    averageMovementMetersPerSecond: number;
    finalTrackingState: string;
    peakThermalState: string;
  }>;
  evidenceLabel: 'direct_depth';
  physicalNLOSStatus: 'blocked_raw_transients_unavailable';
  cameraPermission: 'not_requested' | 'granted' | 'denied' | 'restricted';
  consent: { localValidation: boolean; diagnosticExport: boolean; rawSensorExport: false };
  completionStatus: 'completed' | 'cancelled' | 'failed';
  failureReason?: string;
};

export type NativeSpatialPose = {
  coordinateFrameId: string;
  capturedAtUnixMs: number;
  positionM: [number, number, number];
  transform: number[];
  trackingState: string;
};

type NativeModuleShape = NativeModule & {
  addListener(eventName: 'onLidarFrame', listener: (frame: NativeLidarPointFrame) => void): { remove(): void };
  addListener(eventName: 'onLidarDepthPacket', listener: (packet: NativeLidarDepthPacket) => void): { remove(): void };
  addListener(eventName: 'onCameraPreview', listener: (frame: NativeCameraPreviewFrame) => void): { remove(): void };
  addListener(eventName: 'onBodyTeacherFrame', listener: (frame: NativeBodyTeacherFrame) => void): { remove(): void };
  addListener(eventName: 'onLidarStatus', listener: (status: NativeLidarStatus) => void): { remove(): void };
  addListener(eventName: 'onRoomUpdate', listener: (update: Partial<NativeRoomCapture> & NativeLidarStatus) => void): { remove(): void };
  addListener(eventName: 'onRoomComplete', listener: (room: NativeRoomCapture) => void): { remove(): void };
  addListener(eventName: 'onLidarError', listener: (error: { code: string; message: string }) => void): { remove(): void };
  addListener(eventName: 'onVisibleDepthMetrics', listener: (metrics: NativeVisibleDepthMetrics) => void): { remove(): void };
  addListener(eventName: 'onVisibleDepthDiagnostic', listener: (diagnostic: NativeVisibleDepthDiagnostic) => void): { remove(): void };
  getCapabilities(): Promise<NativeLidarCapabilities>;
  startDepthCapture(options: NativeLidarCaptureOptions): Promise<NativeLidarStatus>;
  stopCapture(): Promise<NativeLidarStatus>;
  startRoomCapture(): Promise<NativeLidarStatus>;
  stopRoomCapture(): Promise<NativeLidarStatus>;
  getLatestRoom(): Promise<NativeRoomCapture | null>;
  getCurrentPose(): Promise<NativeSpatialPose | null>;
  startVisibleDepthValidation(): Promise<NativeLidarStatus>;
  cancelVisibleDepthValidation(): Promise<NativeLidarStatus>;
};

export const nativeRuViewLidar = requireOptionalNativeModule<NativeModuleShape>('RuViewLidar');

export const nativeRuViewLidarEvents = nativeRuViewLidar;
