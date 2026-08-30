import { useCallback, useEffect } from 'react';
import { AppState } from 'react-native';
import { buildSpatialCalibration, lidarRelay, lidarService, unavailableLidarCapabilities } from '@/services/lidar.service';
import { useLidarStore } from '@/stores/lidarStore';
import type { CalibratedNode, CalibrationValidation, ReferenceWalk, SpatialCorrectionModel } from '@/types/lidar';

let runtimeUsers = 0;
let runtimeSubscriptions: Array<{ remove(): void }> = [];
let appStateSubscription: { remove(): void } | null = null;

const mountLidarRuntime = () => {
  runtimeUsers += 1;
  if (runtimeUsers > 1) return;
  if (lidarService.available) {
    lidarService.getCapabilities()
      .then((capabilities) => useLidarStore.getState().setCapabilities(capabilities))
      .catch((error) => useLidarStore.getState().setError(error instanceof Error ? error.message : String(error)));
  } else {
    useLidarStore.getState().setCapabilities(unavailableLidarCapabilities);
  }
  appStateSubscription = AppState.addEventListener('change', (nextState) => {
    if (nextState !== 'active') {
      void lidarService.stopCapture();
      lidarRelay.disconnect();
      useLidarStore.getState().setRelayState('disconnected');
    }
  });
  const events = lidarService.events;
  if (!events) return;
  runtimeSubscriptions = [
    events.addListener('onLidarFrame', (frame) => useLidarStore.getState().ingestFrame(frame)),
    events.addListener('onLidarDepthPacket', (packet) => lidarRelay.send(packet)),
    events.addListener('onCameraPreview', (frame) => useLidarStore.getState().setCameraPreview(frame)),
    events.addListener('onBodyTeacherFrame', (frame) => useLidarStore.getState().setBodyTeacherFrame(frame)),
    events.addListener('onLidarStatus', (status) => useLidarStore.getState().setStatus(status)),
    events.addListener('onRoomComplete', (room) => useLidarStore.getState().setRoom(room)),
    events.addListener('onLidarError', (event) => useLidarStore.getState().setError(event.message)),
    events.addListener('onVisibleDepthMetrics', (metrics) => useLidarStore.getState().setValidationMetrics(metrics)),
    events.addListener('onVisibleDepthDiagnostic', (diagnostic) => useLidarStore.getState().setValidationDiagnostic(diagnostic)),
  ];
};

const unmountLidarRuntime = () => {
  runtimeUsers = Math.max(0, runtimeUsers - 1);
  if (runtimeUsers > 0) return;
  runtimeSubscriptions.forEach((subscription) => subscription.remove());
  runtimeSubscriptions = [];
  appStateSubscription?.remove();
  appStateSubscription = null;
};

export const useLidarCapture = () => {
  const state = useLidarStore();
  const {
    addNode,
    nodes,
    room,
    setCalibration,
    setError,
    setRelayState,
  } = state;

  useEffect(() => {
    mountLidarRuntime();
    return unmountLidarRuntime;
  }, []);

  const run = useCallback(async <T,>(operation: () => Promise<T>) => {
    setError(null);
    try { return await operation(); } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
      return null;
    }
  }, [setError]);

  const markNode = useCallback(async (nodeId: string) => run(async () => {
    const pose = await lidarService.getCurrentPose();
    if (!pose) throw new Error('No tracked spatial pose is available. Start a RoomPlan scan first.');
    if (room && pose.coordinateFrameId !== room.coordinateFrameId) throw new Error('The node pose belongs to a different room scan.');
    if (pose.trackingState !== 'normal') throw new Error('AR tracking is not stable enough to mark this node.');
    const node: CalibratedNode = {
      nodeId: nodeId.trim(),
      role: 'esp32_csi_receiver',
      capturedAtUnixMs: pose.capturedAtUnixMs,
      coordinateFrameId: pose.coordinateFrameId,
      positionM: pose.positionM,
      transform: pose.transform,
      trackingState: pose.trackingState,
    };
    addNode(node);
    return node;
  }), [addNode, room, run]);

  const saveCalibration = useCallback(async (
    referenceWalk?: ReferenceWalk,
    correctionModel?: SpatialCorrectionModel,
    validation?: CalibrationValidation,
  ) => run(async () => {
    if (!room) throw new Error('Complete a RoomPlan scan before saving calibration.');
    if (!nodes.length) throw new Error('Mark at least one ESP32 node before saving calibration.');
    const artifact = await buildSpatialCalibration(room, nodes, referenceWalk, correctionModel, validation);
    setCalibration(artifact);
    return artifact;
  }), [nodes, room, run, setCalibration]);

  return {
    ...state,
    nativeModuleAvailable: lidarService.available,
    startDepth: (includeCameraPreview = false, includeBodyTeacher = false, preserveRoomCoordinateFrame = false) => run(() => lidarService.startDepthCapture({ maxPoints: 1536, maxFramesPerSecond: 5, minimumConfidence: 1, maximumDepthMeters: 8, useSmoothedDepth: true, includeCameraPreview, includeBodyTeacher, maxBodyFramesPerSecond: 15, preserveRoomCoordinateFrame })),
    stopCapture: () => run(() => lidarService.stopCapture()),
    startRoom: () => run(() => lidarService.startRoomCapture()),
    stopRoom: () => run(() => lidarService.stopRoomCapture()),
    startValidation: () => run(() => lidarService.startVisibleDepthValidation()),
    cancelValidation: () => run(() => lidarService.cancelVisibleDepthValidation()),
    getCurrentPose: () => lidarService.getCurrentPose(),
    connectRelay: (endpoint: string) => run(async () => lidarRelay.connect(endpoint, setRelayState)),
    disconnectRelay: () => { lidarRelay.disconnect(); setRelayState('disconnected'); },
    markNode,
    saveCalibration,
  };
};
