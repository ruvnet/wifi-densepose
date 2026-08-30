import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import type {
  CalibratedNode,
  BodyTeacherFrame,
  CameraPreviewFrame,
  LidarCapabilities,
  LidarPointFrame,
  LidarStatus,
  RoomCapture,
  SpatialCalibrationArtifact,
  VisibleDepthDiagnostic,
  VisibleDepthMetrics,
} from '@/types/lidar';
import { validateLidarPointFrame } from '@/types/lidar';
import type { PoseCalibrationArtifact } from '@/types/poseCalibration';

interface LidarState {
  relayState: 'disconnected' | 'connecting' | 'connected' | 'error';
  capabilities: LidarCapabilities | null;
  status: LidarStatus;
  frame: LidarPointFrame | null;
  cameraPreview: CameraPreviewFrame | null;
  bodyTeacherFrame: BodyTeacherFrame | null;
  room: RoomCapture | null;
  nodes: CalibratedNode[];
  calibration: SpatialCalibrationArtifact | null;
  poseCalibration: PoseCalibrationArtifact | null;
  validationMetrics: VisibleDepthMetrics | null;
  validationDiagnostic: VisibleDepthDiagnostic | null;
  error: string | null;
  rejectedFrameCount: number;
  setCapabilities: (value: LidarCapabilities) => void;
  setRelayState: (value: LidarState['relayState']) => void;
  setStatus: (value: LidarStatus) => void;
  ingestFrame: (value: unknown) => void;
  setCameraPreview: (value: CameraPreviewFrame | null) => void;
  setBodyTeacherFrame: (value: BodyTeacherFrame | null) => void;
  setRoom: (value: RoomCapture | null) => void;
  addNode: (value: CalibratedNode) => void;
  removeNode: (nodeId: string) => void;
  setCalibration: (value: SpatialCalibrationArtifact) => void;
  setPoseCalibration: (value: PoseCalibrationArtifact) => void;
  markCalibrationStale: (reason: string) => void;
  setValidationMetrics: (value: VisibleDepthMetrics) => void;
  setValidationDiagnostic: (value: VisibleDepthDiagnostic) => void;
  setError: (value: string | null) => void;
  clearSession: () => void;
}

export const useLidarStore = create<LidarState>()(
  persist(
    (set) => ({
      capabilities: null,
      relayState: 'disconnected',
      status: { state: 'idle' },
      frame: null,
      cameraPreview: null,
      bodyTeacherFrame: null,
      room: null,
      nodes: [],
      calibration: null,
      poseCalibration: null,
      validationMetrics: null,
      validationDiagnostic: null,
      error: null,
      rejectedFrameCount: 0,
      setCapabilities: (capabilities) => set({ capabilities }),
      setRelayState: (relayState) => set({ relayState }),
      setStatus: (status) => set({ status, error: status.state === 'error' ? status.message ?? 'LiDAR error' : null }),
      ingestFrame: (value) => set((state) => validateLidarPointFrame(value)
        ? { frame: value, error: null }
        : { rejectedFrameCount: state.rejectedFrameCount + 1, error: 'Rejected malformed LiDAR point frame.' }),
      setCameraPreview: (cameraPreview) => set({ cameraPreview }),
      setBodyTeacherFrame: (bodyTeacherFrame) => set({ bodyTeacherFrame }),
      setRoom: (room) => set({ room }),
      addNode: (node) => set((state) => ({ nodes: [...state.nodes.filter(({ nodeId }) => nodeId !== node.nodeId), node] })),
      removeNode: (nodeId) => set((state) => ({ nodes: state.nodes.filter((node) => node.nodeId !== nodeId) })),
      setCalibration: (calibration) => set({ calibration }),
      setPoseCalibration: (poseCalibration) => set({ poseCalibration }),
      markCalibrationStale: (reason) => set((state) => state.calibration ? {
        calibration: { ...state.calibration, quality: 'DRAFT', staleness: { ...state.calibration.staleness, state: 'STALE', reason } },
      } : {}),
      setValidationMetrics: (validationMetrics) => set({ validationMetrics }),
      setValidationDiagnostic: (validationDiagnostic) => set({ validationDiagnostic }),
      setError: (error) => set({ error }),
      clearSession: () => set({ status: { state: 'idle' }, frame: null, cameraPreview: null, bodyTeacherFrame: null, room: null, nodes: [], error: null }),
    }),
    {
      name: 'ruview-spatial-calibration',
      version: 3,
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({ calibration: state.calibration, poseCalibration: state.poseCalibration }),
      migrate: (persisted) => {
        const state = persisted as Partial<LidarState>;
        if (state.calibration && !state.calibration.staleness) {
          return {
            ...state,
            calibration: {
              ...state.calibration,
              quality: 'DRAFT',
              staleness: { state: 'STALE', roomFingerprint: 'legacy-unknown', maximumResidualM: 999, reason: 'Legacy calibration must be revalidated with a held-out walk.' },
            },
          };
        }
        return state;
      },
    },
  ),
);
