import { NLOS_TRACK_SCHEMA, type NlosTrackFrame } from '@/types/nlos';

export const createLiveNlosFrameFixture = (
  overrides: Partial<NlosTrackFrame> = {},
): NlosTrackFrame => ({
  schema: NLOS_TRACK_SCHEMA,
  sessionId: 'live-session-1',
  sequence: 1,
  capturedAtUnixMs: 1_700_000_000_000,
  expiresAtUnixMs: 1_700_000_001_000,
  source: 'live',
  evidenceLevel: 'l2_calibrated',
  algorithmVersion: 'nlos-inversion-v1',
  calibrationHash: 'a'.repeat(64),
  provenance: {
    sensorId: 'tof-1',
    sensorModel: 'VL53L8CH',
    firmwareVersion: '1.0.0',
    transientKind: 'raw_histogram',
    histogramPreserved: true,
    transport: 'ruview_server',
  },
  tracks: [
    {
      trackId: 'target-1',
      state: 'tracking',
      positionM: { x: 2.1, y: 1.1, z: 3.4 },
      velocityMps: { x: 0.1, y: 0, z: -0.05 },
      covarianceDiagonalM2: { x: 0.04, y: 0.06, z: 0.08 },
      confidence: 0.88,
      posteriorEntropy: 0.32,
      signalQuality: 0.81,
      modalityContributions: { lidar: 0.7, csi: 0.3 },
    },
  ],
  ...overrides,
});
