import { estimateGestureClockModel, evaluatePoseStudent, extractCsiFeatures, pairPoseTeachingSample, trainPoseStudent } from '@/services/poseCalibration.service';
import { COARSE_TEACHER_JOINTS, type PoseClockModel, type PoseSequence } from '@/types/poseCalibration';
import type { BodyTeacherFrame } from '@/types/lidar';
import type { SensingFrame } from '@/types/sensing';

const rfFrame = (timestamp: number, motion = 1, value = 0): SensingFrame => ({
  timestamp,
  nodes: [{ node_id: 1, rssi_dbm: -65 + value, position: [0, 0, 0], amplitude: [1 + value] }],
  features: { mean_rssi: -65 + value, variance: 4, motion_band_power: motion, breathing_band_power: 0.2, spectral_entropy: 0.4 },
  classification: { motion_level: 'present_moving', presence: true, confidence: 0.9 },
  signal_field: { grid_size: [1, 1, 1], values: [0] },
  persons: [{ confidence: 0.9, keypoints: COARSE_TEACHER_JOINTS.map((name) => ({ name, x: 9, y: 9, z: 9, confidence: 0.9 })) }],
});

const teacherFrame = (capturedAtUnixMs: number, value = 0, handsUp = false): BodyTeacherFrame => ({
  schema: 'ruview.teacher.body.v1', sessionId: 'session', coordinateFrameId: 'frame', capturedAtUnixMs,
  monotonicTimestampSeconds: capturedAtUnixMs / 1000, clockModelId: 'arkit-monotonic+session-wall-offset-v1',
  trackingState: 'normal', source: 'vision-2d+same-frame-scene-depth', evidence: 'MEASURED', visible: true,
  joints: COARSE_TEACHER_JOINTS.map((name, index) => ({
    name, positionM: [value + index * 0.01, (handsUp && name.includes('wrist') ? 2 : 1) + index * 0.01, value * 0.5 + index * 0.01],
    confidence: 0.95, depthMeters: 2,
  })),
  rawCameraPersisted: false, rawDepthPersisted: false, biometricIdentityDerived: false,
});

const clock: PoseClockModel = { model: 'sync-gesture-offset-v1', teacherMinusRfOffsetMs: 100, residualMs: 10, measuredAtUnixMs: 1, passes20MsGate: true };

describe('pose calibration service', () => {
  it('uses a fixed bounded feature schema', () => {
    expect(extractCsiFeatures(rfFrame(1))).toHaveLength(10);
    expect(extractCsiFeatures(rfFrame(1)).every(Number.isFinite)).toBe(true);
  });

  it('estimates a hands-up/motion clock offset and enforces measured resolution', () => {
    const teacher = [teacherFrame(1000), teacherFrame(1033, 0, true), teacherFrame(1066)];
    const rf = [rfFrame(890, 1), rfFrame(900, 9), rfFrame(910, 1)];
    const result = estimateGestureClockModel(teacher, rf);
    expect(result?.teacherMinusRfOffsetMs).toBe(133);
    expect(result?.residualMs).toBeLessThanOrEqual(20);
    expect(result?.passes20MsGate).toBe(true);
  });

  it('rejects a teacher/CSI pair outside 20 ms', () => {
    expect(pairPoseTeachingSample({ teacher: teacherFrame(1000), rfFrame: rfFrame(870), sequenceIndex: 0, clock })).toBeNull();
    expect(pairPoseTeachingSample({ teacher: teacherFrame(1000), rfFrame: rfFrame(900), sequenceIndex: 0, clock })?.associationSkewMs).toBe(0);
  });

  it('fits only coarse joints on seven sequences and scores three held out', () => {
    const sequences: PoseSequence[] = Array.from({ length: 10 }, (_, sequenceIndex) => ({
      index: sequenceIndex,
      attemptedPairs: 12,
      samples: Array.from({ length: 12 }, (_, sampleIndex) => {
        const value = sequenceIndex * 0.08 + sampleIndex * 0.01;
        const sample = pairPoseTeachingSample({
          teacher: teacherFrame(10_000 + sequenceIndex * 1000 + sampleIndex * 20, value),
          rfFrame: rfFrame(9_900 + sequenceIndex * 1000 + sampleIndex * 20, 1 + value, value),
          sequenceIndex,
          clock,
        });
        if (!sample) throw new Error('fixture did not pair');
        return sample;
      }),
    }));
    const model = trainPoseStudent(sequences);
    expect(model?.trainingSequenceCount).toBe(7);
    expect(model?.joints.length).toBeGreaterThanOrEqual(6);
    expect(model?.joints.some((joint) => joint.name.includes('finger'))).toBe(false);
    const evaluation = evaluatePoseStudent(model!, sequences);
    expect(evaluation?.testSequenceCount).toBe(3);
    expect(evaluation?.passesTarget).toBe(true);
  });
});
