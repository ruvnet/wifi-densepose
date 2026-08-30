import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import { InstrumentPanel, instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import { useLidarCapture } from '@/hooks/useLidarCapture';
import { usePoseStream } from '@/hooks/usePoseStream';
import { buildPoseCalibrationArtifact, estimateGestureClockModel, evaluatePoseStudent, pairPoseTeachingSample, trainPoseStudent } from '@/services/poseCalibration.service';
import { wsService } from '@/services/ws.service';
import { useLidarStore } from '@/stores/lidarStore';
import { spacing } from '@/theme/spacing';
import type { BodyTeacherFrame } from '@/types/lidar';
import type { PoseCalibrationEvaluation, PoseClockModel, PoseSequence, PoseStudentModel, PoseTrainingSample } from '@/types/poseCalibration';
import type { SensingFrame } from '@/types/sensing';

const Action = ({ label, onPress, disabled = false }: { label: string; onPress: () => void; disabled?: boolean }) => (
  <Pressable accessibilityRole="button" accessibilityLabel={label} disabled={disabled} onPress={onPress} style={({ pressed }) => [styles.action, disabled && styles.disabled, pressed && !disabled && styles.pressed]}>
    <ThemedText preset="labelMd" style={styles.actionText}>{label}</ThemedText>
  </Pressable>
);

export const PoseTeachingCard = () => {
  const lidar = useLidarCapture();
  const teacherFrame = useLidarStore((state) => state.bodyTeacherFrame);
  const savedCalibration = useLidarStore((state) => state.poseCalibration);
  const setPoseCalibration = useLidarStore((state) => state.setPoseCalibration);
  const { connectionStatus } = usePoseStream();
  const [syncing, setSyncing] = useState(false);
  const [clock, setClock] = useState<PoseClockModel | null>(null);
  const [recordingIndex, setRecordingIndex] = useState<number | null>(null);
  const [sequences, setSequences] = useState<PoseSequence[]>([]);
  const [model, setModel] = useState<PoseStudentModel | null>(null);
  const [evaluation, setEvaluation] = useState<PoseCalibrationEvaluation | null>(null);
  const [message, setMessage] = useState('Start the teacher, then perform one sharp hands-up synchronization gesture.');
  const teacherSync = useRef<BodyTeacherFrame[]>([]);
  const rfSync = useRef<SensingFrame[]>([]);
  const rfRecent = useRef<SensingFrame[]>([]);
  const sequenceSamples = useRef<PoseTrainingSample[]>([]);
  const sequenceAttempts = useRef(0);
  const lastTeacherTimestamp = useRef(0);
  const teaching = lidar.status.state === 'capturing_depth' && Boolean(teacherFrame);
  const roomCalibrationReady = lidar.calibration?.quality === 'VALID' && lidar.calibration.staleness.state === 'CURRENT';
  const teacherAligned = teaching && teacherFrame?.coordinateFrameId === lidar.calibration?.coordinateFrameId;

  useEffect(() => wsService.subscribeRaw((frame) => {
    rfRecent.current = [...rfRecent.current.slice(-255), frame];
    if (syncing) rfSync.current.push(frame);
  }), [syncing]);

  useEffect(() => {
    if (!teacherFrame || teacherFrame.capturedAtUnixMs === lastTeacherTimestamp.current) return;
    lastTeacherTimestamp.current = teacherFrame.capturedAtUnixMs;
    if (syncing) teacherSync.current.push(teacherFrame);
    if (recordingIndex === null || !clock) return;
    sequenceAttempts.current += 1;
    const nearest = rfRecent.current.reduce<SensingFrame | null>((best, candidate) => {
      if (!candidate.timestamp) return best;
      if (!best?.timestamp) return candidate;
      return Math.abs(teacherFrame.capturedAtUnixMs - (candidate.timestamp + clock.teacherMinusRfOffsetMs))
        < Math.abs(teacherFrame.capturedAtUnixMs - (best.timestamp + clock.teacherMinusRfOffsetMs)) ? candidate : best;
    }, null);
    if (!nearest) return;
    const sample = pairPoseTeachingSample({ teacher: teacherFrame, rfFrame: nearest, sequenceIndex: recordingIndex, clock });
    if (sample && !sequenceSamples.current.some((value) => value.pairedAtUnixMs === sample.pairedAtUnixMs)) sequenceSamples.current.push(sample);
  }, [clock, recordingIndex, syncing, teacherFrame]);

  const beginSync = () => {
    teacherSync.current = [];
    rfSync.current = [];
    setClock(null);
    setSyncing(true);
    setMessage('Raise both hands sharply, then lower them. Capturing synchronized teacher and full-rate CSI for 4 seconds…');
    setTimeout(() => {
      setSyncing(false);
      const result = estimateGestureClockModel(teacherSync.current, rfSync.current);
      setClock(result);
      setMessage(result?.passes20MsGate
        ? `Clock aligned; measured sampling residual ${result.residualMs.toFixed(1)} ms.`
        : `Clock gate failed${result ? ` at ${result.residualMs.toFixed(1)} ms` : ''}. Keep the person fully visible and retry with a sharper gesture / higher-rate CSI.`);
    }, 4000);
  };

  const startSequence = () => {
    const index = sequences.length;
    sequenceSamples.current = [];
    sequenceAttempts.current = 0;
    setRecordingIndex(index);
    setMessage(`Recording sequence ${index + 1}/10. Keep the whole body visible; vary walking, sitting, standing, or coarse limb motion.`);
  };

  const finishSequence = () => {
    if (recordingIndex === null) return;
    const next = [...sequences, { index: recordingIndex, samples: sequenceSamples.current, attemptedPairs: sequenceAttempts.current }];
    setSequences(next);
    setRecordingIndex(null);
    if (next.length === 7) {
      const trained = trainPoseStudent(next);
      setModel(trained);
      setMessage(trained ? 'Seven training sequences fit. Record three unseen held-out sequences without refitting.' : 'Training data was insufficient: each sequence needs at least four valid ≤20 ms pairs and six consistently visible joints.');
    } else if (next.length === 10 && model) {
      const result = evaluatePoseStudent(model, next);
      setEvaluation(result);
      if (result && clock && teacherFrame && lidar.calibration) setPoseCalibration(buildPoseCalibrationArtifact(
        teacherFrame.coordinateFrameId,
        lidar.calibration.calibrationId,
        lidar.calibration.staleness.roomFingerprint,
        clock,
        model,
        result,
      ));
      setMessage(result?.passesTarget ? 'Measured held-out PCK target passed.' : 'Held-out target not met. Model remains DRAFT; collect a new dataset after checking timing and visibility.');
    } else {
      setMessage(`Sequence ${next.length}/10 stored in memory (${sequenceSamples.current.length}/${sequenceAttempts.current} valid pairs).`);
    }
  };

  const liveRf = connectionStatus === 'connected';
  return (
    <InstrumentPanel eyebrow="Visible pose teacher / room-specific CSI student" accessory={<ThemedText preset="mono" style={styles.state}>{model ? 'MODEL FIT' : 'DRAFT'}</ThemedText>}>
      <ThemedText preset="bodySm" style={styles.copy}>
        After room calibration, the iPhone temporarily measures visible coarse joints with Vision plus same-frame LiDAR depth in that same room coordinate frame. RuView learns a room-specific CSI mapping, then runs without the phone. This cannot increase WiFi spatial resolution; hands, fingers, identity, and optical through-wall claims are excluded.
      </ThemedText>
      <View style={styles.statusRow}>
        <ThemedText preset="mono" style={liveRf ? styles.valid : styles.draft}>CSI {connectionStatus.toUpperCase()}</ThemedText>
        <ThemedText preset="mono" style={teacherAligned ? styles.valid : styles.draft}>TEACHER {teacherAligned ? `${teacherFrame?.joints.length} JOINTS` : 'OFF/UNALIGNED'}</ThemedText>
        <ThemedText preset="mono" style={clock?.passes20MsGate ? styles.valid : styles.draft}>CLOCK {clock?.passes20MsGate ? '≤20MS' : 'UNSYNCED'}</ThemedText>
      </View>
      <Action label={lidar.status.state === 'capturing_depth' ? 'STOP VISIBLE TEACHER' : 'START ROOM-ALIGNED BODY TEACHER'} disabled={!lidar.capabilities?.sceneDepthSupported || !roomCalibrationReady || lidar.status.state === 'capturing_room' || recordingIndex !== null} onPress={() => { void (lidar.status.state === 'capturing_depth' ? lidar.stopCapture() : lidar.startDepth(false, true, true)); }} />
      <Action label={syncing ? 'CAPTURING SYNC GESTURE…' : 'AUTO-SYNC HANDS-UP GESTURE'} disabled={!teacherAligned || !liveRf || syncing || recordingIndex !== null} onPress={beginSync} />
      <ThemedText preset="bodySm" style={styles.message}>{message}</ThemedText>
      <View style={styles.progress}>
        {Array.from({ length: 10 }, (_, index) => <View key={index} style={[styles.sequence, index < sequences.length && styles.sequenceDone, index === recordingIndex && styles.sequenceRecording]} />)}
      </View>
      {recordingIndex === null ? (
        <Action label={sequences.length < 7 ? `RECORD TRAIN SEQUENCE ${sequences.length + 1}/7` : `RECORD UNSEEN TEST SEQUENCE ${sequences.length - 6}/3`} disabled={!clock?.passes20MsGate || !teacherAligned || !liveRf || sequences.length >= 10 || (sequences.length >= 7 && !model)} onPress={startSequence} />
      ) : (
        <Action label={`FINISH SEQUENCE ${recordingIndex + 1}`} onPress={finishSequence} />
      )}
      <ThemedText preset="mono" style={styles.split}>FIXED SPLIT · SEQUENCES 1–7 TRAIN · 8–10 HELD OUT</ThemedText>
      {model && <ThemedText testID="pose-student-model" preset="bodySm" style={styles.valid}>Student: {model.joints.length} coarse joints · {model.trainingSampleCount} paired samples · raw camera/depth not retained</ThemedText>}
      {evaluation && (
        <View testID="pose-calibration-evaluation" style={styles.result}>
          <ThemedText preset="mono" style={evaluation.passesTarget ? styles.valid : styles.draft}>{evaluation.passesTarget ? 'VALID / MEASURED' : 'DRAFT / TARGET NOT MET'}</ThemedText>
          <ThemedText preset="bodySm" style={styles.copy}>PCK@20cm: {(evaluation.baselinePck20cm * 100).toFixed(1)}% → {(evaluation.calibratedPck20cm * 100).toFixed(1)}% · improvement {(evaluation.improvementFraction * 100).toFixed(1)}%</ThemedText>
          <ThemedText preset="bodySm" style={styles.copy}>Lost poses: {(evaluation.baselineLostPoseRate * 100).toFixed(1)}% → {(evaluation.calibratedLostPoseRate * 100).toFixed(1)}% · 3 unseen sequences</ThemedText>
        </View>
      )}
      {!evaluation && savedCalibration && <ThemedText preset="bodySm" style={savedCalibration.quality === 'VALID' ? styles.valid : styles.draft}>Saved pose calibration: {savedCalibration.quality} · PCK gain {(savedCalibration.evaluation.improvementFraction * 100).toFixed(1)}%</ThemedText>}
    </InstrumentPanel>
  );
};

const styles = StyleSheet.create({
  state: { color: instrumentColors.textSecondary, fontSize: 8 },
  copy: { color: instrumentColors.textSecondary, lineHeight: 18 },
  message: { color: instrumentColors.text, lineHeight: 18, marginVertical: spacing.sm },
  statusRow: { flexDirection: 'row', justifyContent: 'space-between', gap: spacing.xs, marginVertical: spacing.md },
  action: { minHeight: 44, borderWidth: 1, borderColor: instrumentColors.cyanDim, borderRadius: 8, alignItems: 'center', justifyContent: 'center', paddingHorizontal: spacing.sm, marginBottom: spacing.sm },
  actionText: { color: instrumentColors.cyan, fontSize: 10, textAlign: 'center' },
  disabled: { opacity: 0.35 },
  pressed: { opacity: 0.65 },
  valid: { color: instrumentColors.green, fontSize: 9 },
  draft: { color: instrumentColors.warning, fontSize: 9 },
  progress: { flexDirection: 'row', gap: 4, marginVertical: spacing.sm },
  sequence: { height: 7, flex: 1, borderRadius: 4, backgroundColor: instrumentColors.borderStrong },
  sequenceDone: { backgroundColor: instrumentColors.green },
  sequenceRecording: { backgroundColor: instrumentColors.warning },
  split: { color: instrumentColors.textSecondary, fontSize: 7, marginVertical: spacing.sm },
  result: { borderWidth: 1, borderColor: instrumentColors.greenDim, borderRadius: 8, padding: spacing.sm, gap: spacing.xs },
});
