import { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet, TextInput, View } from 'react-native';
import { InstrumentPanel, instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import { useLidarCapture } from '@/hooks/useLidarCapture';
import { fitSpatialCorrection, validateSpatialCorrection } from '@/services/lidar.service';
import { useNlosStore } from '@/stores/nlosStore';
import { spacing } from '@/theme/spacing';
import type { ReferenceWalk, ReferenceWalkSample, SpatialCorrectionModel, CalibrationValidation } from '@/types/lidar';

const Action = ({ label, onPress, disabled = false }: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
}) => (
  <Pressable
    accessibilityRole="button"
    accessibilityLabel={label}
    disabled={disabled}
    onPress={onPress}
    style={({ pressed }) => [styles.action, disabled && styles.disabled, pressed && !disabled && styles.pressed]}
  >
    <ThemedText preset="labelMd" style={styles.actionText}>{label}</ThemedText>
  </Pressable>
);

export const LidarCommissioningCard = () => {
  const lidar = useLidarCapture();
  const nlosFrame = useNlosStore((state) => state.frame);
  const nlosFreshness = useNlosStore((state) => state.freshness);
  const streamStatus = useNlosStore((state) => state.streamStatus);
  const [nodeId, setNodeId] = useState('esp32-s3-01');
  const [relayEndpoint, setRelayEndpoint] = useState('');
  const [phase, setPhase] = useState<'setup' | 'baseline' | 'model_ready' | 'validation' | 'validated'>('setup');
  const [baselineSamples, setBaselineSamples] = useState<ReferenceWalkSample[]>([]);
  const [baselineAttempts, setBaselineAttempts] = useState(0);
  const [validationSamples, setValidationSamples] = useState<ReferenceWalkSample[]>([]);
  const [validationAttempts, setValidationAttempts] = useState(0);
  const [model, setModel] = useState<SpatialCorrectionModel | null>(null);
  const [validation, setValidation] = useState<CalibrationValidation | null>(null);
  const frameRef = useRef(nlosFrame);
  const freshnessRef = useRef(nlosFreshness);
  const sampleInFlight = useRef(false);
  const capturingDepth = lidar.status.state === 'capturing_depth';
  const capturingRoom = lidar.status.state === 'capturing_room';
  const supported = Boolean(lidar.capabilities?.sceneDepthSupported);
  const recording = phase === 'baseline' || phase === 'validation';

  useEffect(() => { frameRef.current = nlosFrame; freshnessRef.current = nlosFreshness; }, [nlosFrame, nlosFreshness]);

  useEffect(() => {
    if (!recording) return undefined;
    const timer = setInterval(() => {
      if (sampleInFlight.current) return;
      sampleInFlight.current = true;
      if (phase === 'baseline') setBaselineAttempts((value) => value + 1);
      else setValidationAttempts((value) => value + 1);
      void lidar.getCurrentPose().then((pose) => {
        const frame = frameRef.current;
        const now = Date.now();
        const track = frame?.tracks
          .filter((candidate) => candidate.state === 'tracking')
          .sort((left, right) => right.confidence - left.confidence)[0];
        const poseAssociationSkewMs = frame ? Math.abs(now - frame.capturedAtUnixMs) : Number.POSITIVE_INFINITY;
        if (!pose || pose.trackingState !== 'normal' || freshnessRef.current !== 'fresh' || frame?.source !== 'live' || !track || poseAssociationSkewMs > 100) return;
        if (lidar.nodes[0] && pose.coordinateFrameId !== lidar.nodes[0].coordinateFrameId) return;
        const sample: ReferenceWalkSample = {
          capturedAtUnixMs: now, coordinateFrameId: pose.coordinateFrameId,
          referencePositionM: pose.positionM,
          rfPositionM: [track.positionM.x, track.positionM.y, track.positionM.z],
          rfTrackId: track.trackId, rfConfidence: track.confidence,
          rfCapturedAtUnixMs: frame.capturedAtUnixMs, poseAssociationSkewMs,
        };
        if (phase === 'baseline') setBaselineSamples((values) => [...values, sample]);
        else setValidationSamples((values) => [...values, sample]);
      }).finally(() => { sampleInFlight.current = false; });
    }, 500);
    return () => clearInterval(timer);
  }, [lidar, phase, recording]);

  const asWalk = (samples: ReferenceWalkSample[], attempts: number): ReferenceWalk => ({
    referenceKind: 'visible_device_path_proxy', evidence: 'MEASURED', samples, attemptedSamples: attempts,
    lostTrackRate: attempts > 0 ? Math.max(0, 1 - samples.length / attempts) : 1,
  });

  const stopBaseline = () => {
    const fitted = fitSpatialCorrection(baselineSamples);
    setModel(fitted);
    setPhase(fitted ? 'model_ready' : 'setup');
    if (!fitted) lidar.setError('Record at least five varied live RF/reference samples; a stationary or missing track cannot calibrate scale and yaw.');
  };

  const stopValidation = () => {
    if (!model) return;
    const result = validateSpatialCorrection(asWalk(baselineSamples, baselineAttempts), asWalk(validationSamples, validationAttempts), model);
    setValidation(result);
    setPhase('validated');
  };

  return (
    <InstrumentPanel
      eyebrow="Guided RuView calibration"
      accessory={<ThemedText preset="mono" style={styles.state}>{lidar.status.state.toUpperCase()}</ThemedText>}
    >
      <ThemedText preset="bodySm" style={styles.boundary}>
        Use the iPhone as an installation instrument: scan the room, mark three or more RuView nodes, then carry the phone on two visible reference walks while authenticated RF tracking is live. It is not required after calibration.
      </ThemedText>
      <View style={styles.capabilityRow}>
        <View style={[styles.dot, supported && styles.dotReady]} />
        <ThemedText testID="lidar-capability" preset="bodySm" style={styles.capabilityText}>
          {!lidar.nativeModuleAvailable
            ? 'Native module unavailable — install the iOS development build'
            : supported
              ? `Scene depth ready${lidar.capabilities?.roomPlanSupported ? ' · RoomPlan ready' : ''}`
              : 'LiDAR is unavailable on this device or simulator'}
        </ThemedText>
      </View>
      <ThemedText preset="mono" style={styles.stepLabel}>1 / ROOM FRAME</ThemedText>
      <View style={styles.actionRow}>
        <Action label={capturingRoom ? 'FINISH ROOM LAST' : 'START ROOM SCAN'} disabled={!lidar.capabilities?.roomPlanSupported || capturingDepth || recording} onPress={() => { void (capturingRoom ? lidar.stopRoom() : lidar.startRoom()); }} />
        <Action label={capturingDepth ? 'STOP DEPTH' : 'CHECK DEPTH'} disabled={!supported || capturingRoom || recording} onPress={() => { void (capturingDepth ? lidar.stopCapture() : lidar.startDepth()); }} />
      </View>
      <View style={styles.metrics}>
        <View style={styles.metric}><ThemedText preset="displayMd" style={styles.metricValue}>{lidar.frame?.pointCount ?? 0}</ThemedText><ThemedText preset="mono" style={styles.metricLabel}>LIVE POINTS</ThemedText></View>
        <View style={styles.metric}><ThemedText preset="displayMd" style={styles.metricValue}>{lidar.room?.surfaceCount ?? 0}</ThemedText><ThemedText preset="mono" style={styles.metricLabel}>SURFACES</ThemedText></View>
        <View style={styles.metric}><ThemedText preset="displayMd" style={styles.metricValue}>{lidar.nodes.length}</ThemedText><ThemedText preset="mono" style={styles.metricLabel}>NODES</ThemedText></View>
      </View>
      <ThemedText preset="mono" style={styles.stepLabel}>2 / MARK 3+ NODES WHILE ROOM SCAN IS ACTIVE</ThemedText>
      <TextInput
        accessibilityLabel="ESP32 node identifier"
        testID="lidar-node-id"
        value={nodeId}
        onChangeText={setNodeId}
        autoCapitalize="none"
        autoCorrect={false}
        maxLength={64}
        placeholder="esp32-s3-01"
        placeholderTextColor={instrumentColors.textSecondary}
        style={styles.input}
      />
      <Action label="MARK NODE AT IPHONE POSITION" disabled={!capturingRoom || !nodeId.trim() || recording} onPress={() => { void lidar.markNode(nodeId); }} />

      <ThemedText preset="mono" style={styles.stepLabel}>3 / REFERENCE WALK · AUTHENTICATED LIVE RF REQUIRED</ThemedText>
      <ThemedText preset="bodySm" style={styles.privacy}>Carry the phone along the participant path. The ARKit device path is a measured proxy, not body-joint ground truth. Keep the room scan active so every sample stays in one coordinate frame.</ThemedText>
      <View style={styles.statusRow}>
        <ThemedText preset="mono" style={streamStatus === 'live' ? styles.valid : styles.draft}>RF {streamStatus.toUpperCase()}</ThemedText>
        <ThemedText preset="mono" style={styles.digest}>BASE {baselineSamples.length}/{baselineAttempts} · CHECK {validationSamples.length}/{validationAttempts}</ThemedText>
      </View>
      {phase === 'baseline' ? (
        <Action label="STOP BASELINE WALK" onPress={stopBaseline} />
      ) : (
        <Action label="START BASELINE WALK" disabled={!capturingRoom || lidar.nodes.length < 3 || streamStatus !== 'live' || phase === 'validation'} onPress={() => { setBaselineSamples([]); setBaselineAttempts(0); setModel(null); setValidation(null); setPhase('baseline'); }} />
      )}
      {model && phase !== 'validation' && (
        <Action label="START HELD-OUT VALIDATION WALK" disabled={!capturingRoom || streamStatus !== 'live'} onPress={() => { setValidationSamples([]); setValidationAttempts(0); setValidation(null); setPhase('validation'); }} />
      )}
      {phase === 'validation' && <Action label="STOP & SCORE VALIDATION" onPress={stopValidation} />}
      {model && (
        <View style={styles.result}>
          <ThemedText preset="mono" style={styles.valid}>CORRECTION MODEL / TARGET</ThemedText>
          <ThemedText preset="bodySm" style={styles.privacy}>Fit: {(model.baselineMedianErrorM).toFixed(2)}m → {(model.fittedMedianErrorM).toFixed(2)}m · {(model.fittedImprovementFraction * 100).toFixed(0)}% lower median error</ThemedText>
          {validation && <ThemedText testID="calibration-validation-result" preset="bodySm" style={validation.passesTarget ? styles.valid : styles.draft}>Held out: {validation.rawMedianErrorM.toFixed(2)}m → {validation.calibratedMedianErrorM.toFixed(2)}m · {(validation.improvementFraction * 100).toFixed(0)}% · lost tracks {(validation.calibratedLostTrackRate * 100).toFixed(0)}% · {validation.passesTarget ? 'PASS' : 'TARGET NOT MET'}</ThemedText>}
        </View>
      )}
      <Action label="GENERATE VERSIONED CALIBRATION" disabled={!lidar.room || !model || !validation} onPress={() => { if (model && validation) void lidar.saveCalibration(asWalk(baselineSamples, baselineAttempts), model, validation); }} />
      <ThemedText preset="mono" style={styles.stepLabel}>OPTIONAL RUFIELD DEPTH INGEST / {lidar.relayState.toUpperCase()}</ThemedText>
      <TextInput
        accessibilityLabel="LiDAR WebSocket relay endpoint"
        testID="lidar-relay-endpoint"
        value={relayEndpoint}
        onChangeText={setRelayEndpoint}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder="wss://ruview.local/ws/lidar?ticket=…"
        placeholderTextColor={instrumentColors.textSecondary}
        style={styles.input}
      />
      <Action
        label={lidar.relayState === 'connected' ? 'DISCONNECT RELAY' : 'CONNECT GEOMETRY RELAY'}
        disabled={lidar.relayState !== 'connected' && !relayEndpoint.trim()}
        onPress={() => lidar.relayState === 'connected' ? lidar.disconnectRelay() : void lidar.connectRelay(relayEndpoint.trim())}
      />
      <ThemedText preset="bodySm" style={styles.privacy}>Explicit session export only. The server requires an admin-scoped bearer or single-use ticket, converts compact depth to a signed P1 summary, and drops raw depth bytes.</ThemedText>
      {lidar.calibration && (
        <View testID="lidar-calibration-result" style={styles.result}>
          <ThemedText preset="mono" style={lidar.calibration.quality === 'VALID' && lidar.calibration.staleness.state === 'CURRENT' ? styles.valid : styles.draft}>{lidar.calibration.quality} / {lidar.calibration.staleness.state} / SHA-256</ThemedText>
          <ThemedText numberOfLines={1} preset="mono" style={styles.digest}>{lidar.calibration.digestSha256}</ThemedText>
          {lidar.calibration.staleness.reason && <ThemedText preset="bodySm" style={styles.draft}>{lidar.calibration.staleness.reason}</ThemedText>}
        </View>
      )}
      {lidar.error && <ThemedText testID="lidar-error" preset="bodySm" style={styles.error}>{lidar.error}</ThemedText>}
      <ThemedText preset="bodySm" style={styles.privacy}>
        Stored: bounded room geometry, node transforms, correction model, validation metrics, room fingerprint, and digest. Raw camera images and raw depth buffers are never persisted. Rescan after nodes move or residuals exceed the recorded threshold.
      </ThemedText>
    </InstrumentPanel>
  );
};

const styles = StyleSheet.create({
  state: { color: instrumentColors.textSecondary, fontSize: 8 },
  boundary: { color: instrumentColors.textSecondary, lineHeight: 19, marginBottom: spacing.md },
  capabilityRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.md },
  capabilityText: { color: instrumentColors.text, flex: 1 },
  dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: instrumentColors.warning },
  dotReady: { backgroundColor: instrumentColors.green },
  actionRow: { flexDirection: 'row', gap: spacing.sm },
  action: { minHeight: 44, flex: 1, borderWidth: 1, borderColor: instrumentColors.cyanDim, borderRadius: 8, alignItems: 'center', justifyContent: 'center', paddingHorizontal: spacing.sm, marginBottom: spacing.sm },
  actionText: { color: instrumentColors.cyan, fontSize: 10, textAlign: 'center' },
  disabled: { opacity: 0.35 },
  pressed: { opacity: 0.65 },
  metrics: { flexDirection: 'row', gap: spacing.sm, marginVertical: spacing.sm },
  metric: { flex: 1, borderWidth: 1, borderColor: instrumentColors.border, borderRadius: 8, padding: spacing.sm },
  metricValue: { color: instrumentColors.green, fontSize: 20 },
  metricLabel: { color: instrumentColors.textSecondary, fontSize: 7 },
  stepLabel: { color: instrumentColors.textSecondary, fontSize: 8, marginTop: spacing.sm, marginBottom: spacing.xs },
  input: { minHeight: 44, borderWidth: 1, borderColor: instrumentColors.borderStrong, borderRadius: 8, color: instrumentColors.text, paddingHorizontal: spacing.md, marginBottom: spacing.sm },
  result: { borderWidth: 1, borderColor: instrumentColors.greenDim, borderRadius: 8, padding: spacing.sm, marginTop: spacing.sm },
  statusRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: spacing.sm, marginVertical: spacing.sm },
  valid: { color: instrumentColors.green, fontSize: 9 },
  draft: { color: instrumentColors.warning, fontSize: 9 },
  digest: { color: instrumentColors.textSecondary, fontSize: 7, marginTop: 4 },
  error: { color: instrumentColors.warning, marginTop: spacing.sm },
  privacy: { color: instrumentColors.textSecondary, lineHeight: 17, marginTop: spacing.sm },
});
