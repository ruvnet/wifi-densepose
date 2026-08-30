import { Pressable, StyleSheet, View } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { colors, spacing } from '@/theme';
import type { LidarStatus, VisibleDepthDiagnostic, VisibleDepthMetrics } from '@/types/lidar';
import type { SensorFusionDisplayFrame } from '@/types/fusion';

export const SensorFusionHUD = ({
  fusion,
  csiConnected,
  csiSimulated,
  csiNodes,
  lidarStatus,
  validationMetrics,
  validationDiagnostic,
  lidarSupported,
  lidarRelayState,
  onToggleDepth,
  onToggleValidation,
  onToggleWallOverlay,
}: {
  fusion: SensorFusionDisplayFrame;
  csiConnected: boolean;
  csiSimulated: boolean;
  csiNodes: number;
  lidarStatus: LidarStatus;
  validationMetrics: VisibleDepthMetrics | null;
  validationDiagnostic: VisibleDepthDiagnostic | null;
  lidarSupported: boolean;
  lidarRelayState: 'disconnected' | 'connecting' | 'connected' | 'error';
  onToggleDepth: () => void;
  onToggleValidation: () => void;
  onToggleWallOverlay: () => void;
}) => {
  const depthActive = lidarStatus.state === 'capturing_depth';
  const validating = lidarStatus.state === 'validating_calibration' || lidarStatus.state === 'validating_wall_scan';
  const poseCalibrationCurrent = fusion.poseCalibration?.quality === 'VALID'
    && fusion.calibration?.quality === 'VALID'
    && fusion.calibration.staleness.state === 'CURRENT'
    && fusion.poseCalibration.spatialCalibrationId === fusion.calibration.calibrationId
    && fusion.poseCalibration.roomFingerprint === fusion.calibration.staleness.roomFingerprint;
  return (
    <View pointerEvents="box-none" style={styles.wrap}>
      <View style={styles.panel}>
        <View style={styles.headingRow}>
          <ThemedText preset="mono" style={styles.heading}>SENSOR FUSION DISPLAY</ThemedText>
          <ThemedText preset="mono" style={styles.overlay}>CO-RENDERED / NOT FUSED</ThemedText>
        </View>
        <View style={styles.sources}>
          <ThemedText preset="mono" style={csiConnected || csiSimulated ? styles.ready : styles.idle}>CSI {csiSimulated ? 'SIM' : csiConnected ? 'LIVE' : 'IDLE'} · {csiNodes} NODES</ThemedText>
          <ThemedText preset="mono" style={fusion.lidarFrame ? styles.ready : styles.idle}>LIDAR {fusion.lidarFrame?.pointCount ?? 0} PTS</ThemedText>
          <ThemedText preset="mono" style={fusion.nlosFreshness === 'fresh' ? styles.ready : styles.idle}>HIDDEN FIELD {fusion.nlosTracks.length} TRACKS</ThemedText>
        </View>
        <ThemedText preset="mono" style={styles.gate}>
          CALIBRATION {fusion.calibration?.quality ?? 'MISSING'} · V1 HIDDEN FIELD HAS NO SHARED FRAME
        </ThemedText>
        <ThemedText preset="mono" style={poseCalibrationCurrent ? styles.ready : styles.gate}>
          POSE STUDENT {poseCalibrationCurrent ? 'VALID/CURRENT' : fusion.poseCalibration ? 'STALE/DRAFT' : 'MISSING'}{fusion.poseCalibration ? ` · PCK20 GAIN ${(fusion.poseCalibration.evaluation.improvementFraction * 100).toFixed(0)}%` : ''}
        </ThemedText>
        <ThemedText preset="mono" style={fusion.transientNlos.status === 'track_stream_available' ? styles.ready : styles.blocked}>
          TRANSIENT/SPAD {fusion.transientNlos.status === 'track_stream_available'
            ? `TRACK STREAM · ${fusion.transientNlos.sensorModel ?? 'SENSOR'} · ${fusion.transientNlos.evidenceLevel ?? 'UNKNOWN'}`
            : 'BLOCKED · RAW PHOTON HISTOGRAMS REQUIRED'}
        </ThemedText>
        {lidarRelayState === 'connected' && (
          <ThemedText accessibilityRole="alert" preset="mono" style={styles.exporting}>DEPTH EXPORT ACTIVE · COMPACT DEPTH LEAVING DEVICE · CAMERA PREVIEW LOCAL ONLY</ThemedText>
        )}
        {fusion.wallOverlay.enabled && (
          <View style={styles.wallNotice}>
            <ThemedText preset="mono" style={fusion.wallOverlay.source === 'live_rf' || fusion.wallOverlay.source === 'calibrated_rf_student' ? styles.ready : styles.blocked}>
              VIDEO + RF POSE · {fusion.wallOverlay.source === 'calibrated_rf_student' ? 'VALIDATED ROOM STUDENT' : fusion.wallOverlay.source === 'live_rf' ? 'UNVALIDATED LIVE RF' : fusion.wallOverlay.source.toUpperCase()} · AGE {fusion.wallOverlay.sourceAgeMs == null ? 'N/A' : `${Math.round(fusion.wallOverlay.sourceAgeMs)}ms`}
            </ThemedText>
            <ThemedText preset="mono" style={styles.gate}>CAMERA/LIDAR: VISIBLE SURFACES ONLY · OCCLUDED POSE: RUView RF · {fusion.wallOverlay.evidenceLabel === 'validated_room_student' ? 'HELD-OUT CALIBRATED' : 'UNVALIDATED'}</ThemedText>
          </View>
        )}
        {validationMetrics && validating && (
          <ThemedText preset="mono" style={styles.metrics}>
            DIRECT DEPTH {validationMetrics.phase.toUpperCase()} · {validationMetrics.phaseSecondsRemaining}s · {validationMetrics.fps.toFixed(0)} FPS · {(validationMetrics.depthCoverage * 100).toFixed(0)}% COVERAGE
          </ThemedText>
        )}
        {validationDiagnostic && !validating && (
          <ThemedText preset="mono" style={styles.metrics}>DIAGNOSTIC {validationDiagnostic.completionStatus.toUpperCase()} · DIRECT_DEPTH · THROUGH-WALL INFERENCE BLOCKED</ThemedText>
        )}
        <View style={styles.actions}>
          <Pressable accessibilityRole="button" disabled={!lidarSupported || validating} onPress={onToggleDepth} style={[styles.button, (!lidarSupported || validating) && styles.disabled]}>
            <ThemedText preset="mono" style={styles.buttonText}>{depthActive ? 'STOP LIDAR' : 'START LIDAR'}</ThemedText>
          </Pressable>
          <Pressable accessibilityRole="button" disabled={!lidarSupported || depthActive} onPress={onToggleValidation} style={[styles.button, (!lidarSupported || depthActive) && styles.disabled]}>
            <ThemedText preset="mono" style={styles.buttonText}>{validating ? 'CANCEL 45s CHECK' : 'RUN 45s CHECK'}</ThemedText>
          </Pressable>
        </View>
        <Pressable accessibilityRole="button" disabled={!lidarSupported || validating} onPress={onToggleWallOverlay} style={[styles.wallButton, (!lidarSupported || validating) && styles.disabled]}>
          <ThemedText preset="mono" style={styles.buttonText}>{fusion.wallOverlay.enabled ? 'STOP WALL OVERLAY' : 'START VIDEO WALL OVERLAY'}</ThemedText>
        </Pressable>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  wrap: { position: 'absolute', top: 48, left: spacing.sm, right: spacing.sm },
  panel: { borderWidth: 1, borderColor: 'rgba(50,184,198,0.45)', borderRadius: 10, backgroundColor: 'rgba(8,12,22,0.88)', padding: spacing.sm },
  headingRow: { flexDirection: 'row', justifyContent: 'space-between', gap: spacing.sm },
  heading: { color: colors.accent, fontSize: 9 },
  overlay: { color: '#ffb65c', fontSize: 7 },
  sources: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, marginTop: 6 },
  ready: { color: '#2bd977', fontSize: 7 },
  idle: { color: colors.textSecondary, fontSize: 7 },
  gate: { color: '#ffb65c', fontSize: 7, marginTop: 5 },
  blocked: { color: '#ff7777', fontSize: 7, marginTop: 5 },
  exporting: { color: '#ff7777', fontSize: 7, marginTop: 5, borderWidth: 1, borderColor: '#ff7777', padding: 5 },
  metrics: { color: colors.textSecondary, fontSize: 7, marginTop: 5 },
  actions: { flexDirection: 'row', gap: spacing.sm, marginTop: 7 },
  wallNotice: { marginTop: 6, borderLeftWidth: 2, borderLeftColor: '#ffb65c', paddingLeft: 6 },
  button: { flex: 1, minHeight: 32, borderWidth: 1, borderColor: 'rgba(50,184,198,0.55)', borderRadius: 6, alignItems: 'center', justifyContent: 'center' },
  buttonText: { color: colors.accent, fontSize: 8 },
  disabled: { opacity: 0.35 },
  wallButton: { minHeight: 32, marginTop: 7, borderWidth: 1, borderColor: '#ffb65c', borderRadius: 6, alignItems: 'center', justifyContent: 'center' },
});
