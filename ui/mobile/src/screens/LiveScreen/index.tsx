import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button, Image, Platform, StyleSheet, View } from 'react-native';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { usePoseStream } from '@/hooks/usePoseStream';
import { useLidarCapture } from '@/hooks/useLidarCapture';
import { useNlosStore } from '@/stores/nlosStore';
import { useLidarStore } from '@/stores/lidarStore';
import { extractCsiFeatures, predictPose } from '@/services/poseCalibration.service';
import { colors, spacing } from '@/theme';
import type { SensorFusionDisplayFrame } from '@/types/fusion';
import type { ConnectionStatus, SensingFrame } from '@/types/sensing';
import { LiveHUD } from './LiveHUD';
import { SensorFusionHUD } from './SensorFusionHUD';

type LiveMode = 'LIVE' | 'SIM' | 'IDLE';

const getMode = (
  status: ConnectionStatus,
  isSimulated: boolean,
  frame: SensingFrame | null,
): LiveMode => {
  if (isSimulated || frame?.source === 'simulated') return 'SIM';
  if (status === 'connected') return 'LIVE';
  return 'IDLE';
};

const isWeb = Platform.OS === 'web';

type ViewerProps = {
  frame: SensingFrame | null;
  fusion: SensorFusionDisplayFrame;
  onReady: () => void;
  onFps: (fps: number) => void;
  onError: (msg: string) => void;
};

const WebLiveViewer = ({ frame, fusion, onReady, onFps, onError }: ViewerProps) => {
  const [Viewer, setViewer] = useState<React.ComponentType<any> | null>(null);

  useEffect(() => {
    import('./GaussianSplatWebView.web').then((mod) => {
      setViewer(() => mod.GaussianSplatWebViewWeb);
    }).catch(() => onError('Failed to load web viewer'));
  }, [onError]);

  if (!Viewer) return null;
  return <Viewer frame={frame} fusion={fusion} onReady={onReady} onFps={onFps} onError={onError} />;
};

const NativeLiveViewer = ({ frame, fusion, onReady, onFps, onError }: ViewerProps) => {
  const webViewRef = useRef<{ postMessage: (message: string) => void } | null>(null);
  const readyRef = useRef(false);
  const [WVComponent, setWVComponent] = useState<React.ComponentType<any> | null>(null);

  const sendFrame = useCallback((nextFrame: SensingFrame) => {
    webViewRef.current?.postMessage(JSON.stringify({
      type: 'FRAME_UPDATE',
      payload: nextFrame,
    }));
  }, []);

  useEffect(() => {
    try {
      const { GaussianSplatWebView } = require('./GaussianSplatWebView');
      setWVComponent(() => GaussianSplatWebView);
    } catch {
      onError('WebView not available on this platform');
    }
  }, [onError]);

  useEffect(() => {
    if (readyRef.current && frame) {
      sendFrame(frame);
    }
  }, [frame, sendFrame]);

  useEffect(() => {
    if (readyRef.current) webViewRef.current?.postMessage(JSON.stringify({ type: 'FUSION_UPDATE', payload: fusion }));
  }, [fusion]);

  if (!WVComponent) return null;

  return (
    <WVComponent
      webViewRef={webViewRef}
      onMessage={(event: any) => {
        try {
          const data = typeof event.nativeEvent.data === 'string'
            ? JSON.parse(event.nativeEvent.data)
            : event.nativeEvent.data;
          if (data.type === 'READY') {
            readyRef.current = true;
            onReady();
            if (frame) sendFrame(frame);
            webViewRef.current?.postMessage(JSON.stringify({ type: 'FUSION_UPDATE', payload: fusion }));
          }
          else if (data.type === 'FPS_TICK') onFps(data.payload?.fps ?? 0);
          else if (data.type === 'ERROR') onError(data.payload?.message ?? 'Unknown error');
        } catch { /* ignore */ }
      }}
      onError={() => onError('WebView renderer failed')}
    />
  );
};

export const LiveScreen = () => {
  const { lastFrame, connectionStatus, isSimulated } = usePoseStream();
  const lidar = useLidarCapture();
  const nlosFrame = useNlosStore((state) => state.frame);
  const nlosFreshness = useNlosStore((state) => state.freshness);
  const poseCalibration = useLidarStore((state) => state.poseCalibration);
  const [ready, setReady] = useState(false);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [viewerKey, setViewerKey] = useState(0);
  const [wallOverlayEnabled, setWallOverlayEnabled] = useState(false);

  const handleReady = useCallback(() => { setReady(true); setError(null); }, []);
  const handleFps = useCallback((f: number) => setFps(Math.max(0, Math.floor(f))), []);
  const handleError = useCallback((msg: string) => { setError(msg); setReady(false); }, []);
  const handleRetry = useCallback(() => { setError(null); setReady(false); setFps(0); setViewerKey((v) => v + 1); }, []);

  const rssi = lastFrame?.features?.mean_rssi;
  const personCount = lastFrame?.estimated_persons ?? (lastFrame?.classification?.presence ? 1 : 0);
  const mode = getMode(connectionStatus, isSimulated, lastFrame);
  const poseCalibrationUsable = poseCalibration?.quality === 'VALID'
    && lidar.calibration?.quality === 'VALID'
    && lidar.calibration.staleness.state === 'CURRENT'
    && poseCalibration.spatialCalibrationId === lidar.calibration.calibrationId
    && poseCalibration.roomFingerprint === lidar.calibration.staleness.roomFingerprint;
  const calibratedPose = useMemo(() => (
    lastFrame && connectionStatus === 'connected' && poseCalibrationUsable && poseCalibration
      ? predictPose(poseCalibration.model, extractCsiFeatures(lastFrame))
      : null
  ), [connectionStatus, lastFrame, poseCalibration, poseCalibrationUsable]);
  const displayFrame = useMemo<SensingFrame | null>(() => lastFrame && calibratedPose ? {
    ...lastFrame,
    source: 'calibrated_csi_student',
    persons: [{ id: 0, confidence: Math.max(0, Math.min(1, lastFrame.classification.confidence)), keypoints: calibratedPose }],
  } : lastFrame, [calibratedPose, lastFrame]);
  const fusion: SensorFusionDisplayFrame = {
    schema: 'ruview.sensor-fusion.display.v1',
    lidarFrame: lidar.frame,
    cameraPreview: wallOverlayEnabled ? lidar.cameraPreview : null,
    nlosTracks: nlosFreshness === 'fresh' ? nlosFrame?.tracks ?? [] : [],
    nlosFreshness,
    calibration: lidar.calibration,
    poseCalibration,
    alignment: 'overlay_only',
    transientNlos: {
      status: nlosFreshness === 'fresh' && nlosFrame?.provenance.histogramPreserved
        && (nlosFrame.provenance.transientKind === 'raw_histogram' || nlosFrame.provenance.transientKind === 'compact_normalized_histogram')
        ? 'track_stream_available'
        : 'blocked_raw_transients_unavailable',
      requiredSchema: 'ruview.nlos.transient.v1',
      requiredMeasurement: 'picosecond_photon_histograms',
      algorithmFamily: 'motion_induced_aperture_sampling',
      sensorModel: nlosFrame?.provenance.sensorModel ?? null,
      evidenceLevel: nlosFrame?.evidenceLevel ?? null,
    },
    wallOverlay: {
      enabled: wallOverlayEnabled,
      source: calibratedPose ? 'calibrated_rf_student' : mode === 'SIM' ? 'simulated' : connectionStatus === 'connected' && personCount > 0 ? 'live_rf' : 'unavailable',
      evidenceLabel: calibratedPose ? 'validated_room_student' : 'rf_pose_hypothesis',
      confidence: Math.max(0, Math.min(1, lastFrame?.classification?.confidence ?? 0)),
      sourceAgeMs: lastFrame?.timestamp ? Math.max(0, Date.now() - lastFrame.timestamp) : null,
      cameraAndLidarVisibleSurfacesOnly: true,
      throughWallClaim: false,
    },
  };

  const toggleWallOverlay = useCallback(async () => {
    const next = !wallOverlayEnabled;
    setWallOverlayEnabled(next);
    lidar.setCameraPreview(null);
    if (next) {
      await lidar.stopCapture();
      await lidar.startDepth(true);
    } else if (lidar.status.state === 'capturing_depth') {
      await lidar.stopCapture();
      await lidar.startDepth(false);
    }
  }, [lidar, wallOverlayEnabled]);

  if (error) {
    return (
      <ThemedView style={styles.fallbackWrap}>
        <ThemedText preset="bodyLg">Live visualization failed</ThemedText>
        <ThemedText preset="bodySm" color="textSecondary" style={styles.errorText}>{error}</ThemedText>
        <Button title="Retry" onPress={handleRetry} />
      </ThemedView>
    );
  }

  return (
    <ErrorBoundary>
      <View testID="live-screen" style={styles.container}>
        {fusion.cameraPreview && (
          <Image
            accessibilityLabel="Live iPhone camera preview; visible surfaces only"
            source={{ uri: `data:image/jpeg;base64,${fusion.cameraPreview.jpegBase64}` }}
            resizeMode="cover"
            style={styles.cameraPreview}
          />
        )}
        {isWeb ? (
          <WebLiveViewer key={viewerKey} frame={displayFrame} fusion={fusion} onReady={handleReady} onFps={handleFps} onError={handleError} />
        ) : (
          <NativeLiveViewer key={viewerKey} frame={displayFrame} fusion={fusion} onReady={handleReady} onFps={handleFps} onError={handleError} />
        )}

        <LiveHUD
          connectionStatus={connectionStatus}
          fps={fps}
          rssi={rssi}
          confidence={lastFrame?.classification?.confidence ?? 0}
          personCount={personCount}
          mode={mode}
        />

        <SensorFusionHUD
          fusion={fusion}
          csiConnected={connectionStatus === 'connected'}
          csiSimulated={mode === 'SIM'}
          csiNodes={lastFrame?.nodes?.length ?? 0}
          lidarStatus={lidar.status}
          validationMetrics={lidar.validationMetrics}
          validationDiagnostic={lidar.validationDiagnostic}
          lidarSupported={Boolean(lidar.capabilities?.sceneDepthSupported)}
          lidarRelayState={lidar.relayState}
          onToggleDepth={() => { void (lidar.status.state === 'capturing_depth' ? lidar.stopCapture() : lidar.startDepth()); }}
          onToggleValidation={() => { void ((lidar.status.state === 'validating_calibration' || lidar.status.state === 'validating_wall_scan') ? lidar.cancelValidation() : lidar.startValidation()); }}
          onToggleWallOverlay={() => { void toggleWallOverlay(); }}
        />

        {!ready && (
          <View style={styles.loadingWrap}>
            <LoadingSpinner />
            <ThemedText preset="bodyMd" style={styles.loadingText}>Loading live renderer</ThemedText>
          </View>
        )}
      </View>
    </ErrorBoundary>
  );
};

export default LiveScreen;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  cameraPreview: { ...StyleSheet.absoluteFillObject, opacity: 0.72 },
  loadingWrap: { ...StyleSheet.absoluteFillObject, backgroundColor: colors.bg, alignItems: 'center', justifyContent: 'center', gap: spacing.md },
  loadingText: { color: colors.textSecondary },
  fallbackWrap: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.md, padding: spacing.lg },
  errorText: { textAlign: 'center' },
});
