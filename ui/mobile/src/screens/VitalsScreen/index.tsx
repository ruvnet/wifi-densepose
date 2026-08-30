import { useEffect, useMemo, useRef, useState } from 'react';
import { ScrollView, StyleSheet, View } from 'react-native';
import { AppleHomeCard } from './AppleHomeCard';
import { BreathingGauge } from './BreathingGauge';
import { HeartRateGauge } from './HeartRateGauge';
import { VitalWaveform } from './VitalWaveform';
import { InstrumentGrid, InstrumentPanel, instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { usePoseStream } from '@/hooks/usePoseStream';
import { usePoseStore } from '@/stores/poseStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useTabScrollToTop } from '@/stores/tabScrollStore';
import { spacing } from '@/theme/spacing';

const FRESHNESS_LIMIT_MS = 3_000;
const HISTORY_LIMIT = 40;
const HISTORY_SAMPLE_INTERVAL_MS = 1_000;
const BREATHING_WAVEFORM_DOMAIN: [number, number] = [6, 30];
const HEART_WAVEFORM_DOMAIN: [number, number] = [40, 120];

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const normalizeConfidence = (value: unknown) => finite(value) ? Math.max(0, Math.min(1, value > 1 ? value / 100 : value)) : null;
const format = (value: unknown, digits = 2) => finite(value) ? value.toFixed(digits) : '—';
const frameTimeMs = (timestamp: number) => timestamp < 10_000_000_000 ? timestamp * 1_000 : timestamp;

const useStableVital = (
  measured: number | undefined,
  available: boolean,
  deadBand: number,
  maximumStep: number,
) => {
  const [stable, setStable] = useState<number | null>(null);
  useEffect(() => {
    if (!available || !finite(measured)) {
      setStable(null);
      return;
    }
    setStable((previous) => {
      if (previous == null) return measured;
      const delta = measured - previous;
      if (Math.abs(delta) < deadBand) return previous;
      return previous + Math.max(-maximumStep, Math.min(maximumStep, delta));
    });
  }, [available, deadBand, maximumStep, measured]);
  return stable;
};

const Reading = ({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) => (
  <View style={styles.reading}>
    <ThemedText preset="mono" style={styles.readingLabel}>{label}</ThemedText>
    <ThemedText preset="displayMd" style={[styles.readingValue, accent && styles.readingAccent]}>{value}</ThemedText>
  </View>
);

const Confidence = ({ value, label }: { value: number | null; label: string }) => (
  <View style={styles.confidence}>
    <View style={styles.confidenceHeading}>
      <ThemedText preset="mono" style={styles.miniLabel}>{label}</ThemedText>
      <ThemedText preset="mono" style={value == null ? styles.dim : styles.green}>{value == null ? 'NO SCORE' : `${Math.round(value * 100)}%`}</ThemedText>
    </View>
    <View style={styles.confidenceTrack}>
      <View style={[styles.confidenceFill, { width: `${(value ?? 0) * 100}%` }]} />
    </View>
  </View>
);

export default function VitalsScreen() {
  const scrollRef = useTabScrollToTop('Vitals');
  const { connectionStatus, lastFrame, isSimulated } = usePoseStream();
  const features = usePoseStore((state) => state.features);
  const classification = usePoseStore((state) => state.classification);
  const serverUrl = useSettingsStore((state) => state.serverUrl);
  const [breathingHistory, setBreathingHistory] = useState<number[]>([]);
  const [heartHistory, setHeartHistory] = useState<number[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const lastRecordedTimestamp = useRef<number | null>(null);

  const sourceIsLive = connectionStatus === 'connected' && !isSimulated && lastFrame?.source !== 'simulated';
  const timestamp = lastFrame?.timestamp;
  const fresh = sourceIsLive && finite(timestamp) && Math.abs(now - frameTimeMs(timestamp)) <= FRESHNESS_LIMIT_MS;
  const vitals = fresh ? lastFrame?.vital_signs : undefined;
  const breathing = vitals?.breathing_bpm ?? vitals?.breathing_rate_bpm;
  const heart = vitals?.hr_proxy_bpm ?? vitals?.heart_rate_bpm;
  const stableBreathing = useStableVital(breathing, fresh, 0.25, 0.75);
  const stableHeart = useStableVital(heart, fresh, 0.75, 2);
  const breathingConfidence = normalizeConfidence(vitals?.breathing_confidence ?? vitals?.confidence);
  const heartConfidence = normalizeConfidence(vitals?.heart_confidence ?? vitals?.heartbeat_confidence ?? vitals?.confidence);
  const classificationConfidence = fresh ? normalizeConfidence(classification?.confidence) : null;
  const nodeId = fresh ? lastFrame?.nodes?.[0]?.node_id ?? null : null;

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1_000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!fresh || !finite(timestamp)) return;
    const measuredAt = frameTimeMs(timestamp);
    if (lastRecordedTimestamp.current != null
      && measuredAt - lastRecordedTimestamp.current < HISTORY_SAMPLE_INTERVAL_MS) return;
    lastRecordedTimestamp.current = measuredAt;
    if (finite(stableBreathing)) setBreathingHistory((history) => [...history, stableBreathing].slice(-HISTORY_LIMIT));
    if (finite(stableHeart)) setHeartHistory((history) => [...history, stableHeart].slice(-HISTORY_LIMIT));
  }, [fresh, stableBreathing, stableHeart, timestamp]);

  const evidenceState = useMemo(() => {
    if (isSimulated || connectionStatus === 'simulated' || lastFrame?.source === 'simulated') return 'SIMULATION HIDDEN';
    if (fresh) return 'MEASURED / FRESH';
    return 'NO FRESH EVIDENCE';
  }, [connectionStatus, fresh, isSimulated, lastFrame?.source]);

  const activity = fresh ? classification?.motion_level?.replaceAll('_', ' ').toUpperCase() ?? 'UNCLASSIFIED' : '—';
  const signalQuality = fresh && finite(lastFrame?.signal_quality_score) ? `${Math.round(lastFrame.signal_quality_score * (lastFrame.signal_quality_score <= 1 ? 100 : 1))}%` : '—';

  return (
    <ThemedView style={styles.screen}>
      <InstrumentGrid />
      <ScrollView ref={scrollRef} testID="vitals-scroll-view" contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <InstrumentPanel
          eyebrow="Live RF vital field"
          accessory={<ThemedText preset="mono" style={fresh ? styles.green : styles.warning}>{evidenceState}</ThemedText>}
          style={styles.hero}
        >
          <ThemedText preset="displayLg" style={styles.heroTitle}>The room has a pulse.</ThemedText>
          <ThemedText preset="bodyMd" style={styles.heroCopy}>
            Live RF physiology estimates appear only when the sensing server sends explicit, fresh measurements. Missing evidence stays missing.
          </ThemedText>
          <View style={styles.heroStats}>
            <Reading label="ACTIVITY" value={activity} accent={fresh} />
            <Reading label="SIGNAL QUALITY" value={signalQuality} accent={fresh} />
            <Reading label="NODE" value={nodeId == null ? '—' : String(nodeId)} />
          </View>
        </InstrumentPanel>

        <View style={styles.gaugesRow}>
          <InstrumentPanel style={styles.gaugeCard} eyebrow="Respiratory field">
            <BreathingGauge available={stableBreathing != null} breathingBpm={stableBreathing} />
            <Confidence value={breathingConfidence} label="MEASUREMENT CONFIDENCE" />
          </InstrumentPanel>
          <InstrumentPanel style={styles.gaugeCard} eyebrow="Cardiac proxy">
            <HeartRateGauge available={stableHeart != null} heartProxyBpm={stableHeart} />
            <Confidence value={heartConfidence} label="MEASUREMENT CONFIDENCE" />
          </InstrumentPanel>
        </View>

        <InstrumentPanel eyebrow="Measured history" accessory={<ThemedText preset="mono" style={styles.dim}>{Math.max(breathingHistory.length, heartHistory.length)} SAMPLES</ThemedText>}>
          <ThemedText preset="mono" style={styles.traceLabel}>BREATHING RATE</ThemedText>
          <VitalWaveform values={breathingHistory} color={instrumentColors.cyan} label="breathing rate" domain={BREATHING_WAVEFORM_DOMAIN} />
          <ThemedText preset="mono" style={styles.traceLabel}>HEART-RATE PROXY</ThemedText>
          <VitalWaveform values={heartHistory} color={instrumentColors.danger} label="heart-rate proxy" domain={HEART_WAVEFORM_DOMAIN} />
        </InstrumentPanel>

        <InstrumentPanel eyebrow="Signal physiology / evidence quality" accessory={<ThemedText preset="mono" style={classificationConfidence == null ? styles.dim : styles.green}>{classificationConfidence == null ? 'UNSCORED' : `${Math.round(classificationConfidence * 100)}% CLASSIFIER`}</ThemedText>}>
          <View style={styles.metricGrid}>
            <Reading label="MEAN RSSI" value={fresh && finite(features?.mean_rssi) ? `${format(features.mean_rssi, 1)} dBm` : '—'} />
            <Reading label="VARIANCE" value={fresh ? format(features?.variance) : '—'} />
            <Reading label="MOTION BAND" value={fresh ? format(features?.motion_band_power, 3) : '—'} />
            <Reading label="BREATH BAND" value={fresh ? format(features?.breathing_band_power, 3) : '—'} />
            <Reading label="SPECTRAL ENTROPY" value={fresh ? format(features?.spectral_entropy, 3) : '—'} />
            <Reading label="PRESENCE" value={fresh && classification ? (classification.presence ? 'DETECTED' : 'CLEAR') : '—'} accent={fresh && Boolean(classification?.presence)} />
          </View>
        </InstrumentPanel>

        <InstrumentPanel eyebrow="Evidence boundary">
          <ThemedText preset="bodySm" style={styles.evidenceCopy}>
            These are RF-derived estimates, not medical measurements or a diagnostic device. If the server does not provide a fresh breathing or heart-rate field, RuView does not invent a BPM value from motion power.
          </ThemedText>
        </InstrumentPanel>

        <AppleHomeCard serverUrl={serverUrl} nodeId={nodeId} />
      </ScrollView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: instrumentColors.background },
  content: { paddingHorizontal: spacing.md, paddingTop: spacing.md, paddingBottom: 84, gap: spacing.md },
  hero: { backgroundColor: '#111820', paddingVertical: spacing.xl },
  heroTitle: { color: instrumentColors.text, fontSize: 31, lineHeight: 36, maxWidth: 310 },
  heroCopy: { color: instrumentColors.textSecondary, lineHeight: 20, maxWidth: 520 },
  heroStats: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, marginTop: spacing.sm },
  reading: { flexGrow: 1, flexBasis: 94, minHeight: 64, borderWidth: 1, borderColor: instrumentColors.border, borderRadius: 10, backgroundColor: 'rgba(7,11,17,0.55)', padding: spacing.sm, justifyContent: 'space-between' },
  readingLabel: { color: instrumentColors.textSecondary, fontSize: 7 },
  readingValue: { color: instrumentColors.text, fontSize: 16, lineHeight: 21 },
  readingAccent: { color: instrumentColors.green },
  gaugesRow: { flexDirection: 'row', gap: spacing.sm, alignItems: 'stretch' },
  gaugeCard: { flex: 1, paddingHorizontal: spacing.sm },
  confidence: { gap: spacing.xs },
  confidenceHeading: { flexDirection: 'row', justifyContent: 'space-between', gap: spacing.xs },
  confidenceTrack: { height: 4, borderRadius: 2, backgroundColor: instrumentColors.border, overflow: 'hidden' },
  confidenceFill: { height: '100%', backgroundColor: instrumentColors.green },
  miniLabel: { color: instrumentColors.textSecondary, fontSize: 6 },
  metricGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  traceLabel: { color: instrumentColors.textSecondary, fontSize: 8 },
  evidenceCopy: { color: instrumentColors.warning, lineHeight: 18 },
  green: { color: instrumentColors.green, fontSize: 8 },
  warning: { color: instrumentColors.warning, fontSize: 8 },
  dim: { color: instrumentColors.textSecondary, fontSize: 8 },
});
