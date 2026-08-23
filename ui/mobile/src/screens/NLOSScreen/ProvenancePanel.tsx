import { StyleSheet, View } from 'react-native';
import { InstrumentPanel, instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import { colors } from '@/theme/colors';
import { spacing } from '@/theme/spacing';
import type { NlosFreshness, NlosStreamStatus, NlosTrackFrame } from '@/types/nlos';

interface ProvenancePanelProps {
  frame: NlosTrackFrame | null;
  freshness: NlosFreshness;
  streamStatus: NlosStreamStatus;
}

export type NlosEvidenceState =
  | 'SYNTHETIC'
  | 'LIVE VERIFIED'
  | 'LIVE UNVERIFIED'
  | 'STALE'
  | 'DISCONNECTED';

const LIVE_ATTEMPT_STATUSES: ReadonlySet<NlosStreamStatus> = new Set([
  'authenticating',
  'connecting',
  'live',
]);

const hasVerifiedLiveEvidence = (frame: NlosTrackFrame): boolean => (
  frame.source === 'live'
  && frame.evidenceLevel === 'l2_calibrated'
);

export const resolveNlosEvidenceState = (
  frame: NlosTrackFrame | null,
  freshness: NlosFreshness,
  streamStatus: NlosStreamStatus,
): NlosEvidenceState => {
  if (freshness === 'stale') return 'STALE';
  if (freshness === 'fresh' && frame?.source === 'synthetic') return 'SYNTHETIC';
  if (freshness === 'fresh' && frame?.source === 'replay') return 'DISCONNECTED';

  if (LIVE_ATTEMPT_STATUSES.has(streamStatus) || frame?.source === 'live') {
    return freshness === 'fresh'
      && streamStatus === 'live'
      && frame !== null
      && hasVerifiedLiveEvidence(frame)
      ? 'LIVE VERIFIED'
      : 'LIVE UNVERIFIED';
  }

  return 'DISCONNECTED';
};

const sourceLabel = (frame: NlosTrackFrame | null): string => {
  if (!frame) return 'UNKNOWN';
  if (frame.source === 'synthetic') return 'SYNTHETIC';
  if (frame.source === 'replay') return 'REPLAY';
  return 'LIVE';
};

const sourceColor = (frame: NlosTrackFrame | null): string => {
  if (!frame) return colors.muted;
  if (frame.source === 'synthetic') return colors.warn;
  if (frame.source === 'replay') return colors.textSecondary;
  return instrumentColors.cyan;
};

const evidenceStateColor = (state: NlosEvidenceState): string => {
  if (state === 'LIVE VERIFIED') return instrumentColors.green;
  if (state === 'SYNTHETIC' || state === 'LIVE UNVERIFIED') {
    return instrumentColors.warning;
  }
  if (state === 'STALE') return instrumentColors.danger;
  return instrumentColors.textSecondary;
};

const humanize = (value: string) => value.replace(/_/g, ' ').toUpperCase();

const ProvenanceRow = ({ label, value }: { label: string; value: string }) => (
  <View style={styles.provenanceRow}>
    <ThemedText preset="bodySm" color="textSecondary" style={styles.provenanceLabel}>{label}</ThemedText>
    <ThemedText preset="bodySm" numberOfLines={1} style={styles.provenanceValue}>{value}</ThemedText>
  </View>
);

export const ProvenancePanel = ({ frame, freshness, streamStatus }: ProvenancePanelProps) => {
  const label = sourceLabel(frame);
  const accent = sourceColor(frame);
  const evidenceState = resolveNlosEvidenceState(frame, freshness, streamStatus);
  const stateAccent = evidenceStateColor(evidenceState);

  return (
    <InstrumentPanel testID="nlos-provenance-panel" eyebrow="Evidence state" style={styles.card}>
      <View style={styles.stateRow}>
        <View style={styles.stateIdentity}>
          <View style={[styles.stateDot, { backgroundColor: stateAccent }]} />
          <ThemedText
            testID="nlos-evidence-state"
            preset="labelLg"
            accessibilityLabel={`NLOS evidence state ${evidenceState}`}
            style={[styles.evidenceState, { color: stateAccent }]}
          >
            {evidenceState}
          </ThemedText>
        </View>
        <ThemedText preset="mono" style={styles.streamStatus}>
          {humanize(streamStatus)}
        </ThemedText>
      </View>

      <View style={styles.badgeRow}>
        <ThemedText testID="nlos-provenance-badge" preset="labelMd" style={[styles.badge, { borderColor: accent, color: accent }]}>
          {label}
        </ThemedText>
        <ThemedText
          testID="nlos-freshness-badge"
          preset="labelMd"
          style={[
            styles.badge,
            {
              borderColor: freshness === 'fresh' ? instrumentColors.greenDim : freshness === 'stale' ? instrumentColors.danger : instrumentColors.border,
              color: freshness === 'fresh' ? instrumentColors.green : freshness === 'stale' ? instrumentColors.danger : instrumentColors.textSecondary,
            },
          ]}
        >
          {freshness.toUpperCase()}
        </ThemedText>
      </View>

      {frame ? (
        <View style={styles.grid}>
          <ProvenanceRow label="Evidence" value={humanize(frame.evidenceLevel)} />
          <ProvenanceRow label="Transient" value={humanize(frame.provenance.transientKind)} />
          <ProvenanceRow label="Histograms" value={frame.provenance.histogramPreserved ? 'PRESERVED' : 'NOT PRESENT'} />
          <ProvenanceRow label="Sensor" value={frame.provenance.sensorModel} />
          <ProvenanceRow label="Sequence" value={String(frame.sequence)} />
        </View>
      ) : (
        <ThemedText preset="bodySm" color="textSecondary">
          No validated frame is available. Unknown evidence is never promoted to live.
        </ThemedText>
      )}
    </InstrumentPanel>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: instrumentColors.panelRaised,
  },
  stateRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  stateIdentity: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  stateDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    shadowColor: instrumentColors.cyan,
    shadowOpacity: 0.8,
    shadowRadius: 5,
  },
  evidenceState: {
    fontSize: 15,
    letterSpacing: 1.1,
  },
  streamStatus: {
    color: instrumentColors.textSecondary,
    fontSize: 10,
    letterSpacing: 1,
  },
  badgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  badge: {
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  grid: {
    gap: spacing.xs,
  },
  provenanceRow: {
    minHeight: 28,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    borderBottomColor: instrumentColors.grid,
    borderBottomWidth: 1,
  },
  provenanceLabel: { width: 82 },
  provenanceValue: { flex: 1 },
});
