import { StyleSheet, View } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { colors } from '@/theme/colors';
import { spacing } from '@/theme/spacing';
import type { NlosFreshness, NlosStreamStatus, NlosTrackFrame } from '@/types/nlos';

interface ProvenancePanelProps {
  frame: NlosTrackFrame | null;
  freshness: NlosFreshness;
  streamStatus: NlosStreamStatus;
}

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
  return colors.success;
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

  return (
    <View style={styles.card}>
      <View style={styles.badgeRow}>
        <ThemedText testID="nlos-provenance-badge" preset="labelMd" style={[styles.badge, { borderColor: accent, color: accent }]}>
          {label}
        </ThemedText>
        <ThemedText testID="nlos-freshness-badge" preset="labelMd" style={{ color: freshness === 'fresh' ? colors.success : freshness === 'stale' ? colors.danger : colors.muted }}>
          {freshness.toUpperCase()}
        </ThemedText>
        <ThemedText preset="bodySm" color="textSecondary">
          {humanize(streamStatus)}
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
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 12,
    padding: spacing.md,
    gap: spacing.md,
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
  provenanceRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  provenanceLabel: { width: 82 },
  provenanceValue: { flex: 1 },
});
