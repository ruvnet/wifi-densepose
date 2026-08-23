import { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, TextInput, useWindowDimensions, View } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useNlosStream } from '@/hooks/useNlosStream';
import { colors } from '@/theme/colors';
import { spacing } from '@/theme/spacing';
import { HiddenTargetVisualization, type NlosViewMode } from './HiddenTargetVisualization';
import { ProvenancePanel } from './ProvenancePanel';

const ViewModePicker = ({ value, onChange }: { value: NlosViewMode; onChange: (value: NlosViewMode) => void }) => (
  <View style={styles.picker}>
    {(['plan', 'perspective'] as const).map((option) => {
      const selected = option === value;
      return (
        <Pressable
          key={option}
          accessibilityRole="button"
          accessibilityState={{ selected }}
          onPress={() => onChange(option)}
          style={[styles.pickerButton, selected && styles.pickerButtonSelected]}
        >
          <ThemedText preset="labelMd" style={{ color: selected ? colors.accent : colors.textSecondary }}>
            {option === 'plan' ? '2D PLAN' : '3D VIEW'}
          </ThemedText>
        </Pressable>
      );
    })}
  </View>
);

export const NLOSScreen = () => {
  const {
    frame,
    freshness,
    streamStatus,
    lastRejectedReason,
    rejectedFrameCount,
    liveCredentialAvailable,
    configureCredential,
    forgetCredential,
    startReplay,
    connectLive,
  } = useNlosStream();
  const [viewMode, setViewMode] = useState<NlosViewMode>('plan');
  const [credentialDraft, setCredentialDraft] = useState('');
  const [credentialError, setCredentialError] = useState(false);
  const { width } = useWindowDimensions();
  const visualizationWidth = useMemo(() => width - spacing.md * 2, [width]);
  const isSynthetic = frame?.source === 'synthetic';
  const visibleTracks = useMemo(
    () => freshness === 'fresh'
      ? frame?.tracks.filter((track) => track.state !== 'unknown') ?? []
      : [],
    [frame, freshness],
  );
  const credentialLengthValid = credentialDraft.length >= 32 && credentialDraft.length <= 512;

  const handleConfigureCredential = () => {
    const configured = configureCredential(credentialDraft);
    setCredentialError(!configured);
    if (configured) setCredentialDraft('');
  };

  return (
    <ThemedView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <View style={{ flex: 1 }}>
            <ThemedText preset="displayMd">RuView NLOS</ThemedText>
            <ThemedText preset="bodySm" color="textSecondary">
              Hidden target hypotheses from a RuView reconstruction server
            </ThemedText>
          </View>
          <ThemedText preset="labelMd" style={{ color: colors.accent }}>LABS</ThemedText>
        </View>

        <View style={styles.notice}>
          <ThemedText preset="bodySm" style={{ color: colors.warn }}>
            This client does not access raw iPhone LiDAR timing data. Safari and Expo display authenticated RuView track frames or visibly watermarked synthetic replay only.
          </ThemedText>
        </View>

        <View testID="nlos-maturity-boundary" style={styles.maturityCard}>
          <ThemedText preset="labelMd" style={{ color: colors.accent }}>SOFTWARE PREVIEW</ThemedText>
          <ThemedText preset="bodySm">
            The software path is implemented and locally validated. Research readiness remains blocked on:
          </ThemedText>
          <ThemedText preset="bodySm" color="textSecondary">OPEN · macOS native compilation</ThemedText>
          <ThemedText preset="bodySm" color="textSecondary">OPEN · Physical VL53L8CH reproduction at 27 fps or better</ThemedText>
          <ThemedText preset="bodySm" color="textSecondary">OPEN · Measured CSI fusion improvement of at least 25 percent</ThemedText>
          <ThemedText preset="bodySm" color="textSecondary">
            Builds, simulators, and synthetic replay do not close hardware or measured-fusion evidence gates.
          </ThemedText>
        </View>

        <ProvenancePanel frame={frame} freshness={freshness} streamStatus={streamStatus} />

        <View style={styles.visualizationCard}>
          <ViewModePicker value={viewMode} onChange={setViewMode} />
          <HiddenTargetVisualization
            tracks={visibleTracks}
            freshness={freshness}
            mode={viewMode}
            width={visualizationWidth}
          />
          {isSynthetic && (
            <View testID="nlos-synthetic-watermark" pointerEvents="none" style={styles.watermark}>
              <ThemedText preset="displayMd" style={styles.watermarkText}>SYNTHETIC</ThemedText>
            </View>
          )}
          {freshness === 'stale' && (
            <View testID="nlos-stale-overlay" pointerEvents="none" style={styles.staleOverlay}>
              <ThemedText preset="labelLg" style={{ color: colors.danger }}>STALE FRAME</ThemedText>
            </View>
          )}
        </View>

        <View style={styles.summaryRow}>
          <View style={styles.metric}>
            <ThemedText testID="nlos-track-count" preset="displayMd">{visibleTracks.length}</ThemedText>
            <ThemedText preset="bodySm" color="textSecondary">TRACKS</ThemedText>
          </View>
          <View style={styles.metric}>
            <ThemedText testID="nlos-mean-confidence" preset="displayMd">
              {visibleTracks.length ? `${Math.round(visibleTracks.reduce((sum, track) => sum + track.confidence, 0) / visibleTracks.length * 100)}%` : 'N/A'}
            </ThemedText>
            <ThemedText preset="bodySm" color="textSecondary">MEAN CONFIDENCE</ThemedText>
          </View>
        </View>

        <View style={styles.actions}>
          <Pressable accessibilityRole="button" onPress={startReplay} style={styles.secondaryButton}>
            <ThemedText preset="labelMd">USE SYNTHETIC REPLAY</ThemedText>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            disabled={!liveCredentialAvailable}
            onPress={connectLive}
            style={[styles.liveButton, !liveCredentialAvailable && styles.disabledButton]}
          >
            <ThemedText preset="labelMd" style={{ color: colors.bg }}>CONNECT AUTHENTICATED LIVE</ThemedText>
          </Pressable>
        </View>

        {!liveCredentialAvailable ? (
          <View style={styles.credentialCard}>
            <ThemedText preset="labelMd">EPHEMERAL LIVE CREDENTIAL</ThemedText>
            <TextInput
              testID="nlos-credential-input"
              accessibilityLabel="Ephemeral NLOS Bearer credential"
              value={credentialDraft}
              onChangeText={(value) => {
                setCredentialDraft(value);
                setCredentialError(false);
              }}
              secureTextEntry
              autoCapitalize="none"
              autoCorrect={false}
              autoComplete="off"
              textContentType="oneTimeCode"
              maxLength={512}
              placeholder="32 to 512 character pairing credential"
              placeholderTextColor={colors.textSecondary}
              style={[styles.credentialInput, credentialError && styles.credentialInputError]}
            />
            <Pressable
              accessibilityRole="button"
              disabled={!credentialLengthValid}
              onPress={handleConfigureCredential}
              style={[styles.credentialButton, !credentialLengthValid && styles.disabledButton]}
            >
              <ThemedText preset="labelMd">UNLOCK AUTHENTICATED LIVE</ThemedText>
            </Pressable>
            <ThemedText preset="bodySm" color="textSecondary">
              A native host or signed in web session may supply this credential automatically. It is held in memory only, sent solely in the ticket request Authorization header, and never stored by this client.
            </ThemedText>
          </View>
        ) : (
          <View style={styles.credentialReadyRow}>
            <ThemedText preset="bodySm" style={{ color: colors.success }}>EPHEMERAL CREDENTIAL READY</ThemedText>
            <Pressable accessibilityRole="button" onPress={forgetCredential}>
              <ThemedText preset="labelMd" color="textSecondary">FORGET</ThemedText>
            </Pressable>
          </View>
        )}
        {lastRejectedReason && (
          <ThemedText testID="nlos-rejection" preset="bodySm" style={{ color: colors.danger }}>
            Rejected {rejectedFrameCount} frame{rejectedFrameCount === 1 ? '' : 's'}; latest reason: {lastRejectedReason}
          </ThemedText>
        )}
      </ScrollView>
    </ThemedView>
  );
};

export default NLOSScreen;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing.md, paddingBottom: spacing.xxxl, gap: spacing.md },
  header: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  notice: {
    backgroundColor: 'rgba(255, 165, 2, 0.08)',
    borderColor: 'rgba(255, 165, 2, 0.4)',
    borderWidth: 1,
    borderRadius: 10,
    padding: spacing.md,
  },
  maturityCard: {
    backgroundColor: colors.surface,
    borderColor: colors.accentDim,
    borderWidth: 1,
    borderRadius: 12,
    padding: spacing.md,
    gap: spacing.sm,
  },
  visualizationCard: {
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 12,
    paddingTop: spacing.sm,
  },
  picker: { flexDirection: 'row', paddingHorizontal: spacing.sm, gap: spacing.sm },
  pickerButton: { flex: 1, alignItems: 'center', paddingVertical: spacing.sm, borderBottomWidth: 2, borderBottomColor: colors.border },
  pickerButtonSelected: { borderBottomColor: colors.accent },
  watermark: { ...StyleSheet.absoluteFill, alignItems: 'center', justifyContent: 'center', transform: [{ rotate: '-18deg' }] },
  watermarkText: { color: 'rgba(255, 165, 2, 0.18)', letterSpacing: 5 },
  staleOverlay: { ...StyleSheet.absoluteFill, backgroundColor: 'rgba(10, 14, 26, 0.7)', alignItems: 'center', justifyContent: 'center' },
  summaryRow: { flexDirection: 'row', gap: spacing.md },
  metric: { flex: 1, backgroundColor: colors.surface, borderRadius: 10, padding: spacing.md },
  actions: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  secondaryButton: { flexGrow: 1, alignItems: 'center', borderColor: colors.border, borderWidth: 1, borderRadius: 8, padding: spacing.md },
  liveButton: { flexGrow: 1, alignItems: 'center', backgroundColor: colors.accent, borderRadius: 8, padding: spacing.md },
  disabledButton: { opacity: 0.35 },
  credentialCard: { backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1, borderRadius: 10, padding: spacing.md, gap: spacing.sm },
  credentialInput: { borderColor: colors.border, borderWidth: 1, borderRadius: 8, padding: spacing.md, color: colors.textPrimary, backgroundColor: colors.bg },
  credentialInputError: { borderColor: colors.danger },
  credentialButton: { alignItems: 'center', borderColor: colors.accent, borderWidth: 1, borderRadius: 8, padding: spacing.md },
  credentialReadyRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: colors.surface, borderRadius: 10, padding: spacing.md },
});
