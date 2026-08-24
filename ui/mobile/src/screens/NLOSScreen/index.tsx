import { useMemo, useState } from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  useWindowDimensions,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  InstrumentGrid,
  InstrumentPanel,
  instrumentColors,
} from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useNlosStream } from '@/hooks/useNlosStream';
import { spacing } from '@/theme/spacing';
import { BetaSetupCard } from './BetaSetupCard';
import { HiddenTargetVisualization, type NlosViewMode } from './HiddenTargetVisualization';
import { ProvenancePanel, resolveNlosEvidenceState } from './ProvenancePanel';

const ViewModePicker = ({
  value,
  onChange,
}: {
  value: NlosViewMode;
  onChange: (value: NlosViewMode) => void;
}) => (
  <View accessibilityRole="tablist" style={styles.picker}>
    {([
      ['plan', '2D', 'PLAN'],
      ['perspective', '3D', 'SCENE'],
      ['cloud', 'LIDAR', 'CLOUD'],
    ] as const).map(([option, eyebrow, label]) => {
      const selected = option === value;
      return (
        <Pressable
          key={option}
          testID={`nlos-view-${option}`}
          accessibilityRole="button"
          accessibilityState={{ selected }}
          onPress={() => onChange(option)}
          style={[styles.pickerButton, selected && styles.pickerButtonSelected]}
        >
          <ThemedText
            preset="mono"
            style={[
              styles.pickerEyebrow,
              { color: selected ? instrumentColors.green : instrumentColors.textSecondary },
            ]}
          >
            {eyebrow}
          </ThemedText>
          <ThemedText
            preset="mono"
            style={[
              styles.pickerLabel,
              { color: selected ? instrumentColors.cyan : instrumentColors.textSecondary },
            ]}
          >
            {label}
          </ThemedText>
        </Pressable>
      );
    })}
  </View>
);

const ScopeChip = ({ label, accent = false }: { label: string; accent?: boolean }) => (
  <View style={[styles.scopeChip, accent && styles.scopeChipAccent]}>
    <View style={[styles.scopeDot, accent && styles.scopeDotAccent]} />
    <ThemedText preset="mono" style={[styles.scopeLabel, accent && styles.scopeLabelAccent]}>
      {label}
    </ThemedText>
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
  const safeAreaInsets = useSafeAreaInsets();
  const visualizationWidth = useMemo(
    () => Math.max(
      260,
      Math.min(width - safeAreaInsets.left - safeAreaInsets.right - spacing.xxxl - 4, 520),
    ),
    [safeAreaInsets.left, safeAreaInsets.right, width],
  );
  const evidenceState = resolveNlosEvidenceState(frame, freshness, streamStatus);
  const isSynthetic = frame?.source === 'synthetic';
  const geometryDisplayable = evidenceState === 'SYNTHETIC' || evidenceState === 'LIVE VERIFIED';
  const visibleTracks = useMemo(
    () => geometryDisplayable
      ? frame?.tracks.filter((track) => track.state !== 'unknown') ?? []
      : [],
    [frame, geometryDisplayable],
  );
  const meanConfidence = visibleTracks.length
    ? `${Math.round(visibleTracks.reduce((sum, track) => sum + track.confidence, 0) / visibleTracks.length * 100)}%`
    : 'N/A';
  const credentialLengthValid = credentialDraft.length >= 32 && credentialDraft.length <= 512;

  const handleConfigureCredential = () => {
    const configured = configureCredential(credentialDraft);
    setCredentialError(!configured);
    if (configured) setCredentialDraft('');
  };

  return (
    <ThemedView style={styles.container}>
      <InstrumentGrid />
      <ScrollView
        testID="nlos-scroll-view"
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
        contentContainerStyle={[
          styles.content,
          {
            paddingTop: spacing.xxl + safeAreaInsets.top,
            paddingRight: spacing.lg + safeAreaInsets.right,
            paddingBottom: 72 + safeAreaInsets.bottom,
            paddingLeft: spacing.lg + safeAreaInsets.left,
          },
        ]}
      >
        <View style={styles.brandBar}>
          <View style={styles.brandIdentity}>
            <View style={styles.brandMark}>
              <View style={styles.brandMarkCore} />
            </View>
            <View>
              <ThemedText preset="labelLg" style={styles.brandName}>RuView NLOS</ThemedText>
              <ThemedText preset="mono" style={styles.brandCaption}>MOBILE INSTRUMENT / 01</ThemedText>
            </View>
          </View>
          <ThemedText preset="mono" style={styles.labsBadge}>LABS</ThemedText>
        </View>

        <InstrumentPanel eyebrow="Consumer NLOS / field viewer" style={styles.hero}>
          <ThemedText preset="displayLg" style={styles.heroTitle}>Track hidden space</ThemedText>
          <ThemedText preset="displayLg" style={styles.heroAccent}>hypotheses.</ThemedText>
          <ThemedText preset="bodyLg" style={styles.heroCopy}>
            Inspect validated reconstruction frames with explicit source, freshness, and confidence. No camera equivalence is implied.
          </ThemedText>
          <View style={styles.scopeRow}>
            <ScopeChip label="VIEWER ONLY" />
            <ScopeChip label="FAIL CLOSED" accent />
          </View>
        </InstrumentPanel>

        <View testID="nlos-capability-boundary" style={styles.notice}>
          <View style={styles.noticeIcon}>
            <ThemedText preset="mono" style={styles.noticeIconText}>!</ThemedText>
          </View>
          <View style={styles.noticeCopy}>
            <ThemedText preset="mono" style={styles.noticeLabel}>SENSOR BOUNDARY</ThemedText>
            <ThemedText preset="bodySm" style={styles.noticeText}>
              This client does not access raw iPhone LiDAR timing data. Safari and Expo display authenticated RuView track frames or visibly watermarked synthetic replay only.
            </ThemedText>
          </View>
        </View>

        <ProvenancePanel frame={frame} freshness={freshness} streamStatus={streamStatus} />

        <InstrumentPanel
          eyebrow="Spatial return"
          accessory={<ThemedText preset="mono" style={styles.panelIndex}>FRAME / 01</ThemedText>}
          style={styles.visualizationCard}
        >
          <ViewModePicker value={viewMode} onChange={setViewMode} />
          <View style={styles.visualizationStage}>
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
                <ThemedText preset="labelLg" style={styles.staleText}>STALE FRAME</ThemedText>
                <ThemedText preset="bodySm" style={styles.staleCaption}>Targets hidden until fresh evidence arrives</ThemedText>
              </View>
            )}
          </View>

          <View style={styles.summaryRow}>
            <View style={styles.metric}>
              <ThemedText testID="nlos-track-count" preset="displayMd" style={styles.metricValue}>
                {visibleTracks.length}
              </ThemedText>
              <ThemedText preset="mono" style={styles.metricLabel}>GATED TRACKS</ThemedText>
            </View>
            <View style={styles.metricDivider} />
            <View style={styles.metric}>
              <ThemedText testID="nlos-mean-confidence" preset="displayMd" style={styles.metricValue}>
                {meanConfidence}
              </ThemedText>
              <ThemedText preset="mono" style={styles.metricLabel}>MEAN CONF.</ThemedText>
            </View>
          </View>
        </InstrumentPanel>

        <View style={styles.actions}>
          <Pressable
            testID="nlos-start-synthetic"
            accessibilityRole="button"
            accessibilityLabel="USE SYNTHETIC REPLAY"
            onPress={startReplay}
            style={({ pressed }) => [styles.secondaryButton, pressed && styles.buttonPressed]}
          >
            <View style={styles.buttonLabelRow}>
              <View style={styles.syntheticButtonDot} />
              <ThemedText preset="labelMd" style={styles.secondaryButtonText}>USE SYNTHETIC REPLAY</ThemedText>
            </View>
            <ThemedText preset="bodySm" style={styles.buttonCaption}>Deterministic and watermarked</ThemedText>
          </Pressable>
          <Pressable
            testID="nlos-connect-live"
            accessibilityRole="button"
            accessibilityLabel="CONNECT AUTHENTICATED LIVE"
            disabled={!liveCredentialAvailable}
            onPress={connectLive}
            style={({ pressed }) => [
              styles.liveButton,
              !liveCredentialAvailable && styles.disabledButton,
              pressed && liveCredentialAvailable && styles.buttonPressed,
            ]}
          >
            <ThemedText preset="labelMd" style={styles.liveButtonText}>CONNECT AUTHENTICATED LIVE</ThemedText>
            <ThemedText preset="bodySm" style={styles.liveButtonCaption}>Ephemeral credential required</ThemedText>
          </Pressable>
        </View>

        {!liveCredentialAvailable ? (
          <InstrumentPanel eyebrow="Secure live pairing" style={styles.credentialCard}>
            <ThemedText preset="bodyMd" style={styles.credentialIntro}>
              Enter a coordinator supplied credential to unlock the authenticated stream for this session.
            </ThemedText>
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
              placeholderTextColor={instrumentColors.textSecondary}
              style={[styles.credentialInput, credentialError && styles.credentialInputError]}
            />
            <Pressable
              testID="nlos-unlock-live"
              accessibilityRole="button"
              accessibilityLabel="UNLOCK AUTHENTICATED LIVE"
              disabled={!credentialLengthValid}
              onPress={handleConfigureCredential}
              style={({ pressed }) => [
                styles.credentialButton,
                !credentialLengthValid && styles.disabledButton,
                pressed && credentialLengthValid && styles.buttonPressed,
              ]}
            >
              <ThemedText preset="labelMd" style={styles.credentialButtonText}>UNLOCK AUTHENTICATED LIVE</ThemedText>
            </Pressable>
            <ThemedText preset="bodySm" style={styles.credentialNote}>
              A native host or signed in web session may supply this credential automatically. It is held in memory only, sent solely in the ticket request Authorization header, and never stored by this client.
            </ThemedText>
          </InstrumentPanel>
        ) : (
          <View style={styles.credentialReadyRow}>
            <View style={styles.readyIdentity}>
              <View style={styles.readyDot} />
              <ThemedText preset="bodySm" style={styles.readyText}>EPHEMERAL CREDENTIAL READY</ThemedText>
            </View>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Forget ephemeral credential"
              onPress={forgetCredential}
              style={styles.forgetButton}
            >
              <ThemedText preset="labelMd" style={styles.forgetText}>FORGET</ThemedText>
            </Pressable>
          </View>
        )}

        {lastRejectedReason && (
          <View style={styles.rejectionCard}>
            <ThemedText testID="nlos-rejection" preset="bodySm" style={styles.rejectionText}>
              Rejected {rejectedFrameCount} frame{rejectedFrameCount === 1 ? '' : 's'}; latest reason: {lastRejectedReason}
            </ThemedText>
          </View>
        )}

        <BetaSetupCard />

        <View testID="nlos-privacy-legend" style={styles.privacyLegend}>
          <ThemedText preset="mono" style={styles.privacyLabel}>PRIVACY DEFAULTS</ThemedText>
          <ThemedText preset="bodySm" style={styles.privacyCopy}>
            Viewer retention: raw RF off, audio off, pairing credential memory only. Connected servers require their own consent and retention controls.
          </ThemedText>
        </View>
      </ScrollView>
    </ThemedView>
  );
};

export default NLOSScreen;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: instrumentColors.background },
  content: {
    width: '100%',
    maxWidth: 620,
    alignSelf: 'center',
    gap: spacing.md,
  },
  brandBar: {
    minHeight: 48,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.md,
  },
  brandIdentity: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  brandMark: {
    width: 38,
    height: 38,
    borderRadius: 19,
    borderColor: instrumentColors.cyanDim,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandMarkCore: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: instrumentColors.green,
    shadowColor: instrumentColors.green,
    shadowOpacity: 0.75,
    shadowRadius: 8,
  },
  brandName: { color: instrumentColors.text, fontSize: 14, letterSpacing: 0.8 },
  brandCaption: { color: instrumentColors.textSecondary, fontSize: 9, letterSpacing: 1.1 },
  labsBadge: {
    color: instrumentColors.cyan,
    borderColor: instrumentColors.cyanDim,
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: 10,
    letterSpacing: 1.3,
  },
  hero: { paddingTop: spacing.xl, paddingBottom: spacing.xl },
  heroTitle: {
    color: instrumentColors.text,
    fontSize: 32,
    lineHeight: 36,
    letterSpacing: -0.8,
  },
  heroAccent: {
    color: instrumentColors.cyan,
    fontSize: 32,
    lineHeight: 36,
    letterSpacing: -0.8,
    textShadowColor: 'rgba(36, 211, 229, 0.3)',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 14,
  },
  heroCopy: {
    maxWidth: 470,
    color: instrumentColors.textSecondary,
    lineHeight: 23,
  },
  scopeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, marginTop: spacing.xs },
  scopeChip: {
    minHeight: 32,
    flexDirection: 'row',
    alignItems: 'center',
    borderColor: instrumentColors.border,
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: spacing.md,
    gap: spacing.sm,
  },
  scopeChipAccent: { borderColor: instrumentColors.greenDim },
  scopeDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: instrumentColors.textSecondary },
  scopeDotAccent: { backgroundColor: instrumentColors.green },
  scopeLabel: { color: instrumentColors.textSecondary, fontSize: 9, letterSpacing: 1 },
  scopeLabelAccent: { color: instrumentColors.green },
  notice: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.md,
    backgroundColor: 'rgba(255, 182, 92, 0.055)',
    borderColor: 'rgba(255, 182, 92, 0.2)',
    borderWidth: 1,
    borderRadius: 12,
    padding: spacing.md,
  },
  noticeIcon: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(255, 182, 92, 0.12)',
    borderColor: 'rgba(255, 182, 92, 0.34)',
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  noticeIconText: { color: instrumentColors.warning, fontSize: 13 },
  noticeCopy: { flex: 1, gap: spacing.xs },
  noticeLabel: { color: instrumentColors.warning, fontSize: 9, letterSpacing: 1.25 },
  noticeText: { color: instrumentColors.textSecondary, lineHeight: 18 },
  panelIndex: { color: instrumentColors.textSecondary, fontSize: 9, letterSpacing: 1 },
  visualizationCard: { paddingHorizontal: 0, paddingBottom: 0 },
  picker: {
    flexDirection: 'row',
    marginHorizontal: spacing.md,
    padding: 3,
    gap: 3,
    backgroundColor: 'rgba(5, 9, 13, 0.7)',
    borderColor: instrumentColors.border,
    borderWidth: 1,
    borderRadius: 12,
  },
  pickerButton: {
    minHeight: 44,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
    borderRadius: 9,
  },
  pickerButtonSelected: {
    backgroundColor: 'rgba(36, 211, 229, 0.1)',
    borderColor: instrumentColors.cyanDim,
    borderWidth: 1,
  },
  pickerEyebrow: { fontSize: 7, letterSpacing: 0.8, textAlign: 'center' },
  pickerLabel: { fontSize: 10, letterSpacing: 0.7, textAlign: 'center' },
  visualizationStage: { position: 'relative', overflow: 'hidden', paddingTop: spacing.xs },
  watermark: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    transform: [{ rotate: '-18deg' }],
  },
  watermarkText: {
    color: 'rgba(255, 182, 92, 0.19)',
    fontSize: 26,
    letterSpacing: 6,
  },
  staleOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: instrumentColors.dimOverlay,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
  },
  staleText: { color: instrumentColors.danger },
  staleCaption: { color: instrumentColors.textSecondary },
  summaryRow: {
    minHeight: 86,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(5, 9, 13, 0.58)',
    borderTopColor: instrumentColors.border,
    borderTopWidth: 1,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  metric: { flex: 1, gap: spacing.xs },
  metricValue: { color: instrumentColors.text, fontSize: 26 },
  metricLabel: { color: instrumentColors.textSecondary, fontSize: 9, letterSpacing: 1.1 },
  metricDivider: { width: 1, height: 42, backgroundColor: instrumentColors.border, marginHorizontal: spacing.md },
  actions: { gap: spacing.sm },
  secondaryButton: {
    minHeight: 62,
    justifyContent: 'center',
    borderColor: 'rgba(255, 182, 92, 0.36)',
    borderWidth: 1,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 182, 92, 0.065)',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    gap: spacing.xs,
  },
  buttonLabelRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  syntheticButtonDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: instrumentColors.warning },
  secondaryButtonText: { color: instrumentColors.warning },
  buttonCaption: { color: instrumentColors.textSecondary },
  liveButton: {
    minHeight: 62,
    justifyContent: 'center',
    borderRadius: 12,
    backgroundColor: instrumentColors.cyan,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    gap: spacing.xs,
  },
  liveButtonText: { color: instrumentColors.background },
  liveButtonCaption: { color: 'rgba(5, 9, 13, 0.68)' },
  disabledButton: { opacity: 0.36 },
  buttonPressed: { opacity: 0.72, transform: [{ scale: 0.992 }] },
  credentialCard: { backgroundColor: instrumentColors.panelRaised },
  credentialIntro: { color: instrumentColors.textSecondary, lineHeight: 20 },
  credentialInput: {
    minHeight: 50,
    borderColor: instrumentColors.borderStrong,
    borderWidth: 1,
    borderRadius: 10,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    color: instrumentColors.text,
    backgroundColor: instrumentColors.background,
    fontFamily: 'Courier New',
    fontSize: 14,
  },
  credentialInputError: { borderColor: instrumentColors.danger },
  credentialButton: {
    minHeight: 48,
    alignItems: 'center',
    justifyContent: 'center',
    borderColor: instrumentColors.cyanDim,
    borderWidth: 1,
    borderRadius: 10,
    paddingHorizontal: spacing.md,
  },
  credentialButtonText: { color: instrumentColors.cyan, textAlign: 'center' },
  credentialNote: { color: instrumentColors.textSecondary, lineHeight: 18 },
  credentialReadyRow: {
    minHeight: 60,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: instrumentColors.panelRaised,
    borderColor: instrumentColors.greenDim,
    borderWidth: 1,
    borderRadius: 12,
    paddingLeft: spacing.lg,
    gap: spacing.md,
  },
  readyIdentity: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  readyDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: instrumentColors.green },
  readyText: { color: instrumentColors.green },
  forgetButton: { minWidth: 72, minHeight: 44, alignItems: 'center', justifyContent: 'center' },
  forgetText: { color: instrumentColors.textSecondary },
  rejectionCard: {
    backgroundColor: 'rgba(255, 100, 120, 0.07)',
    borderColor: 'rgba(255, 100, 120, 0.25)',
    borderWidth: 1,
    borderRadius: 12,
    padding: spacing.md,
  },
  rejectionText: { color: instrumentColors.danger, lineHeight: 18 },
  privacyLegend: {
    borderTopColor: instrumentColors.border,
    borderTopWidth: 1,
    paddingTop: spacing.lg,
    gap: spacing.xs,
  },
  privacyLabel: { color: instrumentColors.green, fontSize: 10, letterSpacing: 1.2 },
  privacyCopy: { color: instrumentColors.textSecondary, lineHeight: 18 },
});
