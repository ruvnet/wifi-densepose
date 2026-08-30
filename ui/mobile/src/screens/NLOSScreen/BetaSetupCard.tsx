import { Linking, Platform, Pressable, StyleSheet, View } from 'react-native';
import { InstrumentPanel, instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import { spacing } from '@/theme/spacing';

export const NLOS_EXPLAINER_URL = 'https://ruview-nlos.ruv.chatgpt.site';
export const NLOS_FEEDBACK_URL = 'https://github.com/ruvnet/RuView/issues/1690';
export const TESTFLIGHT_APP_URL = 'https://apps.apple.com/app/testflight/id899247664';

export type BetaPlatform = 'ios' | 'web' | 'other';

export interface BetaPlatformGuidance {
  label: string;
  steps: readonly string[];
  showTestFlightButton: boolean;
}

export const getBetaPlatformGuidance = (platform: BetaPlatform): BetaPlatformGuidance => {
  if (platform === 'ios') {
    return {
      label: 'NATIVE IOS BETA',
      steps: [
        'Install TestFlight, then open the private RuView invitation supplied by the beta coordinator.',
        'Launch RuView, open Calibration, and allow only the permissions requested for the assigned test.',
        'Run synthetic replay first. Connect live only with a coordinator supplied ephemeral credential.',
      ],
      showTestFlightButton: true,
    };
  }

  if (platform === 'web') {
    return {
      label: 'IPHONE WEB BETA',
      steps: [
        'Open this web build in Safari on your iPhone.',
        'Tap Share, choose Add to Home Screen, then open the installed RuView icon.',
        'Run synthetic replay or connect to an authenticated RuView reconstruction server.',
      ],
      showTestFlightButton: false,
    };
  }

  return {
    label: 'BETA VIEWER',
    steps: [
      'Use the web build for synthetic replay and authenticated RuView track viewing.',
      'Use TestFlight on a supported iPhone or iPad for the native iOS beta.',
      'Report the platform, app version, evidence label, and observed result in the test issue.',
    ],
    showTestFlightButton: false,
  };
};

const currentBetaPlatform = (): BetaPlatform => {
  if (Platform.OS === 'ios') return 'ios';
  if (Platform.OS === 'web') return 'web';
  return 'other';
};

const openTrustedUrl = async (url: string): Promise<void> => {
  try {
    await Linking.openURL(url);
  } catch {
    // The platform owns any launch error UI. No link or credential is persisted.
  }
};

const LinkButton = ({
  label,
  url,
  primary = false,
  testID,
}: {
  label: string;
  url: string;
  primary?: boolean;
  testID?: string;
}) => (
  <Pressable
    testID={testID}
    accessibilityRole="link"
    accessibilityLabel={label}
    accessibilityHint="Opens in your browser"
    onPress={() => { void openTrustedUrl(url); }}
    style={[styles.linkButton, primary && styles.linkButtonPrimary]}
  >
    <ThemedText preset="labelMd" style={primary ? styles.linkButtonPrimaryText : styles.linkButtonText}>
      {label}
    </ThemedText>
  </Pressable>
);

export const BetaSetupCard = () => {
  const guidance = getBetaPlatformGuidance(currentBetaPlatform());

  return (
    <InstrumentPanel
      testID="nlos-beta-setup"
      eyebrow="Governed beta protocol"
      style={styles.card}
      accessibilityLabel="RuView calibration beta setup"
    >
      <View style={styles.headingRow}>
        <ThemedText preset="labelLg" style={styles.sectionLabel}>BETA SETUP</ThemedText>
        <ThemedText preset="labelMd" style={styles.platformBadge}>{guidance.label}</ThemedText>
      </View>

      <ThemedText preset="displayMd" style={styles.title}>Start a governed calibration test</ThemedText>

      <View style={styles.steps}>
        {guidance.steps.map((step, index) => (
          <View key={step} style={styles.stepRow}>
            <ThemedText preset="labelMd" style={styles.stepNumber}>{index + 1}</ThemedText>
            <ThemedText preset="bodyMd" style={styles.stepText}>{step}</ThemedText>
          </View>
        ))}
      </View>

      <View style={styles.boundary}>
        <ThemedText preset="labelMd" style={styles.boundaryLabel}>CAPABILITY BOUNDARY</ThemedText>
        <ThemedText preset="bodyMd" style={styles.boundaryCopy}>
          The web client cannot capture ARKit LiDAR or raw timing data. It only displays synthetic replay or validated tracks produced by a RuView server.
        </ThemedText>
      </View>

      <View style={styles.compatibilityGrid}>
        <View style={styles.compatibilityCell}>
          <ThemedText preset="mono" style={styles.cellLabel}>DEVICE</ThemedText>
          <ThemedText preset="bodySm" style={styles.cellCopy}>
            Compatibility: any supported device can view tracks. A LiDAR equipped iPhone Pro or iPad Pro is needed only for separately assigned hardware capability checks.
          </ThemedText>
        </View>
        <View style={styles.compatibilityCell}>
          <ThemedText preset="mono" style={styles.cellLabel}>EVIDENCE</ThemedText>
          <ThemedText preset="bodySm" style={styles.cellCopy}>
            Evidence labels: L0 synthetic, L1 measured, L2 calibrated, or L3 corroborated, plus fresh, stale, or unknown. Depth-only input is never physical through-wall evidence.
          </ThemedText>
        </View>
      </View>

      <View style={styles.links}>
        {guidance.showTestFlightButton && (
          <LinkButton label="INSTALL TESTFLIGHT" url={TESTFLIGHT_APP_URL} primary />
        )}
        <LinkButton
          testID="nlos-explainer-link"
          label="OPEN EXPLAINER"
          url={NLOS_EXPLAINER_URL}
          primary={!guidance.showTestFlightButton}
        />
        <LinkButton
          testID="nlos-feedback-link"
          label="TEST STEPS AND FEEDBACK"
          url={NLOS_FEEDBACK_URL}
        />
      </View>

      <ThemedText preset="bodySm" style={styles.retentionNote}>
        No credentials are saved by setup. Live pairing credentials remain in memory only and can be forgotten at any time.
      </ThemedText>
    </InstrumentPanel>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: instrumentColors.panelRaised,
  },
  headingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  sectionLabel: { color: instrumentColors.text },
  platformBadge: {
    color: instrumentColors.cyan,
    borderColor: instrumentColors.cyanDim,
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  title: {
    maxWidth: 300,
    color: instrumentColors.text,
    fontSize: 25,
    lineHeight: 30,
    letterSpacing: -0.45,
  },
  steps: { gap: spacing.sm },
  stepRow: {
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.md,
    borderBottomColor: instrumentColors.grid,
    borderBottomWidth: 1,
    paddingBottom: spacing.sm,
  },
  stepNumber: {
    color: instrumentColors.background,
    backgroundColor: instrumentColors.cyan,
    borderRadius: 999,
    width: 28,
    height: 28,
    lineHeight: 28,
    textAlign: 'center',
  },
  stepText: { flex: 1, lineHeight: 21, color: instrumentColors.text },
  boundary: {
    backgroundColor: 'rgba(255, 182, 92, 0.07)',
    borderColor: 'rgba(255, 182, 92, 0.24)',
    borderWidth: 1,
    borderLeftColor: instrumentColors.warning,
    borderLeftWidth: 3,
    borderRadius: 10,
    padding: spacing.md,
    gap: spacing.xs,
  },
  boundaryLabel: { color: instrumentColors.warning },
  boundaryCopy: { color: instrumentColors.text },
  compatibilityGrid: { gap: spacing.sm },
  compatibilityCell: {
    backgroundColor: 'rgba(5, 9, 13, 0.38)',
    borderColor: instrumentColors.border,
    borderWidth: 1,
    borderRadius: 10,
    padding: spacing.md,
    gap: spacing.xs,
  },
  cellLabel: {
    color: instrumentColors.green,
    fontSize: 10,
    letterSpacing: 1.2,
  },
  cellCopy: { color: instrumentColors.textSecondary, lineHeight: 18 },
  links: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  linkButton: {
    minHeight: 48,
    flexGrow: 1,
    flexBasis: 150,
    alignItems: 'center',
    justifyContent: 'center',
    borderColor: instrumentColors.cyanDim,
    borderWidth: 1,
    borderRadius: 10,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  linkButtonPrimary: { backgroundColor: instrumentColors.cyan },
  linkButtonText: { color: instrumentColors.cyan, textAlign: 'center' },
  linkButtonPrimaryText: { color: instrumentColors.background, textAlign: 'center' },
  retentionNote: { color: instrumentColors.textSecondary, lineHeight: 18 },
});
