import { Linking, Platform, Pressable, StyleSheet, View } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { colors } from '@/theme/colors';
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
        'Launch RuView, open NLOS, and allow only the permissions requested for the assigned test.',
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

const LinkButton = ({ label, url, primary = false }: { label: string; url: string; primary?: boolean }) => (
  <Pressable
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
    <View testID="nlos-beta-setup" style={styles.card} accessibilityLabel="RuView NLOS beta setup">
      <View style={styles.headingRow}>
        <ThemedText preset="labelLg">BETA SETUP</ThemedText>
        <ThemedText preset="labelMd" style={styles.platformBadge}>{guidance.label}</ThemedText>
      </View>

      <ThemedText preset="bodyLg" style={styles.title}>Start a governed test in about five minutes</ThemedText>

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
        <ThemedText preset="bodyMd">
          The web client cannot capture ARKit LiDAR or raw timing data. It only displays synthetic replay or validated tracks produced by a RuView server.
        </ThemedText>
      </View>

      <ThemedText preset="bodyMd" color="textSecondary">
        Compatibility: any supported device can view tracks. A LiDAR equipped iPhone Pro or iPad Pro is needed only for separately assigned hardware capability checks.
      </ThemedText>
      <ThemedText preset="bodyMd" color="textSecondary">
        Evidence labels: L0 synthetic, L1 measured, L2 calibrated, or L3 corroborated, plus fresh, stale, or unknown. Depth only input is never physical NLOS evidence.
      </ThemedText>

      <View style={styles.links}>
        {guidance.showTestFlightButton && (
          <LinkButton label="INSTALL TESTFLIGHT" url={TESTFLIGHT_APP_URL} primary />
        )}
        <LinkButton label="OPEN EXPLAINER" url={NLOS_EXPLAINER_URL} primary={!guidance.showTestFlightButton} />
        <LinkButton label="TEST STEPS AND FEEDBACK" url={NLOS_FEEDBACK_URL} />
      </View>

      <ThemedText preset="bodySm" color="textSecondary">
        No credentials are saved by setup. Live pairing credentials remain in memory only and can be forgotten at any time.
      </ThemedText>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderColor: colors.accentDim,
    borderWidth: 1,
    borderRadius: 12,
    padding: spacing.lg,
    gap: spacing.md,
  },
  headingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  platformBadge: {
    color: colors.accent,
    borderColor: colors.accentDim,
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  title: { lineHeight: 23 },
  steps: { gap: spacing.sm },
  stepRow: { flexDirection: 'row', alignItems: 'flex-start', gap: spacing.sm },
  stepNumber: {
    color: colors.bg,
    backgroundColor: colors.accent,
    borderRadius: 999,
    width: 24,
    height: 24,
    lineHeight: 24,
    textAlign: 'center',
  },
  stepText: { flex: 1, lineHeight: 21 },
  boundary: {
    backgroundColor: 'rgba(255, 165, 2, 0.08)',
    borderLeftColor: colors.warn,
    borderLeftWidth: 3,
    padding: spacing.md,
    gap: spacing.xs,
  },
  boundaryLabel: { color: colors.warn },
  links: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  linkButton: {
    minHeight: 44,
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    borderColor: colors.accent,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  linkButtonPrimary: { backgroundColor: colors.accent },
  linkButtonText: { color: colors.accent, textAlign: 'center' },
  linkButtonPrimaryText: { color: colors.bg, textAlign: 'center' },
});
