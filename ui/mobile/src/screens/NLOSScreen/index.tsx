import { useEffect, useMemo, useState } from 'react';
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
import { useLidarStore } from '@/stores/lidarStore';
import { useTabScrollToTop } from '@/stores/tabScrollStore';
import { spacing } from '@/theme/spacing';
import { BetaSetupCard } from './BetaSetupCard';
import { HiddenTargetVisualization, type NlosViewMode } from './HiddenTargetVisualization';
import { LidarCommissioningCard } from './LidarCommissioningCard';
import { PoseTeachingCard } from './PoseTeachingCard';
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

type CalibrationStep = 'connect' | 'room' | 'pose' | 'review';
type StepState = 'complete' | 'current' | 'pending' | 'optional';

const calibrationSteps: ReadonlyArray<{ id: CalibrationStep; number: string; label: string }> = [
  { id: 'connect', number: '01', label: 'CONNECT' },
  { id: 'room', number: '02', label: 'ROOM' },
  { id: 'pose', number: '03', label: 'POSE' },
  { id: 'review', number: '04', label: 'REVIEW' },
];

const GuidedMenu = ({
  active,
  states,
  onChange,
}: {
  active: CalibrationStep;
  states: Record<CalibrationStep, StepState>;
  onChange: (step: CalibrationStep) => void;
}) => (
  <View testID="calibration-guided-menu" style={styles.guideMenuWrap}>
    <View style={styles.guideHeading}>
      <View>
        <ThemedText preset="mono" style={styles.guideEyebrow}>GUIDED CALIBRATION</ThemedText>
        <ThemedText preset="bodySm" style={styles.guideHint}>Complete the active task, then continue.</ThemedText>
      </View>
      <ThemedText preset="mono" style={styles.guideProgress}>{calibrationSteps.findIndex((step) => step.id === active) + 1} / 4</ThemedText>
    </View>
    <View accessibilityRole="tablist" style={styles.guideMenu}>
      {calibrationSteps.map((step) => {
        const selected = active === step.id;
        const state = states[step.id];
        return (
          <Pressable
            key={step.id}
            testID={`calibration-step-${step.id}`}
            accessibilityRole="button"
            accessibilityLabel={`${step.label} calibration step`}
            accessibilityState={{ selected }}
            onPress={() => onChange(step.id)}
            style={[styles.guideStep, selected && styles.guideStepActive]}
          >
            <View style={[styles.guideNumber, state === 'complete' && styles.guideNumberComplete, selected && styles.guideNumberActive]}>
              <ThemedText preset="mono" style={[styles.guideNumberText, (selected || state === 'complete') && styles.guideNumberTextActive]}>{state === 'complete' ? '✓' : step.number}</ThemedText>
            </View>
            <ThemedText preset="mono" style={[styles.guideStepLabel, selected && styles.guideStepLabelActive]}>{step.label}</ThemedText>
            <ThemedText preset="mono" style={[styles.guideState, state === 'complete' && styles.guideStateComplete]}>{state.toUpperCase()}</ThemedText>
          </Pressable>
        );
      })}
    </View>
  </View>
);

const StepIntro = ({ number, title, copy, requirement }: { number: string; title: string; copy: string; requirement: string }) => (
  <View testID="calibration-step-content" style={styles.stepIntro}>
    <ThemedText preset="mono" style={styles.stepIntroNumber}>STEP {number} / 04</ThemedText>
    <ThemedText preset="displayMd" style={styles.stepIntroTitle}>{title}</ThemedText>
    <ThemedText preset="bodyMd" style={styles.stepIntroCopy}>{copy}</ThemedText>
    <View style={styles.requirementRow}><View style={styles.requirementDot} /><ThemedText preset="mono" style={styles.requirementText}>{requirement}</ThemedText></View>
  </View>
);

const GuideAction = ({ label, onPress, secondary = false }: { label: string; onPress: () => void; secondary?: boolean }) => (
  <Pressable accessibilityRole="button" accessibilityLabel={label} onPress={onPress} style={[styles.guideAction, secondary && styles.guideActionSecondary]}>
    <ThemedText preset="labelMd" style={[styles.guideActionText, secondary && styles.guideActionTextSecondary]}>{label}</ThemedText>
  </Pressable>
);

export const NLOSScreen = () => {
  const scrollRef = useTabScrollToTop('Calibration');
  const lidarFrame = useLidarStore((state) => state.frame);
  const calibration = useLidarStore((state) => state.calibration);
  const poseCalibration = useLidarStore((state) => state.poseCalibration);
  const markCalibrationStale = useLidarStore((state) => state.markCalibrationStale);
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
  const [activeStep, setActiveStep] = useState<CalibrationStep>(() => {
    if (calibration?.quality !== 'VALID' || calibration.staleness.state !== 'CURRENT') return 'connect';
    return poseCalibration?.quality === 'VALID' ? 'review' : 'pose';
  });
  const [helpOpen, setHelpOpen] = useState(false);
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
  const spatialReady = calibration?.quality === 'VALID' && calibration.staleness.state === 'CURRENT';
  const poseReady = poseCalibration?.quality === 'VALID';
  const sourceReady = streamStatus === 'live';
  const reviewReady = evidenceState === 'LIVE VERIFIED';
  const stepStates: Record<CalibrationStep, StepState> = {
    connect: sourceReady ? 'complete' : activeStep === 'connect' ? 'current' : 'pending',
    room: spatialReady ? 'complete' : activeStep === 'room' ? 'current' : 'pending',
    pose: poseReady ? 'complete' : activeStep === 'pose' ? 'current' : 'optional',
    review: reviewReady ? 'complete' : activeStep === 'review' ? 'current' : 'pending',
  };

  useEffect(() => {
    if (!calibration || calibration.staleness.state === 'STALE' || frame?.source !== 'live') return;
    if ((frame.evidenceLevel === 'l2_calibrated' || frame.evidenceLevel === 'l3_corroborated')
      && frame.calibrationHash !== calibration.digestSha256) {
      markCalibrationStale('The live RuView stream reports a different calibration hash. Re-synchronize or run a short rescan.');
    }
  }, [calibration, frame, markCalibrationStale]);

  const handleConfigureCredential = () => {
    const configured = configureCredential(credentialDraft);
    setCredentialError(!configured);
    if (configured) setCredentialDraft('');
  };

  return (
    <ThemedView style={styles.container}>
      <InstrumentGrid />
      <ScrollView
        ref={scrollRef}
        testID="nlos-scroll-view"
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
        stickyHeaderIndices={[2]}
        contentContainerStyle={[
          styles.content,
          {
            paddingTop: spacing.lg,
            paddingRight: spacing.lg + safeAreaInsets.right,
            paddingBottom: 72 + safeAreaInsets.bottom,
            paddingLeft: spacing.lg + safeAreaInsets.left,
          },
        ]}
      >
        <InstrumentPanel eyebrow="RuView installation / calibration" style={styles.hero}>
          <ThemedText preset="displayLg" style={styles.heroTitle}>Calibrate the room.</ThemedText>
          <ThemedText preset="displayLg" style={styles.heroAccent}>Then validate sensing.</ThemedText>
          <ThemedText preset="bodyLg" style={styles.heroCopy}>
            Scan visible geometry, align RuView nodes, fit room and pose corrections, and verify improvement on held-out paths before promoting calibration for Live use.
          </ThemedText>
          <View style={styles.scopeRow}>
            <ScopeChip label="INSTALLATION TOOL" />
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
              This client does not access raw iPhone LiDAR timing data. ARKit supplies visible-room depth and geometry for calibration, but cannot see through walls. Hidden-space tracks require authenticated RuView CSI evidence.
            </ThemedText>
          </View>
        </View>

        <GuidedMenu active={activeStep} states={stepStates} onChange={setActiveStep} />

        {activeStep === 'connect' && (
          <>
            <StepIntro number="01" title="Connect a verified source" copy="Use authenticated live RuView evidence for real calibration. Replay is available only to learn the interface and remains visibly synthetic." requirement="LIVE SOURCE REQUIRED BEFORE REFERENCE WALKS" />
            <View style={styles.actions}>
              <Pressable testID="nlos-start-synthetic" accessibilityRole="button" accessibilityLabel="USE CALIBRATION REPLAY" onPress={startReplay} style={({ pressed }) => [styles.secondaryButton, pressed && styles.buttonPressed]}>
                <View style={styles.buttonLabelRow}><View style={styles.syntheticButtonDot} /><ThemedText preset="labelMd" style={styles.secondaryButtonText}>USE CALIBRATION REPLAY</ThemedText></View>
                <ThemedText preset="bodySm" style={styles.buttonCaption}>Practice only · deterministic and watermarked</ThemedText>
              </Pressable>
              <Pressable testID="nlos-connect-live" accessibilityRole="button" accessibilityLabel="CONNECT LIVE CALIBRATION" disabled={!liveCredentialAvailable} onPress={connectLive} style={({ pressed }) => [styles.liveButton, !liveCredentialAvailable && styles.disabledButton, pressed && liveCredentialAvailable && styles.buttonPressed]}>
                <ThemedText preset="labelMd" style={styles.liveButtonText}>CONNECT LIVE CALIBRATION</ThemedText>
                <ThemedText preset="bodySm" style={styles.liveButtonCaption}>{liveCredentialAvailable ? 'Credential ready for this session' : 'Ephemeral credential required'}</ThemedText>
              </Pressable>
            </View>
            {!liveCredentialAvailable ? (
              <InstrumentPanel eyebrow="Secure calibration pairing" style={styles.credentialCard}>
                <ThemedText preset="bodyMd" style={styles.credentialIntro}>Enter a coordinator supplied credential to unlock the authenticated stream for this session.</ThemedText>
                <TextInput testID="nlos-credential-input" accessibilityLabel="Ephemeral calibration bearer credential" value={credentialDraft} onChangeText={(value) => { setCredentialDraft(value); setCredentialError(false); }} secureTextEntry autoCapitalize="none" autoCorrect={false} autoComplete="off" textContentType="oneTimeCode" maxLength={512} placeholder="32 to 512 character pairing credential" placeholderTextColor={instrumentColors.textSecondary} style={[styles.credentialInput, credentialError && styles.credentialInputError]} />
                <Pressable testID="nlos-unlock-live" accessibilityRole="button" accessibilityLabel="UNLOCK AUTHENTICATED LIVE" disabled={!credentialLengthValid} onPress={handleConfigureCredential} style={({ pressed }) => [styles.credentialButton, !credentialLengthValid && styles.disabledButton, pressed && credentialLengthValid && styles.buttonPressed]}>
                  <ThemedText preset="labelMd" style={styles.credentialButtonText}>UNLOCK AUTHENTICATED LIVE</ThemedText>
                </Pressable>
                <ThemedText preset="bodySm" style={styles.credentialNote}>Held in memory only, sent solely in the ticket request Authorization header, and never stored by this client.</ThemedText>
              </InstrumentPanel>
            ) : (
              <View style={styles.credentialReadyRow}><View style={styles.readyIdentity}><View style={styles.readyDot} /><ThemedText preset="bodySm" style={styles.readyText}>EPHEMERAL CREDENTIAL READY</ThemedText></View><Pressable accessibilityRole="button" accessibilityLabel="Forget ephemeral credential" onPress={forgetCredential} style={styles.forgetButton}><ThemedText preset="labelMd" style={styles.forgetText}>FORGET</ThemedText></Pressable></View>
            )}
            {lastRejectedReason && <View style={styles.rejectionCard}><ThemedText testID="nlos-rejection" preset="bodySm" style={styles.rejectionText}>Rejected {rejectedFrameCount} frame{rejectedFrameCount === 1 ? '' : 's'}; latest reason: {lastRejectedReason}</ThemedText></View>}
          </>
        )}

        {activeStep === 'room' && (
          <>
            <StepIntro number="02" title="Measure and align the room" copy="Scan visible geometry, mark at least three RuView nodes, then record separate baseline and held-out walks in one coordinate frame." requirement={sourceReady ? 'LIVE RF READY · KEEP ROOM SCAN ACTIVE' : 'RETURN TO STEP 01 AND CONNECT LIVE RF'} />
            <LidarCommissioningCard />
          </>
        )}

        {activeStep === 'pose' && (
          <>
            <StepIntro number="03" title="Teach coarse pose — optional" copy="Temporarily pair visible iPhone joints with synchronized CSI to fit a room-specific student. Skip this step when spatial localization is the only goal." requirement={spatialReady ? 'VALID ROOM CALIBRATION FOUND' : 'VALID ROOM CALIBRATION REQUIRED TO RECORD'} />
            <PoseTeachingCard />
          </>
        )}

        {activeStep === 'review' && (
          <>
            <StepIntro number="04" title="Review evidence before promotion" copy="Confirm provenance, freshness, calibration identity, visible tracks, and confidence. Missing or stale evidence stays hidden." requirement="PROMOTE ONLY VERIFIED, FRESH, HELD-OUT RESULTS" />
            <ProvenancePanel frame={frame} freshness={freshness} streamStatus={streamStatus} />
            <InstrumentPanel eyebrow="Calibration validation / spatial return" accessory={<ThemedText preset="mono" style={styles.panelIndex}>FRAME / 01</ThemedText>} style={styles.visualizationCard}>
              <ViewModePicker value={viewMode} onChange={setViewMode} />
              <View style={styles.visualizationStage}>
                <HiddenTargetVisualization tracks={visibleTracks} freshness={freshness} mode={viewMode} width={visualizationWidth} lidarFrame={lidarFrame} />
                {isSynthetic && <View testID="nlos-synthetic-watermark" pointerEvents="none" style={styles.watermark}><ThemedText preset="displayMd" style={styles.watermarkText}>SYNTHETIC</ThemedText></View>}
                {freshness === 'stale' && <View testID="nlos-stale-overlay" pointerEvents="none" style={styles.staleOverlay}><ThemedText preset="labelLg" style={styles.staleText}>STALE FRAME</ThemedText><ThemedText preset="bodySm" style={styles.staleCaption}>Targets hidden until fresh evidence arrives</ThemedText></View>}
              </View>
              <View style={styles.summaryRow}><View style={styles.metric}><ThemedText testID="nlos-track-count" preset="displayMd" style={styles.metricValue}>{visibleTracks.length}</ThemedText><ThemedText preset="mono" style={styles.metricLabel}>VALIDATION TRACKS</ThemedText></View><View style={styles.metricDivider} /><View style={styles.metric}><ThemedText testID="nlos-mean-confidence" preset="displayMd" style={styles.metricValue}>{meanConfidence}</ThemedText><ThemedText preset="mono" style={styles.metricLabel}>MEAN CONF.</ThemedText></View></View>
            </InstrumentPanel>
          </>
        )}

        <View style={styles.guideActions}>
          {activeStep !== 'connect' && <GuideAction label="BACK" secondary onPress={() => setActiveStep(activeStep === 'room' ? 'connect' : activeStep === 'pose' ? 'room' : 'pose')} />}
          {activeStep !== 'review' && <GuideAction label={activeStep === 'connect' ? 'CONTINUE TO ROOM' : activeStep === 'room' ? 'CONTINUE TO OPTIONAL POSE' : 'REVIEW RESULTS'} onPress={() => setActiveStep(activeStep === 'connect' ? 'room' : activeStep === 'room' ? 'pose' : 'review')} />}
        </View>

        <Pressable accessibilityRole="button" accessibilityLabel={helpOpen ? 'Hide setup and safety help' : 'Show setup and safety help'} onPress={() => setHelpOpen((value) => !value)} style={styles.helpToggle}>
          <View><ThemedText preset="labelMd" style={styles.helpTitle}>SETUP, SAFETY & TEST HELP</ThemedText><ThemedText preset="bodySm" style={styles.helpCopy}>Device requirements, beta links, evidence labels, and privacy defaults</ThemedText></View>
          <ThemedText preset="mono" style={styles.helpIcon}>{helpOpen ? '−' : '+'}</ThemedText>
        </Pressable>
        {helpOpen && <><BetaSetupCard /><View testID="nlos-privacy-legend" style={styles.privacyLegend}><ThemedText preset="mono" style={styles.privacyLabel}>PRIVACY DEFAULTS</ThemedText><ThemedText preset="bodySm" style={styles.privacyCopy}>Viewer retention: raw RF off, audio off, pairing credential memory only. Connected servers require their own consent and retention controls.</ThemedText></View></>}
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
  guideMenuWrap: {
    backgroundColor: instrumentColors.background,
    borderColor: instrumentColors.border,
    borderWidth: 1,
    borderRadius: 14,
    padding: spacing.sm,
    shadowColor: '#000',
    shadowOpacity: 0.35,
    shadowRadius: 12,
    elevation: 8,
  },
  guideHeading: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.xs, paddingBottom: spacing.sm },
  guideEyebrow: { color: instrumentColors.cyan, fontSize: 9, letterSpacing: 1.2 },
  guideHint: { color: instrumentColors.textSecondary, marginTop: 2 },
  guideProgress: { color: instrumentColors.green, fontSize: 10 },
  guideMenu: { flexDirection: 'row', gap: 4 },
  guideStep: { flex: 1, minHeight: 66, alignItems: 'center', justifyContent: 'center', borderRadius: 9, gap: 3, paddingHorizontal: 2 },
  guideStepActive: { backgroundColor: 'rgba(36, 211, 229, 0.09)', borderColor: instrumentColors.cyanDim, borderWidth: 1 },
  guideNumber: { width: 24, height: 24, borderRadius: 12, borderWidth: 1, borderColor: instrumentColors.borderStrong, alignItems: 'center', justifyContent: 'center' },
  guideNumberActive: { borderColor: instrumentColors.cyan, backgroundColor: 'rgba(36, 211, 229, 0.12)' },
  guideNumberComplete: { borderColor: instrumentColors.greenDim, backgroundColor: 'rgba(43, 217, 119, 0.12)' },
  guideNumberText: { color: instrumentColors.textSecondary, fontSize: 7 },
  guideNumberTextActive: { color: instrumentColors.green },
  guideStepLabel: { color: instrumentColors.textSecondary, fontSize: 7, letterSpacing: 0.4 },
  guideStepLabelActive: { color: instrumentColors.cyan },
  guideState: { color: instrumentColors.textSecondary, fontSize: 5.5 },
  guideStateComplete: { color: instrumentColors.green },
  stepIntro: { borderLeftWidth: 2, borderLeftColor: instrumentColors.cyan, paddingVertical: spacing.sm, paddingLeft: spacing.md, paddingRight: spacing.xs, gap: spacing.xs },
  stepIntroNumber: { color: instrumentColors.green, fontSize: 8, letterSpacing: 1.1 },
  stepIntroTitle: { color: instrumentColors.text, fontSize: 24, lineHeight: 29 },
  stepIntroCopy: { color: instrumentColors.textSecondary, lineHeight: 19 },
  requirementRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginTop: spacing.xs },
  requirementDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: instrumentColors.warning },
  requirementText: { flex: 1, color: instrumentColors.warning, fontSize: 7.5 },
  guideActions: { flexDirection: 'row', gap: spacing.sm },
  guideAction: { flex: 1, minHeight: 48, alignItems: 'center', justifyContent: 'center', borderRadius: 10, backgroundColor: instrumentColors.cyan, paddingHorizontal: spacing.sm },
  guideActionSecondary: { backgroundColor: 'transparent', borderColor: instrumentColors.borderStrong, borderWidth: 1 },
  guideActionText: { color: instrumentColors.background, textAlign: 'center', fontSize: 9 },
  guideActionTextSecondary: { color: instrumentColors.textSecondary },
  helpToggle: { minHeight: 66, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: spacing.md, borderColor: instrumentColors.border, borderWidth: 1, borderRadius: 12, paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  helpTitle: { color: instrumentColors.text, fontSize: 10 },
  helpCopy: { color: instrumentColors.textSecondary, marginTop: 3, maxWidth: 330 },
  helpIcon: { color: instrumentColors.cyan, fontSize: 18 },
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
    fontFamily: 'JetBrainsMono_400Regular',
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
