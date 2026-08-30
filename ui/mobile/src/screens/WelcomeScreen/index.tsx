import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import { Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { InstrumentGrid, InstrumentPanel, instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import type { MainTabsParamList } from '@/navigation/types';
import { useMatStore } from '@/stores/matStore';
import { useNlosStore } from '@/stores/nlosStore';
import { usePoseStore } from '@/stores/poseStore';
import { useTabScrollStore, useTabScrollToTop } from '@/stores/tabScrollStore';

type Destination = Exclude<keyof MainTabsParamList, 'Welcome'>;

const destinations: Array<{ route: Destination; icon: keyof typeof Ionicons.glyphMap; step: string; title: string; copy: string; accent: string }> = [
  { route: 'Calibration', icon: 'scan', step: '01', title: 'Calibrate an installation', copy: 'Scan the room, align RuView nodes, validate walking paths, and optionally teach coarse pose.', accent: instrumentColors.cyan },
  { route: 'Live', icon: 'wifi', step: '02', title: 'Open sensor fusion', copy: 'Inspect live RuView RF, LiDAR-aligned geometry, pose evidence, confidence, and provenance.', accent: instrumentColors.green },
  { route: 'Vitals', icon: 'heart', step: '03', title: 'Review vital signals', copy: 'See only measured breathing and heart evidence, with Apple Home controls gated by real availability.', accent: '#FF6478' },
  { route: 'Zones', icon: 'grid', step: '04', title: 'Inspect rooms and zones', copy: 'View occupancy localization and configured spatial boundaries without invented detections.', accent: '#B98CFF' },
  { route: 'MAT', icon: 'shield-checkmark', step: '05', title: 'Enter incident mode', copy: 'Combine governed WorldGraph topology with live MAT events, survivors, zones, and alerts.', accent: instrumentColors.warning },
  { route: 'Settings', icon: 'settings', step: '06', title: 'Configure connections', copy: 'Set verified sensing and calibration endpoints, test reachability, and adjust appearance.', accent: instrumentColors.textSecondary },
];

const StatePill = ({ label, live }: { label: string; live: boolean }) => (
  <View style={[styles.statePill, live && styles.statePillLive]}><View style={[styles.stateDot, live && styles.stateDotLive]} /><ThemedText preset="mono" style={[styles.stateText, live && styles.stateTextLive]}>{label}</ThemedText></View>
);

export const WelcomeScreen = () => {
  const navigation = useNavigation<BottomTabNavigationProp<MainTabsParamList>>();
  const scrollRef = useTabScrollToTop('Welcome');
  const connectionStatus = usePoseStore((state) => state.connectionStatus);
  const calibrationStatus = useNlosStore((state) => state.streamStatus);
  const matStatus = useMatStore((state) => state.apiStatus);
  const open = (route: Destination) => {
    useTabScrollStore.getState().requestTop(route);
    navigation.navigate(route);
  };

  return (
    <View style={styles.root}>
      <InstrumentGrid />
      <ScrollView ref={scrollRef} testID="welcome-screen" contentContainerStyle={styles.content}>
        <InstrumentPanel testID="welcome-hero" eyebrow="RUVIEW MOBILE / INSTALLATION INSTRUMENT">
          <ThemedText preset="displayLg" style={styles.heroTitle}>Understand the room.<ThemedText preset="displayLg" style={styles.heroAccent}> Trust the evidence.</ThemedText></ThemedText>
          <ThemedText preset="bodyMd" style={styles.heroCopy}>Commission RuView, verify live sensing, and move from room geometry to operational decisions through one evidence-governed workspace.</ThemedText>
          <View style={styles.stateRow}>
            <StatePill label={`RF ${connectionStatus.toUpperCase()}`} live={connectionStatus === 'connected'} />
            <StatePill label={`CAL ${calibrationStatus.replace(/_/g, ' ').toUpperCase()}`} live={calibrationStatus === 'live'} />
            <StatePill label={`MAT ${matStatus.toUpperCase()}`} live={matStatus === 'live'} />
          </View>
          <Pressable accessibilityRole="button" accessibilityLabel="Start calibration" testID="welcome-primary-action" onPress={() => open('Calibration')} style={({ pressed }) => [styles.primaryAction, pressed && styles.pressed]}>
            <View><ThemedText preset="mono" style={styles.actionEyebrow}>RECOMMENDED FIRST STEP</ThemedText><ThemedText preset="labelLg" style={styles.primaryText}>START GUIDED CALIBRATION</ThemedText></View><Ionicons name="arrow-forward" size={20} color="#071015" />
          </Pressable>
        </InstrumentPanel>

        <View style={styles.sectionHeading}><View><ThemedText preset="mono" style={styles.sectionEyebrow}>CHOOSE A WORKSPACE</ThemedText><ThemedText preset="displayMd" style={styles.sectionTitle}>What do you want to do?</ThemedText></View><ThemedText preset="mono" style={styles.sectionCount}>06 TOOLS</ThemedText></View>

        {destinations.map((item) => <Pressable key={item.route} accessibilityRole="button" accessibilityLabel={`Open ${item.route}`} testID={`welcome-open-${item.route.toLowerCase()}`} onPress={() => open(item.route)} style={({ pressed }) => [styles.destination, pressed && styles.pressed]}>
          <View style={[styles.iconShell, { borderColor: `${item.accent}66`, backgroundColor: `${item.accent}10` }]}><Ionicons name={item.icon} size={23} color={item.accent} /></View>
          <View style={styles.destinationCopy}><ThemedText preset="mono" style={[styles.step, { color: item.accent }]}>{item.step} / {item.route.toUpperCase()}</ThemedText><ThemedText preset="labelLg" style={styles.destinationTitle}>{item.title}</ThemedText><ThemedText preset="bodySm" style={styles.destinationBody}>{item.copy}</ThemedText></View>
          <Ionicons name="chevron-forward" size={18} color={instrumentColors.textSecondary} />
        </Pressable>)}

        <InstrumentPanel eyebrow="EVIDENCE BOUNDARY" style={styles.boundary}><ThemedText preset="bodySm" style={styles.boundaryText}>RuView does not manufacture people, pose, vital signs, incident records, or hidden-space claims. Unavailable sources remain unavailable, synthetic practice stays visibly labeled, and live promotion requires verified provenance.</ThemedText></InstrumentPanel>
        <ThemedText preset="mono" style={styles.footer}>RUVIEW / REAL INPUTS · EXPLICIT UNCERTAINTY · FAIL-CLOSED OUTPUTS</ThemedText>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: instrumentColors.background }, content: { padding: 16, paddingBottom: 32, gap: 14 },
  heroTitle: { color: instrumentColors.text, fontSize: 34, lineHeight: 38 }, heroAccent: { color: instrumentColors.cyan, fontSize: 34, lineHeight: 38 }, heroCopy: { color: instrumentColors.textSecondary, lineHeight: 21, maxWidth: 620 },
  stateRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 7 }, statePill: { flexDirection: 'row', alignItems: 'center', gap: 6, borderWidth: 1, borderColor: instrumentColors.border, borderRadius: 999, paddingHorizontal: 9, paddingVertical: 6, backgroundColor: '#0B0E13' }, statePillLive: { borderColor: `${instrumentColors.green}66` }, stateDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: instrumentColors.textSecondary }, stateDotLive: { backgroundColor: instrumentColors.green }, stateText: { color: instrumentColors.textSecondary, fontSize: 8 }, stateTextLive: { color: instrumentColors.green },
  primaryAction: { minHeight: 58, borderRadius: 13, backgroundColor: instrumentColors.green, paddingHorizontal: 16, paddingVertical: 11, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 }, actionEyebrow: { color: 'rgba(7,16,21,.66)', fontSize: 7, letterSpacing: .8 }, primaryText: { color: '#071015', letterSpacing: .6 }, pressed: { opacity: .64 },
  sectionHeading: { paddingHorizontal: 2, paddingVertical: 5, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', gap: 10 }, sectionEyebrow: { color: instrumentColors.green, fontSize: 8, letterSpacing: 1 }, sectionTitle: { color: instrumentColors.text, fontSize: 23 }, sectionCount: { color: instrumentColors.textSecondary, fontSize: 8 },
  destination: { minHeight: 108, flexDirection: 'row', alignItems: 'center', gap: 13, padding: 14, borderRadius: 15, borderWidth: 1, borderColor: instrumentColors.border, backgroundColor: 'rgba(20,24,31,.96)' }, iconShell: { width: 48, height: 48, borderRadius: 15, borderWidth: 1, alignItems: 'center', justifyContent: 'center' }, destinationCopy: { flex: 1, gap: 3 }, step: { fontSize: 8, letterSpacing: .8 }, destinationTitle: { color: instrumentColors.text, fontSize: 16 }, destinationBody: { color: instrumentColors.textSecondary, lineHeight: 17 },
  boundary: { borderColor: 'rgba(255,182,92,.34)' }, boundaryText: { color: instrumentColors.textSecondary, lineHeight: 18 }, footer: { color: instrumentColors.textSecondary, textAlign: 'center', fontSize: 7, letterSpacing: .7, paddingVertical: 7 },
});

export default WelcomeScreen;
