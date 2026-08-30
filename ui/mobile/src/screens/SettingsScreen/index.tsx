import { useEffect, useMemo, useState } from 'react';
import { Ionicons } from '@expo/vector-icons';
import { Alert, Linking, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { InstrumentGrid, InstrumentPanel, instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import { WS_PATH } from '@/constants/websocket';
import { apiService } from '@/services/api.service';
import { wsService } from '@/services/ws.service';
import { useMatStore } from '@/stores/matStore';
import { useNlosStore } from '@/stores/nlosStore';
import { usePoseStore } from '@/stores/poseStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useTabScrollToTop } from '@/stores/tabScrollStore';
import { ThemePicker } from './ThemePicker';
import { ServerUrlInput } from './ServerUrlInput';
import { NlosServerUrlInput } from './NlosServerUrlInput';

const StatusTile = ({ icon, label, value, live }: { icon: keyof typeof Ionicons.glyphMap; label: string; value: string; live: boolean }) => (
  <View style={[styles.statusTile, live && styles.statusTileLive]}><View style={styles.statusIcon}><Ionicons name={icon} size={17} color={live ? instrumentColors.green : instrumentColors.textSecondary} /></View><View style={styles.statusCopy}><ThemedText preset="mono" style={styles.statusLabel}>{label}</ThemedText><ThemedText preset="labelMd" style={[styles.statusValue, live && styles.statusValueLive]} numberOfLines={1}>{value}</ThemedText></View><View style={[styles.statusDot, live && styles.statusDotLive]} /></View>
);

const Palette = () => <View style={styles.palette}>{[instrumentColors.background, instrumentColors.panel, instrumentColors.cyan, instrumentColors.green, instrumentColors.warning, instrumentColors.danger].map((color) => <View key={color} style={[styles.swatch, { backgroundColor: color }]} />)}</View>;

export const SettingsScreen = () => {
  const scrollRef = useTabScrollToTop('Settings');
  const serverUrl = useSettingsStore((state) => state.serverUrl);
  const nlosServerUrl = useSettingsStore((state) => state.nlosServerUrl);
  const theme = useSettingsStore((state) => state.theme);
  const setServerUrl = useSettingsStore((state) => state.setServerUrl);
  const setNlosServerUrl = useSettingsStore((state) => state.setNlosServerUrl);
  const setTheme = useSettingsStore((state) => state.setTheme);
  const sensingStatus = usePoseStore((state) => state.connectionStatus);
  const calibrationStatus = useNlosStore((state) => state.streamStatus);
  const matStatus = useMatStore((state) => state.apiStatus);
  const graphStatus = useMatStore((state) => state.worldGraphStatus);
  const [draftUrl, setDraftUrl] = useState(serverUrl);
  const [draftNlosUrl, setDraftNlosUrl] = useState(nlosServerUrl);

  useEffect(() => { setDraftUrl(serverUrl); }, [serverUrl]);
  useEffect(() => { setDraftNlosUrl(nlosServerUrl); }, [nlosServerUrl]);

  const liveSources = useMemo(() => [sensingStatus === 'connected', calibrationStatus === 'live', matStatus === 'live', graphStatus === 'live'].filter(Boolean).length, [calibrationStatus, graphStatus, matStatus, sensingStatus]);
  const handleSaveUrl = () => {
    const newUrl = draftUrl.trim().replace(/\/+$/, '');
    setDraftUrl(newUrl); setServerUrl(newUrl); wsService.disconnect(); wsService.connect(newUrl); apiService.setBaseUrl(newUrl);
  };
  const handleSaveNlosUrl = () => setNlosServerUrl(draftNlosUrl.trim());
  const handleOpenGitHub = async () => {
    const handled = await Linking.canOpenURL('https://github.com/ruvnet/RuView');
    if (!handled) { Alert.alert('Unable to open link', 'Please open https://github.com/ruvnet/RuView manually.'); return; }
    await Linking.openURL('https://github.com/ruvnet/RuView');
  };

  return (
    <View testID="settings-screen" style={styles.root}>
      <InstrumentGrid />
      <ScrollView ref={scrollRef} testID="settings-scroll-view" keyboardShouldPersistTaps="handled" contentContainerStyle={styles.content}>
        <InstrumentPanel testID="settings-hero" eyebrow="INSTRUMENT CONTROL / LOCAL-FIRST" accessory={<ThemedText preset="mono" style={styles.heroAccessory}>{liveSources} / 4 LIVE</ThemedText>}>
          <ThemedText preset="displayLg" style={styles.heroTitle}>Tune the system.<ThemedText preset="displayLg" style={styles.heroAccent}> Keep every source honest.</ThemedText></ThemedText>
          <ThemedText preset="bodyMd" style={styles.heroBody}>Configure where RuView gets evidence, verify each authority independently, and keep transport security appropriate to the installation.</ThemedText>
          <View style={styles.statusGrid}>
            <StatusTile icon="wifi" label="SENSING" value={sensingStatus.toUpperCase()} live={sensingStatus === 'connected'} />
            <StatusTile icon="scan" label="CALIBRATION" value={calibrationStatus.replace(/_/g, ' ').toUpperCase()} live={calibrationStatus === 'live'} />
            <StatusTile icon="shield-checkmark" label="MAT API" value={matStatus.toUpperCase()} live={matStatus === 'live'} />
            <StatusTile icon="git-network" label="WORLDGRAPH" value={graphStatus.toUpperCase()} live={graphStatus === 'live'} />
          </View>
        </InstrumentPanel>

        <InstrumentPanel eyebrow="SERVER" accessory={<ThemedText preset="mono" style={styles.panelAccessory}>PRIMARY AUTHORITY</ThemedText>}><ServerUrlInput value={draftUrl} onChange={setDraftUrl} onSave={handleSaveUrl} /></InstrumentPanel>
        <InstrumentPanel eyebrow="CALIBRATION SERVER" accessory={<ThemedText preset="mono" style={styles.panelAccessory}>EPHEMERAL ACCESS</ThemedText>}><NlosServerUrlInput value={draftNlosUrl} onChange={setDraftNlosUrl} onSave={handleSaveNlosUrl} /></InstrumentPanel>

        <InstrumentPanel eyebrow="APPEARANCE" accessory={<ThemedText preset="mono" style={styles.panelAccessory}>{theme.toUpperCase()} MODE</ThemedText>}>
          <View style={styles.appearanceHeading}><View><ThemedText preset="labelLg" style={styles.cardTitle}>Instrument lighting</ThemedText><ThemedText preset="bodySm" style={styles.cardCopy}>Choose the interface response while preserving evidence and alert colors.</ThemedText></View><Palette /></View>
          <ThemePicker value={theme} onChange={setTheme} />
        </InstrumentPanel>

        <InstrumentPanel eyebrow="ABOUT" accessory={<ThemedText preset="mono" style={styles.panelAccessory}>BUILD 01</ThemedText>}>
          <View style={styles.aboutBrand}><View style={styles.aboutMark}><Ionicons name="radio" size={25} color={instrumentColors.green} /></View><View style={styles.aboutCopy}><ThemedText preset="displayMd" style={styles.aboutTitle}>RuView Mobile</ThemedText><ThemedText preset="mono" style={styles.version}>WiFi-DensePose Mobile v1.0.0</ThemedText></View></View>
          <View style={styles.factGrid}><View style={styles.fact}><ThemedText preset="mono" style={styles.factLabel}>SOCKET</ThemedText><ThemedText preset="bodySm" style={styles.factValue}>{WS_PATH}</ThemedText></View><View style={styles.fact}><ThemedText preset="mono" style={styles.factLabel}>TRIAGE</ThemedText><ThemedText preset="bodySm" style={styles.factValue}>START / 5 CLASS</ThemedText></View><View style={styles.fact}><ThemedText preset="mono" style={styles.factLabel}>POLICY</ThemedText><ThemedText preset="bodySm" style={styles.factValue}>FAIL CLOSED</ThemedText></View></View>
          <Pressable accessibilityRole="link" accessibilityLabel="View RuView on GitHub" onPress={() => void handleOpenGitHub()} style={({ pressed }) => [styles.githubButton, pressed && styles.pressed]}><Ionicons name="logo-github" size={18} color={instrumentColors.cyan} /><ThemedText preset="labelMd" style={styles.githubText}>VIEW RUVIEW ON GITHUB</ThemedText><Ionicons name="open-outline" size={15} color={instrumentColors.textSecondary} /></Pressable>
        </InstrumentPanel>
        <ThemedText preset="mono" style={styles.footer}>CONNECTION SETTINGS CHANGE AUTHORITIES · THEY NEVER CREATE EVIDENCE</ThemedText>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: instrumentColors.background }, content: { padding: 16, paddingBottom: 32, gap: 14 }, heroAccessory: { color: instrumentColors.green, fontSize: 8 }, heroTitle: { color: instrumentColors.text, fontSize: 31, lineHeight: 35 }, heroAccent: { color: instrumentColors.cyan, fontSize: 31, lineHeight: 35 }, heroBody: { color: instrumentColors.textSecondary, lineHeight: 20, maxWidth: 620 },
  statusGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 }, statusTile: { width: '48%', flexGrow: 1, minHeight: 58, borderWidth: 1, borderColor: instrumentColors.border, borderRadius: 11, backgroundColor: '#0B0E13', padding: 9, flexDirection: 'row', alignItems: 'center', gap: 8 }, statusTileLive: { borderColor: `${instrumentColors.green}55`, backgroundColor: 'rgba(38,217,104,.04)' }, statusIcon: { width: 27, alignItems: 'center' }, statusCopy: { flex: 1 }, statusLabel: { color: instrumentColors.textSecondary, fontSize: 7 }, statusValue: { color: instrumentColors.textSecondary, fontSize: 11 }, statusValueLive: { color: instrumentColors.green }, statusDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: instrumentColors.textSecondary }, statusDotLive: { backgroundColor: instrumentColors.green, shadowColor: instrumentColors.green, shadowOpacity: .8, shadowRadius: 5 },
  panelAccessory: { color: instrumentColors.textSecondary, fontSize: 7, letterSpacing: .6 }, appearanceHeading: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 10 }, cardTitle: { color: instrumentColors.text }, cardCopy: { color: instrumentColors.textSecondary, maxWidth: 240, marginTop: 3 }, palette: { flexDirection: 'row' }, swatch: { width: 13, height: 31, borderWidth: StyleSheet.hairlineWidth, borderColor: 'rgba(255,255,255,.16)' },
  aboutBrand: { flexDirection: 'row', alignItems: 'center', gap: 12 }, aboutMark: { width: 50, height: 50, borderRadius: 16, borderWidth: 1, borderColor: `${instrumentColors.green}66`, backgroundColor: 'rgba(38,217,104,.07)', alignItems: 'center', justifyContent: 'center' }, aboutCopy: { flex: 1 }, aboutTitle: { color: instrumentColors.text }, version: { color: instrumentColors.textSecondary, fontSize: 8 }, factGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 }, fact: { flexGrow: 1, minWidth: '29%', padding: 9, borderRadius: 9, backgroundColor: '#0B0E13', borderLeftWidth: 1, borderLeftColor: instrumentColors.borderStrong }, factLabel: { color: instrumentColors.textSecondary, fontSize: 7 }, factValue: { color: instrumentColors.text, marginTop: 2 }, githubButton: { minHeight: 45, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, borderWidth: 1, borderColor: instrumentColors.borderStrong, borderRadius: 10, backgroundColor: 'rgba(25,212,230,.05)' }, githubText: { flex: 1, color: instrumentColors.cyan, textAlign: 'center' }, pressed: { opacity: .64 }, footer: { color: instrumentColors.textSecondary, fontSize: 7, letterSpacing: .7, textAlign: 'center', paddingVertical: 7 },
});

export default SettingsScreen;
