import { useCallback, useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Platform, Pressable, RefreshControl, ScrollView, StyleSheet, TextInput, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { InstrumentGrid, InstrumentPanel, instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import { matService } from '@/services/mat.service';
import { worldGraphService } from '@/services/worldGraph.service';
import { useMatStore } from '@/stores/matStore';
import { useSettingsStore } from '@/stores/settingsStore';
import { useTabScrollToTop } from '@/stores/tabScrollStore';
import type { Alert, MatStreamMessage, TriageStatus } from '@/types/mat';
import { WorldGraphMap } from './WorldGraphMap';

const TRIAGE_COLORS: Record<TriageStatus, string> = {
  Immediate: instrumentColors.danger, Delayed: instrumentColors.warning, Minor: instrumentColors.green,
  Deceased: instrumentColors.textSecondary, Unknown: '#B98CFF',
};
const errorMessage = (error: unknown): string => {
  if (error instanceof Error) return error.message;
  if (error && typeof error === 'object' && 'message' in error) return String(error.message);
  return 'Unable to reach the MAT service';
};
const endpointHint = (serverUrl: string): string | null => {
  try {
    const host = new URL(serverUrl).hostname;
    if (Platform.OS !== 'web' && (host === 'localhost' || host === '127.0.0.1' || host === '::1')) {
      return 'On a physical iPhone, localhost is the phone itself. In Settings, use this Mac’s Wi-Fi address and the configured RuView HTTP port, for example http://192.168.1.20:3000.';
    }
  } catch { return 'Set a valid RuView sensing-server URL in Settings.'; }
  return null;
};

const StatusPill = ({ label, status }: { label: string; status: 'live' | 'loading' | 'idle' | 'error' | 'connecting' }) => {
  const color = status === 'live' ? instrumentColors.green : status === 'error' ? instrumentColors.danger : status === 'loading' || status === 'connecting' ? instrumentColors.warning : instrumentColors.textSecondary;
  return <View style={[styles.statusPill, { borderColor: `${color}66` }]}><View style={[styles.statusDot, { backgroundColor: color }]} /><ThemedText preset="mono" style={[styles.statusText, { color }]}>{label} · {status.toUpperCase()}</ThemedText></View>;
};

const Metric = ({ value, label, color = instrumentColors.text }: { value: string | number; label: string; color?: string }) => (
  <View style={styles.metric}><ThemedText preset="displayLg" style={[styles.metricValue, { color }]}>{value}</ThemedText><ThemedText preset="mono" style={styles.metricLabel}>{label}</ThemedText></View>
);

export const MATScreen = () => {
  const scrollRef = useTabScrollToTop('MAT');
  const serverUrl = useSettingsStore((state) => state.serverUrl);
  const state = useMatStore();
  const { replaceSnapshot, setApiStatus, setWorldGraph, setWorldGraphStatus, upsertSurvivor, removeSurvivor, upsertAlert } = state;
  const [refreshing, setRefreshing] = useState(false);
  const [worldGraphOrigin, setWorldGraphOrigin] = useState(serverUrl);
  const [worldGraphToken, setWorldGraphToken] = useState('');
  const [worldGraphSessionToken, setWorldGraphSessionToken] = useState('');
  const [acknowledging, setAcknowledging] = useState<string | null>(null);

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setApiStatus('loading');
    setRefreshing(!quiet);
    try {
      matService.configure(serverUrl);
      const snapshot = await matService.fetchSnapshot(state.selectedEventId);
      replaceSnapshot(snapshot);
    } catch (error) {
      setApiStatus('error', errorMessage(error));
    } finally { setRefreshing(false); }
  }, [serverUrl, state.selectedEventId, replaceSnapshot, setApiStatus]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { setWorldGraphOrigin(serverUrl); }, [serverUrl]);
  useEffect(() => {
    if (!state.selectedEventId || state.apiStatus !== 'live') return;
    return matService.openStream(serverUrl, state.selectedEventId, (message: MatStreamMessage) => {
      if ('event_id' in message && message.event_id !== state.selectedEventId) return;
      if (message.type === 'survivor_detected' || message.type === 'survivor_updated') upsertSurvivor(message.survivor);
      else if (message.type === 'survivor_lost') removeSurvivor(message.survivor_id);
      else if (message.type === 'alert_created' || message.type === 'alert_updated') upsertAlert(message.alert);
      else if (message.type === 'zone_scan_complete' || message.type === 'event_status_changed') void refresh(true);
      else if (message.type === 'error') setApiStatus('error', message.message);
    }, (message) => setApiStatus('error', message));
  }, [serverUrl, state.selectedEventId, state.apiStatus, upsertSurvivor, removeSurvivor, upsertAlert, setApiStatus, refresh]);
  const refreshWorldGraph = useCallback(async (token = worldGraphSessionToken) => {
    setWorldGraphStatus('connecting');
    try {
      const { graph, epoch, seq } = await worldGraphService.fetchSnapshot(serverUrl, token);
      setWorldGraph(graph, epoch, seq);
      setWorldGraphStatus('live');
      return true;
    } catch (error) {
      setWorldGraphStatus('error', errorMessage(error));
      return false;
    }
  }, [serverUrl, worldGraphSessionToken, setWorldGraph, setWorldGraphStatus]);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      await refreshWorldGraph();
      if (active) timer = setTimeout(() => { void poll(); }, 3000);
    };
    void poll();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
      worldGraphService.disconnect();
    };
  }, [refreshWorldGraph]);

  const connectWorldGraph = async () => {
    const token = worldGraphToken.trim();
    state.setWorldGraphStatus('connecting');
    try {
      const { graph, epoch, seq } = await worldGraphService.fetchSnapshot(worldGraphOrigin, token);
      state.setWorldGraph(graph, epoch, seq);
      state.setWorldGraphStatus('live');
      setWorldGraphSessionToken(token);
      setWorldGraphToken('');
    } catch (error) { state.setWorldGraphStatus('error', errorMessage(error)); }
  };

  const setScan = async () => {
    try {
      state.setApiStatus('loading');
      await matService.setScanning(state.pipeline?.scanning ? 'pause' : 'start');
      await refresh(true);
    } catch (error) { state.setApiStatus('error', errorMessage(error)); }
  };
  const acknowledge = async (alert: Alert) => {
    try { setAcknowledging(alert.id); state.upsertAlert(await matService.acknowledgeAlert(alert.id)); }
    catch (error) { state.setApiStatus('error', errorMessage(error)); }
    finally { setAcknowledging(null); }
  };

  const selectedEvent = state.events.find((event) => event.id === state.selectedEventId) ?? null;
  const triage = useMemo(() => state.survivors.reduce<Record<TriageStatus, number>>((counts, survivor) => {
    counts[survivor.triage_status] += 1; return counts;
  }, { Immediate: 0, Delayed: 0, Minor: 0, Deceased: 0, Unknown: 0 }), [state.survivors]);
  const nodeCount = state.worldGraph?.nodes.length ?? 0;
  const sensorCount = state.worldGraph?.nodes.filter((node) => node.kind === 'sensor').length ?? 0;
  const trackCount = state.worldGraph?.nodes.filter((node) => node.kind === 'person_track').length ?? 0;
  const connectionHint = endpointHint(serverUrl);

  return (
    <View style={styles.root}>
      <InstrumentGrid />
      <ScrollView ref={scrollRef} testID="mat-screen" contentContainerStyle={styles.content} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => void refresh()} tintColor={instrumentColors.cyan} />}>
        <View style={styles.hero}>
          <View style={styles.heroCopy}>
            <ThemedText preset="labelMd" style={styles.kicker}>MISSION-AWARE TRIAGE / VERIFIED INPUTS</ThemedText>
            <ThemedText preset="displayLg" style={styles.heroTitle}>Incident intelligence,<ThemedText preset="displayLg" style={styles.heroAccent}> fused.</ThemedText></ThemedText>
            <ThemedText preset="bodyMd" style={styles.heroBody}>Live MAT detections are placed inside a governed WorldGraph topology. No client-generated people, vitals, classifications, or alerts.</ThemedText>
          </View>
          <View style={styles.statusStack}>
            <StatusPill label="MAT API" status={state.apiStatus} />
            <StatusPill label="WORLDGRAPH" status={state.worldGraphStatus} />
          </View>
        </View>

        {(state.apiError || state.worldGraphError) && <InstrumentPanel testID="mat-error" eyebrow="SOURCE DIAGNOSTIC" style={styles.errorPanel}>
          {state.apiError && <View style={styles.diagnosticRow}><ThemedText preset="mono" style={styles.diagnosticLabel}>MAT API</ThemedText><ThemedText preset="bodyMd" style={styles.errorText}>{state.apiError}</ThemedText></View>}
          {state.worldGraphError && <View style={styles.diagnosticRow}><ThemedText preset="mono" style={styles.diagnosticLabel}>WORLDGRAPH</ThemedText><ThemedText preset="bodyMd" style={styles.errorText}>{state.worldGraphError}</ThemedText></View>}
          {connectionHint && <ThemedText preset="bodySm" style={styles.repairText}>{connectionHint}</ThemedText>}
          <View style={styles.actions}><Pressable accessibilityRole="button" onPress={() => void refresh()} style={styles.secondaryButton}><Ionicons name="refresh" size={16} color={instrumentColors.cyan} /><ThemedText preset="labelMd" style={styles.secondaryButtonText}>RETRY MAT</ThemedText></Pressable><Pressable accessibilityRole="button" onPress={() => void refreshWorldGraph()} style={styles.secondaryButton}><Ionicons name="git-network" size={16} color={instrumentColors.cyan} /><ThemedText preset="labelMd" style={styles.secondaryButtonText}>RETRY GRAPH</ThemedText></Pressable></View>
          <ThemedText preset="mono" style={styles.muted}>Fail-closed: unavailable sources do not create detections.</ThemedText>
        </InstrumentPanel>}

        <View style={styles.metricRow}>
          <Metric value={state.survivors.length} label="MAT SURVIVORS" color={state.survivors.length ? instrumentColors.warning : instrumentColors.text} />
          <Metric value={trackCount} label="ANON TRACKS" color="#B98CFF" />
          <Metric value={sensorCount} label="GRAPH SENSORS" color={instrumentColors.cyan} />
          <Metric value={state.alerts.filter((alert) => alert.status === 'Pending').length} label="OPEN ALERTS" color={instrumentColors.danger} />
        </View>

        <InstrumentPanel eyebrow="OPERATIONAL TWIN" accessory={<ThemedText preset="mono" style={styles.muted}>NODES {nodeCount} / SEQ {state.worldGraphSeq ?? '—'}</ThemedText>}>
          <WorldGraphMap graph={state.worldGraph} zones={state.zones} survivors={state.survivors} />
          <ThemedText preset="bodySm" style={styles.disclaimer}>Purple markers are privacy-governed anonymous WorldGraph tracks. Only MAT API records appear as survivor markers.</ThemedText>
        </InstrumentPanel>

        <InstrumentPanel eyebrow="INCIDENT CONTROL" accessory={state.pipeline && <StatusPill label="PIPELINE" status={state.pipeline.scanning ? 'live' : 'idle'} />}>
          {state.events.length > 0 ? <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.eventStrip}>{state.events.map((event) => <Pressable key={event.id} onPress={() => state.setSelectedEvent(event.id)} style={[styles.eventChip, event.id === state.selectedEventId && styles.eventChipActive]}><ThemedText preset="mono" style={[styles.eventText, event.id === state.selectedEventId && styles.eventTextActive]}>{event.event_type.replace(/([A-Z])/g, ' $1').trim().toUpperCase()}</ThemedText></Pressable>)}</ScrollView> : <ThemedText preset="bodyMd" style={styles.muted}>No incident events returned. Create an event through the MAT API before scanning.</ThemedText>}
          {selectedEvent && <View><ThemedText preset="labelLg" style={styles.eventTitle}>{selectedEvent.description}</ThemedText><ThemedText preset="mono" style={styles.muted}>{selectedEvent.status.toUpperCase()} · {state.zones.length} ZONES · STARTED {new Date(selectedEvent.start_time).toLocaleString()}</ThemedText></View>}
          <View style={styles.actions}>
            <Pressable accessibilityRole="button" testID="mat-scan-control" disabled={!selectedEvent || state.apiStatus === 'loading'} onPress={() => void setScan()} style={[styles.primaryButton, (!selectedEvent || state.apiStatus === 'loading') && styles.disabled]}>{state.apiStatus === 'loading' ? <ActivityIndicator color="#071015" /> : <Ionicons name={state.pipeline?.scanning ? 'pause' : 'radio'} size={17} color="#071015" />}<ThemedText preset="labelMd" style={styles.primaryButtonText}>{state.pipeline?.scanning ? 'PAUSE SCAN' : 'START SCAN'}</ThemedText></Pressable>
            <Pressable accessibilityRole="button" onPress={() => void refresh()} style={styles.secondaryButton}><Ionicons name="refresh" size={16} color={instrumentColors.cyan} /><ThemedText preset="labelMd" style={styles.secondaryButtonText}>REFRESH</ThemedText></Pressable>
          </View>
          {state.pipeline && <View style={styles.pipelineFacts}><Fact label="BUFFER" value={`${state.pipeline.buffer_duration_secs.toFixed(1)}s`} /><Fact label="SAMPLE" value={`${state.pipeline.sample_rate.toFixed(0)} Hz`} /><Fact label="ML" value={state.pipeline.ml_ready ? 'READY' : state.pipeline.ml_enabled ? 'WARMING' : 'OFF'} /><Fact label="HEARTBEAT" value={state.pipeline.heartbeat_enabled ? 'ON' : 'OFF'} /></View>}
        </InstrumentPanel>

        <InstrumentPanel eyebrow="TRIAGE DISTRIBUTION">
          <View style={styles.triageRow}>{(Object.keys(triage) as TriageStatus[]).map((key) => <View key={key} style={styles.triageItem}><View style={[styles.triageBar, { backgroundColor: TRIAGE_COLORS[key], opacity: triage[key] ? 1 : .25 }]} /><ThemedText preset="displayLg" style={[styles.triageValue, { color: TRIAGE_COLORS[key] }]}>{triage[key]}</ThemedText><ThemedText preset="mono" style={styles.triageLabel}>{key.toUpperCase()}</ThemedText></View>)}</View>
        </InstrumentPanel>

        <InstrumentPanel eyebrow="ACTIVE ALERT QUEUE" accessory={<ThemedText preset="mono" style={styles.muted}>{state.alerts.length} RETURNED</ThemedText>}>
          {state.alerts.length === 0 ? <View style={styles.empty}><Ionicons name="shield-checkmark-outline" size={28} color={instrumentColors.textSecondary} /><ThemedText preset="bodyMd" style={styles.muted}>No active alerts returned by the MAT API.</ThemedText></View> : state.alerts.map((alert) => <View key={alert.id} style={[styles.alert, { borderLeftColor: TRIAGE_COLORS[alert.triage_status] }]}><View style={styles.alertHeading}><ThemedText preset="labelLg" style={styles.alertTitle}>{alert.title}</ThemedText><ThemedText preset="mono" style={{ color: TRIAGE_COLORS[alert.triage_status] }}>{alert.priority.toUpperCase()}</ThemedText></View><ThemedText preset="bodyMd" style={styles.alertMessage}>{alert.message}</ThemedText>{alert.recommended_action && <ThemedText preset="bodySm" style={styles.recommendation}>RECOMMENDED · {alert.recommended_action}</ThemedText>}{alert.status === 'Pending' && <Pressable onPress={() => void acknowledge(alert)} disabled={acknowledging === alert.id} style={styles.ackButton}><ThemedText preset="mono" style={styles.ackText}>{acknowledging === alert.id ? 'ACKNOWLEDGING…' : 'ACKNOWLEDGE'}</ThemedText></Pressable>}</View>)}
        </InstrumentPanel>

        <InstrumentPanel eyebrow="WORLDGRAPH SECURE LINK" accessory={<ThemedText preset="mono" style={styles.muted}>TWIN PROTOCOL / V1</ThemedText>}>
          <ThemedText preset="bodySm" style={styles.disclaimer}>The app automatically reads the governed WorldGraph snapshot from the RuView sensing server. If server authentication is enabled, enter a short-lived read credential and sync again. The credential remains only in memory and the field is cleared after a successful sync.</ThemedText>
          <TextInput accessibilityLabel="WorldGraph server URL" value={worldGraphOrigin} onChangeText={setWorldGraphOrigin} autoCapitalize="none" autoCorrect={false} placeholder="https://worldgraph.example" placeholderTextColor={instrumentColors.textSecondary} style={styles.input} />
          <TextInput accessibilityLabel="WorldGraph short-lived token" value={worldGraphToken} onChangeText={setWorldGraphToken} autoCapitalize="none" autoCorrect={false} secureTextEntry placeholder="Short-lived access token" placeholderTextColor={instrumentColors.textSecondary} style={styles.input} />
          <Pressable testID="worldgraph-connect" onPress={() => void connectWorldGraph()} style={styles.secondaryButton}><Ionicons name="git-network" size={17} color={instrumentColors.cyan} /><ThemedText preset="labelMd" style={styles.secondaryButtonText}>SYNC GOVERNED TWIN</ThemedText></Pressable>
        </InstrumentPanel>

        <ThemedText preset="mono" style={styles.footerNote}>OPERATIONAL AID · VERIFY DETECTIONS WITH APPROVED RESPONSE PROCEDURES · SOURCE PROVENANCE PRESERVED</ThemedText>
      </ScrollView>
    </View>
  );
};

const Fact = ({ label, value }: { label: string; value: string }) => <View style={styles.fact}><ThemedText preset="mono" style={styles.factLabel}>{label}</ThemedText><ThemedText preset="labelLg" style={styles.factValue}>{value}</ThemedText></View>;

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: instrumentColors.background }, content: { padding: 16, paddingBottom: 30, gap: 14 },
  hero: { paddingVertical: 10, gap: 14 }, heroCopy: { gap: 7 }, kicker: { color: instrumentColors.cyan, letterSpacing: 1.3, fontSize: 10 },
  heroTitle: { color: instrumentColors.text, fontSize: 31, lineHeight: 35 }, heroAccent: { color: instrumentColors.green, fontSize: 31, lineHeight: 35 },
  heroBody: { color: instrumentColors.textSecondary, maxWidth: 560, lineHeight: 21 }, statusStack: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  statusPill: { flexDirection: 'row', alignItems: 'center', gap: 6, borderRadius: 999, borderWidth: 1, paddingVertical: 6, paddingHorizontal: 9, backgroundColor: 'rgba(7,10,15,.72)' },
  statusDot: { width: 6, height: 6, borderRadius: 3 }, statusText: { fontSize: 8, letterSpacing: .7 },
  errorPanel: { borderColor: 'rgba(255,100,120,.5)' }, errorText: { color: instrumentColors.danger }, diagnosticRow: { gap: 4 }, diagnosticLabel: { color: instrumentColors.warning, fontSize: 8, letterSpacing: .8 }, repairText: { color: instrumentColors.warning, lineHeight: 18 }, muted: { color: instrumentColors.textSecondary, fontSize: 9, letterSpacing: .55 },
  metricRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 }, metric: { minWidth: '47%', flexGrow: 1, padding: 14, borderRadius: 13, borderWidth: 1, borderColor: instrumentColors.border, backgroundColor: 'rgba(20,24,31,.94)' },
  metricValue: { fontSize: 28, lineHeight: 31 }, metricLabel: { color: instrumentColors.textSecondary, fontSize: 8, letterSpacing: .8, marginTop: 4 },
  disclaimer: { color: instrumentColors.textSecondary, lineHeight: 18 }, eventStrip: { gap: 8 }, eventChip: { borderWidth: 1, borderColor: instrumentColors.border, borderRadius: 999, paddingVertical: 8, paddingHorizontal: 11 }, eventChipActive: { borderColor: instrumentColors.cyan, backgroundColor: 'rgba(25,212,230,.09)' },
  eventText: { color: instrumentColors.textSecondary, fontSize: 8 }, eventTextActive: { color: instrumentColors.cyan }, eventTitle: { color: instrumentColors.text, marginBottom: 4 },
  actions: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 }, primaryButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 7, borderRadius: 11, paddingVertical: 12, paddingHorizontal: 15, backgroundColor: instrumentColors.green }, primaryButtonText: { color: '#071015', letterSpacing: .8 },
  secondaryButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 7, borderRadius: 11, paddingVertical: 11, paddingHorizontal: 14, borderWidth: 1, borderColor: instrumentColors.borderStrong, backgroundColor: 'rgba(25,212,230,.05)' }, secondaryButtonText: { color: instrumentColors.cyan, letterSpacing: .7 }, disabled: { opacity: .35 },
  pipelineFacts: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 }, fact: { flexGrow: 1, minWidth: '21%', borderLeftWidth: 1, borderLeftColor: instrumentColors.borderStrong, paddingLeft: 9 }, factLabel: { color: instrumentColors.textSecondary, fontSize: 7 }, factValue: { color: instrumentColors.text, fontSize: 13 },
  triageRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 7 }, triageItem: { flexGrow: 1, minWidth: '17%', padding: 9, borderRadius: 10, backgroundColor: '#0C1016' }, triageBar: { height: 2, borderRadius: 2, marginBottom: 8 }, triageValue: { fontSize: 22, lineHeight: 24 }, triageLabel: { color: instrumentColors.textSecondary, fontSize: 7 },
  empty: { alignItems: 'center', paddingVertical: 16, gap: 8 }, alert: { borderLeftWidth: 3, padding: 12, borderRadius: 9, backgroundColor: '#0C1016', gap: 7 }, alertHeading: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8 }, alertTitle: { flex: 1, color: instrumentColors.text }, alertMessage: { color: instrumentColors.textSecondary }, recommendation: { color: instrumentColors.warning }, ackButton: { alignSelf: 'flex-start', borderWidth: 1, borderColor: instrumentColors.borderStrong, borderRadius: 7, paddingVertical: 7, paddingHorizontal: 10 }, ackText: { color: instrumentColors.cyan, fontSize: 8 },
  input: { color: instrumentColors.text, backgroundColor: '#0B0E13', borderWidth: 1, borderColor: instrumentColors.border, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 11, fontFamily: 'JetBrainsMono_400Regular', fontSize: 11 }, inlineCode: { color: instrumentColors.cyan, fontSize: 10 }, footerNote: { color: instrumentColors.textSecondary, fontSize: 7, letterSpacing: .65, textAlign: 'center', paddingVertical: 8 },
});

export default MATScreen;
