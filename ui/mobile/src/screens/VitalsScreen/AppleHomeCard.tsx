import { useEffect, useState } from 'react';
import { Platform, Pressable, StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { InstrumentPanel, instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import { appleHomeService } from '@/services/appleHome.service';
import { apiService } from '@/services/api.service';
import { spacing } from '@/theme/spacing';
import type { AppleHomeDiscoveryState, RuViewSemanticEvents } from '@/types/appleHome';

const Action = ({ label, onPress, disabled = false }: { label: string; onPress: () => void; disabled?: boolean }) => (
  <Pressable accessibilityRole="button" accessibilityLabel={label} disabled={disabled} onPress={onPress} style={({ pressed }) => [styles.action, disabled && styles.disabled, pressed && styles.pressed]}>
    <ThemedText preset="mono" style={styles.actionText}>{label}</ThemedText>
  </Pressable>
);

export const AppleHomeCard = ({ serverUrl, nodeId }: { serverUrl: string; nodeId: string | number | null }) => {
  const [discovery, setDiscovery] = useState<AppleHomeDiscoveryState>({ state: 'idle', bridges: [] });
  const [semantic, setSemantic] = useState<RuViewSemanticEvents | null>(null);
  const [semanticError, setSemanticError] = useState<string | null>(null);
  const [homeOpenError, setHomeOpenError] = useState(false);

  const errorMessage = (error: unknown, fallback: string) => {
    if (error instanceof Error) return error.message;
    if (error && typeof error === 'object' && 'message' in error && typeof error.message === 'string') return error.message;
    return fallback;
  };

  useEffect(() => {
    const events = appleHomeService.events;
    const subscription = events?.addListener('onAppleHomeDiscovery', setDiscovery);
    return () => {
      subscription?.remove();
      void appleHomeService.stopDiscovery();
    };
  }, []);

  const discover = async () => {
    setDiscovery({ state: 'searching', bridges: [] });
    try {
      setDiscovery(await appleHomeService.startDiscovery());
    } catch (error) {
      setDiscovery({ state: 'error', bridges: [], error: errorMessage(error, 'Bonjour discovery could not start.') });
    }
  };
  const verifySemanticExport = async () => {
    if (nodeId == null) return;
    apiService.setBaseUrl(serverUrl);
    setSemanticError(null);
    try {
      const value = await appleHomeService.getSemanticEvents(nodeId);
      if (value.privacy_class !== 2 && value.privacy_class !== 3) throw new Error(`Privacy class ${value.privacy_class} is not eligible for Apple Home export.`);
      setSemantic(value);
    } catch (error) {
      setSemantic(null);
      setSemanticError(errorMessage(error, 'Semantic-event endpoint unavailable.'));
    }
  };

  const openHome = async () => {
    setHomeOpenError(false);
    setHomeOpenError(!await appleHomeService.openAppleHome());
  };
  const activeEvents = semantic ? Object.entries(semantic.events).filter(([, event]) => event.active).map(([name]) => name.replaceAll('_', ' ')) : [];

  return (
    <InstrumentPanel eyebrow="Apple Home / local HAP bridge" accessory={<ThemedText preset="mono" style={discovery.bridges.length ? styles.ready : styles.idle}>{discovery.bridges.length ? `${discovery.bridges.length} FOUND` : discovery.state.toUpperCase()}</ThemedText>}>
      <View style={styles.titleRow}>
        <View style={styles.homeGlyph}><Ionicons name="home" size={20} color={instrumentColors.cyan} /></View>
        <View style={styles.titleCopy}>
          <ThemedText preset="displayMd" style={styles.title}>Ambient events in Apple Home</ThemedText>
          <ThemedText preset="bodySm" style={styles.copy}>HomePod or Apple TV acts as the Home Hub. RuView remains the sensor.</ThemedText>
        </View>
      </View>

      <View style={styles.flow}>
        {['RuView CSI', 'HAP bridge', 'Apple Home'].map((label, index) => (
          <View key={label} style={styles.flowItem}>
            <View style={styles.flowNode}><ThemedText preset="mono" style={styles.flowText}>{label}</ThemedText></View>
            {index < 2 && <ThemedText preset="mono" style={styles.arrow}>→</ThemedText>}
          </View>
        ))}
      </View>

      <ThemedText preset="bodySm" style={styles.boundary}>Apple Home receives occupancy, motion, and thresholded semantic events only. Breathing, heart-rate proxy, pose, raw CSI, and identity scores never cross this boundary.</ThemedText>

      <View style={styles.actions}>
        <Action label={discovery.state === 'searching' ? 'SEARCHING LOCAL NETWORK…' : 'DISCOVER HAP BRIDGE'} disabled={Platform.OS !== 'ios' || !appleHomeService.nativeAvailable || discovery.state === 'searching'} onPress={() => { void discover(); }} />
        <Action label="OPEN APPLE HOME" disabled={Platform.OS !== 'ios'} onPress={() => { void openHome(); }} />
      </View>

      {discovery.bridges.map((bridge) => (
        <View testID="apple-home-bridge" key={bridge.id} style={styles.bridge}>
          <View style={styles.bridgeIdentity}>
            <View style={styles.liveDot} />
            <View style={styles.titleCopy}>
              <ThemedText preset="labelMd" style={styles.bridgeName}>{bridge.name}</ThemedText>
              <ThemedText preset="mono" style={styles.bridgeMeta}>{bridge.hostName ?? 'RESOLVING'} · {bridge.port || '—'} · BONJOUR</ThemedText>
            </View>
          </View>
          <ThemedText preset="mono" style={bridge.paired === true ? styles.ready : styles.idle}>{bridge.paired == null ? 'PAIR STATE UNKNOWN' : bridge.paired ? 'PAIRED' : 'AVAILABLE TO PAIR'}</ThemedText>
        </View>
      ))}

      {discovery.error && <ThemedText preset="bodySm" style={styles.error}>{discovery.error}</ThemedText>}
      {!appleHomeService.nativeAvailable && <ThemedText preset="bodySm" style={styles.idle}>Install the native iOS build to perform real Bonjour `_hap._tcp` discovery.</ThemedText>}
      <Action label="VERIFY PRIVACY-GATED EVENT FEED" disabled={nodeId == null || !serverUrl} onPress={() => { void verifySemanticExport(); }} />
      {semantic && <ThemedText testID="apple-home-semantic-status" preset="bodySm" style={styles.ready}>Node {String(semantic.node_id)} · privacy class {semantic.privacy_class} · {activeEvents.length ? `active: ${activeEvents.join(', ')}` : 'no active semantic events'}</ThemedText>}
      {semanticError && <ThemedText preset="bodySm" style={styles.error}>{semanticError}</ThemedText>}
      {homeOpenError && <ThemedText preset="bodySm" style={styles.error}>Apple Home could not be opened on this device.</ThemedText>}
    </InstrumentPanel>
  );
};

const styles = StyleSheet.create({
  titleRow: { flexDirection: 'row', gap: spacing.md, alignItems: 'center' },
  homeGlyph: { width: 46, height: 46, borderRadius: 23, borderWidth: 1, borderColor: instrumentColors.cyanDim, backgroundColor: 'rgba(50,184,198,0.08)', alignItems: 'center', justifyContent: 'center' },
  titleCopy: { flex: 1 },
  title: { color: instrumentColors.text, fontSize: 21 },
  copy: { color: instrumentColors.textSecondary, marginTop: 2, lineHeight: 17 },
  flow: { flexDirection: 'row', alignItems: 'center', marginVertical: spacing.md },
  flowItem: { flexDirection: 'row', alignItems: 'center', flex: 1 },
  flowNode: { flex: 1, minHeight: 38, borderWidth: 1, borderColor: instrumentColors.borderStrong, borderRadius: 8, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 3 },
  flowText: { color: instrumentColors.textSecondary, fontSize: 7, textAlign: 'center' },
  arrow: { color: instrumentColors.green, paddingHorizontal: 3 },
  boundary: { color: instrumentColors.warning, lineHeight: 17, borderLeftWidth: 2, borderLeftColor: instrumentColors.warning, paddingLeft: spacing.sm, marginBottom: spacing.md },
  actions: { flexDirection: 'row', gap: spacing.sm },
  action: { flex: 1, minHeight: 42, borderWidth: 1, borderColor: instrumentColors.cyanDim, borderRadius: 8, alignItems: 'center', justifyContent: 'center', paddingHorizontal: spacing.sm, marginBottom: spacing.sm },
  actionText: { color: instrumentColors.cyan, fontSize: 8, textAlign: 'center' },
  disabled: { opacity: 0.35 },
  pressed: { opacity: 0.65 },
  bridge: { borderWidth: 1, borderColor: instrumentColors.greenDim, borderRadius: 9, padding: spacing.sm, marginBottom: spacing.sm, gap: spacing.xs },
  bridgeIdentity: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  liveDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: instrumentColors.green },
  bridgeName: { color: instrumentColors.text },
  bridgeMeta: { color: instrumentColors.textSecondary, fontSize: 7, marginTop: 2 },
  ready: { color: instrumentColors.green, fontSize: 8 },
  idle: { color: instrumentColors.textSecondary, fontSize: 8 },
  error: { color: instrumentColors.danger, lineHeight: 17, marginBottom: spacing.sm },
});
