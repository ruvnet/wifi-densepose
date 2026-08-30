import { useState } from 'react';
import { Ionicons } from '@expo/vector-icons';
import { Pressable, StyleSheet, TextInput, View } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { instrumentColors } from '@/components/InstrumentPanel';
import { apiService } from '@/services/api.service';
import { isPrivateLanHost, normalizeNlosServerUrl } from '@/utils/nlosServerUrl';

interface NlosServerUrlInputProps {
  value: string;
  onChange: (value: string) => void;
  onSave: () => void;
}

export const NlosServerUrlInput = ({ value, onChange, onSave }: NlosServerUrlInputProps) => {
  const validation = normalizeNlosServerUrl(value);
  const [testResult, setTestResult] = useState<{ ok: boolean; label: string } | null>(null);
  let localHttp = false;
  try { const url = new URL(value); localHttp = url.protocol === 'http:' && isPrivateLanHost(url.hostname); } catch { /* validation renders the error */ }

  const testEndpoint = async () => {
    if (!validation.valid || !validation.normalized) return;
    const start = Date.now();
    try {
      await apiService.getStatusAt(validation.normalized);
      setTestResult({ ok: true, label: `REACHABLE · ${Date.now() - start} MS` });
    } catch {
      setTestResult({ ok: false, label: 'UNREACHABLE · CHECK SERVER AND PORT' });
    }
  };

  return (
    <View>
      <View style={styles.labelRow}><View><ThemedText preset="labelMd" style={styles.label}>Calibration origin</ThemedText><ThemedText preset="mono" style={styles.caption}>ROOM SCAN / POSE TEACHER / VERIFIED REPLAY</ThemedText></View>{validation.valid && <View style={[styles.transportBadge, localHttp && styles.localBadge]}><Ionicons name={localHttp ? 'home-outline' : 'lock-closed-outline'} size={12} color={localHttp ? instrumentColors.warning : instrumentColors.green} /><ThemedText preset="mono" style={[styles.transportText, localHttp && styles.localText]}>{localHttp ? 'LOCAL HTTP' : 'HTTPS'}</ThemedText></View>}</View>
      <TextInput
        testID="nlos-server-url-input"
        value={value}
        onChangeText={onChange}
        autoCapitalize="none"
        autoCorrect={false}
        accessibilityLabel="RuView calibration server URL"
        placeholder="http://192.168.1.166:3000 or https://ruview.example"
        keyboardType="url"
        placeholderTextColor={instrumentColors.textSecondary}
        style={[styles.input, !validation.valid && styles.inputError]}
      />
      {!validation.valid && (
        <ThemedText preset="bodySm" style={styles.error}>
          {validation.error}
        </ThemedText>
      )}
      <ThemedText preset="bodySm" style={styles.note}>
        Private LAN and .local hosts may use HTTP during installation. Public hosts remain HTTPS-only. Live calibration still requires an ephemeral Bearer credential.
      </ThemedText>
      {testResult && <View style={styles.testResult}><View style={[styles.resultDot, { backgroundColor: testResult.ok ? instrumentColors.green : instrumentColors.danger }]} /><ThemedText preset="mono" style={{ color: testResult.ok ? instrumentColors.green : instrumentColors.danger, fontSize: 8 }}>{testResult.label}</ThemedText></View>}
      <View style={styles.actions}><Pressable accessibilityRole="button" onPress={() => void testEndpoint()} disabled={!validation.valid} style={({ pressed }) => [styles.testButton, !validation.valid && styles.disabled, pressed && styles.pressed]}><Ionicons name="pulse" size={15} color={instrumentColors.cyan} /><ThemedText preset="labelMd" style={styles.testText}>TEST ENDPOINT</ThemedText></Pressable><Pressable accessibilityRole="button" onPress={onSave} disabled={!validation.valid} style={({ pressed }) => [styles.saveButton, !validation.valid && styles.disabled, pressed && styles.pressed]}><Ionicons name="checkmark" size={16} color="#071015" /><ThemedText preset="labelMd" style={styles.saveText}>SAVE CALIBRATION SERVER</ThemedText></Pressable></View>
    </View>
  );
};

const styles = StyleSheet.create({
  labelRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8, marginBottom: 10 }, label: { color: instrumentColors.text }, caption: { color: instrumentColors.textSecondary, fontSize: 7, letterSpacing: .6, marginTop: 2 },
  transportBadge: { flexDirection: 'row', alignItems: 'center', gap: 5, borderWidth: 1, borderColor: `${instrumentColors.green}55`, borderRadius: 999, paddingHorizontal: 8, paddingVertical: 5 }, localBadge: { borderColor: `${instrumentColors.warning}55` }, transportText: { color: instrumentColors.green, fontSize: 7 }, localText: { color: instrumentColors.warning },
  input: { borderWidth: 1, borderColor: instrumentColors.border, borderRadius: 11, backgroundColor: '#0B0E13', color: instrumentColors.text, paddingHorizontal: 12, paddingVertical: 12, fontFamily: 'JetBrainsMono_400Regular', fontSize: 10 }, inputError: { borderColor: instrumentColors.danger }, error: { color: instrumentColors.danger, marginTop: 7 }, note: { color: instrumentColors.textSecondary, lineHeight: 18, marginTop: 8 },
  testResult: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 9 }, resultDot: { width: 6, height: 6, borderRadius: 3 }, actions: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 12 }, testButton: { flexGrow: 1, minWidth: 125, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, borderWidth: 1, borderColor: instrumentColors.borderStrong, borderRadius: 10, padding: 11, backgroundColor: 'rgba(25,212,230,.05)' }, testText: { color: instrumentColors.cyan, fontSize: 10 }, saveButton: { flexGrow: 2, minWidth: 180, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, borderRadius: 10, padding: 11, backgroundColor: instrumentColors.green }, saveText: { color: '#071015', fontSize: 10 }, disabled: { opacity: .35 }, pressed: { opacity: .65 },
});
