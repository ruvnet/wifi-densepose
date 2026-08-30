import { useState } from 'react';
import { Ionicons } from '@expo/vector-icons';
import { Pressable, StyleSheet, TextInput, View } from 'react-native';
import { validateServerUrl } from '@/utils/urlValidator';
import { apiService } from '@/services/api.service';
import { ThemedText } from '@/components/ThemedText';
import { instrumentColors } from '@/components/InstrumentPanel';

type ServerUrlInputProps = {
  value: string;
  onChange: (value: string) => void;
  onSave: () => void;
};

export const ServerUrlInput = ({ value, onChange, onSave }: ServerUrlInputProps) => {
  const [testResult, setTestResult] = useState('');

  const validation = validateServerUrl(value);

  const handleTest = async () => {
    if (!validation.valid) {
      setTestResult('✗ Invalid URL');
      return;
    }

    const start = Date.now();
    try {
      await apiService.getStatusAt(value.trim());
      setTestResult(`✓ ${Date.now() - start}ms`);
    } catch {
      setTestResult('✗ Failed');
    }
  };

  return (
    <View>
      <View style={styles.labelRow}><View><ThemedText preset="labelMd" style={styles.label}>Sensing origin</ThemedText><ThemedText preset="mono" style={styles.caption}>CSI / VITALS / ZONES / MAT / WORLDGRAPH</ThemedText></View><View style={styles.transportBadge}><Ionicons name="git-network-outline" size={12} color={instrumentColors.cyan} /><ThemedText preset="mono" style={styles.transportText}>PRIMARY</ThemedText></View></View>
      <TextInput
        testID="sensing-server-url-input"
        accessibilityLabel="Sensing server URL"
        value={value}
        onChangeText={onChange}
        autoCapitalize="none"
        autoCorrect={false}
        placeholder="http://192.168.1.100:8080"
        keyboardType="url"
        placeholderTextColor={instrumentColors.textSecondary}
        style={[styles.input, !validation.valid && styles.inputError]}
      />
      {!validation.valid && (
        <ThemedText preset="bodySm" style={styles.error}>
          {validation.error}
        </ThemedText>
      )}

      <View style={styles.resultRow}><View style={[styles.resultDot, { backgroundColor: testResult.startsWith('✓') ? instrumentColors.green : testResult.startsWith('✗') ? instrumentColors.danger : instrumentColors.textSecondary }]} /><ThemedText preset="mono" style={styles.resultText}>{testResult || 'READY TO TEST CONNECTION'}</ThemedText></View>

      <View style={styles.actions}>
        <Pressable
          onPress={handleTest}
          disabled={!validation.valid}
          style={({ pressed }) => [styles.testButton, !validation.valid && styles.disabled, pressed && styles.pressed]}
        >
          <Ionicons name="pulse" size={15} color={instrumentColors.cyan} /><ThemedText preset="labelMd" style={styles.testText}>Test Connection</ThemedText>
        </Pressable>
        <Pressable
          onPress={onSave}
          disabled={!validation.valid}
          style={({ pressed }) => [styles.saveButton, !validation.valid && styles.disabled, pressed && styles.pressed]}
        >
          <Ionicons name="checkmark" size={16} color="#071015" /><ThemedText preset="labelMd" style={styles.saveText}>Save</ThemedText>
        </Pressable>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  labelRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8, marginBottom: 10 }, label: { color: instrumentColors.text }, caption: { color: instrumentColors.textSecondary, fontSize: 7, letterSpacing: .6, marginTop: 2 }, transportBadge: { flexDirection: 'row', alignItems: 'center', gap: 5, borderWidth: 1, borderColor: instrumentColors.borderStrong, borderRadius: 999, paddingHorizontal: 8, paddingVertical: 5 }, transportText: { color: instrumentColors.cyan, fontSize: 7 },
  input: { borderWidth: 1, borderColor: instrumentColors.border, borderRadius: 11, backgroundColor: '#0B0E13', color: instrumentColors.text, paddingHorizontal: 12, paddingVertical: 12, fontFamily: 'JetBrainsMono_400Regular', fontSize: 10 }, inputError: { borderColor: instrumentColors.danger }, error: { color: instrumentColors.danger, marginTop: 7 }, resultRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 9 }, resultDot: { width: 6, height: 6, borderRadius: 3 }, resultText: { color: instrumentColors.textSecondary, fontSize: 8 },
  actions: { flexDirection: 'row', gap: 8, marginTop: 12 }, testButton: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, borderWidth: 1, borderColor: instrumentColors.borderStrong, borderRadius: 10, padding: 11, backgroundColor: 'rgba(25,212,230,.05)' }, testText: { color: instrumentColors.cyan, fontSize: 10 }, saveButton: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, borderRadius: 10, padding: 11, backgroundColor: instrumentColors.green }, saveText: { color: '#071015', fontSize: 10 }, disabled: { opacity: .35 }, pressed: { opacity: .65 },
});
