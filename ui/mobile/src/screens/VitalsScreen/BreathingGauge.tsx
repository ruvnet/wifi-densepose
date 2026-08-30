import { useMemo } from 'react';
import { View, StyleSheet } from 'react-native';
import { GaugeArc } from '@/components/GaugeArc';
import { colors } from '@/theme/colors';
import { ThemedText } from '@/components/ThemedText';

const BREATHING_MIN_BPM = 0;
const BREATHING_MAX_BPM = 30;
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const deriveBreathingValue = (
  breathingBpm?: number | null,
): number | null => {
  if (typeof breathingBpm === 'number' && Number.isFinite(breathingBpm)) {
    return clamp(breathingBpm, BREATHING_MIN_BPM, BREATHING_MAX_BPM);
  }
  return null;
};

export const BreathingGauge = ({ available, breathingBpm }: { available: boolean; breathingBpm?: number | null }) => {
  const value = useMemo(
    () => available ? deriveBreathingValue(breathingBpm) : null,
    [available, breathingBpm],
  );

  return (
    <View style={styles.container}>
      <ThemedText preset="labelMd" style={styles.label}>
        BREATHING
      </ThemedText>
      <GaugeArc value={value ?? Number.NaN} min={BREATHING_MIN_BPM} max={BREATHING_MAX_BPM} label={value == null ? 'WAITING' : 'RF ESTIMATE'} unit="BPM" color={colors.accent} />
      <ThemedText preset="labelMd" color="textSecondary" style={styles.unit}>
        {value == null ? 'No measured rate' : 'Measured stream'}
      </ThemedText>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  label: {
    color: '#94A3B8',
    letterSpacing: 1,
  },
  unit: {
    marginTop: -12,
    marginBottom: 4,
  },
});
