import { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import { GaugeArc } from '@/components/GaugeArc';
import { colors } from '@/theme/colors';
import { ThemedText } from '@/components/ThemedText';

const HEART_MIN_BPM = 40;
const HEART_MAX_BPM = 120;
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const deriveHeartRate = (
  heartbeat?: number | null,
): number | null => {
  if (typeof heartbeat === 'number' && Number.isFinite(heartbeat)) {
    return clamp(heartbeat, HEART_MIN_BPM, HEART_MAX_BPM);
  }
  return null;
};

export const HeartRateGauge = ({ available, heartProxyBpm }: { available: boolean; heartProxyBpm?: number | null }) => {
  const value = useMemo(
    () => available ? deriveHeartRate(heartProxyBpm) : null,
    [available, heartProxyBpm],
  );

  return (
    <View style={styles.container}>
      <ThemedText preset="labelMd" style={styles.label}>
        HR PROXY
      </ThemedText>
      <GaugeArc
        value={value ?? Number.NaN}
        min={HEART_MIN_BPM}
        max={HEART_MAX_BPM}
        label={value == null ? 'WAITING' : 'RF PROXY'}
        unit="BPM"
        color={colors.danger}
        colorTo={colors.success}
      />
      <ThemedText preset="bodySm" color="textSecondary" style={styles.note}>
        {value == null ? 'No measured proxy' : 'Measured stream'}
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
  note: {
    marginTop: -12,
    marginBottom: 4,
  },
});
