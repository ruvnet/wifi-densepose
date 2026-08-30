import { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Defs, LinearGradient, Path, Stop } from 'react-native-svg';
import { ThemedText } from '@/components/ThemedText';
import { colors } from '@/theme/colors';

const WIDTH = 320;
const HEIGHT = 84;

export const VitalWaveform = ({ values, color, label, domain }: { values: number[]; color: string; label: string; domain?: [number, number] }) => {
  const path = useMemo(() => {
    if (values.length < 2) return '';
    const minimum = domain?.[0] ?? Math.min(...values);
    const maximum = domain?.[1] ?? Math.max(...values);
    const span = Math.max(0.001, maximum - minimum);
    return values.map((value, index) => {
      const x = index / (values.length - 1) * WIDTH;
      const y = HEIGHT - 8 - ((value - minimum) / span) * (HEIGHT - 20);
      return `${index ? 'L' : 'M'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(' ');
  }, [domain, values]);

  return (
    <View style={styles.wrap} accessibilityRole="image" accessibilityLabel={`${label} history with ${values.length} measured samples`}>
      {path ? (
        <Svg width="100%" height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`}>
          <Defs><LinearGradient id="vitalGlow" x1="0" y1="0" x2="1" y2="0"><Stop offset="0" stopColor={colors.accentDim} /><Stop offset="1" stopColor={color} /></LinearGradient></Defs>
          <Path d={path} stroke="url(#vitalGlow)" strokeWidth={3} fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </Svg>
      ) : <ThemedText preset="mono" style={styles.empty}>WAITING FOR 2+ FRESH MEASURED SAMPLES</ThemedText>}
    </View>
  );
};

const styles = StyleSheet.create({
  wrap: { minHeight: HEIGHT, justifyContent: 'center', borderRadius: 10, borderWidth: 1, borderColor: colors.border, overflow: 'hidden', backgroundColor: 'rgba(6,10,18,0.5)' },
  empty: { color: colors.textSecondary, fontSize: 7, textAlign: 'center' },
});
