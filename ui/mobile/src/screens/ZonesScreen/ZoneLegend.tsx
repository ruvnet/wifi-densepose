import { View } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { colors } from '@/theme/colors';
import { spacing } from '@/theme/spacing';
import { instrumentColors } from '@/components/InstrumentPanel';

type LegendStop = {
  label: string;
  color: string;
};

const LEGEND_STOPS: LegendStop[] = [
  { label: 'Quiet', color: '#07151B' },
  { label: 'Trace', color: '#116A76' },
  { label: 'Medium', color: instrumentColors.cyan },
  { label: 'Strong', color: '#23D89B' },
  { label: 'Peak', color: instrumentColors.green },
];

export const ZoneLegend = () => {
  return (
    <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginTop: spacing.md }}>
      {LEGEND_STOPS.map((stop) => (
        <View
          key={stop.label}
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <View
            style={{
              width: 14,
              height: 14,
              borderRadius: 3,
              backgroundColor: stop.color,
              borderColor: colors.border,
              borderWidth: 1,
            }}
          />
          <ThemedText preset="bodySm" style={{ color: colors.textSecondary }}>
            {stop.label}
          </ThemedText>
        </View>
      ))}
    </View>
  );
};
