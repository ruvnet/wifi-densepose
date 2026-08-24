import type { ReactNode } from 'react';
import { StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';
import { ThemedText } from './ThemedText';

export const instrumentColors = {
  background: '#0B0E13',
  panel: '#14181F',
  panelRaised: '#181D25',
  cyan: '#19D4E6',
  green: '#26D968',
  cyanDim: 'rgba(25, 212, 230, 0.34)',
  greenDim: 'rgba(38, 217, 104, 0.32)',
  border: 'rgba(39, 44, 53, 0.92)',
  borderStrong: 'rgba(25, 212, 230, 0.30)',
  grid: 'rgba(25, 212, 230, 0.045)',
  text: '#E7EBEF',
  textSecondary: '#7B899D',
  warning: '#FFB65C',
  danger: '#FF6478',
  dimOverlay: 'rgba(11, 14, 19, 0.82)',
} as const;

interface InstrumentPanelProps {
  children?: ReactNode;
  eyebrow?: string;
  accessory?: ReactNode;
  style?: StyleProp<ViewStyle>;
  testID?: string;
  accessibilityLabel?: string;
}

export const InstrumentGrid = () => (
  <View
    pointerEvents="none"
    accessibilityElementsHidden
    importantForAccessibility="no-hide-descendants"
    style={StyleSheet.absoluteFill}
  >
    <View style={styles.gridColumns}>
      {Array.from({ length: 8 }, (_, index) => (
        <View key={`column-${index}`} style={styles.gridColumn} />
      ))}
    </View>
    <View style={styles.gridRows}>
      {Array.from({ length: 18 }, (_, index) => (
        <View key={`row-${index}`} style={styles.gridRow} />
      ))}
    </View>
  </View>
);

export const InstrumentPanel = ({
  children,
  eyebrow,
  accessory,
  style,
  testID,
  accessibilityLabel,
}: InstrumentPanelProps) => (
  <View testID={testID} accessibilityLabel={accessibilityLabel} style={[styles.panel, style]}>
    <View pointerEvents="none" style={styles.ambientGlow} />
    <View pointerEvents="none" style={styles.accentRail}>
      <View style={styles.accentRailCyan} />
      <View style={styles.accentRailGreen} />
    </View>
    {(eyebrow || accessory) && (
      <View style={styles.headingRow}>
        {eyebrow ? (
          <ThemedText preset="labelMd" style={styles.eyebrow}>{eyebrow}</ThemedText>
        ) : <View />}
        {accessory}
      </View>
    )}
    {children}
  </View>
);

const styles = StyleSheet.create({
  gridColumns: {
    ...StyleSheet.absoluteFillObject,
    flexDirection: 'row',
  },
  gridColumn: {
    flex: 1,
    borderRightColor: instrumentColors.grid,
    borderRightWidth: 1,
  },
  gridRows: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
  },
  gridRow: {
    height: 1,
    backgroundColor: instrumentColors.grid,
  },
  panel: {
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: instrumentColors.panel,
    borderColor: instrumentColors.border,
    borderWidth: 1,
    borderRadius: 16,
    padding: 16,
    gap: 12,
    shadowColor: '#020306',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.72,
    shadowRadius: 20,
    elevation: 3,
  },
  ambientGlow: {
    position: 'absolute',
    top: -76,
    right: -54,
    width: 190,
    height: 190,
    borderRadius: 95,
    backgroundColor: 'rgba(25, 212, 230, 0.045)',
  },
  accentRail: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 2,
    flexDirection: 'row',
  },
  accentRailCyan: { flex: 2, backgroundColor: instrumentColors.cyan },
  accentRailGreen: { flex: 1, backgroundColor: instrumentColors.green },
  headingRow: {
    minHeight: 22,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 8,
  },
  eyebrow: {
    color: instrumentColors.cyan,
    fontSize: 12,
    lineHeight: 16,
    letterSpacing: 1.15,
    textTransform: 'uppercase',
  },
});
