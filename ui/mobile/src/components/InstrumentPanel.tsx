import type { ReactNode } from 'react';
import { StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';
import { ThemedText } from './ThemedText';

export const instrumentColors = {
  background: '#05090D',
  panel: '#091218',
  panelRaised: '#0C171E',
  cyan: '#24D3E5',
  green: '#58F28B',
  cyanDim: 'rgba(36, 211, 229, 0.38)',
  greenDim: 'rgba(88, 242, 139, 0.36)',
  border: 'rgba(65, 204, 219, 0.22)',
  borderStrong: 'rgba(65, 204, 219, 0.42)',
  grid: 'rgba(92, 151, 162, 0.065)',
  text: '#F3F8FA',
  textSecondary: '#91A4AE',
  warning: '#FFB65C',
  danger: '#FF6478',
  dimOverlay: 'rgba(5, 9, 13, 0.78)',
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
    <View pointerEvents="none" style={styles.accentRail}>
      <View style={styles.accentRailCyan} />
      <View style={styles.accentRailGreen} />
    </View>
    <View pointerEvents="none" style={[styles.corner, styles.cornerTopLeft]} />
    <View pointerEvents="none" style={[styles.corner, styles.cornerBottomRight]} />
    {(eyebrow || accessory) && (
      <View style={styles.headingRow}>
        {eyebrow ? (
          <ThemedText preset="mono" style={styles.eyebrow}>{eyebrow}</ThemedText>
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
    borderRadius: 18,
    padding: 16,
    gap: 12,
    shadowColor: instrumentColors.cyan,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.07,
    shadowRadius: 24,
    elevation: 2,
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
  corner: {
    position: 'absolute',
    width: 11,
    height: 11,
    borderColor: instrumentColors.cyanDim,
  },
  cornerTopLeft: {
    top: 7,
    left: 7,
    borderTopWidth: 1,
    borderLeftWidth: 1,
  },
  cornerBottomRight: {
    right: 7,
    bottom: 7,
    borderRightWidth: 1,
    borderBottomWidth: 1,
  },
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
    fontSize: 11,
    lineHeight: 16,
    letterSpacing: 1.4,
    textTransform: 'uppercase',
  },
});
