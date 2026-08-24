import { memo, useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Circle, Line, Rect, Text as SvgText } from 'react-native-svg';
import { instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import type { NlosFreshness, NlosTrack } from '@/types/nlos';
import { buildLidarPointCloud, LIDAR_RELAY_POINT_COUNT } from './lidarPointCloud';

interface LidarPointCloudProps {
  tracks: NlosTrack[];
  freshness: NlosFreshness;
  width: number;
}

interface ProjectedPoint {
  color: string;
  radius: number;
  x: number;
  y: number;
}

const CANVAS_WIDTH = 360;
const CANVAS_HEIGHT = 260;

const rgb = (red: number, green: number, blue: number) => (
  `rgb(${Math.round(red * 255)}, ${Math.round(green * 255)}, ${Math.round(blue * 255)})`
);

export const LidarPointCloud = memo(({ tracks, freshness, width }: LidarPointCloudProps) => {
  const cloud = useMemo(() => buildLidarPointCloud(tracks), [tracks]);
  const points = useMemo(() => {
    const projected: ProjectedPoint[] = [];
    for (let index = 0; index < cloud.totalPointCount; index += 1) {
      const isRelay = index < LIDAR_RELAY_POINT_COUNT;
      if ((isRelay && index % 3 !== 0) || (!isRelay && index % 2 !== 0)) continue;
      const offset = index * 3;
      const x = cloud.positions[offset];
      const y = cloud.positions[offset + 1];
      const z = cloud.positions[offset + 2];
      projected.push({
        x: 180 + x * 34 - z * 8,
        y: 220 - y * 46 + z * 7,
        radius: isRelay ? 1.15 : 1.8,
        color: rgb(cloud.colors[offset], cloud.colors[offset + 1], cloud.colors[offset + 2]),
      });
    }
    return projected;
  }, [cloud]);
  const displayWidth = Math.max(260, Math.min(width, 560));

  return (
    <View
      testID="nlos-lidar-point-cloud"
      accessibilityRole="image"
      accessibilityLabel={`Projected LiDAR reconstruction cloud with ${cloud.targetPointCount} gated target returns from ${tracks.length} hidden target hypotheses. ${freshness} evidence.`}
      style={{ alignSelf: 'center', width: displayWidth, aspectRatio: CANVAS_WIDTH / CANVAS_HEIGHT }}
    >
      <Svg width="100%" height="100%" viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}>
        <Rect x={12} y={12} width={336} height={236} rx={14} fill={instrumentColors.panelRaised} stroke={instrumentColors.border} />
        <Line x1={36} y1={220} x2={328} y2={220} stroke={instrumentColors.borderStrong} />
        <Line x1={180} y1={220} x2={78} y2={176} stroke={instrumentColors.cyanDim} />
        <Line x1={180} y1={220} x2={286} y2={174} stroke={instrumentColors.cyanDim} />
        {points.map((point, index) => (
          <Circle
            key={`cloud-point-${index}`}
            cx={point.x}
            cy={point.y}
            r={point.radius}
            fill={point.color}
            opacity={freshness === 'fresh' ? 0.88 : 0.35}
          />
        ))}
        <SvgText x={24} y={34} fill={instrumentColors.cyan} fontSize={9} letterSpacing={1.1}>LIDAR CLOUD / NATIVE PROJECTION</SvgText>
      </Svg>
      <View pointerEvents="none" style={styles.metricRow}>
        <ThemedText testID="nlos-cloud-target-count" preset="labelLg" style={styles.metricValue}>
          {cloud.targetPointCount}
        </ThemedText>
        <ThemedText preset="mono" style={styles.metricLabel}>GATED TARGET RETURNS</ThemedText>
      </View>
    </View>
  );
});

LidarPointCloud.displayName = 'LidarPointCloud';

const styles = StyleSheet.create({
  metricRow: {
    position: 'absolute',
    left: 24,
    bottom: 24,
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
  },
  metricValue: { color: instrumentColors.green, fontSize: 14, lineHeight: 16 },
  metricLabel: { color: instrumentColors.textSecondary, fontSize: 8, letterSpacing: 0.8 },
});
