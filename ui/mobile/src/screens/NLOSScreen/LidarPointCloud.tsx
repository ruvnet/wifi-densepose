import { Fragment, memo, useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Circle, Line, Rect, Text as SvgText } from 'react-native-svg';
import { instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import type { NlosFreshness, NlosTrack } from '@/types/nlos';
import type { LidarPointFrame } from '@/types/lidar';
import {
  buildLidarPointCloud,
  LIDAR_RELAY_POINT_COUNT,
  resolveLidarTrackCenter,
} from './lidarPointCloudData';

interface LidarPointCloudProps {
  tracks: NlosTrack[];
  freshness: NlosFreshness;
  width: number;
  lidarFrame?: LidarPointFrame | null;
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

export const LidarPointCloud = memo(({ tracks, freshness, width, lidarFrame }: LidarPointCloudProps) => {
  const cloud = useMemo(() => buildLidarPointCloud(tracks), [tracks]);
  const points = useMemo(() => {
    const projected: ProjectedPoint[] = [];
    if (lidarFrame) {
      const stride = Math.max(1, Math.ceil(lidarFrame.pointCount / 900));
      for (let index = 0; index < lidarFrame.pointCount; index += stride) {
        const offset = index * 3;
        const x = lidarFrame.points[offset];
        const y = lidarFrame.points[offset + 1];
        const z = lidarFrame.points[offset + 2];
        const confidence = lidarFrame.confidences[index];
        projected.push({
          x: 180 + x * 34 - z * 8,
          y: 220 - y * 46 + z * 7,
          radius: confidence === 2 ? 1.5 : 1,
          color: confidence === 2 ? instrumentColors.green : instrumentColors.cyan,
        });
      }
      return projected;
    }
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
  }, [cloud, lidarFrame]);
  const displayWidth = Math.max(260, Math.min(width, 560));
  const markers = useMemo(() => tracks.slice(0, 16).map((track) => {
    const [x, y, z] = resolveLidarTrackCenter(track);
    return {
      color: track.state === 'degraded' ? instrumentColors.warning : instrumentColors.green,
      confidence: Math.round(track.confidence * 100),
      id: track.trackId.toUpperCase(),
      x: 180 + x * 34 - z * 8,
      y: 220 - y * 46 + z * 7,
    };
  }), [tracks]);

  return (
    <View
      testID="nlos-lidar-point-cloud"
      accessibilityRole="image"
      accessibilityLabel={lidarFrame
        ? `Live iPhone LiDAR cloud with ${lidarFrame.pointCount} visible-scene points. Tracking ${lidarFrame.trackingState}.`
        : `Projected CSI reconstruction cloud with ${cloud.targetPointCount} gated target returns from ${tracks.length} hidden target hypotheses. ${freshness} evidence.`}
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
        {markers.map((marker) => (
          <Fragment key={`cloud-marker-${marker.id}`}>
            <Line x1={marker.x} y1={marker.y - 20} x2={marker.x} y2={marker.y + 20} stroke={marker.color} opacity={0.55} />
            <Circle cx={marker.x} cy={marker.y} r={11} fill="none" stroke={marker.color} strokeWidth={1.4} />
            <Circle cx={marker.x} cy={marker.y} r={3} fill={marker.color} />
            <SvgText x={marker.x + 15} y={marker.y - 5} fill={instrumentColors.text} fontSize={7}>{marker.id}</SvgText>
            <SvgText x={marker.x + 15} y={marker.y + 6} fill={marker.color} fontSize={6}>{marker.confidence}% CONF.</SvgText>
          </Fragment>
        ))}
        <SvgText x={24} y={34} fill={instrumentColors.cyan} fontSize={9} letterSpacing={1.1}>{lidarFrame ? 'IPHONE LIDAR / VISIBLE SCENE' : 'CSI RECONSTRUCTION CLOUD'}</SvgText>
        <SvgText x={24} y={45} fill={instrumentColors.warning} fontSize={6.5} letterSpacing={0.7}>{lidarFrame ? 'ARKIT SCENE DEPTH / NOT THROUGH-WALL' : 'RECONSTRUCTION / NOT RAW SCAN'}</SvgText>
        <SvgText x={276} y={34} fill={instrumentColors.textSecondary} fontSize={7} letterSpacing={0.7}>{lidarFrame ? 'LIVE' : 'PROJECTED'}</SvgText>
      </Svg>
      <View pointerEvents="none" style={styles.metricRow}>
        <ThemedText testID="nlos-cloud-target-count" preset="labelLg" style={styles.metricValue}>
          {lidarFrame?.pointCount ?? cloud.targetPointCount}
        </ThemedText>
        <ThemedText preset="mono" style={styles.metricLabel}>{lidarFrame ? 'VISIBLE-SCENE POINTS' : 'GATED TARGET RETURNS'}</ThemedText>
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
