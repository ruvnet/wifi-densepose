import React, { memo, useMemo } from 'react';
import { View } from 'react-native';
import Svg, { Circle, Ellipse, Line, Polygon, Rect, Text as SvgText } from 'react-native-svg';
import { instrumentColors } from '@/components/InstrumentPanel';
import type { NlosFreshness, NlosTrack } from '@/types/nlos';
import type { LidarPointFrame } from '@/types/lidar';
import { LidarPointCloud } from './LidarPointCloud';

export type NlosViewMode = 'plan' | 'perspective' | 'cloud';

interface HiddenTargetVisualizationProps {
  tracks: NlosTrack[];
  freshness: NlosFreshness;
  mode: NlosViewMode;
  width: number;
  lidarFrame?: LidarPointFrame | null;
}

interface ProjectedTrack {
  track: NlosTrack;
  x: number;
  y: number;
  radiusX: number;
  radiusY: number;
  velocityX: number;
  velocityY: number;
}

const CANVAS_WIDTH = 360;
const CANVAS_HEIGHT = 260;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const resolveTrackColor = (track: NlosTrack, freshness: NlosFreshness): string => {
  if (freshness !== 'fresh' || track.state === 'unknown') return instrumentColors.textSecondary;
  if (track.state === 'degraded') return instrumentColors.warning;
  return instrumentColors.green;
};

const projectPlan = (track: NlosTrack): ProjectedTrack => {
  const x = 180 + clamp(track.positionM.x, -6, 6) * 24;
  const y = 232 - clamp(track.positionM.z, 0, 8) * 25;
  return {
    track,
    x,
    y,
    radiusX: clamp(Math.sqrt(track.covarianceDiagonalM2.x) * 24, 5, 28),
    radiusY: clamp(Math.sqrt(track.covarianceDiagonalM2.z) * 25, 5, 28),
    velocityX: track.velocityMps.x * 10,
    velocityY: -track.velocityMps.z * 10,
  };
};

const projectPerspective = (track: NlosTrack): ProjectedTrack => {
  const position = track.positionM;
  const x = 180 + (clamp(position.x, -6, 6) - clamp(position.z, 0, 8)) * 17;
  const y = 205 + (clamp(position.x, -6, 6) + clamp(position.z, 0, 8)) * 6 - clamp(position.y, 0, 4) * 25;
  return {
    track,
    x,
    y,
    radiusX: clamp(Math.sqrt(track.covarianceDiagonalM2.x) * 22, 5, 26),
    radiusY: clamp(Math.sqrt(track.covarianceDiagonalM2.y + track.covarianceDiagonalM2.z) * 10, 4, 22),
    velocityX: (track.velocityMps.x - track.velocityMps.z) * 8,
    velocityY: (track.velocityMps.x + track.velocityMps.z - track.velocityMps.y) * 4,
  };
};

const PlanScene = () => (
  <>
    <Rect x={12} y={12} width={336} height={236} rx={14} fill={instrumentColors.panelRaised} stroke={instrumentColors.border} />
    {[60, 108, 156, 204, 252, 300].map((x) => (
      <Line key={`plan-column-${x}`} x1={x} y1={18} x2={x} y2={242} stroke={instrumentColors.grid} />
    ))}
    {[54, 94, 134, 174, 214].map((y) => (
      <Line key={`plan-row-${y}`} x1={18} y1={y} x2={342} y2={y} stroke={instrumentColors.grid} />
    ))}
    <Rect x={13} y={13} width={334} height={80} rx={13} fill="rgba(255, 182, 92, 0.055)" />
    <Line x1={20} y1={94} x2={340} y2={94} stroke={instrumentColors.warning} strokeWidth={2} />
    <SvgText x={24} y={79} fill={instrumentColors.warning} fontSize={9} letterSpacing={1.2}>HIDDEN REGION</SvgText>
    <SvgText x={24} y={110} fill={instrumentColors.textSecondary} fontSize={9} letterSpacing={1}>RELAY SURFACE</SvgText>
    <Circle cx={180} cy={220} r={35} fill="none" stroke={instrumentColors.border} strokeDasharray="2 5" />
    <Circle cx={180} cy={220} r={72} fill="none" stroke={instrumentColors.border} strokeDasharray="2 6" />
    <Circle cx={180} cy={220} r={4} fill={instrumentColors.cyan} />
    <Circle cx={180} cy={220} r={9} fill="none" stroke={instrumentColors.cyanDim} />
    <Line x1={180} y1={211} x2={180} y2={98} stroke={instrumentColors.cyanDim} strokeDasharray="5 5" />
    <Line x1={180} y1={220} x2={252} y2={148} stroke={instrumentColors.greenDim} strokeWidth={1.5} />
    <SvgText x={193} y={232} fill={instrumentColors.textSecondary} fontSize={8} letterSpacing={1}>SENSOR</SvgText>
  </>
);

const PerspectiveScene = () => (
  <>
    <Rect x={12} y={12} width={336} height={236} rx={14} fill={instrumentColors.panelRaised} stroke={instrumentColors.border} />
    <Polygon points="180,34 318,84 180,137 42,84" fill={instrumentColors.panel} stroke={instrumentColors.borderStrong} />
    <Polygon points="42,84 180,137 180,224 42,169" fill="rgba(14, 27, 35, 0.92)" stroke={instrumentColors.border} />
    <Polygon points="180,137 318,84 318,169 180,224" fill="rgba(7, 17, 23, 0.92)" stroke={instrumentColors.border} />
    {[1, 2, 3].map((step) => (
      <React.Fragment key={`perspective-grid-${step}`}>
        <Line x1={42 + step * 34.5} y1={84 + step * 13.25} x2={42 + step * 34.5} y2={169 + step * 13.75} stroke={instrumentColors.grid} />
        <Line x1={318 - step * 34.5} y1={84 + step * 13.25} x2={318 - step * 34.5} y2={169 + step * 13.75} stroke={instrumentColors.grid} />
      </React.Fragment>
    ))}
    <Polygon points="84,69 180,104 276,69 180,34" fill="rgba(255, 182, 92, 0.065)" />
    <Line x1={84} y1={69} x2={180} y2={104} stroke={instrumentColors.warning} strokeWidth={2} />
    <Line x1={180} y1={104} x2={276} y2={69} stroke={instrumentColors.warning} strokeWidth={2} />
    <SvgText x={110} y={59} fill={instrumentColors.warning} fontSize={9} letterSpacing={1}>BEYOND RELAY PLANE</SvgText>
    <Circle cx={180} cy={207} r={4} fill={instrumentColors.cyan} />
    <Circle cx={180} cy={207} r={13} fill="none" stroke={instrumentColors.cyanDim} strokeDasharray="2 3" />
  </>
);

export const HiddenTargetVisualization = memo(({
  tracks,
  freshness,
  mode,
  width,
  lidarFrame,
}: HiddenTargetVisualizationProps) => {
  const projectedTracks = useMemo(
    () => tracks.map(mode === 'plan' ? projectPlan : projectPerspective),
    [mode, tracks],
  );
  const displayWidth = Math.max(260, Math.min(width, 560));

  if (mode === 'cloud') {
    return <LidarPointCloud tracks={tracks} freshness={freshness} width={displayWidth} lidarFrame={lidarFrame} />;
  }

  return (
    <View
      testID="nlos-target-visualization"
      accessibilityRole="image"
      accessibilityLabel={`${mode === 'plan' ? 'Plan' : 'Perspective'} view of ${tracks.length} hidden target hypotheses`}
      style={{ alignSelf: 'center', width: displayWidth, aspectRatio: CANVAS_WIDTH / CANVAS_HEIGHT }}
    >
      <Svg width="100%" height="100%" viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}>
        {mode === 'plan' ? <PlanScene /> : <PerspectiveScene />}
        {projectedTracks.map(({ track, x, y, radiusX, radiusY, velocityX, velocityY }) => {
          const color = resolveTrackColor(track, freshness);
          return (
            <React.Fragment key={track.trackId}>
              <Ellipse
                cx={x}
                cy={y}
                rx={radiusX}
                ry={radiusY}
                fill={`${color}18`}
                stroke={color}
                strokeDasharray="4 3"
              />
              <Line x1={x} y1={y} x2={x + velocityX} y2={y + velocityY} stroke={color} strokeWidth={2} />
              <Circle cx={x} cy={y} r={6 + track.confidence * 4} fill={color} stroke={instrumentColors.text} strokeWidth={1.5} />
              <Circle cx={x} cy={y} r={12 + track.confidence * 5} fill="none" stroke={`${color}55`} />
              <SvgText x={x + 12} y={y - 10} fill={instrumentColors.text} fontSize={10}>
                {track.trackId}
              </SvgText>
            </React.Fragment>
          );
        })}
      </Svg>
    </View>
  );
});

HiddenTargetVisualization.displayName = 'HiddenTargetVisualization';
