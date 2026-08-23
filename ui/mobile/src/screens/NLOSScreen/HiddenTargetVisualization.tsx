import React, { memo, useMemo } from 'react';
import { View } from 'react-native';
import Svg, { Circle, Ellipse, Line, Polygon, Rect, Text as SvgText } from 'react-native-svg';
import { colors } from '@/theme/colors';
import type { NlosFreshness, NlosTrack } from '@/types/nlos';

export type NlosViewMode = 'plan' | 'perspective';

interface HiddenTargetVisualizationProps {
  tracks: NlosTrack[];
  freshness: NlosFreshness;
  mode: NlosViewMode;
  width: number;
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
  if (freshness !== 'fresh' || track.state === 'unknown') return colors.muted;
  if (track.state === 'degraded') return colors.warn;
  return colors.accent;
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
    <Rect x={18} y={18} width={324} height={214} rx={10} fill={colors.surface} stroke={colors.border} />
    <Rect x={19} y={19} width={322} height={74} rx={9} fill="rgba(255, 165, 2, 0.07)" />
    <Line x1={24} y1={94} x2={336} y2={94} stroke={colors.warn} strokeWidth={4} />
    <SvgText x={28} y={84} fill={colors.warn} fontSize={10}>HIDDEN REGION</SvgText>
    <SvgText x={28} y={112} fill={colors.textSecondary} fontSize={10}>RELAY SURFACE</SvgText>
    <Circle cx={180} cy={218} r={5} fill={colors.accent} />
    <Line x1={180} y1={213} x2={180} y2={98} stroke={colors.accentDim} strokeDasharray="5 5" />
    <SvgText x={190} y={222} fill={colors.textSecondary} fontSize={9}>SENSOR</SvgText>
  </>
);

const PerspectiveScene = () => (
  <>
    <Polygon points="180,38 316,86 180,136 44,86" fill={colors.surface} stroke={colors.border} />
    <Polygon points="44,86 180,136 180,220 44,166" fill="rgba(26, 34, 51, 0.7)" stroke={colors.border} />
    <Polygon points="180,136 316,86 316,166 180,220" fill="rgba(17, 24, 39, 0.8)" stroke={colors.border} />
    <Polygon points="84,72 180,106 276,72 180,38" fill="rgba(255, 165, 2, 0.08)" />
    <Line x1={84} y1={72} x2={180} y2={106} stroke={colors.warn} strokeWidth={4} />
    <Line x1={180} y1={106} x2={276} y2={72} stroke={colors.warn} strokeWidth={4} />
    <SvgText x={119} y={62} fill={colors.warn} fontSize={10}>BEYOND RELAY PLANE</SvgText>
    <Circle cx={180} cy={205} r={5} fill={colors.accent} />
  </>
);

export const HiddenTargetVisualization = memo(({
  tracks,
  freshness,
  mode,
  width,
}: HiddenTargetVisualizationProps) => {
  const projectedTracks = useMemo(
    () => tracks.map(mode === 'plan' ? projectPlan : projectPerspective),
    [mode, tracks],
  );
  const displayWidth = Math.max(260, Math.min(width, 560));

  return (
    <View
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
              <Circle cx={x} cy={y} r={6 + track.confidence * 4} fill={color} stroke="#FFFFFF" strokeWidth={1.5} />
              <SvgText x={x + 12} y={y - 10} fill={colors.textPrimary} fontSize={10}>
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
