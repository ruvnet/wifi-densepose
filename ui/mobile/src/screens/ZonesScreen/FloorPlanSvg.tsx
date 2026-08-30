import { useEffect, useMemo } from 'react';
import { View, ViewStyle } from 'react-native';
import Svg, { Circle, G, Line, Rect } from 'react-native-svg';
import Animated, {
  createAnimatedComponent,
  interpolateColor,
  useAnimatedProps,
  useAnimatedStyle,
  useDerivedValue,
  useSharedValue,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';
import {
  Gesture,
  GestureDetector,
} from 'react-native-gesture-handler';
import { spacing } from '@/theme/spacing';
import { instrumentColors } from '@/components/InstrumentPanel';

const GRID_SIZE = 20;
const CELL_COUNT = GRID_SIZE * GRID_SIZE;

type Point = {
  x: number;
  y: number;
};

type FloorPlanSvgProps = {
  gridValues: number[];
  personPositions: Point[];
  size?: number;
  style?: ViewStyle;
};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

const normalizeGrid = (values: number[]): number[] => {
  const normalized = new Array(CELL_COUNT).fill(0);
  const sourceLength = Math.min(values.length, CELL_COUNT);

  for (let i = 0; i < sourceLength; i += 1) {
    const raw = values?.[i];
    normalized[i] = clamp01(typeof raw === 'number' && Number.isFinite(raw) ? raw : 0);
  }

  return normalized;
};

const AnimatedRect = createAnimatedComponent(Rect);

const AnimatedContainer = Animated.View;

const Cell = ({
  index,
  size,
  values,
  progress,
}: {
  index: number;
  size: number;
  values: SharedValue<number[]>;
  progress: SharedValue<number>;
}) => {
  const cellSize = size / GRID_SIZE;
  const x = (index % GRID_SIZE) * cellSize;
  const y = Math.floor(index / GRID_SIZE) * cellSize;

  const animatedProps = useAnimatedProps(() => {
    const intensity = values.value[index] ?? 0;
    const fill = interpolateColor(
      intensity,
      [0, 0.5, 1],
      ['rgb(7, 21, 27)', 'rgb(25, 212, 230)', 'rgb(38, 217, 104)'],
    );
    return {
      fill,
      opacity: 0.05 + intensity * 0.88 + (progress.value - 1) * 0.03,
    };
  }, [index]);

  return <AnimatedRect x={x} y={y} width={cellSize} height={cellSize} rx={1} animatedProps={animatedProps} />;
};

export const FloorPlanSvg = ({ gridValues, personPositions, size = 320, style }: FloorPlanSvgProps) => {
  const normalizedValues = useMemo(() => normalizeGrid(gridValues), [gridValues]);

  const values = useSharedValue(normalizedValues);
  const previousValues = useSharedValue(normalizedValues);
  const targetValues = useSharedValue(normalizedValues);
  const progress = useSharedValue(1);

  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const panStartX = useSharedValue(0);
  const panStartY = useSharedValue(0);

  const panGesture = Gesture.Pan()
    .onStart(() => {
      panStartX.value = translateX.value;
      panStartY.value = translateY.value;
    })
    .onUpdate((event) => {
      translateX.value = panStartX.value + event.translationX;
      translateY.value = panStartY.value + event.translationY;
    })
    .onEnd(() => {
      panStartX.value = translateX.value;
      panStartY.value = translateY.value;
    });

  const panStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
    ],
  }));

  useDerivedValue(() => {
    const interpolated = new Array(CELL_COUNT).fill(0);
    const from = previousValues.value;
    const to = targetValues.value;
    const p = progress.value;

    for (let i = 0; i < CELL_COUNT; i += 1) {
      const start = from[i] ?? 0;
      const end = to[i] ?? 0;
      interpolated[i] = start + (end - start) * p;
    }
    values.value = interpolated;
  });

  useEffect(() => {
    const next = normalizeGrid(normalizedValues);
    previousValues.value = values.value;
    targetValues.value = next;
    progress.value = 0;
    progress.value = withTiming(1, { duration: 500 });
  }, [normalizedValues, previousValues, targetValues, progress, values]);

  const markers = useMemo(() => {
    const cellSize = size / GRID_SIZE;
    return personPositions
      .map((point, idx) => {
        const cx = (Math.max(0, Math.min(GRID_SIZE - 1, point.x)) + 0.5) * cellSize;
        const cy = (Math.max(0, Math.min(GRID_SIZE - 1, point.y)) + 0.5) * cellSize;
        const radius = Math.max(2.8, cellSize * 0.22);

        return <G key={`person-${idx}`}><Circle cx={cx} cy={cy} r={radius * 3.4} fill="rgba(38,217,104,.08)" stroke="rgba(38,217,104,.42)" strokeWidth={1} strokeDasharray="3 3" /><Circle cx={cx} cy={cy} r={radius} fill={instrumentColors.green} stroke="#E7EBEF" strokeWidth={1.8} /></G>;
      });
  }, [personPositions, size]);

  return (
    <View style={[{ overflow: 'hidden', paddingBottom: spacing.xs }, style]}>
      <GestureDetector gesture={panGesture}>
        <AnimatedContainer style={panStyle}>
          <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
            <Rect x={0} y={0} width={size} height={size} rx={12} fill="#070B10" />
            {Array.from({ length: 9 }, (_, index) => { const offset = ((index + 1) * size) / 10; return <Line key={`grid-v-${index}`} x1={offset} y1={0} x2={offset} y2={size} stroke="rgba(25,212,230,.07)" strokeWidth={1} />; })}
            {Array.from({ length: 9 }, (_, index) => { const offset = ((index + 1) * size) / 10; return <Line key={`grid-h-${index}`} x1={0} y1={offset} x2={size} y2={offset} stroke="rgba(25,212,230,.07)" strokeWidth={1} />; })}
            {Array.from({ length: CELL_COUNT }).map((_, index) => (
              <Cell
                key={`cell-${index}`}
                index={index}
                size={size}
                values={values}
                progress={progress}
              />
            ))}
            {markers}
          </Svg>
        </AnimatedContainer>
      </GestureDetector>
    </View>
  );
};
