import { useMemo } from 'react';
import type { PersonDetection, SignalField } from '@/types/sensing';

const GRID_SIZE = 20;
const CELL_COUNT = GRID_SIZE * GRID_SIZE;

type Point = {
  x: number;
  y: number;
};

const clamp01 = (value: number): number => {
  if (Number.isNaN(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
};

const parseNumber = (value: unknown): number | null => {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

export const mapWorldPositionsToField = (persons: PersonDetection[] | undefined): Point[] => (
  persons?.flatMap((person) => {
    if (!person.position) return [];
    const [worldX, , worldZ] = person.position;
    if (!Number.isFinite(worldX) || !Number.isFinite(worldZ)) return [];
    // Inverse of the sensing server's field-cell → room-coordinate transform.
    return [{ x: worldX / 0.6 + GRID_SIZE / 2, y: worldZ / 0.5 + GRID_SIZE / 2 }];
  }) ?? []
);

export const deriveFieldPositions = (values: number[], requestedCount: number): Point[] => {
  const count = Math.max(0, Math.min(16, Math.floor(requestedCount)));
  if (count === 0 || values.length === 0) return [];

  const candidates = values
    .slice(0, CELL_COUNT)
    .map((value, index) => ({
      x: index % GRID_SIZE,
      y: Math.floor(index / GRID_SIZE),
      value: parseNumber(value) ?? 0,
    }))
    .sort((a, b) => b.value - a.value);

  const positions: Point[] = [];
  for (const candidate of candidates) {
    const separated = positions.every(({ x, y }) => {
      const dx = candidate.x - x;
      const dy = candidate.y - y;
      return dx * dx + dy * dy >= 9;
    });
    if (separated) positions.push({ x: candidate.x, y: candidate.y });
    if (positions.length === count) break;
  }
  return positions;
};

export const useOccupancyGrid = (
  signalField: SignalField | null,
  estimatedPersonCount = 0,
  persons?: PersonDetection[],
): { gridValues: number[]; personPositions: Point[] } => {
  const gridValues = useMemo(() => {
    const sourceValues = signalField?.values;

    if (!sourceValues || sourceValues.length === 0) {
      return new Array(CELL_COUNT).fill(0);
    }

    const normalized = new Array(CELL_COUNT).fill(0);
    const sourceLength = Math.min(CELL_COUNT, sourceValues.length);

    for (let i = 0; i < sourceLength; i += 1) {
      const value = parseNumber(sourceValues[i]);
      normalized[i] = clamp01(value ?? 0);
    }

    return normalized;
  }, [signalField?.values]);

  const personPositions = useMemo(() => {
    const positions = mapWorldPositionsToField(persons);

    if (positions.length > 0) {
      return positions
        .map(({ x, y }) => ({
          x: Math.max(0, Math.min(GRID_SIZE - 1, x)),
          y: Math.max(0, Math.min(GRID_SIZE - 1, y)),
        }))
        .slice(0, 16);
    }

    return deriveFieldPositions(gridValues, estimatedPersonCount);
  }, [estimatedPersonCount, gridValues, persons]);

  return {
    gridValues,
    personPositions,
  };
};
