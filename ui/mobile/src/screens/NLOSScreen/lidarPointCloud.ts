import type { NlosTrack } from '@/types/nlos';

export const LIDAR_RELAY_POINT_COUNT = 288;
export const LIDAR_POINTS_PER_TRACK = 96;
export const LIDAR_MAX_TRACKS = 16;

export interface LidarPointCloudData {
  positions: Float32Array;
  colors: Float32Array;
  relayPointCount: number;
  targetPointCount: number;
  totalPointCount: number;
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

const clamp = (value: number, minimum: number, maximum: number) => (
  Math.max(minimum, Math.min(maximum, value))
);

const writePoint = (
  positions: Float32Array,
  colors: Float32Array,
  index: number,
  point: readonly [number, number, number],
  color: readonly [number, number, number],
) => {
  const offset = index * 3;
  positions[offset] = point[0];
  positions[offset + 1] = point[1];
  positions[offset + 2] = point[2];
  colors[offset] = color[0];
  colors[offset + 1] = color[1];
  colors[offset + 2] = color[2];
};

export const buildLidarPointCloud = (tracks: readonly NlosTrack[]): LidarPointCloudData => {
  const visibleTracks = tracks.slice(0, LIDAR_MAX_TRACKS);
  const targetPointCount = visibleTracks.length * LIDAR_POINTS_PER_TRACK;
  const totalPointCount = LIDAR_RELAY_POINT_COUNT + targetPointCount;
  const positions = new Float32Array(totalPointCount * 3);
  const colors = new Float32Array(totalPointCount * 3);
  let pointIndex = 0;

  for (let row = 0; row < 12; row += 1) {
    for (let column = 0; column < 12; column += 1) {
      const x = -3.85 + column * 0.7;
      const z = 2.2 - row * 0.55;
      const scanRipple = Math.sin(column * 1.7 + row * 0.8) * 0.025;
      writePoint(
        positions,
        colors,
        pointIndex,
        [x, scanRipple, z],
        [0.08, 0.52 + row * 0.008, 0.61 + column * 0.006],
      );
      pointIndex += 1;
    }
  }

  for (let row = 0; row < 12; row += 1) {
    for (let column = 0; column < 12; column += 1) {
      const x = -3.85 + column * 0.7;
      const y = 0.15 + row * 0.25;
      const relayRipple = Math.cos(column * 1.3 + row * 0.9) * 0.018;
      const edgeMix = row > 9 ? 0.42 : 0;
      writePoint(
        positions,
        colors,
        pointIndex,
        [x, y, 0.72 + relayRipple],
        [0.12 + edgeMix, 0.58 + edgeMix * 0.42, 0.66 - edgeMix * 0.3],
      );
      pointIndex += 1;
    }
  }

  visibleTracks.forEach((track, trackIndex) => {
    const uncertaintyX = clamp(Math.sqrt(track.covarianceDiagonalM2.x), 0.08, 0.85);
    const uncertaintyY = clamp(Math.sqrt(track.covarianceDiagonalM2.y), 0.08, 0.85);
    const uncertaintyZ = clamp(Math.sqrt(track.covarianceDiagonalM2.z), 0.08, 0.85);
    const centerX = clamp(track.positionM.x, -6, 6) * 0.55;
    const centerY = 0.45 + clamp(track.positionM.y, 0, 4) * 0.5;
    const centerZ = 0.35 - clamp(track.positionM.z, 0, 8) * 0.58;
    const degraded = track.state === 'degraded';
    const intensity = 0.58 + clamp(track.confidence, 0, 1) * 0.42;

    for (let sample = 0; sample < LIDAR_POINTS_PER_TRACK; sample += 1) {
      const normalizedY = 1 - 2 * ((sample + 0.5) / LIDAR_POINTS_PER_TRACK);
      const radial = Math.sqrt(Math.max(0, 1 - normalizedY * normalizedY));
      const theta = sample * GOLDEN_ANGLE + trackIndex * 0.73;
      const shell = 0.5 + ((sample * 37 + trackIndex * 17) % 47) / 94;
      const directionX = Math.cos(theta) * radial;
      const directionZ = Math.sin(theta) * radial;

      writePoint(
        positions,
        colors,
        pointIndex,
        [
          centerX + directionX * uncertaintyX * shell,
          centerY + normalizedY * uncertaintyY * shell,
          centerZ + directionZ * uncertaintyZ * shell,
        ],
        degraded
          ? [intensity, 0.52 * intensity, 0.16]
          : [0.22 * intensity, intensity, 0.48 * intensity],
      );
      pointIndex += 1;
    }
  });

  return {
    positions,
    colors,
    relayPointCount: LIDAR_RELAY_POINT_COUNT,
    targetPointCount,
    totalPointCount,
  };
};
