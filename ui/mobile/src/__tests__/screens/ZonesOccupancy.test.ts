import { deriveFieldPositions } from '@/screens/ZonesScreen/useOccupancyGrid';

describe('Zones hardware occupancy mapping', () => {
  it('places an estimated person at the strongest field cell', () => {
    const values = new Array(400).fill(0);
    values[7 * 20 + 12] = 1;
    expect(deriveFieldPositions(values, 1)).toEqual([{ x: 12, y: 7 }]);
  });

  it('keeps multiple estimated people spatially separated', () => {
    const values = new Array(400).fill(0);
    values[2 * 20 + 2] = 1;
    values[15 * 20 + 16] = 0.9;
    const positions = deriveFieldPositions(values, 2);
    expect(positions).toHaveLength(2);
    expect(positions).toContainEqual({ x: 2, y: 2 });
    expect(positions).toContainEqual({ x: 16, y: 15 });
  });
});
