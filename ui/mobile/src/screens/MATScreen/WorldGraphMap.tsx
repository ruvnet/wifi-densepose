import { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Circle, G, Line, Polygon, Rect, Text as SvgText } from 'react-native-svg';
import { ThemedText } from '@/components/ThemedText';
import { instrumentColors } from '@/components/InstrumentPanel';
import type { ScanZone, Survivor, TriageStatus, ZoneBounds } from '@/types/mat';
import type { WorldBounds, WorldGraphSnapshot, WorldNode } from '@/types/worldGraph';

const WIDTH = 720;
const HEIGHT = 390;
const PAD = 34;
const TRIAGE: Record<TriageStatus, string> = {
  Immediate: '#FF6478', Delayed: '#FFB65C', Minor: '#26D968', Deceased: '#7B899D', Unknown: '#B98CFF',
};

type Point = [number, number];
type DrawableBounds =
  | { shape: 'rectangle'; minX: number; minY: number; maxX: number; maxY: number }
  | { shape: 'circle'; centerX: number; centerY: number; radius: number }
  | { shape: 'polygon'; vertices: Point[] };
const drawableBounds = (bounds: WorldBounds | ZoneBounds): DrawableBounds => {
  if ('shape' in bounds) {
    if (bounds.shape === 'rectangle') return { shape: 'rectangle', minX: bounds.min_e, minY: bounds.min_n, maxX: bounds.max_e, maxY: bounds.max_n };
    if (bounds.shape === 'circle') return { shape: 'circle', centerX: bounds.center_e, centerY: bounds.center_n, radius: bounds.radius_m };
    return { shape: 'polygon', vertices: bounds.vertices };
  }
  if (bounds.type === 'rectangle') return { shape: 'rectangle', minX: bounds.min_x, minY: bounds.min_y, maxX: bounds.max_x, maxY: bounds.max_y };
  if (bounds.type === 'circle') return { shape: 'circle', centerX: bounds.center_x, centerY: bounds.center_y, radius: bounds.radius };
  return { shape: 'polygon', vertices: bounds.vertices };
};
const boundsPoints = (bounds: WorldBounds): Point[] => {
  if (bounds.shape === 'rectangle') return [[bounds.min_e, bounds.min_n], [bounds.max_e, bounds.max_n]];
  if (bounds.shape === 'circle') return [[bounds.center_e - bounds.radius_m, bounds.center_n - bounds.radius_m], [bounds.center_e + bounds.radius_m, bounds.center_n + bounds.radius_m]];
  return bounds.vertices;
};
const matBoundsPoints = (bounds: ZoneBounds): Point[] => {
  if (bounds.type === 'rectangle') return [[bounds.min_x, bounds.min_y], [bounds.max_x, bounds.max_y]];
  if (bounds.type === 'circle') return [[bounds.center_x - bounds.radius, bounds.center_y - bounds.radius], [bounds.center_x + bounds.radius, bounds.center_y + bounds.radius]];
  return bounds.vertices;
};

export const WorldGraphMap = ({ graph, zones, survivors }: { graph: WorldGraphSnapshot | null; zones: ScanZone[]; survivors: Survivor[] }) => {
  const nodes = graph?.nodes ?? [];
  const rooms = nodes.filter((node): node is Extract<WorldNode, { kind: 'room' | 'zone' }> => node.kind === 'room' || node.kind === 'zone');
  const walls = nodes.filter((node): node is Extract<WorldNode, { kind: 'wall' }> => node.kind === 'wall');
  const sensors = nodes.filter((node): node is Extract<WorldNode, { kind: 'sensor' }> => node.kind === 'sensor');
  const tracks = nodes.filter((node): node is Extract<WorldNode, { kind: 'person_track' }> => node.kind === 'person_track');
  const transform = useMemo(() => {
    const points: Point[] = [
      ...rooms.flatMap((node) => boundsPoints(node.bounds_enu)),
      ...walls.flatMap((node) => [[node.a.east_m, node.a.north_m], [node.b.east_m, node.b.north_m]] as Point[]),
      ...sensors.map((node) => [node.position.east_m, node.position.north_m] as Point),
      ...tracks.map((node) => [node.last_position.east_m, node.last_position.north_m] as Point),
      ...zones.flatMap((zone) => matBoundsPoints(zone.bounds)),
      ...survivors.flatMap((survivor) => survivor.location ? [[survivor.location.x, survivor.location.y] as Point] : []),
    ];
    const xs = points.map(([x]) => x); const ys = points.map(([, y]) => y);
    const minX = xs.length ? Math.min(...xs) : 0; const maxX = xs.length ? Math.max(...xs) : 10;
    const minY = ys.length ? Math.min(...ys) : 0; const maxY = ys.length ? Math.max(...ys) : 10;
    const scale = Math.min((WIDTH - PAD * 2) / Math.max(maxX - minX, 1), (HEIGHT - PAD * 2) / Math.max(maxY - minY, 1));
    return { point: (x: number, y: number): Point => [PAD + (x - minX) * scale, HEIGHT - PAD - (y - minY) * scale], scale };
  }, [rooms, walls, sensors, tracks, zones, survivors]);

  const renderBounds = (key: string, bounds: WorldBounds | ZoneBounds, color: string, fill: string) => {
    const drawable = drawableBounds(bounds);
    if (drawable.shape === 'rectangle') {
      const [x1, y1] = transform.point(drawable.minX, drawable.minY); const [x2, y2] = transform.point(drawable.maxX, drawable.maxY);
      return <Rect key={key} x={Math.min(x1, x2)} y={Math.min(y1, y2)} width={Math.abs(x2 - x1)} height={Math.abs(y2 - y1)} rx={8} fill={fill} stroke={color} strokeWidth={2} />;
    }
    if (drawable.shape === 'circle') {
      const [cx, cy] = transform.point(drawable.centerX, drawable.centerY);
      return <Circle key={key} cx={cx} cy={cy} r={drawable.radius * transform.scale} fill={fill} stroke={color} strokeWidth={2} />;
    }
    const vertices = drawable.vertices.map(([x, y]) => transform.point(x, y).join(',')).join(' ');
    return <Polygon key={key} points={vertices} fill={fill} stroke={color} strokeWidth={2} />;
  };

  const empty = !graph && zones.length === 0 && survivors.length === 0;
  return (
    <View testID="worldgraph-map" style={styles.shell}>
      <Svg width="100%" height={230} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} accessibilityLabel="WorldGraph incident topology">
        <Rect x={0} y={0} width={WIDTH} height={HEIGHT} rx={16} fill="#090D13" />
        {Array.from({ length: 10 }, (_, i) => <Line key={`v${i}`} x1={(i + 1) * WIDTH / 11} y1={0} x2={(i + 1) * WIDTH / 11} y2={HEIGHT} stroke="rgba(25,212,230,.06)" />)}
        {Array.from({ length: 6 }, (_, i) => <Line key={`h${i}`} x1={0} y1={(i + 1) * HEIGHT / 7} x2={WIDTH} y2={(i + 1) * HEIGHT / 7} stroke="rgba(25,212,230,.06)" />)}
        {rooms.map((node) => <G key={node.id}>{renderBounds(`world-${node.id}`, node.bounds_enu, node.kind === 'room' ? '#19D4E6' : '#26D968', node.kind === 'room' ? 'rgba(25,212,230,.07)' : 'rgba(38,217,104,.08)')}</G>)}
        {zones.map((zone) => renderBounds(`mat-${zone.id}`, zone.bounds, '#FFB65C', 'rgba(255,182,92,.045)'))}
        {walls.map((wall) => { const a = transform.point(wall.a.east_m, wall.a.north_m); const b = transform.point(wall.b.east_m, wall.b.north_m); return <Line key={wall.id} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#A9B4C2" strokeWidth={4} />; })}
        {sensors.map((sensor) => { const p = transform.point(sensor.position.east_m, sensor.position.north_m); return <G key={sensor.id}><Circle cx={p[0]} cy={p[1]} r={9} fill="#19D4E6"/><Circle cx={p[0]} cy={p[1]} r={18} fill="none" stroke="rgba(25,212,230,.5)" /></G>; })}
        {tracks.map((track) => { const p = transform.point(track.last_position.east_m, track.last_position.north_m); return <G key={track.id}><Circle cx={p[0]} cy={p[1]} r={11} fill="#B98CFF"/><SvgText x={p[0] + 16} y={p[1] + 5} fill="#DCC7FF" fontSize={13}>T{track.track_id}</SvgText></G>; })}
        {survivors.flatMap((survivor) => survivor.location ? [(() => { const p = transform.point(survivor.location!.x, survivor.location!.y); const color = TRIAGE[survivor.triage_status]; return <G key={survivor.id}><Circle cx={p[0]} cy={p[1]} r={Math.max(12, survivor.location!.uncertainty_radius * transform.scale)} fill="none" stroke={color} strokeDasharray="5 5"/><Circle cx={p[0]} cy={p[1]} r={10} fill={color}/><SvgText x={p[0] + 15} y={p[1] + 5} fill={color} fontSize={13}>S-{survivor.id.slice(0, 4)}</SvgText></G>; })()] : [])}
        {empty && <SvgText x={WIDTH / 2} y={HEIGHT / 2} textAnchor="middle" fill="#7B899D" fontSize={20}>WAITING FOR VERIFIED TOPOLOGY</SvgText>}
      </Svg>
      <View style={styles.legend}>
        <LegendDot color={instrumentColors.cyan} label="SENSOR" />
        <LegendDot color="#B98CFF" label="ANONYMOUS TRACK" />
        <LegendDot color={instrumentColors.warning} label="MAT ZONE" />
        <LegendDot color={instrumentColors.danger} label="MAT SURVIVOR" />
      </View>
    </View>
  );
};

const LegendDot = ({ color, label }: { color: string; label: string }) => <View style={styles.legendItem}><View style={[styles.dot, { backgroundColor: color }]} /><ThemedText preset="mono" style={styles.legendText}>{label}</ThemedText></View>;
const styles = StyleSheet.create({
  shell: { overflow: 'hidden', borderRadius: 14, borderWidth: 1, borderColor: instrumentColors.borderStrong, backgroundColor: '#090D13' },
  legend: { padding: 10, flexDirection: 'row', flexWrap: 'wrap', gap: 12, borderTopWidth: 1, borderTopColor: instrumentColors.border },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 5 }, dot: { width: 7, height: 7, borderRadius: 4 },
  legendText: { color: instrumentColors.textSecondary, fontSize: 8, letterSpacing: .7 },
});
