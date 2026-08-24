import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import * as THREE from 'three';
import { instrumentColors } from '@/components/InstrumentPanel';
import { ThemedText } from '@/components/ThemedText';
import type { NlosFreshness, NlosTrack } from '@/types/nlos';
import {
  buildLidarPointCloud,
  LIDAR_MAX_TRACKS,
  resolveLidarTrackCenter,
} from './lidarPointCloudData';

interface LidarPointCloudProps {
  tracks: NlosTrack[];
  freshness: NlosFreshness;
  width: number;
}

type RendererState = 'initializing' | 'ready' | 'fallback';

const CANVAS_ASPECT_RATIO = 360 / 260;

interface WebSceneState {
  geometry: THREE.BufferGeometry;
  material: THREE.PointsMaterial;
  markerGroup: THREE.Group;
  markerSignature: string;
  points: THREE.Points;
  render: () => void;
}

const disposeMaterial = (material: THREE.Material | THREE.Material[]) => {
  if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
  else material.dispose();
};

const clearGroup = (group: THREE.Group) => {
  group.traverse((entry) => {
    if (entry instanceof THREE.Mesh || entry instanceof THREE.Line) {
      entry.geometry.dispose();
      disposeMaterial(entry.material);
    }
  });
  group.clear();
};

const rgb = (red: number, green: number, blue: number) => (
  `rgb(${Math.round(red * 255)}, ${Math.round(green * 255)}, ${Math.round(blue * 255)})`
);

export const LidarPointCloud = memo(({ tracks, freshness, width }: LidarPointCloudProps) => {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<WebSceneState | null>(null);
  const [rendererState, setRendererState] = useState<RendererState>('initializing');
  const cloud = useMemo(() => buildLidarPointCloud(tracks), [tracks]);
  const displayWidth = Math.max(260, Math.min(width, 560));
  const fallbackPoints = useMemo(() => {
    const height = displayWidth / CANVAS_ASPECT_RATIO;
    const points: Array<{ color: string; left: number; top: number }> = [];
    for (let index = 0; index < cloud.totalPointCount; index += 1) {
      const isRelay = index < cloud.relayPointCount;
      if ((isRelay && index % 3 !== 0) || (!isRelay && index % 2 !== 0)) continue;
      const offset = index * 3;
      const x = cloud.positions[offset];
      const y = cloud.positions[offset + 1];
      const z = cloud.positions[offset + 2];
      points.push({
        left: ((180 + x * 34 - z * 8) / 360) * displayWidth,
        top: ((220 - y * 46 + z * 7) / 260) * height,
        color: rgb(cloud.colors[offset], cloud.colors[offset + 1], cloud.colors[offset + 2]),
      });
    }
    return points;
  }, [cloud, displayWidth]);
  const assignHost = useCallback((node: unknown) => {
    hostRef.current = node as HTMLDivElement | null;
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;

    let animationFrame = 0;
    let resizeObserver: ResizeObserver | null = null;
    let disposed = false;
    let renderer: THREE.WebGLRenderer | null = null;
    const listeners: Array<() => void> = [];

    try {
      renderer = new THREE.WebGLRenderer({
        alpha: false,
        antialias: false,
        powerPreference: 'high-performance',
      });
      renderer.setClearColor(0x071017, 1);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.domElement.dataset.testid = 'nlos-lidar-point-cloud-canvas';
      renderer.domElement.dataset.ready = 'false';
      renderer.domElement.setAttribute('aria-hidden', 'true');
      renderer.domElement.style.position = 'absolute';
      renderer.domElement.style.inset = '0';
      renderer.domElement.style.zIndex = '0';
      renderer.domElement.style.touchAction = 'none';
      host.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x071017);
      scene.fog = new THREE.FogExp2(0x071017, 0.055);
      const camera = new THREE.PerspectiveCamera(42, CANVAS_ASPECT_RATIO, 0.1, 40);
      let yaw = 0.55;
      let pitch = 0.34;
      let radius = 8.6;
      let dragging = false;
      let pointerX = 0;
      let pointerY = 0;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(0), 3));
      const material = new THREE.PointsMaterial({
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        opacity: 0.92,
        size: 0.065,
        sizeAttenuation: true,
        transparent: true,
        vertexColors: true,
      });
      const points = new THREE.Points(geometry, material);
      scene.add(points);

      const markerGroup = new THREE.Group();
      scene.add(markerGroup);

      const floorGrid = new THREE.GridHelper(8, 16, 0x174957, 0x102934);
      floorGrid.position.y = -0.04;
      floorGrid.position.z = -1.25;
      scene.add(floorGrid);

      const relayPlane = new THREE.GridHelper(8, 16, 0xffb65c, 0x16414b);
      relayPlane.rotation.x = Math.PI / 2;
      relayPlane.position.set(0, 1.55, 0.72);
      scene.add(relayPlane);

      const sensor = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.11, 0),
        new THREE.MeshBasicMaterial({ color: 0x24d3e5, wireframe: true }),
      );
      sensor.position.set(0, 0.18, 2.45);
      scene.add(sensor);

      const rayGeometry = new THREE.BufferGeometry().setFromPoints([
        sensor.position,
        new THREE.Vector3(0, 1.2, 0.72),
        new THREE.Vector3(0.9, 1.1, -2.1),
      ]);
      const rayMaterial = new THREE.LineBasicMaterial({ color: 0x58f28b, opacity: 0.42, transparent: true });
      const ray = new THREE.Line(rayGeometry, rayMaterial);
      scene.add(ray);

      const sizeRenderer = () => {
        if (!renderer || disposed) return;
        const nextWidth = Math.max(1, host.clientWidth || 360);
        const nextHeight = Math.max(1, host.clientHeight || 260);
        camera.aspect = nextWidth / nextHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(nextWidth, nextHeight, false);
      };
      const renderScene = () => {
        if (!renderer || disposed) return;
        const horizontalRadius = radius * Math.cos(pitch);
        camera.position.set(
          Math.sin(yaw) * horizontalRadius,
          1.25 + Math.sin(pitch) * radius,
          Math.cos(yaw) * horizontalRadius - 0.65,
        );
        camera.lookAt(0, 1.15, -1.05);
        renderer.render(scene, camera);
      };

      sceneRef.current = {
        geometry,
        markerGroup,
        markerSignature: '',
        material,
        points,
        render: renderScene,
      };

      const onPointerDown = (event: PointerEvent) => {
        dragging = true;
        pointerX = event.clientX;
        pointerY = event.clientY;
        renderer?.domElement.setPointerCapture?.(event.pointerId);
      };
      const onPointerMove = (event: PointerEvent) => {
        if (!dragging) return;
        yaw += (event.clientX - pointerX) * 0.008;
        pitch = THREE.MathUtils.clamp(pitch + (event.clientY - pointerY) * 0.006, -0.1, 0.85);
        pointerX = event.clientX;
        pointerY = event.clientY;
        renderScene();
      };
      const onPointerUp = (event: PointerEvent) => {
        dragging = false;
        renderer?.domElement.releasePointerCapture?.(event.pointerId);
      };
      const onWheel = (event: WheelEvent) => {
        event.preventDefault();
        radius = THREE.MathUtils.clamp(radius + event.deltaY * 0.008, 4.8, 13);
        renderScene();
      };
      const onContextLost = (event: Event) => {
        event.preventDefault();
        setRendererState('fallback');
      };
      const onContextRestored = () => {
        setRendererState('ready');
        renderScene();
      };

      const canvas = renderer.domElement;
      canvas.addEventListener('pointerdown', onPointerDown);
      canvas.addEventListener('pointermove', onPointerMove);
      canvas.addEventListener('pointerup', onPointerUp);
      canvas.addEventListener('pointercancel', onPointerUp);
      canvas.addEventListener('wheel', onWheel, { passive: false });
      canvas.addEventListener('webglcontextlost', onContextLost);
      canvas.addEventListener('webglcontextrestored', onContextRestored);
      listeners.push(
        () => canvas.removeEventListener('pointerdown', onPointerDown),
        () => canvas.removeEventListener('pointermove', onPointerMove),
        () => canvas.removeEventListener('pointerup', onPointerUp),
        () => canvas.removeEventListener('pointercancel', onPointerUp),
        () => canvas.removeEventListener('wheel', onWheel),
        () => canvas.removeEventListener('webglcontextlost', onContextLost),
        () => canvas.removeEventListener('webglcontextrestored', onContextRestored),
      );

      sizeRenderer();
      renderScene();
      canvas.dataset.ready = 'true';
      setRendererState('ready');

      if (typeof ResizeObserver !== 'undefined') {
        resizeObserver = new ResizeObserver(() => {
          sizeRenderer();
          renderScene();
        });
        resizeObserver.observe(host);
      } else {
        window.addEventListener('resize', sizeRenderer);
        listeners.push(() => window.removeEventListener('resize', sizeRenderer));
      }

      const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
      if (!reduceMotion) {
        const animate = () => {
          if (disposed) return;
          if (!dragging) yaw += 0.0009;
          renderScene();
          animationFrame = window.requestAnimationFrame(animate);
        };
        animationFrame = window.requestAnimationFrame(animate);
      }

      return () => {
        disposed = true;
        sceneRef.current = null;
        window.cancelAnimationFrame(animationFrame);
        resizeObserver?.disconnect();
        listeners.forEach((remove) => remove());
        scene.remove(points, markerGroup, floorGrid, relayPlane, sensor, ray);
        clearGroup(markerGroup);
        geometry.dispose();
        material.dispose();
        floorGrid.geometry.dispose();
        disposeMaterial(floorGrid.material);
        relayPlane.geometry.dispose();
        disposeMaterial(relayPlane.material);
        sensor.geometry.dispose();
        disposeMaterial(sensor.material);
        rayGeometry.dispose();
        rayMaterial.dispose();
        renderer?.dispose();
        if (canvas.parentNode === host) host.removeChild(canvas);
      };
    } catch {
      sceneRef.current = null;
      setRendererState('fallback');
      return () => {
        disposed = true;
        window.cancelAnimationFrame(animationFrame);
        resizeObserver?.disconnect();
        listeners.forEach((remove) => remove());
        renderer?.dispose();
        const canvas = renderer?.domElement;
        if (canvas?.parentNode === host) host.removeChild(canvas);
      };
    }
  }, []);

  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;

    const currentPosition = state.geometry.getAttribute('position') as THREE.BufferAttribute;
    if (currentPosition.count === cloud.totalPointCount) {
      (currentPosition.array as Float32Array).set(cloud.positions);
      currentPosition.needsUpdate = true;
      const currentColor = state.geometry.getAttribute('color') as THREE.BufferAttribute;
      (currentColor.array as Float32Array).set(cloud.colors);
      currentColor.needsUpdate = true;
    } else {
      const nextGeometry = new THREE.BufferGeometry();
      nextGeometry.setAttribute('position', new THREE.BufferAttribute(cloud.positions.slice(), 3));
      nextGeometry.setAttribute('color', new THREE.BufferAttribute(cloud.colors.slice(), 3));
      state.geometry.dispose();
      state.geometry = nextGeometry;
      state.points.geometry = nextGeometry;
    }
    state.geometry.computeBoundingSphere();
    state.material.opacity = freshness === 'fresh' ? 0.92 : 0.34;
    state.material.needsUpdate = true;

    const markerTracks = tracks.slice(0, LIDAR_MAX_TRACKS);
    const markerSignature = markerTracks.map((track) => track.state).join('|');
    if (markerSignature !== state.markerSignature) {
      clearGroup(state.markerGroup);
      state.markerSignature = markerSignature;
      markerTracks.forEach((track) => {
        const degraded = track.state === 'degraded';
        const color = degraded ? 0xffb65c : 0x58f28b;
        const opacity = freshness === 'fresh' ? 0.82 : 0.28;
        const horizontalRing = new THREE.Mesh(
          new THREE.TorusGeometry(0.31, 0.016, 5, 40),
          new THREE.MeshBasicMaterial({ color, opacity, transparent: true }),
        );
        horizontalRing.rotation.x = Math.PI / 2;
        state.markerGroup.add(horizontalRing);

        const core = new THREE.Mesh(
          new THREE.OctahedronGeometry(0.08, 0),
          new THREE.MeshBasicMaterial({ color, wireframe: true }),
        );
        state.markerGroup.add(core);

        const pillarGeometry = new THREE.BufferGeometry();
        pillarGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
        const pillar = new THREE.Line(
          pillarGeometry,
          new THREE.LineBasicMaterial({ color, opacity: opacity * 0.62, transparent: true }),
        );
        state.markerGroup.add(pillar);
      });
    }

    markerTracks.forEach((track, index) => {
      const [x, y, z] = resolveLidarTrackCenter(track);
      const degraded = track.state === 'degraded';
      const opacity = freshness === 'fresh' ? 0.82 : 0.28;
      const horizontalRing = state.markerGroup.children[index * 3] as THREE.Mesh;
      horizontalRing.position.set(x, y, z);
      (horizontalRing.material as THREE.MeshBasicMaterial).opacity = opacity;

      const core = state.markerGroup.children[index * 3 + 1] as THREE.Mesh;
      core.position.set(x, y, z);
      (core.material as THREE.MeshBasicMaterial).opacity = degraded ? 0.82 : 1;
      (core.material as THREE.MeshBasicMaterial).transparent = degraded;

      const pillar = state.markerGroup.children[index * 3 + 2] as THREE.Line;
      const pillarPosition = pillar.geometry.getAttribute('position') as THREE.BufferAttribute;
      (pillarPosition.array as Float32Array).set([
        x, Math.max(0.02, y - 0.52), z,
        x, y + 0.52, z,
      ]);
      pillarPosition.needsUpdate = true;
      (pillar.material as THREE.LineBasicMaterial).opacity = opacity * 0.62;
    });
    state.render();
  }, [cloud, freshness, tracks]);

  return (
    <View
      ref={assignHost}
      testID="nlos-lidar-point-cloud"
      accessibilityRole="image"
      accessibilityLabel={`Interactive Three.js LiDAR reconstruction cloud with ${cloud.targetPointCount} gated target returns from ${tracks.length} hidden target hypotheses. ${freshness} evidence.`}
      style={[styles.host, { width: displayWidth, aspectRatio: CANVAS_ASPECT_RATIO }]}
    >
      {rendererState === 'fallback' ? (
        <View pointerEvents="none" style={styles.fallbackLayer}>
          {fallbackPoints.map((point, index) => (
            <View
              key={`${index}-${point.left}-${point.top}`}
              style={[
                styles.fallbackPoint,
                { backgroundColor: point.color, left: point.left, top: point.top },
              ]}
            />
          ))}
        </View>
      ) : null}
      <View pointerEvents="none" style={styles.topHud}>
        <View style={styles.titleStack}>
          <ThemedText preset="mono" style={styles.hudTitle}>LIDAR POINT CLOUD</ThemedText>
          <ThemedText testID="nlos-cloud-boundary-label" preset="mono" style={styles.boundaryLabel}>
            RECONSTRUCTION / NOT RAW SCAN
          </ThemedText>
        </View>
        <View style={styles.rendererStack}>
          <ThemedText preset="mono" style={styles.rendererLabel}>
            {rendererState === 'fallback' ? 'STATIC FALLBACK' : 'THREE.JS / WEBGL'}
          </ThemedText>
          <ThemedText preset="mono" style={styles.lockLabel}>
            {tracks.length ? `${String(tracks.length).padStart(2, '0')} TRACK LOCK` : 'NO TARGET LOCK'}
          </ThemedText>
        </View>
      </View>
      <View pointerEvents="none" style={styles.depthScale}>
        <ThemedText preset="mono" style={styles.depthLabel}>RELAY</ThemedText>
        <View style={styles.depthLine} />
        <ThemedText preset="mono" style={styles.depthLabel}>HIDDEN</ThemedText>
      </View>
      {tracks[0] ? (
        <View pointerEvents="none" style={styles.trackChip}>
          <View style={styles.trackChipDot} />
          <View>
            <ThemedText preset="mono" style={styles.trackChipId}>
              {tracks[0].trackId.toUpperCase()}
            </ThemedText>
            <ThemedText preset="mono" style={styles.trackChipMeta}>
              {Math.round(tracks[0].confidence * 100)}% CONFIDENCE
            </ThemedText>
          </View>
        </View>
      ) : null}
      <View pointerEvents="none" style={styles.bottomHud}>
        <View>
          <ThemedText testID="nlos-cloud-target-count" preset="labelLg" style={styles.metricValue}>
            {cloud.targetPointCount}
          </ThemedText>
          <ThemedText preset="mono" style={styles.metricLabel}>GATED TARGET RETURNS</ThemedText>
        </View>
        <ThemedText preset="mono" style={styles.gestureHint}>
          {rendererState === 'ready' ? 'DRAG TO ORBIT' : 'WEBGL UNAVAILABLE'}
        </ThemedText>
      </View>
    </View>
  );
});

LidarPointCloud.displayName = 'LidarPointCloud';

const styles = StyleSheet.create({
  host: {
    position: 'relative',
    alignSelf: 'center',
    overflow: 'hidden',
    backgroundColor: '#071017',
    borderColor: instrumentColors.border,
    borderWidth: 1,
    borderRadius: 14,
  },
  fallbackLayer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 1,
  },
  fallbackPoint: {
    position: 'absolute',
    width: 2,
    height: 2,
    borderRadius: 1,
  },
  topHud: {
    position: 'absolute',
    top: 12,
    left: 12,
    right: 12,
    zIndex: 2,
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 8,
  },
  titleStack: { gap: 3 },
  hudTitle: { color: instrumentColors.cyan, fontSize: 9, letterSpacing: 1.15 },
  boundaryLabel: { color: instrumentColors.warning, fontSize: 6.5, letterSpacing: 0.7 },
  rendererStack: { alignItems: 'flex-end', gap: 3 },
  rendererLabel: { color: instrumentColors.textSecondary, fontSize: 8, letterSpacing: 0.8 },
  lockLabel: { color: instrumentColors.green, fontSize: 7, letterSpacing: 0.7 },
  depthScale: {
    position: 'absolute',
    left: 12,
    top: 70,
    bottom: 54,
    zIndex: 2,
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  depthLine: { flex: 1, width: 1, marginVertical: 5, backgroundColor: instrumentColors.cyanDim },
  depthLabel: { color: instrumentColors.textSecondary, fontSize: 6, letterSpacing: 0.55 },
  trackChip: {
    position: 'absolute',
    top: 54,
    right: 12,
    zIndex: 2,
    minHeight: 34,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 9,
    paddingVertical: 6,
    borderRadius: 8,
    borderColor: instrumentColors.greenDim,
    borderWidth: 1,
    backgroundColor: 'rgba(5, 14, 18, 0.76)',
  },
  trackChipDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: instrumentColors.green },
  trackChipId: { color: instrumentColors.text, fontSize: 7, letterSpacing: 0.65 },
  trackChipMeta: { color: instrumentColors.green, fontSize: 6.5, letterSpacing: 0.5 },
  bottomHud: {
    position: 'absolute',
    left: 12,
    right: 12,
    bottom: 10,
    zIndex: 2,
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    gap: 8,
  },
  metricValue: { color: instrumentColors.green, fontSize: 16, lineHeight: 18 },
  metricLabel: { color: instrumentColors.textSecondary, fontSize: 7, letterSpacing: 0.7 },
  gestureHint: { color: instrumentColors.textSecondary, fontSize: 8, letterSpacing: 0.8 },
});
