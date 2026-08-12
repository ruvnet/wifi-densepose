import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const TEST_DIR = dirname(fileURLToPath(import.meta.url));
const rendererSource = readFileSync(resolve(TEST_DIR, '../utils/pose-renderer.js'));
const rendererModuleUrl = `data:text/javascript;base64,${rendererSource.toString('base64')}`;
const { PoseRenderer, getRenderableKeypoints } = await import(rendererModuleUrl);

function createCanvas() {
  const calls = { fill: 0, stroke: 0 };
  const gradient = { addColorStop() {} };
  const context = {
    arc() {},
    beginPath() {},
    clearRect() {},
    createLinearGradient: () => gradient,
    createRadialGradient: () => gradient,
    fill() { calls.fill += 1; },
    fillRect() {},
    fillText() {},
    lineTo() {},
    moveTo() {},
    stroke() { calls.stroke += 1; },
    strokeRect() {},
  };
  const canvas = { width: 320, height: 240, getContext: () => context };
  return { calls, canvas };
}

test('a completely unscored set gets display-only confidence', () => {
  const source = [
    { x: 0.25, y: 0.25, confidence: 0 },
    { x: 0.75, y: 0.75, confidence: 0 },
  ];

  const renderable = getRenderableKeypoints(source);

  assert.ok(renderable.every((keypoint) => keypoint.confidence === 0.5));
  assert.ok(source.every((keypoint) => keypoint.confidence === 0));
});

test('a mixed scored set preserves rejected zero-confidence keypoints', () => {
  const source = [
    { x: 0.25, y: 0.25, confidence: 0 },
    { x: 0.75, y: 0.75, confidence: 0.8 },
  ];

  assert.equal(getRenderableKeypoints(source), source);
});

test('default skeleton mode draws zero-confidence coordinates', () => {
  const { calls, canvas } = createCanvas();
  const keypoints = Array.from({ length: 17 }, (_, index) => ({
    x: 80 + (index % 5) * 35,
    y: 40 + Math.floor(index / 5) * 50,
    confidence: 0,
  }));
  const renderer = new PoseRenderer(canvas, {
    enableSmoothing: false,
    showConfidence: false,
    showZones: false,
  });

  renderer.renderSkeletonMode({ persons: [{ confidence: 0.9, keypoints }] });

  assert.ok(calls.stroke > 0, 'expected visible skeleton connections');
  assert.equal(calls.fill, keypoints.length, 'expected every unscored keypoint to render');
  assert.ok(keypoints.every((keypoint) => keypoint.confidence === 0));
});
