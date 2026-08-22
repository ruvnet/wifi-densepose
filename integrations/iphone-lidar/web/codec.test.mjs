import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeLiDARPacket, depthToPointCloud } from './codec.mjs';

function b64(bytes) {
  return Buffer.from(bytes).toString('base64');
}

test('decodes u16 millimeter depth and confidence', () => {
  const packet = {
    type: 'ruview.lidar.depth.v1',
    intrinsics: { fx: 100, fy: 100, cx: 1, cy: 1, imageWidth: 2, imageHeight: 2 },
    pose: { matrix: Array(16).fill(0) },
    depth: {
      width: 2,
      height: 2,
      encoding: 'u16le-mm+u8-confidence',
      millimetersBase64: b64([0xe8,0x03,0xd0,0x07,0xb8,0x0b,0xa0,0x0f]),
      confidenceBase64: b64([2,2,1,0]),
    },
    provenance: { sensor: 'test', source: 'live', privacyClass: 'geometry-only', sequence: 1, timestampNs: 1, schema: 'ruview.lidar.depth.v1' },
  };

  const frame = decodeLiDARPacket(packet);
  assert.deepEqual(Array.from(frame.depth.meters), [1,2,3,4]);
  assert.deepEqual(Array.from(frame.depth.confidence), [2,2,1,0]);
});

test('rejects malformed payload length', () => {
  assert.throws(() => decodeLiDARPacket({
    type: 'ruview.lidar.depth.v1',
    intrinsics: { fx: 100, fy: 100, cx: 1, cy: 1, imageWidth: 2, imageHeight: 2 },
    pose: { matrix: Array(16).fill(0) },
    depth: { width: 2, height: 2, encoding: 'u16le-mm+u8-confidence', millimetersBase64: b64([1,2]), confidenceBase64: b64([1,1,1,1]) },
  }), /length mismatch/);
});

test('rejects invalid dimensions, intrinsics, pose, and base64', () => {
  const valid = {
    type: 'ruview.lidar.depth.v1',
    intrinsics: { fx: 100, fy: 100, cx: 0, cy: 0, imageWidth: 1, imageHeight: 1 },
    pose: { matrix: Array(16).fill(0) },
    depth: {
      width: 1,
      height: 1,
      encoding: 'u16le-mm+u8-confidence',
      millimetersBase64: b64([0xe8, 0x03]),
      confidenceBase64: b64([2]),
    },
  };

  assert.throws(() => decodeLiDARPacket({ ...valid, depth: { ...valid.depth, width: 0 } }), /positive integer/);
  assert.throws(() => decodeLiDARPacket({ ...valid, intrinsics: { ...valid.intrinsics, fx: 0 } }), /intrinsics/);
  assert.throws(() => decodeLiDARPacket({ ...valid, pose: { matrix: [1] } }), /pose/);
  assert.throws(() => decodeLiDARPacket({
    ...valid,
    depth: { ...valid.depth, millimetersBase64: '!!!!' },
  }), /canonical base64/);
});

test('projects depth into a point cloud and honors confidence', () => {
  const frame = {
    intrinsics: { fx: 100, fy: 100, cx: 0, cy: 0, imageWidth: 2, imageHeight: 2 },
    depth: { width: 2, height: 2, meters: Float32Array.from([1,1,1,1]), confidence: Uint8Array.from([2,0,2,0]) },
  };
  const points = depthToPointCloud(frame, 1);
  assert.equal(points.length, 2);
  assert.deepEqual(points[0], [0, 0, -1]);
});
