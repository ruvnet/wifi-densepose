export function decodeLiDARPacket(packet) {
  if (!packet || packet.type !== 'ruview.lidar.depth.v1') {
    throw new Error('Unsupported LiDAR packet type');
  }

  const { depth } = packet;
  if (!depth || depth.encoding !== 'u16le-mm+u8-confidence') {
    throw new Error('Unsupported depth encoding');
  }

  assertPositiveInteger(depth.width, 'depth.width');
  assertPositiveInteger(depth.height, 'depth.height');
  assertIntrinsics(packet.intrinsics);
  if (!packet.pose || !Array.isArray(packet.pose.matrix) || packet.pose.matrix.length !== 16
      || packet.pose.matrix.some((value) => !Number.isFinite(value))) {
    throw new Error('Invalid camera pose');
  }

  const mmBytes = base64ToBytes(depth.millimetersBase64, 'millimetersBase64');
  const confidence = base64ToBytes(depth.confidenceBase64, 'confidenceBase64');
  const expectedPixels = depth.width * depth.height;

  if (!Number.isSafeInteger(expectedPixels) || expectedPixels > 1_000_000) {
    throw new Error('Depth dimensions exceed the supported pixel limit');
  }

  if (mmBytes.byteLength !== expectedPixels * 2) {
    throw new Error(`Depth payload length mismatch: expected ${expectedPixels * 2}, got ${mmBytes.byteLength}`);
  }
  if (confidence.byteLength !== expectedPixels) {
    throw new Error(`Confidence payload length mismatch: expected ${expectedPixels}, got ${confidence.byteLength}`);
  }

  const view = new DataView(mmBytes.buffer, mmBytes.byteOffset, mmBytes.byteLength);
  const meters = new Float32Array(expectedPixels);
  for (let i = 0; i < expectedPixels; i += 1) {
    meters[i] = view.getUint16(i * 2, true) / 1000;
  }

  return {
    ...packet,
    depth: {
      width: depth.width,
      height: depth.height,
      meters,
      confidence,
    },
  };
}

export function depthToPointCloud(frame, confidenceThreshold = 1) {
  const { width, height, meters, confidence } = frame.depth;
  const { fx, fy, cx, cy, imageWidth, imageHeight } = frame.intrinsics;

  assertPositiveInteger(width, 'depth.width');
  assertPositiveInteger(height, 'depth.height');
  assertIntrinsics(frame.intrinsics);
  if (meters.length !== width * height || confidence.length !== width * height) {
    throw new Error('Decoded depth array length mismatch');
  }
  if (!Number.isFinite(confidenceThreshold) || confidenceThreshold < 0 || confidenceThreshold > 255) {
    throw new Error('Invalid confidence threshold');
  }

  const sx = width / imageWidth;
  const sy = height / imageHeight;
  const scaledFx = fx * sx;
  const scaledFy = fy * sy;
  const scaledCx = cx * sx;
  const scaledCy = cy * sy;

  const points = [];
  for (let v = 0; v < height; v += 1) {
    for (let u = 0; u < width; u += 1) {
      const index = v * width + u;
      const z = meters[index];
      if (!Number.isFinite(z) || z <= 0 || confidence[index] < confidenceThreshold) continue;

      const x = ((u - scaledCx) / scaledFx) * z;
      const y = ((v - scaledCy) / scaledFy) * z;
      points.push([x, y === 0 ? 0 : -y, -z]);
    }
  }
  return points;
}

function assertPositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
}

function assertIntrinsics(intrinsics) {
  if (!intrinsics
      || !Number.isFinite(intrinsics.fx) || intrinsics.fx <= 0
      || !Number.isFinite(intrinsics.fy) || intrinsics.fy <= 0
      || !Number.isFinite(intrinsics.cx)
      || !Number.isFinite(intrinsics.cy)) {
    throw new Error('Invalid camera intrinsics');
  }
  assertPositiveInteger(intrinsics.imageWidth, 'intrinsics.imageWidth');
  assertPositiveInteger(intrinsics.imageHeight, 'intrinsics.imageHeight');
}

function base64ToBytes(value, name) {
  if (typeof value !== 'string' || value.length === 0 || value.length % 4 !== 0
      || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)) {
    throw new Error(`${name} must be canonical base64`);
  }

  if (typeof Buffer !== 'undefined') {
    return Uint8Array.from(Buffer.from(value, 'base64'));
  }

  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}
