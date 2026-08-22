import { decodeLiDARPacket, depthToPointCloud } from './codec.mjs';

const canvas = document.querySelector('#view');
const ctx = canvas.getContext('2d');
const status = document.querySelector('#status');
const fpsEl = document.querySelector('#fps');
const pointsEl = document.querySelector('#points');
const seqEl = document.querySelector('#seq');
const latencyEl = document.querySelector('#latency');
const sensorEl = document.querySelector('#sensor');

let lastFrameAt = performance.now();
let yaw = 0.3;
let pitch = -0.15;
let scale = 120;

function connect() {
  const token = new URLSearchParams(location.search).get('token');
  if (!token) {
    status.textContent = 'TOKEN REQUIRED';
    status.dataset.state = 'warn';
    return;
  }

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/ws/lidar?token=${encodeURIComponent(token)}`);

  socket.addEventListener('open', () => {
    status.textContent = 'LIVE';
    status.dataset.state = 'live';
  });

  socket.addEventListener('close', () => {
    status.textContent = 'RECONNECTING';
    status.dataset.state = 'warn';
    setTimeout(connect, 1000);
  });

  socket.addEventListener('message', (event) => {
    try {
      const raw = JSON.parse(event.data);
      const frame = decodeLiDARPacket(raw);
      const points = depthToPointCloud(frame, 1);
      render(points);

      const now = performance.now();
      const delta = now - lastFrameAt;
      lastFrameAt = now;
      fpsEl.textContent = delta > 0 ? (1000 / delta).toFixed(1) : '0.0';
      pointsEl.textContent = points.length.toLocaleString();
      seqEl.textContent = frame.provenance.sequence;
      latencyEl.textContent = Math.max(0, Date.now() - Number(frame.provenance.timestampNs / 1_000_000)).toFixed(0);
      sensorEl.textContent = frame.provenance.sensor;
    } catch (error) {
      console.error(error);
      status.textContent = 'FRAME ERROR';
      status.dataset.state = 'warn';
    }
  });
}

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function render(points) {
  resize();
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#091018';
  ctx.fillRect(0, 0, w, h);

  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);

  ctx.fillStyle = '#58e0d2';
  for (let i = 0; i < points.length; i += 1) {
    let [x, y, z] = points[i];
    const rx = x * cy - z * sy;
    const rz = x * sy + z * cy;
    const ry = y * cp - rz * sp;
    const rz2 = y * sp + rz * cp;
    const perspective = 1 / Math.max(0.35, 1.8 - rz2 * 0.12);
    const px = w / 2 + rx * scale * perspective;
    const py = h / 2 + ry * scale * perspective;
    if (px >= 0 && px < w && py >= 0 && py < h) ctx.fillRect(px, py, 1.4, 1.4);
  }
}

canvas.addEventListener('pointermove', (event) => {
  if (!event.buttons) return;
  yaw += event.movementX * 0.006;
  pitch += event.movementY * 0.006;
});

canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  scale = Math.max(40, Math.min(400, scale - event.deltaY * 0.2));
}, { passive: false });

connect();
