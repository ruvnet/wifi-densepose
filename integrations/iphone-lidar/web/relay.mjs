import { randomBytes, timingSafeEqual } from 'node:crypto';
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { WebSocket, WebSocketServer } from 'ws';

const root = fileURLToPath(new URL('.', import.meta.url));
const staticFiles = new Set(['index.html', 'app.mjs', 'codec.mjs', 'styles.css']);
const maxPayloadBytes = 2_000_000;

function constantTimeEqual(left, right) {
  const leftBytes = Buffer.from(left, 'utf8');
  const rightBytes = Buffer.from(right, 'utf8');
  return leftBytes.length === rightBytes.length && timingSafeEqual(leftBytes, rightBytes);
}

function rejectUpgrade(socket, status, message) {
  const body = `${message}\n`;
  socket.end(
    `HTTP/1.1 ${status}\r\nConnection: close\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`,
  );
}

export function createLiDARRelay({ token, rootDirectory = root } = {}) {
  const accessToken = token || randomBytes(24).toString('hex');
  const clients = new Set();

  const server = http.createServer(async (req, res) => {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405, { allow: 'GET, HEAD' }).end();
      return;
    }

    let pathname;
    try {
      pathname = decodeURIComponent(new URL(req.url || '/', 'http://localhost').pathname);
    } catch {
      res.writeHead(400).end('bad path');
      return;
    }

    const filename = pathname === '/' ? 'index.html' : pathname.slice(1);
    if (!staticFiles.has(filename)) {
      res.writeHead(404).end('not found');
      return;
    }

    try {
      const data = await readFile(join(rootDirectory, filename));
      const contentType = {
        '.html': 'text/html; charset=utf-8',
        '.mjs': 'text/javascript; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
      }[extname(filename)] || 'application/octet-stream';
      res.writeHead(200, {
        'content-type': contentType,
        'cache-control': 'no-store',
        'content-security-policy': "default-src 'self'; connect-src 'self' ws: wss:; img-src 'self'; style-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        'referrer-policy': 'no-referrer',
        'x-content-type-options': 'nosniff',
      });
      if (req.method === 'HEAD') res.end();
      else res.end(data);
    } catch {
      res.writeHead(404).end('not found');
    }
  });

  const wss = new WebSocketServer({ noServer: true, maxPayload: maxPayloadBytes });

  server.on('upgrade', (req, socket, head) => {
    let url;
    try {
      url = new URL(req.url || '/', 'http://localhost');
    } catch {
      rejectUpgrade(socket, '400 Bad Request', 'bad websocket URL');
      return;
    }

    if (url.pathname !== '/ws/lidar') {
      rejectUpgrade(socket, '404 Not Found', 'not found');
      return;
    }

    const suppliedToken = url.searchParams.get('token') || '';
    if (!constantTimeEqual(suppliedToken, accessToken)) {
      rejectUpgrade(socket, '401 Unauthorized', 'valid LiDAR relay token required');
      return;
    }

    wss.handleUpgrade(req, socket, head, (websocket) => {
      wss.emit('connection', websocket, req);
    });
  });

  wss.on('connection', (socket) => {
    clients.add(socket);
    socket.on('close', () => clients.delete(socket));
    socket.on('error', () => socket.terminate());
    socket.on('message', (data, isBinary) => {
      if (isBinary) return;

      let packet;
      try {
        packet = JSON.parse(data.toString());
      } catch {
        return;
      }

      if (packet?.type !== 'ruview.lidar.depth.v1') return;

      for (const peer of clients) {
        if (peer !== socket && peer.readyState === WebSocket.OPEN) {
          peer.send(data.toString());
        }
      }
    });
  });

  return {
    accessToken,
    server,
    async listen(port = 0, host = '127.0.0.1') {
      await new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(port, host, () => {
          server.off('error', reject);
          resolve();
        });
      });
      return server.address();
    },
    async close() {
      for (const peer of clients) peer.terminate();
      await new Promise((resolve) => wss.close(resolve));
      if (server.listening) {
        await new Promise((resolve, reject) => {
          server.close((error) => (error ? reject(error) : resolve()));
        });
      }
    },
  };
}

function configuredPort(value) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > 65_535) {
    throw new Error(`PORT must be an integer from 1 to 65535; received ${value}`);
  }
  return parsed;
}

const directInvocation = process.argv[1]
  && pathToFileURL(process.argv[1]).href === import.meta.url;

if (directInvocation) {
  const port = configuredPort(process.env.PORT || '8787');
  const host = process.env.HOST || '0.0.0.0';
  const relay = createLiDARRelay({ token: process.env.RUVIEW_LIDAR_TOKEN });
  await relay.listen(port, host);
  const encodedToken = encodeURIComponent(relay.accessToken);
  console.log(`RuView iPhone LiDAR relay listening on ${host}:${port}`);
  console.log(`Browser: http://<host>:${port}/?token=${encodedToken}`);
  console.log(`Native endpoint: ws://<host>:${port}/ws/lidar?token=${encodedToken}`);
}
