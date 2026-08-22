import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';

const root = fileURLToPath(new URL('.', import.meta.url));
const port = Number(process.env.PORT || 8787);
const clients = new Set();

const server = http.createServer(async (req, res) => {
  const path = req.url === '/' ? '/index.html' : req.url?.split('?')[0] || '/index.html';
  if (path.includes('..')) {
    res.writeHead(400).end('bad path');
    return;
  }

  try {
    const data = await readFile(join(root, path));
    const contentType = {
      '.html': 'text/html; charset=utf-8',
      '.js': 'text/javascript; charset=utf-8',
      '.mjs': 'text/javascript; charset=utf-8',
      '.css': 'text/css; charset=utf-8',
    }[extname(path)] || 'application/octet-stream';
    res.writeHead(200, { 'content-type': contentType, 'cache-control': 'no-store' });
    res.end(data);
  } catch {
    res.writeHead(404).end('not found');
  }
});

const wss = new WebSocketServer({ server, path: '/ws/lidar' });

wss.on('connection', (socket) => {
  clients.add(socket);
  socket.on('close', () => clients.delete(socket));
  socket.on('message', (data, isBinary) => {
    if (isBinary || data.length > 2_000_000) return;

    let packet;
    try {
      packet = JSON.parse(data.toString());
    } catch {
      return;
    }

    if (packet?.type !== 'ruview.lidar.depth.v1') return;

    for (const peer of clients) {
      if (peer !== socket && peer.readyState === peer.OPEN) {
        peer.send(data);
      }
    }
  });
});

server.listen(port, '0.0.0.0', () => {
  console.log(`RuView iPhone LiDAR relay: http://0.0.0.0:${port}`);
  console.log(`Native endpoint: ws://<host>:${port}/ws/lidar`);
});
