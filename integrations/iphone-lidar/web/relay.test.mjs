import assert from 'node:assert/strict';
import http from 'node:http';
import test from 'node:test';
import { WebSocket } from 'ws';
import { createLiDARRelay } from './relay.mjs';

function connect(url) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    socket.once('open', () => resolve(socket));
    socket.once('error', reject);
  });
}

function request(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (response) => {
      response.resume();
      response.once('end', () => resolve(response));
    }).once('error', reject);
  });
}

test('relay requires a token, limits static files, and forwards LiDAR frames', async (t) => {
  const token = 'test-token-for-relay';
  const relay = createLiDARRelay({ token });
  const address = await relay.listen(0, '127.0.0.1');
  const httpBase = `http://127.0.0.1:${address.port}`;
  const wsBase = `ws://127.0.0.1:${address.port}/ws/lidar`;
  const sockets = [];

  t.after(async () => {
    for (const socket of sockets) socket.terminate();
    await relay.close();
  });

  const indexResponse = await request(`${httpBase}/?token=${token}`);
  assert.equal(indexResponse.statusCode, 200);
  assert.match(indexResponse.headers['content-security-policy'], /frame-ancestors 'none'/);

  const sourceResponse = await request(`${httpBase}/relay.mjs`);
  assert.equal(sourceResponse.statusCode, 404);

  await new Promise((resolve, reject) => {
    const unauthorized = new WebSocket(wsBase);
    unauthorized.once('unexpected-response', (_request, response) => {
      assert.equal(response.statusCode, 401);
      response.resume();
      resolve();
    });
    unauthorized.once('open', () => reject(new Error('unauthorized websocket opened')));
    unauthorized.once('error', () => {});
  });

  const sender = await connect(`${wsBase}?token=${encodeURIComponent(token)}`);
  const receiver = await connect(`${wsBase}?token=${encodeURIComponent(token)}`);
  sockets.push(sender, receiver);

  const received = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timed out waiting for relayed frame')), 2_000);
    receiver.once('message', (data, isBinary) => {
      clearTimeout(timer);
      resolve({ data, isBinary });
    });
  });

  const frame = { type: 'ruview.lidar.depth.v1', provenance: { sequence: 7 } };
  sender.send(JSON.stringify(frame));
  const message = await received;

  assert.equal(message.isBinary, false);
  assert.deepEqual(JSON.parse(message.data.toString()), frame);
});
