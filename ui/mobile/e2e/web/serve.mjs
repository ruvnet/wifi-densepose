import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';

const host = '127.0.0.1';
const port = 4173;
const root = resolve(process.cwd(), 'dist-e2e');

const contentTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
};

const resolveRequestPath = (rawUrl) => {
  const pathname = decodeURIComponent(new URL(rawUrl ?? '/', `http://${host}`).pathname);
  const normalized = normalize(pathname).replace(/^(\.\.[/\\])+/, '');
  const requested = resolve(root, `.${normalized}`);
  if (!requested.startsWith(`${root}/`) && requested !== root) return null;
  if (existsSync(requested) && statSync(requested).isFile()) return requested;
  const nestedIndex = join(requested, 'index.html');
  if (existsSync(nestedIndex) && statSync(nestedIndex).isFile()) return nestedIndex;
  return join(root, 'index.html');
};

const server = createServer((request, response) => {
  const path = resolveRequestPath(request.url);
  if (!path || !existsSync(path)) {
    response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    response.end('Not found');
    return;
  }

  response.writeHead(200, {
    'cache-control': 'no-store',
    'content-type': contentTypes[extname(path)] ?? 'application/octet-stream',
    'x-content-type-options': 'nosniff',
  });
  createReadStream(path).pipe(response);
});

server.listen(port, host, () => {
  process.stdout.write(`RuView mobile E2E server listening on http://${host}:${port}\n`);
});
