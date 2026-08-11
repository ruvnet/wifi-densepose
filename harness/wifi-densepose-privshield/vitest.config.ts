// SPDX-License-Identifier: MIT
// Strips the `#!/usr/bin/env node` shebang from importable entrypoints (e.g.
// bin/cli.js) before Vite parses them — Vite/esbuild (used internally by
// Vitest) does NOT strip shebangs, so importing a shebanged module throws
// `SyntaxError: Invalid or unexpected token`. No effect on direct CLI
// execution.
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [
    {
      name: 'strip-shebang',
      enforce: 'pre',
      transform(code: string) {
        if (code.startsWith('#!')) {
          return { code: code.replace(/^#![^\n]*/, ''), map: null };
        }
        return null;
      },
    },
  ],
});
