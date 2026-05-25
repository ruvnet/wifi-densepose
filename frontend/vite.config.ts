import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8123',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: true,
  },
  optimizeDeps: {
    // Allow WASM async import via dynamic import()
    exclude: [],
  },
  // WASM async import support: vite handles .wasm?init natively
  assetsInclude: ['**/*.wasm'],
});
