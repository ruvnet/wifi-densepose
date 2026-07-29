// SPDX-License-Identifier: MIT
// Prefer the packaged WebAssembly kernel while retaining an honest fallback.

import { readFileSync } from 'node:fs';
import { loadKernel } from '@metaharness/kernel';

const PKG = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const MCP_SPEC = Object.freeze({
  name: 'homecore',
  command: ['npx', '-y', `${PKG.name}@${PKG.version}`, 'mcp', 'start'],
});

let cached;
let loading;

/**
 * Load the metaharness kernel with WASM as the default requested backend.
 * An explicit METAHARNESS_KERNEL_BACKEND value remains authoritative.
 */
export async function loadHomecoreKernel() {
  if (cached) return cached;
  if (loading) return loading;
  loading = initializeKernel();
  try {
    cached = await loading;
    return cached;
  } finally {
    loading = undefined;
  }
}

async function initializeKernel() {
  const explicit = process.env.METAHARNESS_KERNEL_BACKEND;
  if (explicit) {
    return Object.freeze({
      ...await loadKernel(),
      homecoreRequestedBackend: explicit,
    });
  }

  process.env.METAHARNESS_KERNEL_BACKEND = 'wasm';
  try {
    return Object.freeze({
      ...await loadKernel(),
      homecoreRequestedBackend: 'wasm',
    });
  } catch (wasmError) {
    delete process.env.METAHARNESS_KERNEL_BACKEND;
    const fallback = await loadKernel();
    return Object.freeze({
      ...fallback,
      homecoreRequestedBackend: 'wasm',
      wasmFallbackReason: wasmError instanceof Error ? wasmError.message : String(wasmError),
    });
  } finally {
    if (explicit === undefined) delete process.env.METAHARNESS_KERNEL_BACKEND;
    else process.env.METAHARNESS_KERNEL_BACKEND = explicit;
  }
}

export async function getKernelStatus({ strict = false } = {}) {
  const kernel = await loadHomecoreKernel();
  const info = kernel.kernelInfo();
  const validationError = kernel.mcpValidate(JSON.stringify(MCP_SPEC));
  const wasm = kernel.backend === 'wasm';
  const requestedBackend = kernel.homecoreRequestedBackend || 'wasm';
  return {
    ok: validationError === null && (!strict || wasm),
    preferredBackend: 'wasm',
    requestedBackend,
    resolvedBackend: kernel.backend,
    strict,
    info,
    mcpSpec: MCP_SPEC,
    mcpValidation: validationError,
    fallbackReason: kernel.wasmFallbackReason || null,
    note: wasm
      ? 'The packaged WebAssembly kernel is active.'
      : requestedBackend === 'wasm'
        ? `WASM was unavailable; the reported ${kernel.backend} fallback backend is active.`
        : `The operator explicitly requested ${requestedBackend}; the reported ${kernel.backend} backend is active.`,
  };
}

export { MCP_SPEC };
