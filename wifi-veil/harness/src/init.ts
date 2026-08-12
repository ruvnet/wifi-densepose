// SPDX-License-Identifier: MIT
// The harness's `wifi-veil-harness init` entry (typed mirror of
// the JS command in bin/cli.js; the published CLI uses the JS version so no
// build is required for `init`).

import { loadKernel } from '@metaharness/kernel';
import adapter from '@metaharness/host-claude-code';

const HARNESS_NAME = 'wifi-veil-harness';

async function main(): Promise<number> {
  const kernel = await loadKernel();
  const info = kernel.kernelInfo();
  console.log(`${HARNESS_NAME} — kernel ${info.version} (${kernel.backend})`);
  console.log(`Host adapter: ${adapter.name}`);
  console.log(`Run \`${HARNESS_NAME} doctor\` to verify the install.`);
  return 0;
}

main()
  .then((c) => process.exit(c))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
