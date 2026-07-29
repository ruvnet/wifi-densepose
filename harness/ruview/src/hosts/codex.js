// SPDX-License-Identifier: MIT
import { runProcess } from '../process-runner.js';
import { assertTrustedRuViewRepo } from '../repo-trust.js';
export function buildCodexArgs(root, { write = false } = {}) {
  return ['exec', '-', '-C', root, '--sandbox', write ? 'workspace-write' : 'read-only',
    '--ephemeral', '--json', '--strict-config', '--ignore-user-config', '--ignore-rules'];
}
export async function runCodex({
  prompt, repoRoot, trustedRoot = repoRoot, allowWrite = false, confirm = false,
  command = 'codex', commandArgs = [], ...runOptions
}) {
  if (typeof prompt !== 'string' || !prompt.trim()) throw new TypeError('prompt must be a non-empty string');
  const root = assertTrustedRuViewRepo(repoRoot, { trustedRoot });
  const write = allowWrite === true && confirm === true;
  return runProcess(command, [...commandArgs, ...buildCodexArgs(root, { write })], { ...runOptions, cwd: root, input: prompt });
}
export default Object.freeze({ name: 'codex', run: runCodex, buildArgs: buildCodexArgs });
