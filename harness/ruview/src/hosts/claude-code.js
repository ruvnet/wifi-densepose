// SPDX-License-Identifier: MIT
import { runProcess } from '../process-runner.js';
import { assertTrustedRuViewRepo } from '../repo-trust.js';
export function buildClaudeCodeArgs({ write = false } = {}) {
  return ['-p', '--safe-mode', '--output-format', 'json', '--no-session-persistence', '--permission-mode', write ? 'acceptEdits' : 'plan',
    '--allowedTools', write ? 'Read,Grep,Glob,Edit,Write' : 'Read,Grep,Glob'];
}
export async function runClaudeCode({
  prompt, repoRoot, trustedRoot = repoRoot, allowWrite = false, confirm = false,
  command = 'claude', commandArgs = [], ...runOptions
}) {
  if (typeof prompt !== 'string' || !prompt.trim()) throw new TypeError('prompt must be a non-empty string');
  const root = assertTrustedRuViewRepo(repoRoot, { trustedRoot });
  const write = allowWrite === true && confirm === true;
  return runProcess(command, [...commandArgs, ...buildClaudeCodeArgs({ write })], { ...runOptions, cwd: root, input: prompt });
}
export default Object.freeze({ name: 'claude-code', run: runClaudeCode, buildArgs: buildClaudeCodeArgs });
