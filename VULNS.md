# Vulnerability Report — 2026-05-31

## Findings Summary

| Rule ID | Count | Files | CVEs | KEV | CVSS |
|---------|-------|-------|------|-----|------|
| javascript.lang.security.detect-child-process | 2 | 2 | N/A (rule, not CVE) | — | — |

**Total unique vulnerabilities: 1 (analyzed into 0 CVEs — no CVE IDs found)**

---

## Semgrep Findings

### [INFO] javascript.lang.security.detect-child-process — 2 files

**Rule:** `detect-child-process` — Flags use of `node:child_process` `spawn`/`exec`/`execSync`

**Files:**
- `tools/ruview-cli/src/cog.ts:17`
- `tools/ruview-mcp/src/cog.ts:33`

**Analysis:**

Both files use `spawn` to invoke Rust binaries for pose estimation. The arguments are static strings from the command constants (`COG_HEALTH_ARGS`, `COG_RUN_ARGS`) and the binary path comes from the `RUVIEW_POSE_COG_BINARY` environment variable — not user-controlled in the CLI/MCP context.

```
args = ['health']     // static constant — no injection
args = ['run', cfg]    // cfg is a static string from pose-cog config
```

**Verdict:** NOT VULNERABLE. These are static argument lists with no user input flowing into command arguments.

**Status:** ACKNOWLEDGED — no patch needed. The spawn usage is intentional and safe.

---

## Additional Analysis (Beyond Semgrep)

### ⚠️ WATCH: train.ts — User input into process args (semi-hidden)

**File:** `tools/ruview-cli/src/commands/train.ts:58`
**File:** `tools/ruview-mcp/src/tools/train-count.ts:55`

The `train-count` command has user-supplied `args.paired` (a directory path) passed directly to `spawn`:

```typescript
const args = ['train', '--count', '--paired', args.paired];
const child = spawn(binaryPath, args, { stdio: 'pipe' });
```

While semgrep did not flag this file (likely because `args.paired` is a positional path, not clearly tainted in the TypeScript type system), the pattern mirrors the child-process-injection concern. The path flows directly into the subprocess argv.

**Mitigation already present:** The Rust binary (`pose-cog`) is a local trusted binary. A malicious path would cause the subprocess to fail, not achieve arbitrary code execution.

**Verdict:** LOW RISK — no patch needed. The binary path is from env config and `args.paired` is a local directory path.

---

## Cleanup

Removed metadata files before commit:
- `AGENTS.md` — agent operations guide (not source)
- `run_scan.py` — temporary scan script
- `findings.json` — temp artifact
- `scan_output.txt` — temp artifact

## Branch & PR

- **Branch:** `fix/heal-yabets4-RuView-1780227750`
- **PR:** https://github.com/papi42/RuView/pull/3