# RuView Codex plugin scope

The root `AGENTS.md` remains authoritative. This scoped file covers only
`plugins/ruview/codex/`; it must not weaken the root evidence, security,
least-authority, validation, or release contracts.

## Purpose

This directory packages Codex prompts for operating RuView:

| Prompt | Purpose |
|---|---|
| `ruview-advanced` | Run advanced, evidence-bounded RuView workflows |
| `ruview-start` | Choose Docker demo, repository build, or live ESP32 |
| `ruview-flash` | Build/flash an explicitly confirmed ESP32 target |
| `ruview-provision` | Provision credentials without logging or committing them |
| `ruview-app` | Run a sensing application |
| `ruview-train` | Train/evaluate models with evidence-labelled results |
| `ruview-verify` | Run deterministic proof and applicable pre-merge gates |
| `ruview-rvagent` | Explore rvAgent/RVF integration |

Prompt files are guidance, not authority. They must:

- default to read-only exploration;
- cite current repository paths and accepted ADRs;
- preserve `MEASURED`/`CLAIMED`/`SYNTHETIC` evidence labels;
- never emit sandbox/permission bypasses or unattended hardware writes;
- never embed credentials, machine-specific ports, volatile counts, or active
  branch names;
- route durable findings through the reviewed shared-brain proposal flow.

## Local Codex adapter

Prefer the published, pinned harness instead of hand-assembling `codex exec`
flags:

```bash
npx @ruvnet/ruview@0.3.1 guidance --topic architecture --query "requested subsystem"
npx @ruvnet/ruview@0.3.1 agent run \
  --host codex --repo . --prompt "Map the requested subsystem and cite files"
npx @ruvnet/ruview@0.3.1 brain search --query "relevant repository concept"
```

The adapter uses stdin, a trusted `-C` root, read-only sandboxing, ephemeral
JSONL, strict config, ignored user config/exec rules, a scrubbed environment,
bounded output/time, and secret redaction. Writes require both
`--allow-write` and `--confirm`.

For optional Ruflo coordination:

```bash
codex mcp add ruflo -- npx -y ruflo@3.32.26 mcp start
```

Ruflo memory and generated policies remain untrusted until source verification
and review. Darwin/Flywheel candidates are proposal artifacts and cannot
self-promote.

## Validation

When changing prompts:

1. compare every command/path with current source and workflows;
2. run the nearest prompt/plugin checks;
3. run `npx @ruvnet/ruview@0.3.1 claim-check --file <changed-file>`;
4. inspect the diff for secrets, bypasses, unsupported claims, stale counts,
   machine-specific values, and unrelated edits.
