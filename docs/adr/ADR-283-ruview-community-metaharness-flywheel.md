# ADR-283: RuView community metaharness and verified learning flywheel

| Field | Value |
|---|---|
| Status | Accepted — P0/P1 implemented |
| Date | 2026-07-28 |
| Builds on | ADR-182, ADR-263, ADR-265 |

## Decision

Extend `harness/ruview` as the single contributor automation boundary for
repository exploration, development, debugging, testing and release
preparation. The published package remains runtime-dependency-free.

Repository exploration starts with a read-only guidance tool. Its reviewed
catalog records capability maturity, fixed source paths, focused validation
commands, and explicit limitations. In a checkout those citations are checked
for existence; outside a checkout they are labelled as a packaged snapshot.
Optional shared-brain matches remain cited evidence rather than instructions.

Two local hosts are supported with executable contracts:

- Claude Code uses non-interactive `claude -p --safe-mode`, JSON output, no
  session persistence, plan mode, and only read/search tools by default.
- Codex uses `codex exec -`, a trusted `-C` root, `read-only` sandbox,
  ephemeral sessions, strict config parsing, ignored user config/exec rules and
  JSONL output.

Both use shell-free subprocesses, stdin prompts, allowlisted environments,
bounded output/time, secret redaction and realpath-based RuView checkout
validation. Write mode requires two explicit flags and never uses permission or
sandbox bypasses.

## Shared brain

The public brain is committed JSONL, not a shared mutable database. Canonical
records are reviewed, bounded, source-relative, source-cited and content
digested. Secret-shaped and instruction-shaped submissions are quarantined.
Community learning enters through ordinary proposal pull requests.

Ruflo/AgentDB may build local semantic indexes and private overlays from that
corpus. Those indexes, raw transcripts, credentials and personal/CSI data are
not committed. This provides a common brain without turning retrieved text into
executable policy.

## Darwin and Flywheel

The seven policy surfaces are explicit in `flywheel/genome.json`. Evolution is
human-initiated and each Darwin candidate may mutate only one surface.
Contributor runs produce untrusted `.metaharness/` artifacts.

Promotion is conjunctive:

1. the frozen anchor cannot regress;
2. the holdout must improve;
3. legacy and security tests pass;
4. no blocked action or secret exposure occurs;
5. corpus, files and gate fingerprints verify;
6. a maintainer reviews and approves the replay bundle.

Flywheel signatures establish bundle integrity, not maintainer authority.
Authority comes from protected-branch review and release provenance. CI never
autonomously promotes or publishes an evolved candidate.

## Consequences

Contributors can explore RuView with either major local CLI and share durable
findings without sharing secrets. Improvements become reproducible proposals
with frozen evaluation evidence. The cost is a larger development-only npm
lockfile, a 128 KiB unpacked-package budget (the current tarball is below that
bound), and explicit maintenance of the corpus, genome and gate.
