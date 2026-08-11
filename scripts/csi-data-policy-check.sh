#!/usr/bin/env bash
#
# csi-data-policy-check.sh — ADR-296 CSI data-incident repository guard.
#
# WHY (ADR-296): raw CSI recordings are person data (they encode breathing,
# movement, and presence) and CLAUDE.md prohibits committing CSI or person
# data. A stale `.gitignore` rule let ~64.6 MB of raw captures reach the tree
# under `data/recordings/` and `v2/data/recordings/`. This check is the
# mechanical guard that prevents the incident from getting worse: it fails when
# CSI-format files or oversized JSONL captures are tracked/staged.
#
# WHAT IT FLAGS:
#   * `*.csi.jsonl`      — raw CSI capture stream (person data)
#   * `*.csi.meta.json`  — capture sidecar metadata
#   * `*.jsonl` larger than CSI_POLICY_MAX_JSONL_BYTES (~5 MB default) — a
#     capture-sized JSONL blob that almost never belongs in git.
#
# DETERMINISTIC / OFFLINE: no network, no clock, no randomness. It only reads
# the file list git already knows about (or a list you pass in) and file sizes.
#
# USAGE:
#   scripts/csi-data-policy-check.sh                 # scan tracked files (git ls-files)
#   scripts/csi-data-policy-check.sh --staged        # scan the staged set (pre-commit)
#   scripts/csi-data-policy-check.sh --files-from -   # scan a newline list on stdin
#   scripts/csi-data-policy-check.sh --files-from FILE
#   scripts/csi-data-policy-check.sh --self-test     # run built-in self-tests
#
# EXIT CODES: 0 = clean, 1 = policy violation, 2 = usage/environment error.
#
# ------------------------------------------------------------------------------
# ALLOWLIST (synthetic test fixtures)
# ------------------------------------------------------------------------------
# Tests may use only synthetic or expressly-consented minimal fixtures (ADR-296).
# A file whose path matches an allow pattern is exempt. Patterns come from:
#   * the file `scripts/csi-data-policy.allow` (one glob per line, `#` comments), and
#   * the env var `CSI_POLICY_ALLOW` (colon-separated globs).
# Patterns are shell globs matched against the repo-relative path, e.g.
#   scripts/tests/fixtures/csi-policy/*.csi.jsonl
#
# ------------------------------------------------------------------------------
# BASELINE (acknowledged pre-existing incident, ADR-296)
# ------------------------------------------------------------------------------
# The tree today ALREADY contains the incident recordings under
# `data/recordings/` and `v2/data/recordings/`. Removing them is destructive and
# gated on data-owner sign-off (ADR-296 "Decision"), so this guard is EXPECTED to
# fail on the current tree — that failure documents the incident.
#
# Once the owner removes those files, or to acknowledge them in the interim
# without weakening the guard for NEW files, point `CSI_POLICY_BASELINE` at a
# file listing the acknowledged repo-relative paths (one per line, `#` comments,
# globs allowed). Baseline-matched files are reported as "acknowledged" and do
# NOT fail the check; every other violation still fails. This is the intended
# mechanism to make the CI job green in a follow-up once remediation lands.
#
set -euo pipefail

# --- configuration -----------------------------------------------------------
# ~5 MB default. Override with CSI_POLICY_MAX_JSONL_BYTES for tests/tuning.
MAX_JSONL_BYTES="${CSI_POLICY_MAX_JSONL_BYTES:-5242880}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLOW_FILE="${CSI_POLICY_ALLOW_FILE:-$SCRIPT_DIR/csi-data-policy.allow}"

# --- allow / baseline pattern loading ----------------------------------------
# Read glob patterns from a file (skip blank lines and `#` comments) into the
# named array. Missing files are treated as empty (not an error).
read_patterns() {
  local file="$1" __arrname="$2" line
  eval "$__arrname=()"
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    # trim leading/trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    eval "$__arrname+=(\"\$line\")"
  done < "$file"
}

ALLOW_PATTERNS=()
read_patterns "$ALLOW_FILE" ALLOW_PATTERNS
# Append colon-separated CSI_POLICY_ALLOW entries.
if [[ -n "${CSI_POLICY_ALLOW:-}" ]]; then
  local_ifs="$IFS"; IFS=':'
  for p in $CSI_POLICY_ALLOW; do [[ -n "$p" ]] && ALLOW_PATTERNS+=("$p"); done
  IFS="$local_ifs"
fi

BASELINE_PATTERNS=()
if [[ -n "${CSI_POLICY_BASELINE:-}" ]]; then
  [[ -f "$CSI_POLICY_BASELINE" ]] || {
    echo "ERROR: CSI_POLICY_BASELINE points at a missing file: $CSI_POLICY_BASELINE" >&2
    exit 2
  }
  read_patterns "$CSI_POLICY_BASELINE" BASELINE_PATTERNS
fi

# Return 0 if $1 matches any glob in the array named by $2.
matches_any() {
  local path="$1" __arrname="$2" pat
  local -a arr
  eval "arr=(\"\${${__arrname}[@]}\")"
  for pat in "${arr[@]:-}"; do
    [[ -z "$pat" ]] && continue
    # shellcheck disable=SC2053  # intentional glob match
    [[ "$path" == $pat ]] && return 0
  done
  return 1
}

# --- per-file classification -------------------------------------------------
# Echo a violation reason for $1, or nothing if the file is fine. CSI-format
# files are always flagged; other .jsonl files are flagged only when oversized.
classify() {
  local f="$1"
  case "$f" in
    *.csi.jsonl)     echo "CSI capture stream (*.csi.jsonl)"; return 0 ;;
    *.csi.meta.json) echo "CSI capture metadata (*.csi.meta.json)"; return 0 ;;
  esac
  case "$f" in
    *.jsonl)
      local size
      size="$(file_size "$f")"
      if [[ "$size" -gt "$MAX_JSONL_BYTES" ]]; then
        echo "oversized JSONL capture (${size} bytes > ${MAX_JSONL_BYTES})"
        return 0
      fi
      ;;
  esac
  return 0
}

# Best-effort byte size for a repo-relative path. Prefer the working-tree file;
# fall back to git's stored blob so the check works on a bare/partial checkout.
# Prints 0 when the size cannot be determined (glob rules still apply).
file_size() {
  local f="$1" sz
  if [[ -f "$f" ]]; then
    sz="$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)"
    echo "${sz:-0}"; return 0
  fi
  if command -v git >/dev/null 2>&1; then
    sz="$(git cat-file -s ":$f" 2>/dev/null || git cat-file -s "HEAD:$f" 2>/dev/null || echo 0)"
    echo "${sz:-0}"; return 0
  fi
  echo 0
}

# --- core scan ---------------------------------------------------------------
# Read newline-separated repo-relative paths on stdin and enforce policy.
# Exits 1 if any non-baseline, non-allowlisted violation is found.
scan_stdin() {
  local violations=0 acknowledged=0 f reason
  while IFS= read -r f || [[ -n "$f" ]]; do
    [[ -z "$f" ]] && continue
    reason="$(classify "$f")"
    [[ -z "$reason" ]] && continue
    if matches_any "$f" ALLOW_PATTERNS; then
      continue  # synthetic fixture, expressly allowed
    fi
    if matches_any "$f" BASELINE_PATTERNS; then
      echo "ack:   $f — $reason (acknowledged baseline, ADR-296)" >&2
      acknowledged=$((acknowledged + 1))
      continue
    fi
    echo "BLOCK: $f — $reason" >&2
    violations=$((violations + 1))
  done

  if [[ "$acknowledged" -gt 0 ]]; then
    echo "note: $acknowledged file(s) acknowledged via CSI_POLICY_BASELINE (ADR-296)." >&2
  fi

  if [[ "$violations" -gt 0 ]]; then
    echo "" >&2
    echo "FAIL: $violations CSI/person-data policy violation(s) (ADR-296)." >&2
    echo "      Raw CSI is person data and must not be tracked in git. See" >&2
    echo "      docs/adr/ADR-296-csi-data-incident-repo-controls.md." >&2
    echo "      Synthetic test fixtures can be allowlisted in $ALLOW_FILE." >&2
    return 1
  fi
  echo "OK: no CSI/person-data policy violations." >&2
  return 0
}

# --- input sources -----------------------------------------------------------
emit_tracked() {
  command -v git >/dev/null 2>&1 || { echo "ERROR: git not found" >&2; exit 2; }
  git ls-files
}

emit_staged() {
  command -v git >/dev/null 2>&1 || { echo "ERROR: git not found" >&2; exit 2; }
  git diff --cached --name-only --diff-filter=ACMR
}

# --- self-tests --------------------------------------------------------------
# Deterministic, offline. Builds synthetic fixtures in a temp dir and asserts
# the check flags a *.csi.jsonl / oversized JSONL and passes allowlisted ones.
self_test() {
  local tmp rc pass=0 fail=0
  tmp="$(mktemp -d)"

  mkdir -p "$tmp/fixtures"
  printf '{"csi":[1,2,3]}\n' > "$tmp/real.csi.jsonl"
  printf '{"schema":1}\n'     > "$tmp/real.csi.meta.json"
  printf '{"note":"ok"}\n'    > "$tmp/small.jsonl"
  printf '{"synthetic":true}\n' > "$tmp/fixtures/synthetic.csi.jsonl"
  # Oversized JSONL: 40 bytes, checked against a 10-byte threshold below.
  printf '%0.sX' {1..40} > "$tmp/big.jsonl"; printf '\n' >> "$tmp/big.jsonl"

  assert() { # desc expected_rc actual_rc
    if [[ "$2" -eq "$3" ]]; then echo "  PASS: $1"; pass=$((pass+1));
    else echo "  FAIL: $1 (expected rc=$2, got rc=$3)"; fail=$((fail+1)); fi
  }

  echo "self-test: fixtures in $tmp"

  # 1. A raw *.csi.jsonl must be blocked.
  rc=0; printf '%s\n' "$tmp/real.csi.jsonl" | scan_stdin >/dev/null 2>&1 || rc=$?
  assert "blocks *.csi.jsonl" 1 "$rc"

  # 2. A *.csi.meta.json must be blocked.
  rc=0; printf '%s\n' "$tmp/real.csi.meta.json" | scan_stdin >/dev/null 2>&1 || rc=$?
  assert "blocks *.csi.meta.json" 1 "$rc"

  # 3. A small, ordinary .jsonl must pass.
  rc=0; printf '%s\n' "$tmp/small.jsonl" | scan_stdin >/dev/null 2>&1 || rc=$?
  assert "passes small ordinary .jsonl" 0 "$rc"

  # 4. An oversized .jsonl must be blocked (tiny threshold, deterministic).
  rc=0; CSI_POLICY_MAX_JSONL_BYTES=10 bash "$0" --files-from - <<<"$tmp/big.jsonl" >/dev/null 2>&1 || rc=$?
  assert "blocks oversized .jsonl" 1 "$rc"

  # 5. An allowlisted synthetic fixture must pass despite matching *.csi.jsonl.
  rc=0; CSI_POLICY_ALLOW="$tmp/fixtures/*.csi.jsonl" bash "$0" --files-from - \
        <<<"$tmp/fixtures/synthetic.csi.jsonl" >/dev/null 2>&1 || rc=$?
  assert "passes allowlisted synthetic fixture" 0 "$rc"

  # 6. A baseline-acknowledged CSI file must pass (job made green post-cleanup).
  local bl="$tmp/baseline.txt"; printf '%s\n' "$tmp/real.csi.jsonl" > "$bl"
  rc=0; CSI_POLICY_BASELINE="$bl" bash "$0" --files-from - \
        <<<"$tmp/real.csi.jsonl" >/dev/null 2>&1 || rc=$?
  assert "passes baseline-acknowledged file" 0 "$rc"

  echo "self-test: $pass passed, $fail failed"
  rm -rf "$tmp"
  [[ "$fail" -eq 0 ]]
}

# --- entrypoint --------------------------------------------------------------
main() {
  local mode="tracked" from=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --staged)      mode="staged" ;;
      --tracked)     mode="tracked" ;;
      --files-from)  mode="files-from"; from="${2:-}"; shift ;;
      --self-test)   mode="self-test" ;;
      -h|--help)     grep '^#' "$0" | sed 's/^#\s\{0,1\}//'; exit 0 ;;
      *) echo "ERROR: unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift
  done

  case "$mode" in
    self-test)  self_test ;;
    tracked)    emit_tracked | scan_stdin ;;
    staged)     emit_staged  | scan_stdin ;;
    files-from)
      if [[ "$from" == "-" || -z "$from" ]]; then
        scan_stdin
      else
        [[ -f "$from" ]] || { echo "ERROR: --files-from file not found: $from" >&2; exit 2; }
        scan_stdin < "$from"
      fi
      ;;
  esac
}

main "$@"
