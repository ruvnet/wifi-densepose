#!/usr/bin/env bash
# WiFi Veil CI guard — enforces the project's honesty / anti-slop invariants so
# they cannot silently regress. This is a STATIC scan; the deterministic proof
# witness, tests, clippy, and the C-core test run in the main CI jobs.
#
# It scans only git-tracked files under the current tree, so untracked local
# scratch never fails CI and build outputs (target/) are ignored. Run locally
# from the repo root:  bash scripts/ci-guard.sh
#
# Every check prints PASS/FAIL; the script exits non-zero if any check fails.
set -u

fail=0
pass() { printf 'PASS  %s\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1"; fail=1; }

# All tracked files under cwd (works both in-monorepo-subdir and standalone).
mapfile -t TRACKED < <(git ls-files -- .)
# Convenience filtered lists.
code_files()  { printf '%s\n' "${TRACKED[@]}" | grep -E '\.(rs|c|h|js|ts)$' || true; }
doc_files()   { printf '%s\n' "${TRACKED[@]}" | grep -E '\.md$' || true; }

# ---------------------------------------------------------------------------
# 1. No forbidden artifacts / telemetry / scratch committed.
# ---------------------------------------------------------------------------
artifacts=$(printf '%s\n' "${TRACKED[@]}" | grep -E \
  '(^|/)(\.claude-flow/|node_modules/|target/)|\.o$|(^|/)test_veil_shield$|(^|/)Cargo\.lock$|(^|/)tests/probe.*\.rs$|(^|/)(tmp_|scratch_).*' \
  || true)
if [ -n "$artifacts" ]; then
  bad "forbidden artifacts/telemetry/scratch are tracked:"
  printf '        %s\n' $artifacts
else
  pass "no telemetry / build artifacts / lockfile / scratch files tracked"
fi

# ---------------------------------------------------------------------------
# 2. No debug/scratch leftovers in source (mock-probe / slop markers).
# ---------------------------------------------------------------------------
markers='panic!\("probe|dbg!\(|println!\("DEBUG|console\.log\("DEBUG|TODO\(ai\)|FIXME\(slop\)|XXX SLOP|LOREM IPSUM'
hits=$(code_files | xargs -r grep -nEI "$markers" 2>/dev/null || true)
if [ -n "$hits" ]; then
  bad "debug/scratch/slop markers in source:"
  printf '        %s\n' "$hits"
else
  pass "no debug/scratch/slop markers in source"
fi

# ---------------------------------------------------------------------------
# 3. Honesty labels present where evidence discipline requires them.
#    Every firmware provider README must carry the SYNTHETIC label; the top
#    firmware README and the root README must carry the compliance disclaimer.
# ---------------------------------------------------------------------------
for p in firmware/openwifi firmware/openwrt firmware/nexmon firmware/esp32; do
  if [ -f "$p/README.md" ]; then
    if grep -qi 'SYNTHETIC' "$p/README.md"; then
      pass "$p/README.md carries SYNTHETIC evidence label"
    else
      bad "$p/README.md is missing the SYNTHETIC evidence label"
    fi
  fi
done
for f in firmware/README.md README.md; do
  if [ -f "$f" ]; then
    if grep -qiE 'never jamming|not jamming|not a jammer' "$f"; then
      pass "$f carries the 'never jamming' compliance disclaimer"
    else
      bad "$f is missing the 'never jamming' compliance disclaimer"
    fi
  fi
done

# ---------------------------------------------------------------------------
# 4. No dishonest hardware-success claims. Honest 'not yet MEASURED / TODO(hw) /
#    build-only' language is REQUIRED elsewhere; here we forbid only phrases that
#    assert silicon validation that does not exist. (Conservative denylist to
#    avoid false positives on the many honest negated mentions.)
# ---------------------------------------------------------------------------
dishonest='hardware[- ]validated|validated on (real )?silicon|flashed and verified|[^n]verified on silicon|confirmed on hardware|MEASURED on (real )?hardware'
# Exclude honest negated/hedged mentions (the discipline itself): "not/never
# validated on silicon", "NOT hardware-validated", "unverified", "nothing is
# validated", "SYNTHETIC ... not hardware-validated", roadmap/TODO framing, etc.
negation='\bnot\b|\bnever\b|\bno\b|\bnothing\b|\bwithout\b|unverified|unvalidated|\bwould\b|\bplanned\b|\bbefore\b|not yet|TODO|SYNTHETIC'
hwhits=$( { doc_files; code_files; } | xargs -r grep -nEiI "$dishonest" 2>/dev/null \
  | grep -viE "$negation" || true)
if [ -n "$hwhits" ]; then
  bad "dishonest hardware-success claim(s) (no captured log exists):"
  printf '        %s\n' "$hwhits"
else
  pass "no dishonest hardware-validation claims"
fi

# ---------------------------------------------------------------------------
# 5. No stale monorepo identifiers in the standalone code surface. The crate is
#    `wifi-veil` (lib `wifi_veil`); the old `wifi-densepose-privshield` name must
#    not survive in code / manifests (docs may cite the historical ADR filename).
# ---------------------------------------------------------------------------
codeset=$(printf '%s\n' "${TRACKED[@]}" | grep -E '\.(rs|toml)$|harness/(bin|src)/.*\.(js|ts)$|harness/package\.json$|harness/\.harness/manifest\.json$' || true)
if [ -n "$codeset" ]; then
  stale=$(printf '%s\n' "$codeset" | xargs -r grep -nEI 'wifi[_-]densepose[_-]privshield' 2>/dev/null || true)
  if [ -n "$stale" ]; then
    bad "stale monorepo crate/harness identifier in code/manifests:"
    printf '        %s\n' "$stale"
  else
    pass "no stale monorepo identifiers in code/manifests"
  fi
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "ci-guard: FAILED"
  exit 1
fi
echo "ci-guard: all invariants hold"
