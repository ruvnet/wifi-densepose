// SPDX-License-Identifier: MIT
// RuView harness guardrails — the "prove everything" rule made executable.
//
// The project was accused of AI-slop; the cultural fix is that every accuracy
// number must be tagged MEASURED (with a reproducer) or CLAIMED/SYNTHETIC, and
// the retracted "100% accuracy" framing must never reappear untagged. This module
// is the static enforcement of that, shared by the `ruview.claim_check` MCP tool,
// the `npx ruview claim-check` CLI, and the claude-code pre-output hook.

/** Phrases that signal a quantitative accuracy claim. */
const METRIC_TERMS = [
  'accuracy', 'pck', 'pck@', 'f1', 'precision', 'recall', 'map', 'auc',
  'iou', 'mpjpe', 'error rate', 'detection rate', 'true positive',
];

/** Tags that make a claim honest (case-insensitive). */
const HONEST_TAGS = ['measured', 'claimed', 'synthetic', 'unvalidated', 'baseline'];

/** Reproducer references that count as evidence backing a MEASURED claim. */
const REPRODUCER_HINTS = [
  'verify.py', 'witness', 'mean-pose', 'mean pose', 'held-out', 'held out',
  'baseline', 'reproduce', 'sha256', 'boot log', 'pck@20 vs', 'expected_features',
];

const PERCENT_RE = /\b(\d{1,3}(?:\.\d+)?)\s?%/g;
// "perfect" / "100%" framing is the specific retracted claim — always high severity.
// NOTE: no trailing \b after "%": "%"→" " is non-word→non-word, so a trailing \b
// never matches and would silently miss "100%". Bare 100% is only damning next to a
// metric term (see claimCheck); the word phrases are inherently accuracy claims.
const PERFECT_PCT_RE = /\b100(?:\.0+)?\s?%/;
const PERFECT_WORD_RE = /perfect accuracy|flawless|never (?:wrong|fails)/i;

/**
 * Lint a block of text for untagged or overstated accuracy claims.
 * @param {string} text
 * @returns {{ok: boolean, findings: Array<{severity:'high'|'medium', line:number, excerpt:string, reason:string, suggestion:string}>}}
 */
export function claimCheck(text) {
  const findings = [];
  if (typeof text !== 'string' || text.length === 0) {
    return { ok: true, findings };
  }
  const lines = text.split(/\r?\n/);

  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (!line) return;
    const lower = line.toLowerCase();

    const hasPercent = PERCENT_RE.test(line);
    PERCENT_RE.lastIndex = 0; // reset stateful global regex
    const mentionsMetric = METRIC_TERMS.some((t) => lower.includes(t));
    if (!hasPercent && !mentionsMetric) return;

    const tagged = HONEST_TAGS.some((t) => lower.includes(t));
    const hasReproducer = REPRODUCER_HINTS.some((h) => lower.includes(h));
    const perfect = PERFECT_WORD_RE.test(line) || (mentionsMetric && PERFECT_PCT_RE.test(line));

    if (perfect && !lower.includes('retract')) {
      findings.push({
        severity: 'high',
        line: i + 1,
        excerpt: clip(line),
        reason: 'States perfect/100% accuracy — this is the exact framing the project retracted.',
        suggestion: 'Replace with a held-out number vs the mean-pose baseline, tagged MEASURED, or mark the old claim "retracted".',
      });
      return;
    }

    // A metric/percent with no honesty tag at all.
    if (!tagged) {
      findings.push({
        severity: 'medium',
        line: i + 1,
        excerpt: clip(line),
        reason: 'Accuracy claim is not tagged MEASURED / CLAIMED / SYNTHETIC.',
        suggestion: 'Tag it. If MEASURED, name the reproducer (verify.py, witness bundle, held-out vs mean-pose).',
      });
      return;
    }

    // Tagged MEASURED but cites no reproducer — still a gap.
    if (lower.includes('measured') && !hasReproducer) {
      findings.push({
        severity: 'medium',
        line: i + 1,
        excerpt: clip(line),
        reason: 'Tagged MEASURED but cites no reproducer/evidence.',
        suggestion: 'Add the evidence path: verify.py VERDICT, witness bundle, or held-out PCK vs the mean-pose baseline.',
      });
    }
  });

  return { ok: findings.length === 0, findings };
}

function clip(s, n = 120) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

/** Convenience: a one-line human summary for CLI output. */
export function summarize(result) {
  if (result.ok) return 'claim-check: PASS — no untagged or overstated accuracy claims.';
  const high = result.findings.filter((f) => f.severity === 'high').length;
  return `claim-check: ${result.findings.length} finding(s) (${high} high) — accuracy claims need MEASURED/CLAIMED tags + a reproducer.`;
}
