/**
 * Validates that tokens.css contains all 16 documented HOMECORE design tokens.
 * Reads the file from disk and checks for each CSS custom property name.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const tokensPath = resolve(__dirname, '../styles/tokens.css');
const css = readFileSync(tokensPath, 'utf-8');

/**
 * The 16 design tokens from ADR-131 §9 / HOMECORE-FRONTEND-design-recon.md §1.
 * 4 surfaces + 2 text + 6 accent/status + 2 border/ring + 2 radius = 16 tokens.
 */
const REQUIRED_TOKENS = [
  // Surfaces (4)
  '--hc-bg',
  '--hc-surface-card',
  '--hc-surface-elevated',
  '--hc-surface-overlay',
  // Text (2)
  '--hc-text',
  '--hc-text-muted',
  // Accent palette (6)
  '--hc-primary',
  '--hc-primary-fg',
  '--hc-accent',
  '--hc-accent-fg',
  '--hc-destructive',
  '--hc-warning',
  // Borders & rings (2)
  '--hc-border',
  '--hc-ring',
  // Radii (2)
  '--hc-radius',
  '--hc-radius-sm',
] as const;

describe('tokens.css', () => {
  it('contains all 16 documented design tokens', () => {
    for (const token of REQUIRED_TOKENS) {
      expect(css, `Missing token: ${token}`).toContain(token);
    }
  });

  it('has exactly 16 (or more) --hc- custom properties', () => {
    const matches = css.match(/--hc-[\w-]+\s*:/g) ?? [];
    // De-duplicate (token may appear in comments)
    const unique = new Set(matches.map(m => m.replace(/\s*:/, '')));
    expect(unique.size).toBeGreaterThanOrEqual(16);
  });

  it('defines the teal primary token with the correct hue value', () => {
    // --hc-primary must reference HSL hue 185 (teal, from cognitum-v0)
    expect(css).toMatch(/--hc-primary\s*:\s*hsl\(185/);
  });

  it('defines the green accent token (#26d867)', () => {
    // --hc-accent must reference HSL 142 70% 50%
    expect(css).toMatch(/--hc-accent\s*:\s*hsl\(142/);
  });
});
