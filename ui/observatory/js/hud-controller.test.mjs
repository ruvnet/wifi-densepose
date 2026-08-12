// Regression coverage for issue #1527.
//
// This exercises HudController against a minimal DOM boundary. It verifies
// rendered text synchronization, not a browser, animation timing, or hardware.

import { test } from 'node:test';
import assert from 'node:assert/strict';

function makeElement() {
  const classes = new Set();
  return {
    textContent: '',
    innerHTML: '',
    value: '',
    className: '',
    style: {},
    classList: {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      contains: (name) => classes.has(name),
    },
  };
}

const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, makeElement());
  return elements.get(id);
}

globalThis.document = {
  getElementById(id) {
    if (id === 'rssi-sparkline') return { getContext: () => null };
    return element(id);
  },
};

const { HudController } = await import('./hud-controller.js');

test('each HUD frame repairs a stale scenario caption even when the key is unchanged', () => {
  elements.clear();
  const hud = new HudController({ settings: {} });
  const demoData = {
    _autoMode: true,
    currentScenario: 'single_breathing',
  };
  const data = {
    vital_signs: { heart_rate_bpm: 72, breathing_rate_bpm: 14 },
    features: { mean_rssi: -48, variance: 0.3, motion_band_power: 0.02 },
    classification: { confidence: 0.9, presence: true, motion_level: 'present' },
    estimated_persons: 1,
  };

  // Reproduce the stale-load state: internal scenario tracking already agrees
  // with the frame, while the visible caption still describes an empty room.
  hud._currentScenarioKey = 'single_breathing';
  element('scenario-description').textContent =
    'Baseline calibration with no human presence in the monitored zone.';

  hud.updateHUD(data, demoData);

  assert.equal(
    element('scenario-description').textContent,
    'Detecting vital signs through WiFi signal micro-variations.',
  );
  assert.equal(element('persons-value').textContent, 1);
});
