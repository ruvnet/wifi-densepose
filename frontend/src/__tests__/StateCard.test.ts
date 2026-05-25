/**
 * Unit tests for <hc-state-card>.
 * Verifies that the component renders entity_id and state value into the DOM.
 *
 * Uses jsdom (via vitest environment) — no real browser required.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import type { StateView } from '../api/types.js';

// Register the custom element before tests run
beforeAll(async () => {
  // jsdom does not support Lit's adoptedStyleSheets; suppress the error.
  if (typeof document !== 'undefined' && !document.adoptedStyleSheets) {
    Object.defineProperty(document, 'adoptedStyleSheets', { value: [], writable: true });
  }
  await import('../components/StateCard.js');
});

function makeState(overrides: Partial<StateView> = {}): StateView {
  return {
    entity_id: 'light.living_room',
    state: 'on',
    attributes: { brightness: 255 },
    last_changed: '2026-05-25T10:00:00Z',
    last_updated: '2026-05-25T10:00:00Z',
    context: { id: 'abc123', user_id: null, parent_id: null },
    ...overrides,
  };
}

describe('StateCard', () => {
  it('renders entity_id in the DOM', async () => {
    const el = document.createElement('hc-state-card') as HTMLElement & { state: StateView };
    el.state = makeState();
    document.body.appendChild(el);

    // Lit renders synchronously into shadow root after a microtask
    await el.updateComplete;

    const shadowRoot = el.shadowRoot!;
    const entityEl = shadowRoot.querySelector('.entity-id');
    expect(entityEl).not.toBeNull();
    expect(entityEl!.textContent).toContain('light.living_room');

    document.body.removeChild(el);
  });

  it('renders the state value', async () => {
    const el = document.createElement('hc-state-card') as HTMLElement & { state: StateView };
    el.state = makeState({ state: 'off' });
    document.body.appendChild(el);

    await el.updateComplete;

    const stateEl = el.shadowRoot!.querySelector('.state-value');
    expect(stateEl).not.toBeNull();
    expect(stateEl!.textContent).toBe('off');

    document.body.removeChild(el);
  });

  it('applies .off badge class for unavailable state', async () => {
    const el = document.createElement('hc-state-card') as HTMLElement & { state: StateView };
    el.state = makeState({ state: 'unavailable' });
    document.body.appendChild(el);

    await el.updateComplete;

    const badge = el.shadowRoot!.querySelector('.badge.off');
    expect(badge).not.toBeNull();

    document.body.removeChild(el);
  });
});

// Augment for updateComplete
declare global {
  interface HTMLElement {
    updateComplete: Promise<boolean>;
  }
}
