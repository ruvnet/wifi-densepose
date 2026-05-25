/**
 * `<hc-state-card>` — renders one HOMECORE entity state in the cognitum-v0 card style.
 * Uses Lit 3 (LitElement + html/css template tags).
 */

import { LitElement, html, css, nothing } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import type { StateView } from '../api/types.js';

@customElement('hc-state-card')
export class StateCard extends LitElement {
  @property({ type: Object }) state!: StateView;
  /** Optional: icon SVG string (use `iconSvg()` from lucide.ts) */
  @property({ type: String }) iconSvg?: string;

  static styles = css`
    :host {
      display: block;
    }

    .card {
      background: var(--hc-gradient-card, linear-gradient(180deg, #181c24 0%, #111318 100%));
      border: 1px solid hsl(220 15% 18% / 0.5);
      border-radius: var(--hc-radius, 0.75rem);
      box-shadow: var(--hc-shadow-card, 0 8px 32px -8px hsl(220 25% 2% / 0.8));
      padding: 1.25rem;
      transition: transform 200ms, border-color 200ms;
    }

    .card:hover {
      transform: translateY(-2px);
      border-color: hsl(185 80% 50% / 0.4);
    }

    .header {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }

    .icon-wrap {
      flex-shrink: 0;
      width: 38px;
      height: 38px;
      border-radius: var(--hc-radius-sm, 0.4rem);
      background: hsl(220 20% 14%);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--hc-primary, #19d4e5);
    }

    .meta { flex: 1; min-width: 0; }

    .entity-id {
      font-family: var(--hc-font-mono, 'JetBrains Mono', monospace);
      font-size: 0.6875rem;
      font-weight: 600;
      color: var(--hc-text-muted, #7b899d);
      text-overflow: ellipsis;
      overflow: hidden;
      white-space: nowrap;
      letter-spacing: 0.05em;
    }

    .state-value {
      font-family: var(--hc-font-display, 'Outfit', system-ui, sans-serif);
      font-size: 1.125rem;
      font-weight: 600;
      color: var(--hc-text, #e6eaee);
      letter-spacing: -0.02em;
      margin-top: 0.2rem;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      padding: 0.15rem 0.5rem;
      border-radius: 9999px;
      border: 1px solid var(--hc-border, #272b34);
      font-family: var(--hc-font-mono, monospace);
      font-size: 0.6875rem;
      font-weight: 600;
    }

    .badge.on  { color: #26d867; border-color: hsl(142 70% 50% / 0.4); }
    .badge.off { color: #d22c2c; border-color: hsl(0 65% 50% / 0.4); }

    .timestamp {
      font-family: var(--hc-font-mono, monospace);
      font-size: 0.625rem;
      color: var(--hc-text-muted, #7b899d);
      margin-top: 0.75rem;
    }
  `;

  private badgeClass(state: string): string {
    const s = state.toLowerCase();
    if (s === 'on' || s === 'open' || s === 'home' || s === 'running') return 'on';
    if (s === 'off' || s === 'closed' || s === 'away' || s === 'unavailable') return 'off';
    return '';
  }

  render() {
    if (!this.state) return nothing;
    const { entity_id, state, last_updated } = this.state;
    const badge = this.badgeClass(state);

    return html`
      <div class="card" part="card">
        <div class="header">
          ${this.iconSvg
            ? html`<div class="icon-wrap" .innerHTML=${this.iconSvg}></div>`
            : nothing}
          <div class="meta">
            <div class="entity-id" title=${entity_id}>${entity_id}</div>
            <div class="state-value">${state}</div>
          </div>
          <span class="badge ${badge}">${state}</span>
        </div>
        <div class="timestamp">updated ${new Date(last_updated).toLocaleTimeString()}</div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'hc-state-card': StateCard;
  }
}
