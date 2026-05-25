/**
 * `<hc-app-shell>` — top-level layout: sticky header + horizontal sidenav + content slot.
 * Page shell mirrors cognitum-v0's appbar + wrap layout (ADR-131 §3).
 */

import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';

export interface NavItem {
  id: string;
  label: string;
  /** Raw SVG string for the icon */
  iconSvg?: string;
}

const DEFAULT_NAV: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'states', label: 'States' },
  { id: 'services', label: 'Services' },
  { id: 'settings', label: 'Settings' },
];

@customElement('hc-app-shell')
export class AppShell extends LitElement {
  @property({ type: String }) locationName = 'HOMECORE';
  @property({ type: String }) version = '0.1.0';
  @property({ type: Array }) navItems: NavItem[] = DEFAULT_NAV;
  @state() private activeId = 'dashboard';

  static styles = css`
    :host { display: block; min-height: 100dvh; background: var(--hc-bg, #0b0e13); }

    /* ── Appbar ── */
    .appbar {
      position: sticky;
      top: 0;
      z-index: 50;
      background: hsl(220 25% 6% / 0.9);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      border-bottom: 1px solid hsl(220 15% 18% / 0.8);
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 0 1.25rem;
      height: 3.25rem;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-family: var(--hc-font-display, 'Outfit', system-ui, sans-serif);
      font-weight: 600;
      font-size: 0.9375rem;
      color: var(--hc-text, #e6eaee);
      white-space: nowrap;
      flex-shrink: 0;
    }

    .brand-icon {
      width: 32px;
      height: 32px;
      border-radius: 0.4rem;
      background: var(--hc-primary, #19d4e5);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--hc-primary-fg, #0b0e13);
      font-size: 1rem;
      font-weight: 700;
    }

    .nav {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      overflow-x: auto;
      scrollbar-width: none;
      flex: 1;
      mask-image: linear-gradient(to right, black calc(100% - 24px), transparent);
    }
    .nav::-webkit-scrollbar { display: none; }

    .nav-link {
      position: relative;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.4rem 0.7rem;
      border-radius: 0.4rem;
      font-family: var(--hc-font-display, 'Outfit', system-ui, sans-serif);
      font-size: 0.8125rem;
      font-weight: 500;
      color: var(--hc-text-muted, #7b899d);
      background: transparent;
      border: none;
      cursor: pointer;
      white-space: nowrap;
      transition: color 150ms, background 150ms;
    }

    .nav-link:hover {
      color: var(--hc-text, #e6eaee);
      background: hsl(220 20% 14%);
    }

    .nav-link:focus-visible {
      outline: 2px solid hsl(185 80% 50% / 0.6);
      outline-offset: 1px;
    }

    .nav-link:active { transform: translateY(1px); }

    .nav-link.active { color: var(--hc-primary, #19d4e5); }

    .nav-link.active::after {
      content: '';
      position: absolute;
      bottom: -2px;
      left: 0.7rem;
      right: 0.7rem;
      height: 2px;
      background: var(--hc-primary, #19d4e5);
      border-radius: 9999px;
    }

    .version-chip {
      font-family: var(--hc-font-mono, 'JetBrains Mono', monospace);
      font-size: 0.6875rem;
      color: var(--hc-text-muted, #7b899d);
      white-space: nowrap;
      flex-shrink: 0;
    }

    /* ── Main content ── */
    main {
      max-width: 1400px;
      margin-inline: auto;
      padding-inline: 1.25rem;
      padding-block: 1.5rem;
    }

    /* ── Footer ── */
    footer {
      border-top: 1px solid hsl(220 15% 18%);
      text-align: center;
      padding: 1rem 1.25rem;
      font-family: var(--hc-font-mono, 'JetBrains Mono', monospace);
      font-size: 0.75rem;
      color: var(--hc-text-muted, #7b899d);
    }
  `;

  private onNavClick(id: string) {
    this.activeId = id;
    this.dispatchEvent(new CustomEvent('hc-navigate', { detail: { id }, bubbles: true, composed: true }));
  }

  render() {
    return html`
      <header class="appbar" part="appbar">
        <div class="brand">
          <div class="brand-icon" aria-hidden="true">H</div>
          ${this.locationName}
        </div>
        <nav class="nav" aria-label="Primary navigation">
          ${this.navItems.map(item => html`
            <button
              class="nav-link ${this.activeId === item.id ? 'active' : ''}"
              @click=${() => this.onNavClick(item.id)}
              aria-current=${this.activeId === item.id ? 'page' : 'false'}
            >${item.label}</button>
          `)}
        </nav>
        <span class="version-chip">v${this.version}</span>
      </header>

      <main part="content">
        <slot></slot>
      </main>

      <footer part="footer">
        HOMECORE &mdash; ${this.locationName} &mdash; v${this.version}
      </footer>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'hc-app-shell': AppShell;
  }
}
