# HOMECORE-FRONTEND Design Recon — ADR-131

**Source:** cognitum-one/v0-appliance dashboard at `http://cognitum-v0:9000/`
**Captured:** 2026-05-25 by browser-recon agent (session `20260525-181819-adr131-recon`)
**Pages fetched:** dashboard, cogs, seeds, edge, analytics, settings, cluster, tailscale, aidefence, guide (all HTTP 200)
**Auth:** dashboard is unauthenticated; `/api/*` requires bearer token — all recon confined to dashboard pages

---

## 1. Color Palette

The entire UI is dark-only. There is no light mode and no `prefers-color-scheme` media query anywhere in the stylesheet. Every surface is drawn from a tight family of near-black navy blues with two accent hues: a cool teal (`--primary`) and a green (`--accent`).

### Core tokens (hex conversions from HSL source)

| CSS variable | HSL value | Hex | Role |
|---|---|---|---|
| `--background` | `220 25% 6%` | `#0b0e13` | Page background, modal overlay base |
| `--foreground` | `210 20% 92%` | `#e6eaee` | Body text, headings |
| `--primary` | `185 80% 50%` | `#19d4e5` | Teal — active nav underline, CTA borders, ring focus, brand slash |
| `--primary-foreground` | `220 25% 6%` | `#0b0e13` | Text on filled primary buttons |
| `--accent` | `142 70% 50%` | `#26d867` | Green — secondary CTA, success state, deploy button text |
| `--accent-foreground` | `220 25% 6%` | `#0b0e13` | Text on filled accent buttons |
| `--secondary` | `220 20% 14%` | `#1c212a` | Button fill, pill-tab background |
| `--card` | `220 20% 10%` | `#14171e` | Card surface (also popover) |
| `--surface-elevated` | `220 20% 12%` | `#181c24` | Slightly elevated card variant |
| `--surface-overlay` | `220 20% 8%` | `#111318` | Modal scrim, sticky navbar |
| `--muted` | `220 15% 15%` | `#20242b` | Muted chip backgrounds, scrollbar track |
| `--muted-foreground` | `215 15% 55%` | `#7b899d` | Secondary text, labels, timestamps |
| `--border` | `220 15% 18%` | `#272b34` | All borders (at 50% opacity by default) |
| `--destructive` | `0 65% 50%` | `#d22c2c` | Error state, danger button |
| `--ring` | `185 80% 50%` | `#19d4e5` | Focus ring (same hue as primary) |

### Semantic status colors (inline, not variables)

| State | Color | Hex | Usage |
|---|---|---|---|
| Online / success | `hsl(142 70% 50%)` | `#26d867` | `.badge.online`, `.dot.up`, `.heat-cell.up` |
| Warning | `hsl(38 80% 60%)` | `#e69940` | `.badge.unpaired`, `.hero-dot.warn`, banner backgrounds |
| Error / offline | `hsl(0 65% 50%)` | `#d22c2c` | `.badge.offline`, `.badge.danger`, `.dot.down` |
| Info (log line) | `hsl(205 80% 65%)` | `#4db8f5` | Log viewer `.info` class |
| Paired | `hsl(185 80% 50%)` | `#19d4e5` | `.badge.paired` (same as primary) |

---

## 2. Typography

### Font families

The CSS declares two font families via CSS custom properties:

- `--font-display: 'Outfit', system-ui, sans-serif` — all headings, nav items, buttons, card titles, KPI values. Outfit is a modern geometric sans loaded locally (no Google Fonts outbound call; the source comment says "ship from local chrome.css fallback").
- `--font-mono: 'JetBrains Mono', monospace` — timestamps, port numbers, version strings, table cells, log output, KPI labels, chip text.

### Type scale

| Token name / usage | Size | Weight | Notes |
|---|---|---|---|
| Hero title (`h1.hero-title`) | `clamp(1.5rem, 2.4vw, 2.1rem)` | 600 | Fluid, capped at ~33.6px |
| Page h1 (`.page`) | `1.5rem` (24px) | 600 | All inner pages |
| Section heading (`.row-h h2`) | `1.125rem` (18px) | 700 | Section openers on Cogs/Dashboard |
| Card title (`.card-title`) | `0.9375rem` (15px) | 600 | |
| Body / button | `0.8125rem` (13px) | 400/500 | Default body, nav links, buttons |
| Secondary body / lede | `0.875rem` (14px) | 400 | Page lede text |
| Small label | `0.75rem` (12px) | 400–600 | Table cells, modal sub-text |
| Micro label | `0.6875rem` (11px) | 600 | Section eyebrows, uppercase KPI labels, badge text |
| Mono micro | `0.625rem` (10px) | 400 | Heatmap cells, chip category text |

Letter-spacing: `0.1em` on section eyebrows (`.section h2`), `0.08em` on filter-rail headings and chip category text, `-0.02em` on all `h1–h4` display headings. Line-height for body is `1.5`; lede text uses `1.45`.

---

## 3. Layout Primitives

### Page shell

```
┌─────────────────────────────────────────────────────────┐
│  .appbar  (sticky, z-50, backdrop-filter:blur(8px))     │
│  [brand-mark] [brand-text] [nav links scrollable]       │
├─────────────────────────────────────────────────────────┤
│  .wrap  (max-width: 1400px, padding: 1.5rem 1.25rem)   │
│    ┌── .hero (full-width, gradient bg, radial accents) │
│    ├── .kpi-grid  (auto-fill, min 170px columns)       │
│    ├── .section > h2 (eyebrow) + content               │
│    └── .grid / .grid-2 / .grid-3  (auto-fit)          │
├─────────────────────────────────────────────────────────┤
│  footer.appfoot  (border-top, centered text)           │
└─────────────────────────────────────────────────────────┘
```

**Appbar:** `position: sticky; top: 0; z-index: 50`. Background is the page background at 90% opacity with 8px blur backdrop-filter, so the page content bleeds through. Nav links overflow-scroll horizontally with a right-fade mask gradient.

**Active nav state:** primary-colored text + a 2px bottom border line (`::after` pseudo-element) positioned at bottom: -2px of the link. Hover reveals secondary background fill on the link.

**Content wrap:** max-width 1400px, centered, 1.25rem horizontal padding. Inner page sections are separated by margin-bottom spacing in multiples of 0.75rem (base unit = 12px at 16px root).

### Cogs page: app-store sub-navigation

The Cogs page adds a sticky secondary nav bar (`.subnav`) at `top: 3.25rem` (just below the appbar). Tabs are borderless buttons with a 2px bottom underline indicator when active. A `flex: 1` spacer pushes a gear icon to the right edge.

### Card patterns

Three card variants, all sharing the same surface gradient and border:

1. **Standard card (`.card`)** — `background: var(--gradient-card)` (linear 180deg from `--surface-elevated` to `--surface-overlay`), 1px border at 50% opacity, `--radius` (0.75rem), `box-shadow` 8px/32px dark drop shadow.
2. **KPI card (`.kpi`)** — 38px icon square left + text right, same gradient, 1rem/1.125rem padding, smaller vertical rhythm.
3. **Empty-state card (`.empty-card`)** — dashed 1px border (instead of solid), centered text, optional compact variant. The headline in `.empty-card h3` uses the primary teal, body explains what to do next.

### Spacing rhythm

Base unit is 4px. Gaps between grid items are universally `0.75rem` (12px). Card padding is `1.25rem` (20px) for standard, `0.875rem` (14px) for compact. Section margin-bottom is `1.5rem` (24px). The hero section uses `1.75rem` (28px) horizontal padding.

---

## 4. Component Vocabulary

### Navigation components

- **Appbar** — sticky top bar with brand + horizontal nav links. Brand mark is a 32px rounded SVG icon square.
- **Nav link** — 0.4rem × 0.7rem padding, 0.4rem radius, transitions on color + background. Active state: primary text + 2px underline pseudo-element. Mobile: wraps below brand row at 720px.
- **Sub-nav / secondary tab bar** (`.subnav`) — app-store style horizontal tab strip, sticky under appbar. Used exclusively on Cogs.
- **Pill tabs** (`.pill-tabs` + `.pill-tab`) — smaller rounded-rect tab group for in-card filter switching. Active state fills with primary color.
- **Page tabs** (`.page-tabs`) — used on Analytics for domain view switching. Underline-style, same pattern as sub-nav but at content level.

### Card & data display

- **Card** (`.card`) — base data container with gradient surface, subtle border, shadow.
- **KPI tile** (`.kpi`, `.kpi-tile`) — metric display with icon, label (uppercase micro mono), large value, and optional sub-line. Two variants: `.kpi` (icon-left layout) and `.kpi-tile` (stack layout, used on Seeds/Edge/AIDefence).
- **Node card** (`.node`) — cluster member card with mono metadata rows. Key-value pairs in `.node-meta` with dimmed label prefix (`.l` class).
- **Cog card** (`.cog`) — product-catalog card with emoji icon, name, description, category chips, and a "Get" pill button. Hover lifts 2px with primary glow border.
- **Pick card** (`.pick-card`) — horizontal-scroll featured card (220px fixed width), snap-scroll container. Smaller emoji + name + category + pill CTA.
- **Category tile small** (`.cat-tile-sm`) — 180px min-width grid item, emoji + name + count.
- **Category tile large** (`.cat-tile-big`) — 16:9 aspect-ratio card, full-bleed with gradient per category.
- **Nav tile** (`.nav-tile`) — dashboard home navigation card with icon square, title, description, and a chevron arrow that translates +2px on hover.
- **Architecture action card** (`.arch-card`, `.arch-action-card`) — setup wizard launcher cards on the dashboard.

### Status & feedback

- **Badge** (`.badge`) — pill with 1px border, 11px mono text. Variants: `role-master` (teal), `role-worker` (green), `online` (green), `offline` (red), `unknown` (muted), `paired` (teal), `unpaired` (amber), `danger` (red).
- **Dot** (`.dot`) — 8px circle status indicator. `.up` glows green with box-shadow, `.down` is red, default is muted gray.
- **Hero dot** (`.hero-dot`) — 7px circle in the dashboard hero status row. Same three states: `.ok` (green glow), `.warn` (amber glow), `.down` (red glow).
- **Op-pill** (`.op-pill`) — "operational status" pill with colored dot inside. Used in dashboard architecture hub.
- **AI pill / status chip** (`.pill` on AIDefence, `.md-badge` in cluster) — inline classification badge at 0.68rem. States: `.ok`, `.warn`, `.bad`.
- **Chip** (`.chip`) — tiny category/difficulty label, all-caps, 0.5625rem, pill-shaped. Category-colored variants (`.cat-ai`, `.cat-health`, `.cat-security`, etc.) each get a hue-appropriate 15% opacity background.

### Actions

- **Button** (`.btn`) — 0.5rem × 0.875rem padding, 0.4rem radius, secondary fill. Variants: `.primary` (filled teal, 600 weight, box-shadow), `.outline` (transparent fill), `.danger` (red tint), `.sm` (compact).
- **Hero button** (`.hero-btn`) — slightly larger, display-font, 0.9rem padding, glass-effect dark fill. `.primary` variant uses the green accent gradient.
- **Pill CTA** (`.get`, `.pget`) — full pill-radius (9999px), primary-tint background at rest, fills solid on hover. Used on cog cards and pick cards.
- **Gear button** (`.gear-btn`) — icon-only square button, transparent at rest, border appears on hover.
- **Context menu** (`.ctx-menu`) — dark card dropdown (min-width 180px), each item is a full-width button with secondary hover fill.
- **Copy button** (`.copy-btn`) — positioned absolute in `.copy-row`, 0.7rem opacity at rest, `.copied` state turns green/accent.

### Forms & inputs

- **Input** — all `<input>`, `<textarea>`, `<select>` inherit dark theme globally. Focus ring: 2px solid primary at 30% opacity (`box-shadow: 0 0 0 2px hsl(var(--ring) / 0.3)`). Checkboxes and radios use `accent-color: hsl(var(--primary))`.
- **Collapsible section** (`.coll`, `.coll-h`, `.coll-body`) — used in Settings page. Header row is clickable with `user-select: none`. Body `display: none` by default, revealed on expand.
- **Key-value row** (`.kv`) — 3-column grid (160px label | 1fr value | auto action) for settings display.
- **Filters rail** (`.filters-rail`) — sticky sidebar on Cogs/Apps tab. Sticky at `top: 7rem` (below both navbars). Contains checkboxes, a range input, and a reset button.
- **Range input** — native `<input type="range">` styled with `accent-color: hsl(var(--primary))`.

### Data visualization

- **Heatmap** (`.heatmap`) — CSS grid of 14px × variable cells. 60 time columns, label column at 90px. Cell states: `up` (green 70%), `down` (red 70%), `empty` (muted 30%).
- **Bar chart** (`.bar-list` + `.bar-row` + `.bar-fill`) — horizontal bar list, 3-col grid (120px label | 1fr bar | 30px value). Bar fill transitions width in 0.3s.
- **uPlot time-series** (`.uplot-host`) — 200px height host container; actual charting via uPlot library.
- **Three.js 3D** — importmap for `three` + `OrbitControls` in Analytics page, for 3D sensor visualization.
- **Log box** (`pre.logbox`) — monospace pre-formatted block, max-height 30rem, overflow-y scroll. Dark background on dark background gives subtle separation via border.
- **OTA row table** (`.ota-row`) — 3-col grid (160px | 80px | 1fr) for firmware OTA records.

### Overlays

- **Modal** (`.modal-bg` + `.modal`) — fixed inset, 70% opacity blur-backdrop scrim. Modal itself is card-surfaced, max-width 560px. Result states: `.modal-result.ok` (green tint) and `.modal-result.err` (red tint).
- **Detail modal** (`.detail-modal-bg` + `.detail-modal`) — larger variant (max 820px, 2rem padding) used on Cog detail view. Header has emoji, name, meta chips; sections below are tabbed.
- **Keyboard shortcut tag** (`.kb`) — small monospace tag with secondary background, used inline in Settings and Tailscale pages to show keyboard shortcuts.

---

## 5. Iconography

All icons are inline SVG, 24×24 viewBox, `fill: none`, `stroke: currentColor`, `stroke-width: 2`. The path geometry is **Lucide Icons** — confirmed by comparing the Sun/gear/shield/grid/activity paths against Lucide's source. Key examples observed:

- Sun/rays (brand mark, dashboard hero)
- Settings/gear (nav, subnav gear button)
- Activity/pulse (KPI signal icon)
- Bar chart 3 (analytics KPI)
- Grid 2×2 (cluster/cog layout)
- Shield with checkmark (AIDefence)
- House (home nav tile)
- Book-open (guide nav)

No external icon font is used. Every icon is self-contained in the HTML at point of use — no sprite sheet.

---

## 6. Dark Mode

The design is **dark-only**. There is no `prefers-color-scheme: light` media query in `v0-chrome.css` or any page-level stylesheet. The color system is entirely designed around the dark palette above. The source comments explicitly note that `fonts.googleapis.com` is blocked for Tailnet isolation, reinforcing that this is an always-dark appliance UI, not a consumer product that needs theming.

Surface hierarchy (light to dark, within the dark palette):
1. `--surface-elevated` (`#181c24`) — slightly lighter card variant
2. `--card` (`#14171e`) — standard card
3. `--surface-overlay` (`#111318`) — modal/sticky appbar base
4. `--background` (`#0b0e13`) — page root

The appbar uses `background: hsl(var(--background) / 0.9)` + `backdrop-filter: blur(8px)` so content underneath bleeds through as a translucency effect.

---

## 7. Notable Interactions

- **Nav hover:** 150ms color + background transition, no translate. Active state uses a 2px pseudo-element underline that animates in via opacity.
- **Nav link active press:** `transform: translateY(1px)` on `:active` at 50ms — very subtle tactile response.
- **Card hover:** `transform: translateY(-2px)` at 200ms on cards and cog items. Border shifts from `--border/0.5` to `primary/0.4` on hover. On the nav tiles, box-shadow deepens.
- **Hero button hover:** `transform: translateY(-1px)` + border-color shift to primary at 70%.
- **Pick card hover:** translateY(-2px) + primary-glow box-shadow.
- **Focus ring:** 2px solid primary at 30% opacity as box-shadow — uses `outline: none` everywhere and replaces it with the ring shadow. nav links use `outline: 2px solid hsl(var(--primary)/0.6); outline-offset: 1px` for focus-visible.
- **Bar fill animation:** `transition: width 0.3s` on bar chart fill elements for data-load entrance.
- **Modal backdrop:** `backdrop-filter: blur(4px)` on modal scrim, `blur(6px)` on the Cog detail modal.
- **Copy button feedback:** `.copied` state class swaps border and text to accent green, visible for a short duration (JS-controlled).
- **Pill CTA:** Background fills from 15% opacity teal to 100% solid on hover — a strong affordance for primary actions.
- **Scroll fade mask:** The nav bar has `mask-image: linear-gradient(to right, black calc(100% - 24px), transparent)` to fade out the rightmost item, hinting at horizontal scroll.
- **Cogs hero carousel:** Paginator dots expand from 0.55rem circles to 1.5rem pill shape (border-radius 0.4rem) when active — a distinctive indicator pattern.

---

## 8. HA-Parity Opportunities

For ADR-131 P2, the following comparisons are relevant between this design and Home Assistant's frontend (`home-assistant-main`):

| HOMECORE component | Cognitum V0 pattern | HA equivalent | Better reference |
|---|---|---|---|
| KPI metric card | `.kpi` — icon + label + value | `ha-statistic-card`, `sensor-badge` | **Cognitum** — cleaner dense layout; HA's is more verbose |
| Status badge/pill | `.badge` + `.chip` — pill with 1px border | `ha-label-badge`, `state-badge` | **HA** — HA has more state variants and i18n built in |
| Dark surface cards | `--gradient-card` linear gradient | HA uses flat `var(--card-background-color)` | **Cognitum** — gradient gives depth HA lacks |
| Toggle/switch | `accent-color` native checkbox | HA `ha-switch` (Material) | **HA** — purpose-built, accessible, animated |
| Navigation | Horizontal sticky nav, underline indicator | HA sidebar (vertical) | Neither — HOMECORE needs a new shell; Cognitum's horizontal bar is appropriate for appliance context |
| Heatmap timeline | CSS grid `.heatmap` | No HA equivalent | **Cognitum** — take this pattern directly |
| Bar chart | CSS-only `.bar-fill` bar list | HA uses Recharts | **Cognitum** — zero-dep CSS bars good for simple metrics; use for small cards |
| Time-series chart | uPlot `.uplot-host` | HA uses ApexCharts / Recharts | **HA** — ApexCharts has more features, better RTL support |
| Modal | `.modal-bg` blur-backdrop | HA `ha-dialog` (Material) | **HA** — a11y and focus-trap already solved |
| Toast / alert banner | `.modal-result.ok/err` inline result + `.cl-banner.warn/err` | HA `ha-alert` | **HA** — HA's alerts are more composable |
| Focus ring | `box-shadow` ring pattern | HA uses `:focus-visible` outline | **HA** — HA's approach has better browser compatibility |
| Chip (category) | `.chip.cat-*` per-category color mapping | HA `ha-chip` | **Cognitum** — the category-specific hue mapping is richer |

---

## 9. Design Tokens for HOMECORE-FRONTEND P1

Concrete CSS variable names and starting values for the TypeScript+WASM frontend to adopt. These follow the Cognitum V0 source directly, adjusted where needed for HOMECORE context.

```css
:root {
  /* Surfaces */
  --hc-bg:                   hsl(220 25% 6%);      /* #0b0e13 — page root */
  --hc-surface-card:         hsl(220 20% 10%);     /* #14171e — card fill */
  --hc-surface-elevated:     hsl(220 20% 12%);     /* #181c24 — raised panel */
  --hc-surface-overlay:      hsl(220 20% 8%);      /* #111318 — modal/nav base */

  /* Text */
  --hc-text:                 hsl(210 20% 92%);     /* #e6eaee — primary text */
  --hc-text-muted:           hsl(215 15% 55%);     /* #7b899d — secondary/label */

  /* Accent palette */
  --hc-primary:              hsl(185 80% 50%);     /* #19d4e5 — teal, primary actions */
  --hc-primary-fg:           hsl(220 25% 6%);      /* #0b0e13 — text on primary */
  --hc-accent:               hsl(142 70% 50%);     /* #26d867 — green, success/CTA */
  --hc-accent-fg:            hsl(220 25% 6%);      /* #0b0e13 — text on accent */
  --hc-destructive:          hsl(0 65% 50%);       /* #d22c2c — error/danger */
  --hc-warning:              hsl(38 80% 60%);      /* #e69940 — warning/amber */

  /* Borders & rings */
  --hc-border:               hsl(220 15% 18%);     /* #272b34 — subtle border */
  --hc-ring:                 hsl(185 80% 50%);     /* #19d4e5 — focus ring */

  /* Radii */
  --hc-radius:               0.75rem;              /* cards, modals */
  --hc-radius-sm:            0.4rem;               /* buttons, inputs, chips */
  --hc-radius-pill:          9999px;               /* badges, CTA pills */

  /* Typography */
  --hc-font-display:         'Outfit', system-ui, sans-serif;
  --hc-font-mono:            'JetBrains Mono', monospace;

  /* Shadows */
  --hc-shadow-card:          0 8px 32px -8px hsl(220 25% 2% / 0.8);
  --hc-shadow-glow:          0 0 60px -10px hsl(185 80% 50% / 0.3);

  /* Gradients */
  --hc-gradient-card:        linear-gradient(180deg, hsl(220 20% 12%) 0%, hsl(220 20% 8%) 100%);
}
```

**Notes for P1 implementation:**

- Adopt Outfit + JetBrains Mono from Google Fonts in development; ship local fallbacks for production (Tailnet appliances block outbound font requests per the Cognitum source comment).
- The `--hc-ring` focus approach should be implemented as `box-shadow: 0 0 0 2px hsl(var(--hc-ring) / 0.3)` combined with `outline: none` — matches Cognitum's pattern and avoids the offset-gap issue in Firefox.
- Add `--hc-gradient-hero` and `--hc-gradient-glow` when the dashboard hero section is built; keep them out of the P1 design-token foundation to avoid premature complexity.
- The `--hc-warning` amber is not in the Cognitum `:root` block (it is inline throughout) — elevating it to a token is a deliberate improvement for HOMECORE.
