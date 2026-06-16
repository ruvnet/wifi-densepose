# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-06-04
- Primary product surfaces: Desktop RuView UI under `ui/`
- Evidence reviewed: `ui/index.html`, `ui/style.css`, `ui/app.js`

## Brand
- Personality: Vintage Swiss graphic design, precise, technical, poster-like.
- Trust signals: Live hardware status, explicit unavailable states, clear typographic hierarchy.
- Avoid: Synthetic demo polish, soft SaaS gradients, rounded card-heavy presentation.

## Product goals
- Goals: Make live sensing state readable at a glance and visually distinctive.
- Non-goals: Marketing landing page, fake telemetry, decorative illustration.
- Success signals: Status, node, metric, and sensing surfaces remain scannable on desktop and mobile.

## Personas and jobs
- Primary personas: Local operator, hardware debugger, demo observer.
- User jobs: Confirm hardware data availability, inspect person counts, monitor stream health.
- Key contexts of use: Steam Deck or desktop browser near live hardware.

## Information architecture
- Primary navigation: Tabbed sections for dashboard, hardware, live demo, architecture, performance, applications, sensing, and training.
- Core routes/screens: `index.html`, `pose-fusion.html`, `observatory.html`.
- Content hierarchy: System truth first, technical dashboards second, explanatory material lower.

## Design principles
- Principle 1: Use grid, type, and hard rules as the primary visual system.
- Principle 2: Live/unavailable data states must be more important than decorative copy.
- Tradeoffs: The style is intentionally flatter and sharper than modern card UI.

## Visual language
- Color: Warm paper, black ink, Swiss red primary, blue live accent, yellow warning/accent.
- Typography: Helvetica-style sans serif with mono labels for machine/status metadata.
- Spacing/layout rhythm: 48px grid background, boxed modular panels, dense but readable status grids.
- Shape/radius/elevation: Square corners, hard borders, no shadows.
- Motion: Minimal; avoid motion that implies live data when unavailable.
- Imagery/iconography: Prefer geometric color blocks and typographic markers over emoji decoration.

## Components
- Existing components to reuse: Header, nav tabs, dashboard panels, status cells, stats, mobile drawer.
- New/changed components: Swiss theme override layer in `ui/style.css`.
- Variants and states: Healthy/live blue, warning yellow, error red, unavailable neutral.
- Token/component ownership: CSS custom properties in `ui/style.css`.

## Accessibility
- Target standard: Practical WCAG AA contrast for core text and status surfaces.
- Keyboard/focus behavior: Preserve visible focus rings.
- Contrast/readability: Black on warm paper, white on black/red headers.
- Screen-reader semantics: Preserve existing roles and live regions.
- Reduced motion and sensory considerations: Respect existing reduced-motion rules.

## Responsive behavior
- Supported breakpoints/devices: Desktop browser and narrow Steam Deck/mobile widths.
- Layout adaptations: Header collapses to one column, stat grids reduce to two then one column.
- Touch/hover differences: Hover color is decorative only; active state remains explicit.

## Interaction states
- Loading: Existing text/status placeholders remain visible.
- Empty: Unavailable hardware states are explicit, not masked by mock visuals.
- Error: Red hard-rule treatment.
- Success: Blue live treatment.
- Disabled: Existing opacity treatment.
- Offline/slow network: Neutral unavailable state with no synthetic frames.

## Content voice
- Tone: Direct, technical, terse.
- Terminology: Use live, unavailable, hardware, packets, persons.
- Microcopy rules: Do not imply simulated/demo data is real.

## Implementation constraints
- Framework/styling system: Static HTML/CSS/JS served by `/home/deck/bin/ruview-ui-server.py`.
- Design-token constraints: Use existing CSS variables and append scoped overrides.
- Performance constraints: No new runtime dependencies.
- Compatibility constraints: Keep launcher-served static files browser-native.
- Test/screenshot expectations: CSS syntax check plus live launcher smoke check.

## Open questions
- [ ] Whether secondary pages should receive a deeper layout-specific Swiss redesign beyond shared tokens.
