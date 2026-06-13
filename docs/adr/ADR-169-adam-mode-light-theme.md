# ADR-169: adam-mode — light theme toggle for the three.js realtime demo

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-06-02 |
| **Deciders** | ruv |
| **Codename** | **adam-mode** |
| **Scope** | `examples/three.js/demos/05-skinned-realtime.html` (primary), demos 01–04 (follow-on) |
| **Relates to** | ADR-019 (sensing-only UI), ADR-035 (live sensing UI accuracy) |
| **Tracking issue** | none yet |

---

## 1. Context

`examples/three.js/demos/05-skinned-realtime.html` (build stamp `2026-05-15-fps-tune`) is the live MediaPipe → Mixamo retargeting + ESP32 CSI overlay demo. It currently ships a single, opinionated **dark theme**:

- Body `--bg: #050507` (near-black), `--text: #d8c69a` (warm beige).
- Amber accents (`--amber: #ffb840`, `--amber-hot: #ffe09f`) on panels and controls.
- Two full-screen overlays: a radial-vignette `.overlay-frame` and a 50%-opacity CRT-style `.scanlines` layer.
- Three.js scene matches: `scene.background = new THREE.Color(0x050507)` and `scene.fog = new THREE.FogExp2(0x050507, 0.06)` (lines 269–270).

The dark/amber CRT aesthetic is intentional for screen-recording and "command-centre" feel, but it has real failure modes:

1. **Daylight visibility** — Demoing the live capture on a laptop in a sunlit room is unreadable; the dark background absorbs ambient glare and the amber-on-dark contrast disappears.
2. **Recording for embedded/print contexts** — When the demo's screen is captured for documentation, blog posts, or HA blueprints, the dark theme bleeds into surrounding white content and looks heavy.
3. **Accessibility** — A subset of users with light-sensitive retinas (the inverse of typical photophobia) report the high amber-on-near-black combination strains them; high-contrast light themes are easier.
4. **Operator pairing with a light-mode IDE** — Many operators run a light-mode browser alongside a dark-mode IDE and want the demo to match the browser, not the IDE.

A toggle is the right answer because none of these reasons are universal — some sessions and some users want each mode.

### 1.1 What this ADR is *not*

- Not a redesign. The amber accent stays; only the surface colours and overlays swap. The information density, panel layout, and three.js scene geometry are unchanged.
- Not a multi-theme system. We add exactly two themes: the existing dark (default, unnamed) and **adam-mode** (light). Future themes would need a new ADR.
- Not a backend / data-model change. Pure presentation.
- Not yet propagated to demos 01–04. Those follow-on after adam-mode lands on demo 05 and is validated.

## 2. Decision

Add a **client-side theme toggle** to `05-skinned-realtime.html` that switches between the existing dark theme and a new light theme called **adam-mode**, driven by a `data-theme="adam"` attribute on `<body>` plus a sibling `:root[data-theme="adam"]` CSS block that re-defines the existing custom properties. A new toggle button in the existing `#helpers` panel switches between modes and persists the choice in `localStorage` under the key `ruview.theme`.

### 2.1 CSS — the colour swap

Add immediately after the existing `:root { ... }` block in `<style>`:

```css
:root[data-theme="adam"] {
    --bg: #f6f2ea;
    --bg-panel: rgba(252, 250, 246, 0.92);
    --amber: #b8741a;        /* deeper amber, readable on cream */
    --amber-hot: #8a5612;    /* deepest amber for emphasis text */
    --cyan: #1a6f8a;         /* slate cyan */
    --magenta: #a8348a;      /* slate magenta */
    --text: #2a241c;         /* near-black warm */
    --text-mute: #7a6f5d;    /* warm grey */
    --green: #1f7a32;        /* forest green */
    --red: #b03a1a;          /* burnt sienna */
    --border: rgba(184, 116, 26, 0.28);
}
```

Every existing element already reads from these custom properties, so the swap is automatic for panels, text, borders, and bar fills. No per-element CSS rewrites required.

### 2.2 Overlay handling

The vignette and scanlines are dark-theme aesthetics. In adam-mode they would muddy the cream background. Two new rules:

```css
:root[data-theme="adam"] .overlay-frame {
    background:
        radial-gradient(ellipse at center, transparent 70%, rgba(184,116,26,0.10) 100%),
        linear-gradient(180deg, rgba(184,116,26,0.06) 0%, transparent 18%, transparent 82%, rgba(184,116,26,0.08) 100%);
}
:root[data-theme="adam"] .scanlines {
    opacity: 0.15;
    mix-blend-mode: multiply;
}
```

The vignette is preserved but inverted in colour and lightened; scanlines drop to 15 % opacity and switch from `overlay` to `multiply` blend so they read as faint paper texture rather than CRT lines.

### 2.3 Three.js scene reactivity

Two scene colours are hard-coded at construction (lines 269–270). Replace them with a function call that reads the current theme:

```js
function themeSceneColors(theme) {
    return theme === 'adam'
        ? { bg: 0xf6f2ea, fogDensity: 0.025 }
        : { bg: 0x050507, fogDensity: 0.06 };
}
function applySceneTheme(theme) {
    const c = themeSceneColors(theme);
    scene.background = new THREE.Color(c.bg);
    scene.fog = new THREE.FogExp2(c.bg, c.fogDensity);
    renderer.setClearColor(c.bg, 1.0);
}
```

Called once after `renderer` is constructed, then again from the toggle handler.

`scene.fog` density drops in adam-mode because exponential fog on a light background reads as "haze" much more strongly than on dark — 0.06 → 0.025 keeps the falloff visible without losing the figure into the background.

### 2.4 UI toggle

Add to the `#helpers` panel (top of its labels list):

```html
<label class="theme-toggle">
    <input type="checkbox" id="adam-mode-toggle">
    <span>adam-mode (light)</span>
    <span class="swatch" style="background: var(--amber)"></span>
</label>
```

Handler:

```js
const THEME_KEY = 'ruview.theme';
const root = document.documentElement;
const toggle = document.getElementById('adam-mode-toggle');

function applyTheme(theme) {
    if (theme === 'adam') {
        root.setAttribute('data-theme', 'adam');
        toggle.checked = true;
    } else {
        root.removeAttribute('data-theme');
        toggle.checked = false;
    }
    applySceneTheme(theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch (_) {}
}

const initialTheme = (() => {
    try { return localStorage.getItem(THEME_KEY) || 'dark'; }
    catch (_) { return 'dark'; }
})();
applyTheme(initialTheme);

toggle.addEventListener('change', e => {
    applyTheme(e.target.checked ? 'adam' : 'dark');
});
```

### 2.5 Why "adam-mode" as the codename

The user picked the name. It is a project-specific brand — distinct from the generic "light mode" terminology that other modes (`--theme=high-contrast`, `--theme=print`) may eventually need. Keeping a codename makes the toggle searchable in the codebase, the localStorage key portable across the demo set, and avoids ambiguity if dark itself is later renamed.

The string `"adam"` is the only literal value the `data-theme` attribute and the `localStorage` key ever take. `"dark"` is the implicit default (no attribute, no stored value).

### 2.6 Rejected alternatives

| Alternative | Rejected because |
|---|---|
| Use `prefers-color-scheme: light` only, no toggle | Operators frequently want the opposite of their OS preference for screen-recording or daylight desk use. Auto-only frustrates the actual use case. |
| Ship two separate HTML files (`05-…-dark.html`, `05-…-light.html`) | Doubles maintenance for every future demo edit. No path to per-session toggle. |
| Build a full multi-theme system with a runtime registry | Premature. Two themes don't need a registry; the `data-theme="adam"` attribute is the registry. |
| Use Tailwind / DaisyUI / a CSS framework | Demos are intentionally stand-alone single-file HTML for portability. No build step exists; adding one for theming is wrong shape. |
| Adopt the cognitum-v0 / HOMECORE design tokens (`--hc-*` from `examples/frontend/`) | That design system is dark-only by intent (ADR-131). adam-mode is the light counterpart needed in *demo* contexts, not HA dashboard contexts. |
| Make adam-mode the default | Breaks the dark-aesthetic recording context this demo was originally built for. Default stays dark; toggle stays opt-in. |

## 3. Consequences

### 3.1 Positive

- Demo is usable in daylight, in printed documentation, on light-mode browsers, and by users who find the dark-amber combination fatiguing.
- Toggle persists across reloads via `localStorage` — set once, sticks.
- No structural change to information density, panel layout, or three.js scene geometry. Operators familiar with the dark theme can switch and still find every readout in the same place.
- Implementation is contained — a single `<style>` block addition, a single button, a ~25-line JS handler, and a swap of two scene-construction lines.

### 3.2 Negative

- Two themes to maintain. Any future colour change requires updating both `:root` blocks. Mitigated by keeping the existing custom-property names — adam-mode's values are the only edits.
- The vignette + scanlines lose some of the CRT charm in adam-mode. Tradeoff accepted by design.
- One additional `localStorage` slot consumed per origin (`ruview.theme`).
- The amber accent in adam-mode (`#b8741a`) is visibly different from the dark-mode amber (`#ffb840`) — they share the same CSS variable name but a screenshot from each mode is not pixel-comparable. This is the correct call for accessibility (the bright amber is unreadable on cream) but does mean side-by-side comparisons need both screenshots labelled.

### 3.3 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Future demo edits update one `:root` block and forget the other | Medium | A lint script in `scripts/` could grep both blocks for matching key sets; documented as P2 follow-up. |
| `localStorage` blocked by privacy settings | Low | All accesses are wrapped in try/catch; falls back to dark. |
| Three.js fog density of 0.025 still washes out the model on adam-mode | Low | Empirically tuned during implementation; if it does, drop to 0.015 or remove fog entirely in adam-mode. |
| User on a high-DPI display sees scanlines as visible paper texture even at 15 % opacity | Low | If reported, drop to 8 % or hide scanlines entirely in adam-mode. |

## 4. Implementation plan

Tiny scope — single file. No swarm needed.

1. Add `:root[data-theme="adam"]` CSS block and the two overlay overrides.
2. Refactor scene background + fog into the two helper functions `themeSceneColors()` and `applySceneTheme()`.
3. Add `<label>` markup and handler script.
4. Verify in a browser at http://127.0.0.1:8765/examples/three.js/demos/05-skinned-realtime.html — toggle on, reload, confirm adam-mode persists; toggle off, reload, confirm dark persists.
5. Smoke-screenshot both modes; commit.

Acceptance criteria:

- Toggle checkbox visible in `#helpers` panel.
- Clicking the toggle swaps colours within one frame.
- Reload preserves last choice.
- Three.js scene background follows the toggle (no dark frame visible behind a light HUD or vice-versa).
- Existing dark-theme appearance is byte-identical when toggle is off.

## 5. Test plan

- Manual visual check in two themes (no automated visual regression — demos aren't in the CI test loop today).
- `view-source` confirms the new CSS block, the toggle markup, and the handler are present.
- DevTools `localStorage` shows `ruview.theme` after a toggle.
- Three.js inspector (or a `console.log(scene.background.getHexString())`) confirms scene colour swap.

## 6. Follow-on work (out of scope for this ADR)

- Roll adam-mode into demos 01–04. Each demo has its own `<style>` block; the same `data-theme="adam"` selector and the same JS handler can be copied.
- Honor `prefers-color-scheme: light` on first load *if* `localStorage` has no stored choice. Trivial three-line addition.
- Add a high-contrast theme for accessibility (separate ADR).
- Lint script that asserts both `:root` blocks declare the same custom-property names.

## 7. Related ADRs

- [ADR-019](ADR-019-sensing-only-ui-mode.md) — sensing-only UI mode (Gaussian splats viewer)
- [ADR-035](ADR-035-live-sensing-ui-accuracy.md) — live sensing UI accuracy norms (which this demo follows)
- [ADR-131](docs/adr/ADR-131-...) — HOMECORE / cognitum-v0 design tokens (dark-only, separate context)
