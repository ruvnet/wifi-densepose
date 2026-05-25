# @ruvnet/homecore-frontend

HOMECORE web UI — built with Lit 3, TypeScript, and Vite.
Design system mirrors the cognitum-v0 / v0-appliance dashboard (ADR-131).

## Quick start

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The Vite dev server proxies `/api` → `http://localhost:8123`, so you need a
`homecore-api-server` (or the `wifi-densepose-sensing-server` crate) running on `:8123`.

## Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start Vite dev server on port 5173 |
| `npm run build` | TypeScript compile + Vite production bundle → `dist/` |
| `npm run lint` | ESLint on `src/` |
| `npm test` | Vitest unit tests (3 suites, jsdom) |

## Package layout

```
frontend/
  src/
    api/
      client.ts        # fetch + WebSocket client (REST + WS)
      types.ts         # TypeScript types matching homecore-api JSON shapes
    components/
      AppShell.ts      # <hc-app-shell> — header + nav + content slot
      StateCard.ts     # <hc-state-card> — single entity state card
    icons/
      lucide.ts        # Tree-shaken Lucide icon wrapper
    styles/
      tokens.css       # 16 CSS custom properties (--hc-*)
      base.css         # Typography reset, page shell, nav layout
    __tests__/         # Vitest unit tests
  index.html           # Shell loading src/main.ts
  vite.config.ts
  tsconfig.json
  vitest.config.ts
```

## Design system

Colors, typography, and components mirror the cognitum-v0 dashboard
(`http://cognitum-v0:9000/`). Dark-only; no light-mode. Key tokens:

- `--hc-primary` `#19d4e5` — teal (active nav, focus ring, CTA borders)
- `--hc-accent` `#26d867` — green (success, secondary CTA)
- `--hc-bg` `#0b0e13` — near-black navy page root
- Font: Outfit (display) + JetBrains Mono (mono)
- Icons: Lucide (SVG, `stroke: currentColor`, no icon font)

See `docs/design/HOMECORE-FRONTEND-design-recon.md` for the full recon.

## Architecture notes

- Components are standard Lit `LitElement` custom elements — compatible with
  any HTML page and with Home Assistant's Lit-based frontend.
- The REST client uses `fetch`; the WS client uses `WebSocket`. Both accept a
  bearer token and are fully typed against the Rust `homecore-api` JSON shapes.
- WASM: `vite.config.ts` enables `.wasm` asset import. Hook up via dynamic
  `import('/path/to/module.wasm?init')` when WASM bindings are ready.
