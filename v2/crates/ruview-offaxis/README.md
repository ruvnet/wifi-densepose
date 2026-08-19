# ruview-offaxis

Clean-room **off-axis (head-coupled) perspective projection** in Rust, with a
wasm-bindgen surface for browser demos. Implements ADR-324.

The screen becomes a window: given the physical screen's corners and the
viewer's eye position, the crate produces the asymmetric frustum and
screen-aligned view matrix that keep the screen plane fixed while everything
behind and in front of it moves with true parallax.

## Clean-room statement

ADR-324 records that the prior-art repository (`icurtis1/off-axis-sneaker`)
publishes **no license**. No code, assets, or derived text from it appear in
this crate. The implementation follows the published math only:

- Robert Kooima, *Generalized Perspective Projection* (2008) — the
  `pa`/`pb`/`pc` screen-corner frustum formulation.
- Casiez, Roussel & Vogel, *1€ Filter* (CHI 2012) — adaptive tracking-noise
  smoothing.

## What's in the crate

| Module | Contents |
|---|---|
| `projection` | `Screen` (3 corners, any orientation), `off_axis()` → asymmetric `projection` + screen-aligned `view` matrix (column-major `f64`, the three.js `Matrix4.elements` layout). All failure modes are typed errors — never NaN matrices. |
| `filter` | `OneEuro` / `OneEuro3` one-euro filter. Timestamps are injected; the crate never reads a clock. |
| `rf` | `field_peak()` — strongest-cell extraction for `/ws/sensing` `signal_field` grids, mirroring the sensing server's `field_localize.rs` constants (`X_SCALE 0.6`, `Z_SCALE 0.5`, `PEAK_THRESHOLD 0.35`). `CoarseParallax` — the **Tier B** stage: deadband + gain + hard clamp + one-euro. `ScreenCalibration` — physical screen (cm) + normalized-head → metric eye mapping (**Tier A** input hook). |
| `wasm` (wasm32 only) | `OffAxisCamera` and `RfParallax` wasm-bindgen classes. |

The native core is **dependency-free**; wasm-bindgen is pulled only when
compiling for `wasm32`.

## Honesty contract (repo rule — read before demoing)

- A single-link CSI field peak is a *representation of field energy*, *not*
  metric localization and *never* a head position (see the caveat in
  `wifi-densepose-sensing-server/src/field_localize.rs`). The Tier B path is
  therefore **coarse body parallax by construction**: deadbanded,
  gain-limited, hard-clamped. Do not present it as head tracking; the demo
  labels it on screen at all times.
- Numeric defaults (gains, deadbands, filter cutoffs) are interaction-design
  choices — `CLAIMED`, not measured performance.
- The benchmark numbers below are `MEASURED` with the stated reproducer on
  the stated machine; re-run locally before relying on them.

## Native quick start

```rust
use ruview_offaxis::{off_axis, Screen, Vec3};

// 60 cm × 34 cm screen centered at the origin; eye 65 cm out, 10 cm right.
let screen = Screen::centered(0.60, 0.34)?;
let oa = off_axis(&screen, Vec3::new(0.10, 0.0, 0.65), 0.05, 100.0)?;
let mvp: [f64; 16] = oa.view_projection(); // column-major, GL/three.js layout
# Ok::<(), ruview_offaxis::OffAxisError>(())
```

Key invariant (unit-tested for a grid of eye positions and for tilted
screens): the physical screen corners always project exactly to the NDC
corners — `pa→(−1,−1)`, `pb→(1,−1)`, `pc→(−1,1)`, `pd→(1,1)` — and points on
the screen plane are eye-invariant. That is the mathematical definition of
"the screen is a window".

## Building the WASM package

Generated artifacts are not committed (repo rule). Build once:

```bash
cd v2
rustup target add wasm32-unknown-unknown
cargo build -p ruview-offaxis --target wasm32-unknown-unknown --release

# Install the matching CLI once: cargo install wasm-bindgen-cli --version 0.2.114
wasm-bindgen --target web --out-dir crates/ruview-offaxis/pkg \
    target/wasm32-unknown-unknown/release/ruview_offaxis.wasm
```

Output: `pkg/ruview_offaxis.js` + `pkg/ruview_offaxis_bg.wasm`
(≈54 KB wasm, `MEASURED` for this crate at wasm-bindgen 0.2.114; `wasm-opt -Oz`
can shrink it further if you have binaryen). The demo at
`examples/three.js/demos/07-off-axis-window.html` loads this path directly —
build, then open the demo. A Node smoke test of the same flow lives in the
PR's validation notes.

## three.js integration (the whole wiring)

```js
import init, { OffAxisCamera, RfParallax } from './pkg/ruview_offaxis.js';
await init();

// Physical calibration in cm — measure your actual screen.
const cam = new OffAxisCamera(60, 34, 65, 0.05, 100.0);
cam.set_filter(1.2, 0.4); // one-euro: min_cutoff Hz, beta

const camera = new THREE.PerspectiveCamera();
camera.matrixAutoUpdate = false; // WASM owns every matrix

const view = new THREE.Matrix4();
function onFrame(eyeX, eyeY, eyeZ) {          // metres, screen space
  cam.update_eye(eyeX, eyeY, eyeZ, performance.now() / 1000);
  camera.projectionMatrix.fromArray(cam.projection());
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  view.fromArray(cam.view());
  camera.matrixWorld.copy(view).invert();
  camera.matrixWorldInverse.copy(view);
}
```

Scene convention: the screen plane is `z = 0`; content behind the screen has
`z < 0`; content with `z > 0` "pops out". Do **not** update `camera.aspect`
on resize — the frustum is determined by the physical screen, not the
viewport.

### Input tiers (ADR-324)

- **Tier A (fine tracker)** — feed any head tracker through
  `cam.update_normalized(nx, ny, depthScale, lateralRangeM, t)`; `nx`/`ny`
  are normalized image coordinates from whatever fine tracker the host runs
  (which stays entirely in the browser). RF adds presence gating around it.
- **Tier B (RF only, labeled)** — `RfParallax.update(Float32Array, nx, nz, t)`
  with `/ws/sensing` `signal_field` values, then `rf.eye()` →
  `cam.update_eye(...)`. Keep the on-screen "coarse body parallax — not head
  tracking" label; the clamps in the Rust core bound the excursion but the
  label is what keeps the demo honest.

## Benchmarks

`MEASURED` — reproducer: `cd v2 && cargo bench -p ruview-offaxis`.
Environment for the numbers below: Linux x86_64 container (shared/virtualized
CPU), rustc 1.89.0, criterion 0.5, 2026-08-16. Treat them as order-of-
magnitude; re-run on your hardware.

| Benchmark | Time (median) |
|---|---|
| `off_axis_projection` (frustum + view build) | ~76 ns |
| `view_projection` combined (4×4 multiply) | ~27 ns |
| `one_euro3_step` (3-axis filter step) | ~60 ns |
| `field_peak_20x20` (live grid size) | ~488 ns |
| `field_peak_100x100` | ~12.3 µs |
| `tier_b_full_frame_20x20` (scan → parallax → projection) | ~598 ns |

The full Tier B per-frame path costs well under a microsecond — under 0.01%
of a 60 Hz frame budget. The argmax scan was the only hot spot found; it was
rewritten branch-light for a measured −18% (20×20) / −33% (100×100) before
these numbers were taken. End-to-end *motion-to-photon* latency (RF capture →
render) has **not** been measured and is dominated by the sensing pipeline,
not this crate; no figure is claimed.

## Validation

```bash
cd v2
cargo test -p ruview-offaxis          # 23 unit tests + doctest
cargo clippy -p ruview-offaxis --all-targets   # zero warnings
cargo build -p ruview-offaxis --target wasm32-unknown-unknown --release
```

## References

- ADR-324 — decision record, tier definitions, licensing analysis
- `docs/adr/ADR-282-*` — L0–L5 evidence ladder (labels used above)
- `v2/crates/wifi-densepose-sensing-server/src/field_localize.rs` — the
  field-peak honesty caveat this crate inherits
