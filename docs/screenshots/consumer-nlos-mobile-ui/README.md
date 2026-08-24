# Consumer NLOS mobile UI review captures

These images are deterministic review captures from the production Expo web
export at a 390 by 844 viewport. The Playwright flow navigates the real mobile
application, exercises the synthetic replay control, verifies provenance and
watermark requirements, opens the deterministic Three.js LiDAR point cloud,
verifies the reconstruction boundary, target lock, and confidence HUD, and
writes the PNG files in this directory.

The visual system is pinned to `cognitum-one/website` commit
`0288734c3426ca9125ef4cb2e067ef057c09f3ce`. Outfit supplies display and body
roles, while JetBrains Mono supplies instrumentation and metric roles. Both
font families are bundled locally. Screenshot generation makes no runtime font
request.

The captures contain only the disconnected state, governed setup copy, and the
built in synthetic fixture. The point cloud is generated locally from gated
track hypotheses and schematic relay geometry. It is not a raw sensor point
cloud. The captures use no live endpoint, credential, sensor data, person data,
or remote asset. They are not native iPhone screenshots, physical LiDAR
evidence, performance evidence, or proof of optical NLOS capability.

Regenerate them from `ui/mobile` with `npm run e2e:web`. The executable design
and evidence contract is documented in
`docs/adr/ADR-342-cognitum-inspired-mobile-instrument-ui.md`.
