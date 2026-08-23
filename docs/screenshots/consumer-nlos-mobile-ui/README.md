# Consumer NLOS mobile UI review captures

These images are deterministic review captures from the production Expo web
export at a 390 by 844 viewport. The Playwright flow navigates the real mobile
application, exercises the synthetic replay control, verifies provenance and
watermark requirements, and writes the PNG files in this directory.

The captures contain only the disconnected state, governed setup copy, and the
built in synthetic fixture. They use no live endpoint, credential, sensor data,
person data, or remote asset. They are not native iPhone screenshots, physical
LiDAR evidence, performance evidence, or proof of optical NLOS capability.

Regenerate them from `ui/mobile` with `npm run e2e:web`. The executable design
and evidence contract is documented in
`docs/adr/ADR-342-cognitum-inspired-mobile-instrument-ui.md`.
