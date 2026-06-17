---
name: onboard
description: Zero-to-sensing path picker for RuView (WiFi-DensePose) — pick docker-demo, repo-build, or live-esp32 and run the next concrete step.
---

# onboard

Get a newcomer from nothing to a working RuView setup. **First fact to set:** WiFi
sensing infers *coarse* pose/presence/breathing from Channel State Information — it
is **not a camera**, and any accuracy number must be MEASURED against a baseline
(use the `verify` skill / `ruview.claim_check` tool). Never present WiFi output as
camera-grade.

## Pick a path

Run `ruview.onboard {path}` or decide from:

1. **docker-demo** — fastest, no hardware. Replays sample CSI into the dashboard.
   `docker run -p 8000:8000 ruvnet/wifi-densepose` → open `http://localhost:8000`.
   Use to see what it looks like.
2. **repo-build** — for developers. `cd v2 && cargo test --workspace --no-default-features`
   (1,031+ tests pass), then `cargo run -p wifi-densepose-cli -- --help`.
3. **live-esp32** — a real install. Flash a node (`provision-node` skill), point it at
   the sensing-server, then `calibrate-room`. This is the only path that senses a real room.

## Then

- Live sensing → go to **provision-node**, then **calibrate-room**.
- Evaluating a model/claim → go to **verify** and run `ruview.claim_check` on any
  report before you quote a number.
