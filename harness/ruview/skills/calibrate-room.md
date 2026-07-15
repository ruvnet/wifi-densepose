---
name: calibrate-room
description: Run the ADR-151 per-room calibration pipeline — baseline → enroll → extract → train → a bank of small specialists (presence/posture/breathing/heartbeat/restlessness/anomaly).
---

# calibrate-room

Turn a provisioned node + sensing-server into a working room model. Pure-Rust,
edge-deployable (ADR-151). Use the `ruview_calibrate` tool (installed
`wifi-densepose` binary, else `cargo run -p wifi-densepose-cli`).

## Sequence

1. **baseline** — capture the empty room (Welford amplitude + von Mises phase). Leave
   the room empty.
   `ruview_calibrate {step: "baseline"}`
2. **enroll** — record the occupant(s) doing the target activities.
   `ruview_calibrate {step: "enroll"}`
3. **train-room** — train the bank of small specialists from baseline + enrollment.
   `ruview_calibrate {step: "train-room"}`
4. **room-watch** — live presence/posture/breathing from the trained room.
   `ruview_calibrate {step: "room-watch"}`  (or the `room-watch` skill)

## Honesty

The specialists are calibrated to *this* room; cross-room transfer is a separate
problem (LoRA recalibration, ADR-079 P9). Report which room a number came from, and
tag presence/vitals accuracy MEASURED only with a held-out check — run
`ruview_claim_check` on the writeup.
