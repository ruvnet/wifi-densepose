# RuView harness — agent operating notes

You are operating **RuView** (WiFi-DensePose), a camera-free WiFi-CSI sensing system.

## The one rule: prove everything

This project was accused of AI-slop; the fix is hard discipline. Before you quote ANY
accuracy number:

1. It must be tagged **MEASURED** (with a reproducer named), **CLAIMED**, or **SYNTHETIC**.
2. Pose PCK is quoted only as a **delta over the mean-pose baseline** on a leakage-free
   held-out split. (A mean-pose predictor already scores ~50% PCK.)
3. Run `ruview.claim_check` on any report/PR/model-card. It flags untagged numbers and
   the retracted "100%/perfect accuracy" framing.
4. Firmware is "hardware-validated" only with a captured **boot log on real silicon** —
   never on a build-passes signal.

## Tools

`ruview.onboard`, `ruview.claim_check`, `ruview.verify`, `ruview.node_monitor`,
`ruview.calibrate`, `ruview.node_flash`. All fail-closed. Mutating/hardware tools
(`node_flash`) require explicit confirmation and are Windows/ESP-IDF gated.

## Skills

`onboard` · `provision-node` · `calibrate-room` · `train-pose` · `verify`
(`npx @ruvnet/ruview skill <name>`).

## Don'ts

- Don't present WiFi sensing as camera-grade.
- Don't echo or commit WiFi passwords / secrets.
- Don't merge or release firmware without a real boot log.
- Don't report a PCK without its mean-pose baseline.
