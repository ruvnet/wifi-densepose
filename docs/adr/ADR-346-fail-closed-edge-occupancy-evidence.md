# ADR 346: Fail closed ESP32 occupancy evidence

## Status

Accepted and implemented. Physical qualification is required after each firmware build.

## Date

2026 08 31

## Context

The ESP32 Tier 2 pipeline produces two different signals. Presence is a debounced room level decision. Person count is a bounded subcarrier diversity heuristic. A live four node installation emitted packets with `presence=false` and `n_persons=3` or `4`. The server eventually gated the aggregate room count, but raw WebSocket consumers and diagnostics could still treat the contradictory count as occupancy evidence.

That contradiction is more dangerous than a missed optional count. It can contaminate empty room calibration, train a room model on false labels, and encourage a product claim that the firmware cannot support. The count is not identity, pose, or a validated multi person estimator.

## Decision

1. Firmware person slots are subordinate to the debounced presence gate.
2. When presence is false, the firmware clears slot activity, slot history, candidate count, persistence streak, and stable count.
3. The serialized person count is always zero when presence is false and is clamped to `EDGE_MAX_PERSONS` when presence is true.
4. The sensing server repeats the invariant for older firmware. A contradictory or out of range count becomes zero and carries `person_count_valid=false`.
5. Fused CSI plus mmWave packets use either CSI presence or mmWave presence as the supporting presence condition.
6. The node inventory and WebSocket diagnostics expose person count validity. Consumers must not infer a person from an invalid count.
7. No count accuracy claim is created by this change. The firmware output remains a heuristic until a leakage free, held out physical dataset demonstrates otherwise.
8. OTA admission uses the selected update partition size rather than a stale fixed 900 KB ceiling. The status endpoint reports that same hardware bound, while image validation and authenticated OTA remain mandatory.

## Security and privacy

The change retains no raw CSI or personal data. It reduces authority by preventing a secondary heuristic from asserting occupancy after the primary gate has closed. The host validates packet length, magic, range, and logical consistency before using count evidence.

## Consequences

Older firmware remains wire compatible. Invalid count evidence becomes visibly unavailable instead of silently affecting calibration. A true multi person event can still be undercounted when the presence gate is false, which is the intended fail closed behavior. Current C6 images larger than 900 KB can use the installed 1,900,544 byte OTA slots after one serial upgrade, without weakening the OTA authentication gate.

The largest uncertainty is whether the current presence gate itself generalizes across the installed rooms. The fix path is a room bound empty baseline plus the fixed room selective held out protocol, not a global threshold reduction.

## Evidence and acceptance

MEASURED before implementation on 2026 08 31: four live nodes streamed for 86 seconds with zero transport errors, while edge packets repeatedly contradicted `presence=false` with counts of three or four.

Software acceptance requires:

1. Host firmware tests prove absent plus four active slots serializes zero.
2. Rust parser tests prove contradictory and out of range counts fail closed.
3. The node API exposes count validity without breaking older firmware.

Physical acceptance requires the updated firmware on a confirmed board, a captured boot log, five minutes of live packets, zero logical count contradictions, and no increase in transport errors. Accuracy remains unmeasured until labelled held out sequences are recorded.

Physical occupancy qualification completed for ESP32 C6 node 4 on 2026 08 31. The five minute run observed 242 edge packets, including 61 absent packets, with zero logical count contradictions and zero parse errors. See `docs/validation/2026-08-31-esp32-c6-occupancy-integrity.md`.

ESP32 C6 node 7 was subsequently identified, upgraded to firmware 0.8.8, and transport qualified for five minutes with zero fused presence count contradictions and zero steady state transport errors. Its controlled empty room sequence remains required before occupancy qualification. See `docs/validation/2026-08-31-esp32-c6-node7-rate-aware-sensing.md`. Other nodes remain unqualified until separately identified and upgraded.
