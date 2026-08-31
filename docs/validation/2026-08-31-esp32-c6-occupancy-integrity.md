# ESP32 C6 occupancy evidence qualification

## Scope

This record qualifies the fail closed person count invariant in ADR 346 on one physically attached ESP32 C6. It does not qualify person counting accuracy, identity, pose, room separation, or vital sign accuracy.

## Hardware and firmware

| Field | Measured value |
|---|---|
| Board | ESP32 C6 QFN40 revision 0.2 |
| Logical node | 4 |
| Firmware before | 0.7.0 |
| Firmware after | 0.8.4 development build |
| App image | 1,051,168 bytes |
| App SHA 256 | `f9470a31b82612f1740f0cf0943ddb78917cd58784ba16d2e9d57cc8fb39364c` |
| OTA slot size | 1,900,544 bytes |
| CSI stream target | Preserved from NVS |

The device partition table was read before the update. NVS, OTA metadata, bootloader, and partition table were not overwritten. A private recovery copy was created outside the repository and excluded from version control.

## Software gates

| Gate | Result |
|---|---|
| Firmware host tests | PASS, 54 assertions across encoding, vital evidence, and mmWave detection |
| Rust sensing server package | PASS, 532 library tests plus all package integration and documentation tests |
| Mobile Jest suite | PASS, 164 suites and 1,223 tests |
| Mobile TypeScript | PASS |
| Mobile ESLint | PASS |
| Mobile security verifier | PASS |
| Repository wide Rust formatting | PREEXISTING DRIFT outside this change; changed code builds and package tests pass |

## Physical result

MEASURED on 2026 08 31 from the live local sensing WebSocket for 300 seconds:

| Node | Firmware state | Edge packets | Absent packets | Absent with nonzero count | Result |
|---|---|---:|---:|---:|---|
| 4 | Updated | 242 | 61 | 0 | PASS |
| 3 | Unupdated control | 216 | 216 | 216 | Expected control failure |
| 7 | Unupdated control | 280 | 278 | 278 | Expected control failure |

Node 4 reduced the targeted logical contradiction from observed to zero, a 100 percent reduction for this invariant during this run. This is not a person count accuracy result.

The WebSocket run had zero JSON parse errors and one expected client close at completion. The sensing server remained ready with `engine_error_count=0`. A separate 45 second serial observation recorded 120 log lines, 17 CSI callback markers, zero ENOMEM backoffs, and zero other error lines.

The updated OTA status endpoint reports the selected 1,900,544 byte partition rather than the stale 921,600 byte constant. The 1,051,168 byte image therefore fits with 849,376 bytes of partition headroom.

## Remaining qualification

Nodes 3 and 7 still demonstrate the old contradictory behavior and must be upgraded only after their network identity, OTA credential, and rollback path are verified. The current run had no labelled ground truth, so multi person fidelity and adjacent room rejection remain unmeasured.

## Subsequent node 7 status

Later on 2026 08 31, node 7 was separately identified, backed up, upgraded to firmware 0.8.8, and transport qualified for five minutes. That later occupied room run had zero fused presence count contradictions but did not produce the 30 absent edge packets required to supersede the historical control result above. See `docs/validation/2026-08-31-esp32-c6-node7-rate-aware-sensing.md`.

## Acceptance test

Repeat a five minute capture after every firmware change. Pass only when every updated node has at least 30 absent packets, zero packets where `presence=false` and `n_persons>0`, zero parser errors, and a ready sensing server with zero engine errors.
