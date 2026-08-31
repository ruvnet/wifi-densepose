# ESP32 C6 rate aware sensing qualification

## Scope

This record qualifies ADR 347 on one physically attached ESP32 C6 and verifies
that the same source compiles for ESP32 S3. It measures transport cadence, edge
DSP cadence, process stability, and end to end sensing delivery. It does not
qualify heartbeat, respiration, gesture, pose, identity, or person count
accuracy against labelled ground truth.

## Hardware and firmware

| Field | Measured value |
|---|---|
| Board | ESP32 C6 QFN40 revision 0.2 |
| Logical node | 4 |
| Firmware before | 0.8.4 |
| Firmware after | 0.8.8 development build |
| C6 app image | 1,051,552 bytes |
| C6 app SHA 256 | `f2ea422c9b99ec13c7a168afc2b019229642769ffabfd8f29a85978770236e87` |
| OTA slot size | 1,900,544 bytes |
| OTA headroom | 848,992 bytes, 45 percent |
| S3 compile image | 1,127,104 bytes |
| S3 compile SHA 256 | `63e4f0c484d79e7dd37eb28275951c8beb924e6908942f7fec0b90d92109129c` |

Only the application partition at offset `0x20000` was flashed. WiFi
credentials, node identity, sensing server target, bootloader, partition table,
OTA metadata, and NVS were preserved. The pre update OTA application was read
to a private recovery file outside the repository. Its SHA 256 is
`a2e503f1622b2f3f9c1cfce0a07ba34b9fc5d6a413b6346311622af1fe18a6d8`.

The OTA status endpoint reported firmware 0.8.8 running from `ota_0` after the
update. The sensing server health endpoint remained ready with ESP32 input.

## Software gates

| Gate | Result |
|---|---|
| Rate estimator and occupancy host tests | PASS, 30 assertions |
| ADR 110 encoding host tests | PASS, 21 assertions |
| mmWave frame predicate host tests | PASS, 8 assertions |
| ESP32 C6 IDF 5.4 ARM64 build | PASS |
| ESP32 S3 IDF 5.4 ARM64 build | PASS, compile only |
| Image checksum and validation hash | PASS |
| Repository diff whitespace check | PASS |
| Local libFuzzer aggregate | NOT RUN, local Xcode toolchain lacks `libclang_rt.fuzzer_osx.a` |

For this C6 record, the S3 result was source and toolchain validation only and
no S3 runtime claim is made here. The later physical S3 Tier 0 transport run is
recorded separately in
`docs/validation/2026-08-31-esp32-s3-rate-aware-transport.md`.

## Measured rate correction

The pre update 20 second C6 baseline delivered a mean 34.05 raw callbacks per
second, median 34.5, and range 28 through 37. An intermediate 0.8.7 physical
run requested 10 Hz edge DSP but converged to 8.0 through 8.4 Hz while raw CSI
remained 30 through 40 packets per second. This proved that C6 Tier 2 compute,
not the raw transport, was the limiting path.

Firmware 0.8.8 therefore keeps the 50 Hz probe and independent raw network
path, but sets the C6 Tier 2 DSP clock to its measured sustainable 8 Hz. The
phase preserving sampler prevents callback jitter from shifting the configured
clock, and the filter estimator follows processed timestamps rather than raw
probe intent.

## Five minute physical result

MEASURED on 2026 08 31 after flashing firmware 0.8.8:

| Device observation | Result |
|---|---:|
| Duration | 300.64 seconds |
| Controller ticks | 300 |
| Raw callback mean | 34.92 pps |
| Raw callback range | 22 through 41 pps |
| Edge DSP mean | 8.00 Hz |
| Edge DSP range | 8.00 through 8.00 Hz |
| ENOMEM events | 0 |
| UDP send failures | 0 |
| Other steady state errors | 0 |
| Watchdogs, panics, or reboots | 0 |

| End to end WebSocket observation | Result |
|---|---:|
| Duration | 300.01 seconds |
| Sensing frames | 26,786 |
| JSON parse errors | 0 |
| Reconnects | 0 |
| Frames containing node 4 | 26,148 |
| Node 4 frame coverage | 97.62 percent |
| Node 4 stale frames | 0 |
| Maximum node 4 inference age | 176 ms |
| Maximum WebSocket frame gap | 110 ms |
| Nodes per frame | 0 through 4 |
| Fused `presence=false` with nonzero count contradictions | 0 |

The boot log emitted one expected iTWT negotiation error because the access
point rejected the requested target wake time parameters. Firmware immediately
selected its documented opportunistic CSI fallback. No iTWT or other error
recurred during the five minute steady state window.

## Result and limitation

ADR 347 timing and transport acceptance passes on the attached C6. Raw
throughput did not regress relative to the short baseline, the edge clock now
matches the rate the temporal filters actually receive, and node 4 was never
stale when present in the live sensing service. The separate occupancy
qualification recorded 61 absent node 4 packets with zero contradictions for
the unchanged fail closed invariant. This run did not repeat an empty room
sequence because the room was occupied during qualification.

The largest remaining uncertainty is inference accuracy. Stable timing removes
one source of feature distortion but cannot prove better heartbeat, respiration,
gesture, or multi person classification without synchronized held out labels.

## Acceptance test

Repeat this five minute procedure after any timing, WiFi, filter, or task
scheduling change. Pass only when raw callback yield remains at least 20 pps,
DSP cadence remains within one hertz of the configured target, the device has
zero steady state ENOMEM, send failure, watchdog, panic, and reboot events, the
server has zero parse failures and reconnects, node 4 stays fresh, and fused
presence count contradictions remain zero.
