# ESP32 C6 node 7 rate aware sensing qualification

## Scope

This record qualifies the firmware 0.8.8 timing and transport path on a second
physically attached ESP32 C6. It measures raw callback cadence, edge DSP
cadence, process stability, and end to end sensing delivery. It does not
qualify heartbeat, respiration, pose, identity, room separation, or person
count accuracy against labelled ground truth.

## Hardware and firmware

| Field | Measured value |
|---|---|
| Board | ESP32 C6 QFN40 revision 0.2 |
| Logical node | 7 |
| Firmware before | 0.8.4 |
| Firmware after | 0.8.8 development build |
| App image | 1,051,552 bytes |
| App SHA 256 | `eab7561d65e302dc33e9331ac591763a46f92fb3fa9f824fef0e9b541daddb3f` |
| OTA slot size | 1,900,544 bytes |
| OTA headroom | 848,992 bytes, 45 percent |

Only the application partition at offset `0x20000` was flashed. WiFi
credentials, logical node identity, sensing server target, channel, edge tier,
bootloader, partition table, OTA metadata, and NVS were preserved. The pre
update application was copied to a private recovery file outside the
repository. Its SHA 256 is
`f5ebc5e0142425adae16e9180bf298ef444ea8862ba0d8809c310a39fe45721d`.
The device partition table was also read before the update and had SHA 256
`0a8d2f192a8fff209d6c75ab639fcf8aa2f43c64abb732c6e74596fbd6971dca`.

The post update boot log reported firmware 0.8.8, node 7, channel 10, Tier 2,
an 8 Hz edge DSP cadence, and the preserved sensing server target. The OTA
status endpoint reported firmware 0.8.8 running from `ota_0`, with `ota_1` as
the next partition and the correct 1,900,544 byte limit. The sensing server
health endpoint remained ready with ESP32 input.

## Before and after

The pre update baseline was a 20 second observation on the same attached board.
The post update observation was a five minute steady state run after boot.

| Observation | Before 0.8.4 | After 0.8.8 |
|---|---:|---:|
| Raw callback mean | 39.05 pps | 36.32 pps |
| Raw callback range | 35 through 42 pps | 24 through 42 pps |
| Server CSI FPS mean | 46.23 Hz | 48.88 Hz |
| WebSocket parser errors | 0 | 0 |
| WebSocket reconnects | 0 | 0 |

Raw callback mean changed by negative 7.0 percent while the server CSI FPS
estimate changed by positive 5.7 percent. Both remain above the 20 pps
transport floor. The result is transport neutral rather than an accuracy lift;
the room and WiFi traffic were not controlled between the two windows.

Firmware 0.8.4 did not expose the edge DSP cadence used by the temporal
filters. Firmware 0.8.8 held that separately governed clock at exactly 8.0 Hz
for every controller sample while preserving the higher rate raw network path.

## Five minute physical result

MEASURED on 2026 08 31 after flashing firmware 0.8.8:

| Device observation | Result |
|---|---:|
| Duration | 300.70 seconds |
| Controller samples | 300 |
| Raw callback mean | 36.32 pps |
| Raw callback range | 24 through 42 pps |
| Edge DSP mean | 8.00 Hz |
| Edge DSP range | 8.00 through 8.00 Hz |
| ENOMEM events | 0 |
| UDP send failures | 0 |
| ESP NOW nonzero failure lines | 0 |
| Other steady state errors | 0 |
| Watchdogs, panics, or reboots | 0 |

| End to end WebSocket observation | Result |
|---|---:|
| Duration | 300.06 seconds |
| Sensing frames | 36,254 |
| JSON parse errors | 0 |
| WebSocket errors or reconnects | 0 |
| Frames containing node 7 | 35,313 |
| Node 7 frame coverage | 97.40 percent |
| Node 7 stale frames | 0 |
| Maximum node 7 staleness | 972 ms |
| Maximum node 7 inference age | 483 ms |
| Maximum WebSocket frame gap | 108 ms |
| Nodes per frame | 0 through 5 |
| Fused presence count contradictions | 0 |

One ENOMEM backoff occurred during startup and recovered in 210 ms. No memory
backoff or send failure recurred in the separate five minute steady state
window. The boot log also reported the documented fail closed OTA behavior:
the status service was available, but image upload remained rejected because
this node has no provisioned OTA signing secret.

## Result and limitation

The node 7 timing and transport update passes. Its configuration survived, the
edge DSP clock remained phase stable at the measured sustainable C6 rate, and
the live service received fresh node 7 data throughout the run. This does not
complete the ADR 346 occupancy qualification for node 7 because the room was
not held empty and the live aggregate did not expose 30 absent edge packets.

The largest uncertainty remains inference accuracy. Timing stability cannot
prove better vital, motion, room separation, or multi person estimates without
synchronized held out labels and a controlled empty room sequence.

## Acceptance test

Repeat this five minute procedure after timing, WiFi, filter, or scheduling
changes. Pass transport only when raw callback yield remains at least 20 pps,
DSP cadence stays within one hertz of the configured target, the device has
zero steady state memory backoff, send failure, watchdog, panic, and reboot
events, the server has zero parse failures and reconnects, and the updated node
stays fresh. Complete occupancy qualification separately with at least 30
absent edge packets and zero absent packets carrying a nonzero person count.
