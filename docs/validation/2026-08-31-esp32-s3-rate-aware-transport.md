# ESP32 S3 rate aware transport qualification

## Scope

This record qualifies the firmware 0.8.8 raw transport path on one physically
attached ESP32 S3. The node retained its existing Tier 0 configuration, so this
run does not qualify the S3 edge DSP rate, temporal filters, heartbeat,
respiration, gesture, pose, identity, person count, or localization accuracy.

## Hardware and firmware

| Field | Measured value |
|---|---|
| Board | ESP32 S3 QFN56 revision 0.2 with 2 MB embedded PSRAM |
| Logical node | 1 |
| Firmware before | 0.8.4 |
| Firmware after | 0.8.8 development build |
| Edge tier | 0, raw passthrough |
| App image | 1,127,104 bytes |
| App SHA 256 | `b531c76900c07d0d6f6e864a5f28afff3e71f124777af97358bb405b34e339a2` |
| OTA slot size | 2,097,152 bytes |
| OTA headroom | 970,048 bytes, 46 percent |

The production partition table was read from the device before the update.
Only the application partition at offset `0x20000` was flashed. WiFi
credentials, logical node identity, channel, sensing server target, bootloader,
partition table, OTA metadata, and NVS were preserved. A private recovery copy
of the prior 2 MB application partition was saved outside the repository. Its
SHA 256 is
`14e72c060c4f1a465f739949b873f6aade5b8899a8909f6c5bb591ee49837c1b`.

The post update boot log reported firmware 0.8.8, logical node 1, channel 4,
the preserved UDP target, Tier 0 raw passthrough, and successful CSI streaming.
The OTA status endpoint reported firmware 0.8.8 running from `ota_0` with the
correct 2,097,152 byte update limit. The sensing server health endpoint remained
ready with ESP32 input.

## Software and image gates

| Gate | Result |
|---|---|
| Firmware encoding, vitals, occupancy, and mmWave host tests | PASS, 59 assertions |
| Firmware provisioning Python tests | PASS, 14 tests |
| ESP32 S3 IDF 5.4 ARM64 clean build | PASS |
| Image target detection | PASS, ESP32 S3 |
| Image checksum | PASS |
| Image validation hash | PASS |
| Application partition fit | PASS, 46 percent free |
| Physical flash write verification | PASS |
| Preserved runtime configuration | PASS |

The build excluded the optional WASM3 source because it was not present in the
firmware checkout. The boot log therefore reported WASM Tier 3 disabled. That
is not a regression introduced by this update and is outside this transport
qualification.

## Before and after comparison

The pre update baseline was a 20 second serial and WebSocket capture on firmware
0.8.4. The post update stability observation was 300 seconds on firmware 0.8.8.

| Observation | Before 0.8.4 | After 0.8.8 | Change |
|---|---:|---:|---:|
| Raw CSI yield mean | 27.80 pps | 28.03 pps | plus 0.83 percent |
| Server CSI FPS mean | 39.58 | 39.13 | minus 1.15 percent |
| Node frame coverage | 100 percent | 100 percent | unchanged |
| Maximum node staleness | 1,571 ms | 1,565 ms | minus 0.38 percent |
| Maximum WebSocket frame gap | 111 ms | 112 ms | plus 0.90 percent |
| Device or parser errors | 0 | 0 | unchanged |

These small movements are operationally neutral and within uncontrolled room
and WiFi variation. Firmware 0.8.8 did not regress the raw transport. Because
Tier 0 bypasses the DSP task, this run provides no evidence that temporal
features or inference accuracy improved.

## Five minute physical result

MEASURED on 2026 08 31 after flashing firmware 0.8.8:

| Device observation | Result |
|---|---:|
| Duration | 300 seconds |
| Controller yield samples | 300 |
| Raw CSI yield mean | 28.03 pps |
| Raw CSI yield range | 22 through 34 pps |
| ENOMEM or stack errors | 0 |
| UDP send failures | 0 |
| ESP NOW send failures | 0 |
| Unexpected resets | 0 |
| Watchdogs or panics | 0 |

| End to end WebSocket observation | Result |
|---|---:|
| Duration | 300.03 seconds |
| Sensing frames | 10,559 |
| Frames containing node 1 | 10,559 |
| Node 1 frame coverage | 100 percent |
| Source offline frames | 0 |
| JSON parse errors | 0 |
| WebSocket errors | 0 |
| Early closes | 0 |
| Maximum node staleness | 1,565 ms |
| Maximum WebSocket frame gap | 112 ms |

## Result and limitation

The ESP32 S3 raw transport acceptance passes. Firmware 0.8.8 booted from the
existing slot, retained the installation configuration, sustained the prior raw
CSI delivery rate, and completed the burn with zero transport or runtime
errors. The result extends ADR 347 physical coverage to the S3 transport path.

The largest remaining uncertainty is S3 Tier 2 behavior and inference accuracy.
The log line reporting the configured 20 Hz DSP cadence is not proof that DSP
ran because the preserved Tier 0 setting explicitly disables the DSP task. A
separate, rollback protected Tier 2 qualification with synchronized held out
labels is required before making heartbeat, respiration, motion, or accuracy
claims.

## Acceptance test

Repeat this five minute procedure after any S3 timing, WiFi, transport, or task
scheduling change. Pass only when raw callback yield remains at least 20 pps,
node frame coverage remains at least 99 percent, and the device and server have
zero send failures, offline frames, parser errors, WebSocket errors, watchdogs,
panics, and unexpected resets. Qualify Tier 2 separately and require its
measured DSP cadence to stay within one hertz of the configured target.
