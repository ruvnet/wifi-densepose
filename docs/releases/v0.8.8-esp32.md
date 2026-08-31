# RuView ESP32 firmware 0.8.8

Firmware 0.8.8 is a reliability and correctness release for ESP32-S3 and
ESP32-C6 RuView nodes. It makes the timing used by signal processing explicit,
prevents contradictory occupancy output, and improves update diagnostics.

## What changed

### Empty means zero people

Older firmware could report `presence=false` and a nonzero person count in the
same edge packet. That was internally contradictory and could contaminate an
empty-room calibration. Firmware 0.8.8 clears the count whenever the presence
gate is closed. The sensing server repeats the same check when it receives data
from older nodes.

This is a consistency fix, not proof that the heuristic can count multiple
people accurately. See
[ADR 346](../adr/ADR-346-fail-closed-edge-occupancy-evidence.md).

### Stable time scale on ESP32-C6

Raw CSI and on-device signal processing now have separate clocks. The C6 keeps
raw CSI moving over the network while its Tier 2 filters process a stable 8 Hz
sample stream. The S3 retains its 20 Hz DSP default. A phase-preserving sampler
keeps callback jitter from shifting those clocks.

The result is a correct time base for motion and vital-band features. It does
not by itself prove that heartbeat, respiration, gesture, or pose estimates are
more accurate. See
[ADR 347](../adr/ADR-347-rate-aware-esp32-temporal-sensing.md).

### Better diagnostics and safer updates

The one-second controller log now shows both raw callback yield and DSP rate.
The OTA status endpoint reports the actual selected application partition size
instead of a fixed 900 KB assumption. Firmware upload remains fail closed when
the node has no provisioned OTA signing secret.

## Measured hardware validation

All results below are physical measurements from 2026-08-31. They are not
simulator claims.

| Board | Duration | Raw CSI mean | DSP clock | Live coverage | Steady-state transport errors |
|-------|---------:|-------------:|----------:|--------------:|------------------------------:|
| ESP32-C6 node 4 | 300.64 s | 34.92 pps | 8.00 Hz | 97.62% | 0 |
| ESP32-C6 node 7 | 300.70 s | 36.32 pps | 8.00 Hz | 97.40% | 0 |
| ESP32-S3 node 1 | 300 s | 28.03 pps | Tier 0 | 100.00% | 0 |

The first C6 empty-room qualification observed 61 absent packets with zero
nonzero counts. The second C6 was transport-qualified in an occupied room and
still needs its own controlled empty-room sequence. Full evidence is recorded
in:

1. [C6 timing and transport](../validation/2026-08-31-esp32-c6-rate-aware-sensing.md)
2. [C6 occupancy integrity](../validation/2026-08-31-esp32-c6-occupancy-integrity.md)
3. [Second C6 timing and transport](../validation/2026-08-31-esp32-c6-node7-rate-aware-sensing.md)
4. [S3 transport](../validation/2026-08-31-esp32-s3-rate-aware-transport.md)

## Choose the correct download

| Release file | Target |
|--------------|--------|
| `esp32-csi-node-v0.8.8-s3-8mb-flash-bundle.zip` | ESP32-S3 with 8 MB flash |
| `esp32-csi-node-v0.8.8-s3-4mb-flash-bundle.zip` | ESP32-S3 with 4 MB flash |
| `esp32-csi-node-v0.8.8-c6-4mb-flash-bundle.zip` | ESP32-C6 using the supported 4 MB partition layout |
| `esp32-csi-node-v0.8.8-s3-8mb.bin` | S3 8 MB application only |
| `esp32-csi-node-v0.8.8-s3-4mb.bin` | S3 4 MB application only |
| `esp32-csi-node-v0.8.8-c6-4mb.bin` | C6 application only |

Never mix S3 and C6 images. Confirm the chip and physical flash before writing.

## Install or update

For a fresh installation, extract the matching bundle and follow its included
`FLASHING.md`. The standard offsets are:

| Image | Offset |
|-------|-------:|
| Bootloader | `0x0000` |
| Partition table | `0x8000` |
| OTA metadata | `0xf000` |
| Application | `0x20000` |

For an existing provisioned node:

1. Back up the current application partition.
2. Confirm the exact chip, flash layout, logical node, and serial port.
3. Read `http://DEVICE_IP:8032/ota/status`.
4. Use an application-only serial update at `0x20000` only when the running
   partition is `ota_0` and the image matches the board.
5. Reboot and confirm version 0.8.8, the preserved node identity, channel, and
   sensing-server target.
6. Run a five-minute burn-in before returning the node to calibration duty.

The full bundle does not contain an NVS image. A four-offset install therefore
preserves the existing WiFi and node settings, but operators should still keep
a backup before changing firmware.

## What this release does not prove

Firmware 0.8.8 does not prove medical-grade vital signs, accurate person
counting, identity, dense pose, through-wall video, or room separation. Those
claims require synchronized references and leakage-free held-out sequences.

The practical next acceptance test is a controlled empty-room capture with at
least 30 absent edge packets per updated node, zero absent packets carrying a
nonzero count, and zero transport or parser errors. Accuracy evaluation then
needs held-out occupied, movement, heartbeat-reference, and adjacent-room
sequences.
