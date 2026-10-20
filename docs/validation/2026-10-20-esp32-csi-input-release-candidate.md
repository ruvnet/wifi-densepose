# ESP32 CSI input release candidate, 2026-10-20

Status: INCOMPLETE. No hardware qualification or release acceptance is claimed.
Base source: `a1ca023ee` on `origin/main`. Changes cover #2286, the bounded
timeout and host-traffic mitigation for #2287, and onboarding fix #2302.

## Software evidence

MEASURED host-test outcomes with SYNTHETIC RF/device inputs:

- The complete firmware `make host_tests` suite passed under
  `espressif/idf:v5.4`, including the onboarding console regression.
- `test_edge_stbc.c` includes the production DSP implementation. Ordinary
  HT20, STBC, and maximum-size payloads advance the processing state; empty
  input does not. The test passes for both pre-HE and HE budgets.
- The DSP regression passes address and undefined-behavior sanitizers.
- `test_self_ping.c` executes the production setup against a synthetic AP and
  verifies that timeout equals interval and repeated setup is idempotent.
- Both new regressions fail against the original `origin/main` C sources:
  STBC does not advance the frame count; ping timeout exceeds its interval.
- A helper-only UDP test delivered 49 datagrams to an owned local socket at
  a requested 50 Hz for one second. Public/broadcast destinations and invalid
  rate, duration, and port inputs were rejected before traffic. This is not
  an ESP32 CSI measurement.

Reproduce the host checks from `firmware/esp32-csi-node/test`:

```sh
make host_tests
cc -std=c99 -fsanitize=address,undefined -g -Istubs -I../main \
  test_edge_stbc.c stubs/esp_stubs.c -o /tmp/edge-stbc-sanitized -lm
/tmp/edge-stbc-sanitized
cc -std=c99 -DCONFIG_SOC_WIFI_HE_SUPPORT=1 -Istubs -I../main \
  test_edge_stbc.c stubs/esp_stubs.c -o /tmp/edge-stbc-he -lm
/tmp/edge-stbc-he
```

## Hardware boundary

MEASURED identification only: the attached UART bridge enumerated as COM11;
esptool identified an ESP32-C6 revision 0.2 with 8 MB flash. Passive UART
monitoring and a non-writing onboarding identity request returned no runtime
lines. The standard C6 build selects native USB serial/JTAG, so an independent
CP210x UART bridge is not sufficient evidence of console availability.

Board form factor and flash-target confirmation remain pending. No candidate
was flashed. No boot log, sustained CSI capture, packet cadence, OTA recovery,
or sensing-accuracy validation has been obtained. The C6 cannot independently
qualify the S3-specific STBC correction on S3 hardware.

A private read-only backup of the first 128 KiB (bootloader, partition map,
NVS, and OTA selection data) succeeded at 115200 baud. Full 8 MiB reads at
921600 and 460800 baud failed with corrupt serial data. This is not a complete
recovery backup. The existing map has two application slots at `0x20000` and
`0x1f0000`, each `0x1d0000` bytes. Any future flash must preserve this map or
explicitly migrate it after a complete verified backup. No private backup is
included in this repository.

## Remaining release gates

- Finish the target build matrix and inspect binary size and configuration.
- Preserve the attached board's existing firmware, partition map, and NVS
  privately before any authorized flash; never commit those backups.
- Capture a candidate boot/runtime log and at least five minutes of actual
  CSI delivery. Verify callback yield, DSP cadence, edge packets, reset/panic
  absence, and fail-closed absent-person count behavior.
- Compare raw and edge delivery with and without bounded LAN UDP traffic.
- Qualify native USB onboarding on an IDF 6 device; host stubs alone cannot
  demonstrate that the physical USB RX ring is functioning.
- Resolve required CI failures. PR #2302's Rust audit artifact reports
  advisories affecting quinn-proto, rustls, wasmtime, and zip. Do not suppress
  the audit to make a release green.
- Review the exact release diff, bump firmware version, and publish only
  after terminal successful required checks and physical evidence review.

The firmware release workflow currently marks tag builds as prereleases.
There is no authorization implied here to relabel unqualified targets stable.
