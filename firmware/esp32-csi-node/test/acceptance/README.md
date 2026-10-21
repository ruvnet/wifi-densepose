# Configuration API acceptance checks

These scripts exercise the node configuration API (`GET /config`, `POST
/config`) against a **real node**. They are bench tools, not part of the
host test suite: they need a flashed board, a network, and the node's OTA
PSK.

They exist because the API's most important property cannot be proven by a
unit test: a request that the node refuses must leave every stored value
untouched, on the node itself, while it is live.

## Contents

- [What each script proves](#what-each-script-proves)
- [Requirements](#requirements)
- [Running the request checks](#running-the-request-checks)
- [Running the failure injection](#running-the-failure-injection)
- [Checking the boot log reader without a board](#checking-the-boot-log-reader-without-a-board)
- [Secrets](#secrets)
- [What these checks do not cover](#what-these-checks-do-not-cover)
- [See also](#see-also)

## What each script proves

| Script | Needs a board | Proves |
|---|---|---|
| `config_api_acceptance.sh` | no, only the network | a rejected request changes nothing; the second change in the trial window is refused with `409`; out-of-range values are refused |
| `config_api_failure_injection.sh` | yes, over USB | when the recovery bank cannot be written, the node answers `5xx` and no live value changes |
| `serial_boot_ip.py` | yes, over USB | reads the boot log and reports the address the node got |
| `make_trial_full_nvs.py` | no | builds the NVS image the failure injection needs |

The host-side unit test for the validation rules is
[`../test_config_rules.c`](../test_config_rules.c) and runs in CI. These
scripts cover the part that unit test cannot reach: the wire behaviour of a
running node.

## Requirements

| For | You need |
|---|---|
| `config_api_acceptance.sh` | `curl`, `jq`, network access to the node |
| `config_api_failure_injection.sh` | the above, plus `python3`, `esptool`, the `esp_idf_nvs_partition_gen` module, and the board on this host's USB |
| `serial_boot_ip.py` | `python3` (standard library only) |
| `make_trial_full_nvs.py` | `python3` and `esp_idf_nvs_partition_gen` |

`esp_idf_nvs_partition_gen` comes from Espressif (Apache-2.0):

```bash
python3 -m venv ~/venv && ~/venv/bin/pip install esp-idf-nvs-partition-gen esptool
```

## Running the request checks

The PSK is read from a file, never from the command line. See
[OTA key provisioning](../../README.md#ota-key-provisioning) for how to
create one.

```bash
./config_api_acceptance.sh http://192.168.0.115:8032 ~/.ruview-ota-psk
```

The node URL can also come from `$RUVIEW_NODE`, and the PSK path from
`$RUVIEW_PSK_FILE`.

The script prints one line per check and ends with a summary. It exits
non-zero if any check failed.

| Check | Expected |
|---|---|
| a payload with a type error | `400`, nothing written |
| a payload with a range error | `400`, nothing written |
| `vital_window=4096` | `400` (the loader applies 32..256) |
| `top_k_count=64` | `400` (the loader applies 1..32) |
| `channel_hop_count=14` | `400` (the channel list holds 6) |
| `vital_window=256`, `top_k_count=32` | `200`, the node reboots into them |
| a second change inside the trial window | `409` |
| a third change inside the trial window | `409` |

The last two checks point the node at a network that does not exist, so the
trial fails and the node restores its own values. Leave it long enough to
revert before using the node again.

## Running the failure injection

This one writes to the board. **The node is unreachable over the network
while the script runs**: the flash resets it, and the NVS image it writes
holds a full `cfg_trial` namespace.

```bash
SIZES="120" ./config_api_failure_injection.sh /dev/ttyUSB0 ~/.ruview-ota-psk
```

The script tries several filler sizes until the recovery bank write fails.
The size that works is hardware dependent; the range below is the one that
worked on an ESP32-S3 DevKitC-1 with 16 MB flash and 8 MB PSRAM.

| Filler entries | Result |
|---|---|
| 40, 60, 80, 100 | the bank still writes, the request succeeds |
| 120 | the bank write fails, the node answers `500` |
| 150 | the node no longer has room for its own WiFi state and does not come up |

Filling the whole partition is not a usable injection: the node dies in a
loop instead of refusing the request. Only the `cfg_trial` namespace is
filled.

To check a node that is **already** in the failure state, without touching
the board, point the script at it:

```bash
RUVIEW_ASSERT_ONLY=http://192.168.0.115:8032 \
  ./config_api_failure_injection.sh /dev/ttyUSB0 ~/.ruview-ota-psk
```

## Checking the boot log reader without a board

`serial_boot_ip.py` reads a serial port, so it can be exercised against a
pseudo terminal with a synthetic boot log. This is the shape of the check:

```python
import os, pty, subprocess
master, slave = pty.openpty()
p = subprocess.Popen([sys.executable, "serial_boot_ip.py",
                      "--port", os.ttyname(slave), "--no-reset"])
os.write(master, b"I (2345) esp_netif: wifi:sta: Got IP: 192.168.0.115\r\n")
print(p.communicate()[0].decode().strip())   # IP=192.168.0.115
```

Use `--no-reset` on a pty: it has no modem control lines, so the reset
pulse is skipped.

## Secrets

The PSK and the WiFi password are read from files and are never echoed, but
the **generated NVS CSV holds both in clear text**. Treat it as a secret
until it is deleted, and keep it out of the repository.

The tools never write a secret into a command line, a process argument, or
a log. `make_trial_full_nvs.py` reads the WiFi password from
`~/.wifi-pass` and the PSK from `~/.ruview-ota-psk` by default; both paths
can be overridden with `--wifi-pass-file` and `--ota-psk-file`.

## What these checks do not cover

They are hardware checks. A pass says the node behaved as specified on the
board it ran on; it does not qualify a different board, a different flash
size, or sensing accuracy. The repo rules on claims apply: report a result
as `MEASURED` only with the reproducer, and keep the evidence.

## See also

- [Remote Configuration (no USB)](../../README.md#remote-configuration-no-usb) — the API these checks drive
- [`../test_config_rules.c`](../test_config_rules.c) — the host-side unit test for the same validation rules
- [`../../config_push.py`](../../config_push.py) — the supported way to read and change a node's configuration
- [`../../provision.py`](../../provision.py) — USB provisioning, which writes the same NVS keys
