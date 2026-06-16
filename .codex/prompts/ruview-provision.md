# /ruview-provision — provision an ESP32 sensing node

Write NVS config to a RuView ESP32 node. Args: `$ARGUMENTS` (expect `--port`, `--ssid`, `--password`, `--target-ip`, optional `--channel`, `--filter-mac`). Default port `COM8`.

First get the authoritative flag list: `python firmware/esp32-csi-node/provision.py --help` (on Windows prefix `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` — the help text has non-ASCII and crashes under cp1252). Then run:

```
python firmware/esp32-csi-node/provision.py --port COM8 \
  --ssid "<SSID>" --password "<PW>" --target-ip <SINK_IP> --target-port 5005 --node-id <0-255> \
  [--channel <N>] [--filter-mac <AA:BB:CC:DD:EE:FF>] [--hop-channels 1,6,11 --hop-dwell 200] \
  [--tdm-slot <i> --tdm-total <n>] [--edge-tier 0|1|2] [--pres-thresh 50] [--fall-thresh 15000] \
  [--vital-win 300] [--vital-int 1000] [--subk-count 32] \
  [--seed-url http://10.1.10.236 --seed-token <bearer> --zone lobby] [--swarm-hb 30] [--swarm-ingest 5] [--dry-run]
```

Trade-offs:
- `--channel <N>` pins the node to one WiFi channel (set it to the AP's channel). Omit it and pass `--hop-channels 1,6,11` for the firmware's multi-band hopping schedule (more sensing bandwidth, uses neighbour APs as illuminators; `--hop-dwell` ms per channel).
- `--filter-mac <MAC>` restricts CSI capture to one transmitter (cleaner signal); omit for all transmitters (more data, more noise).
- `--edge-tier` 0/1/2 = off / stats / vitals (ADR-041). `--tdm-slot`/`--tdm-total` slot a multi-node mesh. `--fall-thresh 15000` ≈ 15.0 rad/s² (raise to cut false falls).

⚠️ **Issue #391:** flashing rewrites the *entire* `csi_cfg` NVS namespace — every key not on the CLI is erased. Pass the full set you want; warn before re-provisioning a working node. `--dry-run` builds the NVS binary without flashing; `--force-partial` allows config without WiFi creds (knowingly).

Fleet provisioning: `python scripts/generate_nvs_matrix.py` (subprocess-first — the `esp_idf_nvs_partition_gen` API changed across versions).

Verify: serial monitor (pyserial on `COM8`, 115200) should show `adaptive_ctrl` ticks + `csi_collector: CSI cb #… len=128 rssi=… ch=…` lines; the sink `cd v2 && cargo run -p wifi-densepose-sensing-server` should report incoming UDP frames if `--target-ip` points at this host. If no frames: wrong channel, MAC filter too tight, target-ip not this host, or WiFi creds wrong — re-run with corrected args.
