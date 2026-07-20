# ADR-270 Vendor RF Providers

RuView exposes a capability-safe Rust provider layer for vendor sensing and RF
telemetry. It never converts RSSI, occupancy, location or network inventory into
complex CSI.

## API

- `GET /api/v1/rf/vendors` — all provider descriptors and access states.
- `GET /api/v1/rf/vendors/latest` — latest validated event per vendor.
- `GET /api/v1/rf/vendors/:vendor/latest` — latest event for one stable vendor ID.
- `POST /api/v1/rf/vendors/:vendor/events` — ingest the vendor's documented
  sidecar/webhook payload through its strict provider decoder. This `/api/v1/*`
  route uses the server's bearer-token policy when configured.

Stable IDs are `origin_ai`, `plume`, `mist`, `netgear`, `electric_imp`,
`rf_solutions`, `linksys`, `luma`, `google_nest`, and `wifigarden`.

## Deterministic simulator

```bash
cd v2
cargo run -p wifi-densepose-hardware --bin vendor-rf-sim -- \
  --vendor plume --frames 100 --output plume.jsonl

# Stream canonical synthetic events to the sensing server UDP port.
cargo run -p wifi-densepose-hardware --bin vendor-rf-sim -- \
  --vendor mist --frames 100 --udp 127.0.0.1:5005 --realtime
```

Supported simulator names are `origin-ai`, `plume`, `mist`, `netgear`,
`electric-imp`, `rf-solutions`, `luma`, and `google-nest`. Linksys is refused
because its sensing service is discontinued. Wifigarden is refused until a
contracted event schema exists.

Every synthetic event includes `synthetic: true`, a deterministic sequence and
timestamp, and a source ending in `-sim-01`.

Canonical UDP JSON is accepted only when `synthetic: true`. Live vendor payloads
must use the HTTP ingestion route so provider-specific schemas, metric allowlists,
access states and bounds cannot be bypassed.

## Live/provider payloads

Provider decoders are strict, bounded and reject unknown schema fields. Origin
paths and credentials are supplied by the commercial contract. Plume uses a
read-only allow-listed OVSDB request plan. Mist and NETGEAR configurations use
regional HTTPS endpoints with redacted tokens. Electric Imp, RF Solutions and
Luma accept only allow-listed scalar metrics. Google Nest remains network-only.

Credentials are never embedded in fixtures or descriptors. Linksys returns
`Unsupported`; Wifigarden returns `ContractRequired`. These are usable,
test-covered provider outcomes—not simulated integrations.

## Hardware honesty

All descriptors remain `hardware_validated: false` until exact hardware/cloud
versions, lawful access, repeatable captures, calibration where applicable, and
fixture publication rights have been verified. Passing the simulator and API
tests validates RuView software only.
