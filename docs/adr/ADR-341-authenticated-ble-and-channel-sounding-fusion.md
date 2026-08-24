# ADR-341: Authenticated BLE anchors and external Channel Sounding fusion

- **Status**: Accepted for implementation, hardware validation pending
- **Date**: 2026-08-23
- **Deciders**: ruv
- **Tags**: BLE, ESP32-S3, Channel Sounding, identity, sensor fusion, privacy

## Context

RuView needs a low-cost way to add short-horizon identity evidence and
micromotion research inputs to ESP32-S3 CSI nodes. Three premises require
correction before choosing the architecture:

1. ESP32-S3 can scan ordinary BLE advertisements and RSSI, but the supported
   ESP-IDF interface does not expose raw Bluetooth CTE IQ samples. Firmware
   cannot turn the S3 into a coherent CTE radar by configuration.
2. An iPhone's background BLE traffic is neither a stable identifier nor a
   RuView authentication token. Private addresses rotate, RSSI is not position,
   and a phone on a table is not proof of which body produced a CSI signal.
3. ESP32-S3 cannot acquire Bluetooth 6 Channel Sounding phase and RTT. Those
   primitives require a capable companion radio. Monotonic timestamps from two
   independent chips cannot be compared without synchronization.

This ADR adds two strictly separated paths:

1. A bounded-duty, passive S3 scanner for a RuView-specific authenticated,
   rotating BLE service token. It emits only a pseudonym, RSSI, TTL and evidence
   quality. It emits no BLE MAC, civil identity, raw advertisement or vital
   sign.
2. A default-off UART ingress for a separate Bluetooth 6 Channel
   Sounding-capable radio. The companion sends calibrated phase and timing
   primitives in an authenticated fixed frame. The S3 validates and forwards
   primitives. Host fusion may estimate respiration but must abstain under
   motion, incoherence, expiry or cross-source conflict.

All simulator output is labelled **SYNTHETIC**. No hardware or clinical
performance is asserted by this implementation.

## Decision

### BLE identity scanning

`CONFIG_BLE_IDENTITY_SCAN_ENABLE` is compile-time default off. A compiled image
also requires `ble_enable=1`, `ble_key_id`, and an exact 32-byte `ble_secret` in
NVS. Missing key material fails closed.

The scanner uses the ESP-IDF NimBLE passive extended-scan event path. The
default 50 ms window per 1000 ms interval is 5 percent controller scan duty.
Firmware rejects a configuration above 25 percent. It does not request scan
responses, keep BLE addresses, or accept general phone advertisements.
Controller duplicate suppression is disabled because a long-running scan must
observe repeated authenticated tokens to refresh the three-second evidence
TTL. The bounded scan window limits receive work, and unauthenticated payloads
are discarded before telemetry emission.

ESP-IDF 5.4 gates the extended discovery event structure behind extended
advertising support and otherwise defaults the NimBLE transport event buffer
to 70 bytes. The scanner therefore additionally requires
`CONFIG_BT_NIMBLE_EXT_ADV=y` and
`CONFIG_BT_NIMBLE_TRANSPORT_EVT_SIZE=257`. It rejects NimBLE reports
whose data status is incomplete or truncated, so a partial authenticated field
can never be parsed as a complete token. RuView advertisers keep the complete
advertising payload at or below 200 bytes; the canonical token-only payload is
52 bytes including its AD length and type octets.
Operators compare CSI packet yield with BLE off and on; a regression beyond the
deployment budget is a rollback condition.

The advertiser carries the vendor UUID
`6f31a840-5d65-4d69-9f09-c511b1e00100`. UUID bytes appear little endian on the
air. The 50-byte service record requires extended advertising; it cannot fit in
either a legacy 31-byte advertisement or its 31-byte scan response.

#### BLE service token v1

The AD element uses type `0x21`, Service Data with 128-bit UUID. The AD length
and type bytes precede this payload and are not included below.

| Payload offset | Size | Field | Validation |
|---:|---:|---|---|
| 0 | 16 | RuView service UUID, little endian | Exact match |
| 16 | 1 | token version | Must be 1 |
| 17 | 1 | key id | Must match provisioned key selector |
| 18 | 4 | Unix epoch minute, little endian | Host and scanner freshness window |
| 22 | 4 | advertiser nonce, little endian | Covered by HMAC, not forwarded |
| 26 | 8 | rotating pseudonym | Not a BLE MAC or civil identity |
| 34 | 16 | HMAC-SHA256 tag, first 128 bits | Constant-time comparison |

The HMAC covers the raw 16 UUID bytes plus offsets 16 through 33. A deployment
rotates the pseudonym at least once per token epoch and uses separate keys from
the Channel Sounding companion. A shared scanner key authenticates membership
in the provisioned deployment; it does not prevent relay and a compromised
scanner can forge advertiser tokens.

#### BLE telemetry v1

The scanner forwards a fixed 36-byte little-endian record with magic
`0xC51100B1`:

| Offset | Size | Field | Host mapping |
|---:|---:|---|---|
| 0 | 4 | magic | rvCSI packet discriminator |
| 4 | 1 | version | `source_contract.version = 1` |
| 5 | 1 | gateway node id | authenticated source id after ADR-305 verification |
| 6 | 1 | flags | bit 0 authenticated token, bit 1 scanner time verified, bit 2 extended advert |
| 7 | 1 | key id | `identity.key_id` |
| 8 | 4 | sequence | short-horizon gateway replay guard |
| 12 | 4 | observed boot ms | diagnostic only, not Unix time |
| 16 | 2 | TTL ms | `identity.ttl_ms`, maximum 5000 |
| 18 | 2 | quality permille | `identity.confidence`, evidence quality rather than identity probability |
| 20 | 1 | RSSI dBm | `identity.rssi_dbm` |
| 21 | 1 | TX power dBm | `identity.tx_power_dbm`, 127 means unavailable |
| 22 | 2 | reserved | Must be zero |
| 24 | 8 | rotating pseudonym | `identity.pseudonymous_token` |
| 32 | 4 | token epoch minute | host freshness recheck |

rvCSI maps the record to capability fields
`source_capability = { ble_scan: true, cte_iq: false, channel_sounding: false,
identity_kind: rotating_pseudonym }`. It must not map the token to a person
name, account, phone MAC or stable device identifier. The BLE telemetry no
longer contains the advertiser HMAC. It is accepted only inside the
authenticated gateway envelope defined below. The host verifies the envelope
before parsing the inner record and requires its gateway node id to match the
telemetry node id. TTL lifetimes are half open: evidence is live at host time
`t` only when `received_at <= t < received_at + ttl`.

### Authenticated gateway envelope

BLE telemetry and Channel Sounding primitives share the existing UDP data
plane, so neither is sent as a bare record. A bounded sender task wraps each
sanitized payload in this variable-length little-endian envelope. BLE and UART
callbacks use a nonblocking queue and never perform UDP I/O directly.

| Offset | Size | Field | Validation |
|---:|---:|---|---|
| 0 | 4 | magic `RVAE`, numeric `0x45415652` | Exact match |
| 4 | 1 | version | Must be 1 |
| 5 | 1 | payload type | 1 is BLE telemetry, 2 is Channel Sounding |
| 6 | 1 | flags | bit 0 means gateway monotonic receive time; other bits rejected |
| 7 | 1 | gateway key id | Exact enrolled selector |
| 8 | 2 | total frame length | Must equal datagram length |
| 10 | 2 | payload length | Exactly 36 for type 1 or 72 for type 2 |
| 12 | 1 | gateway node id | Exact enrolled node when configured |
| 13 | 3 | reserved | Must be zero |
| 16 | 4 | gateway sequence | Nonzero and strictly newer within the boot session |
| 20 | 8 | random gateway boot nonce | Nonzero authenticated replay namespace |
| 28 | 8 | gateway receive time in boot microseconds | Kept distinct from host receipt time |
| 36 | 4 | receive timing uncertainty in microseconds | Policy bounded on the host |
| 40 | N | sanitized payload | Exact type-specific size |
| 40 plus N | 16 | HMAC-SHA256 tag, first 128 bits | Constant-time comparison |

The tag covers the 12-byte domain `RuView/GW/v1` followed by offsets 0 through
`39 + N`. A third, independent 32-byte `radio_secret` protects this boundary.
The random boot nonce makes a legitimate gateway sequence reset explicit. The
host keys replay state by gateway node, key id and boot nonce, and separately
retains companion source session state. Host receipt time is never substituted
for the authenticated gateway receive time.

The default queue holds 16 records. The scanner admits at most 40 parsed RuView
tokens per second before advertiser HMAC work, and UART admits at most 100 valid
frames per second before companion HMAC work. Queue overflow drops evidence and
increments a rate-limited counter. Radio evidence uses normal UDP backpressure,
not the small low-rate priority control path.

This symmetric envelope supplies online source authentication and integrity.
It does not provide ADR-305 nonrepudiation and does not encrypt the payload.
Production evidence chains still add the ADR-305 signature or authenticated
transport witness. Deployments that treat rotating pseudonyms or phase samples
as confidential use WireGuard, DTLS, or an equivalent encrypted network path.

### External Channel Sounding companion

`CONFIG_CHANNEL_SOUNDING_INGRESS_ENABLE` is compile-time default off and limited
to ESP32-S3 in this implementation. NVS additionally requires `cs_enable=1`,
`cs_key_id`, a distinct exact 32-byte `cs_secret`, and an exact nonzero enrolled
`cs_source_id`. UART2 is the default to avoid the existing UART1 mmWave probe;
UART0 is rejected because it carries console and provisioning traffic. Pin
selection must be checked against the actual board before enabling.

The contract uses `sample_age_us`, not the companion monotonic clock. On receipt,
the gateway assigns its own monotonic time and approximates capture time as
`receive_time - sample_age_us`. `timing_uncertainty_us` remains attached to the
measurement. This does not claim synchronized clocks.

#### Authenticated Channel Sounding frame v1

The fixed 72-byte little-endian layout is:

| Offset | Size | Field | Validation and host mapping |
|---:|---:|---|---|
| 0 | 4 | magic `RVCS`, numeric `0x53435652` | packet discriminator |
| 4 | 1 | version | Must be 1 |
| 5 | 1 | flags | bit 0 calibrated, bit 1 gross motion; other bits rejected |
| 6 | 1 | key id | Must match separate companion key |
| 7 | 1 | reserved | Must be zero |
| 8 | 2 | frame length | Must be 72 |
| 10 | 2 | Bluetooth RF channel index | 0 through 78 |
| 12 | 4 | source sequence | strictly newer per source, wrap aware |
| 16 | 4 | sample age microseconds | maximum configured age, default 2 seconds |
| 20 | 4 | opaque companion source id | nonzero, provisioned capability join |
| 24 | 2 | quality permille | 0 through 1000, minimum default 600 |
| 26 | 2 | timing uncertainty microseconds | maximum 10000 |
| 28 | 4 | signed phase milliradians | minus 3142 through 3142 |
| 32 | 4 | signed RTT picoseconds | 0 through 250000 |
| 36 | 4 | signed frequency offset Hz | minus 500000 through 500000 |
| 40 | 4 | companion source session id | Nonzero authenticated boot or rekey namespace |
| 44 | 4 | Channel Sounding procedure id | Nonzero grouping key |
| 48 | 2 | procedure step index | Less than step count |
| 50 | 2 | procedure step count | 4 through 79 |
| 52 | 16 | HMAC-SHA256 tag, first 128 bits | constant-time comparison |
| 68 | 4 | IEEE CRC32 over offsets 0 through 67 | framing corruption check |

The HMAC input is the 12-byte domain string `RuView/CS/v1` followed by frame
offsets 0 through 51. Domain separation prevents a valid BLE token or another
protocol object from being reused as a companion measurement. CRC is not an
authenticator; it allows cheap corruption rejection before HMAC. The host maps
the source to
`source_capability = { ble_scan: false, cte_iq: false,
channel_sounding: true, phase: calibrated_flag, rtt: true }` only after HMAC,
freshness, sequence and capability enrollment checks pass.

The companion sends primitives only. A procedure is a coherent group of steps,
not proof that every advertised step was observed. Host estimators require
an exact complete set of step indexes, unique channels, and consistent step
counts. A valid procedure contains 4 through 79 steps because Bluetooth RF
channel indexes are limited to 0 through 78. Estimators reject mixed source
sessions, mixed gateway boot scopes, duplicate channels, duplicate steps,
incomplete procedures, or inconsistent metadata. The companion does not send
`respiration_bpm`, `heart_rate_bpm`, identity or clinical labels.

### Host fusion and deterministic replay

`ruview-fusion::radio_fusion` repeats all Channel Sounding bounds, CRC and HMAC
checks. It assigns gateway receive time, retains timing uncertainty, circularly
centres calibrated phase, and estimates a nonclinical respiratory component
only from complete coherent procedures within one enrolled source session and
gateway boot scope. It abstains under gross motion, insufficient duration,
incomplete procedures, source mixing or weak spectral concentration.

The host must pass parsed BLE and Channel Sounding records through the bounded
`RadioReplayGuard` before fusion. Gateway and BLE sequences are strictly newer;
only the companion Channel Sounding sequence uses serial-number wrap semantics.
All maps are bounded. A companion sequence reset requires a new authenticated
nonzero source session id. A gateway sequence reset requires a new authenticated
random boot nonce.

The sensing server snapshots replay high-water marks in versioned private JSON.
Raw eight-byte BLE pseudonyms are replaced by one-way replay fingerprints before
serialization. A separate private lock file gives the runtime exclusive
ownership. Missing replay state fails closed unless the operator passes the
explicit one-shot initialization option; the runtime rejects that option once
a snapshot exists, so it must be removed after creation. Deletion of an
established snapshot requires rotation of all enrolled keys. Secret and replay
files are opened once without following symbolic links and validated from their
file descriptors.

HMAC, estimation, and storage run in a dedicated ordered worker behind a bounded
256-record queue. The worker group commits at most once per second or every 256
records. The shared UDP loop admits only the exact 92-byte BLE and 128-byte
Channel Sounding RVAE envelopes before allocation and queueing. The worker
releases no update until the replay snapshot is durable. P4 and P5 WebSocket
export is closed by default. The local override is not a subject consent receipt
and is accepted only with an authenticated loopback server and a private
append-only audit log. A newly created audit entry and its parent directory are
synced before export is enabled. Opened file identity and canonical-path checks
prevent the audit log from aliasing replay state, its lock, or any gateway,
pseudonym, or companion secret. An incomplete audit tail, audit write failure,
or audit sync failure stops the worker and disconnects its queue. Exact P0 phase,
RTT, frequency offset, channel, and step vectors never enter the WebSocket
message. Aggregate decisions expire five seconds after host receipt.

The live host boundary supports multiple independently keyed gateways. Its
current scope ends at authenticated BLE and Channel Sounding admission plus
standalone aggregate publication. CSI track association and CSI plus Channel
Sounding rate fusion exist in the deterministic library simulation but are not
wired into the production sensing-server track manager by this ADR. That
follow-on cannot be represented as live or measured until its own integration
and hardware gate passes. TTLs are half open, so evidence is expired at the
exact expiry timestamp.

CSI and Channel Sounding respiration estimates combine only when live and
within the configured disagreement threshold. BLE association receives
geometry-derived track likelihoods from an upstream localizer. RSSI alone is
not treated as coordinates. Associations are one-to-one, TTL bounded and
fail closed on ambiguity or an incompatible duplicate pseudonym.

The built-in deterministic replay includes two tracks that approach, overlap
and cross. It binds both before the crossing, abstains at exact overlap, and
rebinds each rotating token to its original privacy-preserving track afterward.
Separate cases verify spoof conflict, TTL expiry, Channel Sounding motion
abstention and CSI plus Channel Sounding rate fusion. Every output is marked
`SYNTHETIC`; it is not hardware evidence.

## Security and privacy analysis

BLE, Channel Sounding and gateway-envelope tags truncate HMAC-SHA256 to 128 bits. Under a
uniform forgery model, an attacker making `q` independent online attempts has
success probability at most approximately `q / 2^128`. Even one billion
attempts gives approximately `2.9 × 10^-30`. The practical risks are therefore
key extraction, shared-key blast radius, token relay, compromised firmware and
mis-enrollment rather than blind tag guessing.

Controls are:

1. Separate BLE, companion and gateway-envelope keys with explicit key ids and
   protocol domain separation.
2. Exact 32-byte secrets provisioned from files that are never printed or
   persisted in the local provisioning state JSON.
3. Secure boot, flash encryption and NVS encryption for production devices.
4. Short token epochs, host freshness validation, packet TTL and sequence
   replay checks.
5. No BLE address or raw advertisement egress. Only a rotating eight-byte
   pseudonym leaves the scanner.
6. Geometry consistency and fail-closed abstention for relay or crossing
   ambiguity. HMAC authenticates a token but cannot prove physical proximity.
7. Authenticated gateway envelopes before accepting either inner payload. A LAN
   source address or an inner packet flag is not sufficient.
8. Bounded queues, ingress rates, exact pre-allocation payload sizes and replay
   maps. Invalid input cannot allocate unbounded state or block a radio callback.
9. Encrypted transport when LAN disclosure of pseudonyms or biological phase
   primitives is outside the deployment threat model.
10. Multiple gateway secrets are unique; the host selects an enrollment by the
    unauthenticated node and key selectors, then authenticates the complete
    envelope before accepting either selector as provenance.
11. Session capacity remains fail closed. Retiring a gateway boot or companion
    session is an administrative key-rotation operation because automatic
    eviction could make an old authenticated capture replayable.

The eight-byte rotating pseudonym has 64 bits of collision space. At 10000
simultaneous tokens, random collision probability is approximately
`2.7 × 10^-12` per epoch. Collision detection still abstains because a collision
could also be an intentional conflict.

## Consequences

The incremental firmware cost is BLE controller/host memory only when the BLE
option is compiled. Default builds retain current behavior. BLE radio use may
reduce CSI yield because WiFi and BLE share 2.4 GHz silicon. The default scan
duty is 5 percent, with a hard 25 percent ceiling, but real coexistence cost is
hardware and traffic dependent.

The architecture gains identity evidence, not absolute identity. A phone or
beacon must run the RuView token protocol; ordinary iPhone background traffic
does not qualify. It gains a path for Bluetooth 6 phase measurements, not
Bluetooth 6 capability on ESP32-S3.

## Rollback

Set `ble_enable=0` and `cs_enable=0`, or build with both Kconfig options off.
No existing packet magic, CSI path or vitals packet changes. Remove all three
provisioned secrets during decommissioning. Roll back BLE if measured CSI packet
yield or downstream presence quality breaches the deployment baseline.

## Validation

1. Host C tests parse malformed BLE AD elements, confirm privacy-minimized
   telemetry, validate Channel Sounding CRC, bounds, sample age and wrap-aware
   sequence handling.
2. Provisioning tests confirm separate BLE, Channel Sounding and gateway
   envelope secrets become NVS blobs, never enter the additive local state, and
   cannot leak into a fallback CSV.
3. Rust tests verify both HMAC layers, tamper rejection, wrong-key rejection,
   gateway boot replay, companion session replay, expiry, two crossing tracks,
   spoof conflict, motion abstention and synthetic respiration fusion.
4. ESP-IDF 5.4.2 clean builds passed with both features disabled and with BLE
   identity plus UART companion ingress enabled. The enabled image is 1,327,888
   bytes with 37 percent of the smallest application partition free. The
   default image is 1,139,536 bytes with 46 percent free. The feature cost is
   188,352 bytes. These are compile and link receipts, not hardware receipts.
5. Host tests verify one-shot replay initialization, exclusive locking,
   descriptor-based private file checks, two independently keyed gateways,
   exact pre-copy RVAE sizes, audit path isolation, fatal audit failure, and
   commit plus audit before P5 publication.
6. Real hardware acceptance requires captured ESP32-S3 boot/runtime logs,
   measured CSI yield before and after BLE enablement, and a capable Bluetooth
   6 companion capture. Until then the implementation is **CLAIMED** and all
   replay results are **SYNTHETIC**.

## Primary references

1. [Bluetooth SIG Channel Sounding overview](https://www.bluetooth.com/learn-about-bluetooth/feature-enhancements/channel-sounding/)
2. [Bluetooth Core Specification 6.0 feature overview](https://www.bluetooth.com/core-specification-6-feature-overview/)
3. [Bluetooth Low Energy primer](https://www.bluetooth.com/bluetooth-le-primer/)
4. [ESP32-S3 Bluetooth Low Energy feature support](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/ble/ble-feature-support-status.html)
5. [ESP32-S3 NimBLE device discovery guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/ble/get-started/ble-device-discovery.html)

## Acceptance test

On two ESP32-S3 gateways plus one enrolled Channel Sounding companion, replay
two rotating BLE tokens through a physical track crossing. Pass only if both
tokens bind before and after, exact overlap produces abstention, stale and
forged tokens produce no binding, motion suppresses respiration within one
fusion cycle, and BLE enablement keeps measured CSI packet yield within the
operator's predeclared regression budget. Record boot logs and raw counters;
the deterministic simulator alone cannot pass this hardware gate.
