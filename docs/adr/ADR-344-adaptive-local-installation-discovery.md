# ADR-344: Adaptive local installation discovery

## Status

Accepted — local advertisement and mobile discovery implemented; physical
peer-link and hosted-relay qualification pending.

## Context

RuView installations previously depended on a manually stored IP address.
DHCP changes, access-point changes, and client isolation could therefore leave
a healthy sensing installation unreachable from the mobile app. Repeated
authentication-policy log entries did not prove that the selected endpoint was
reachable.

Discovery must make commissioning recoverable without turning a service name
into authentication, leaking sensor data, scanning arbitrary subnets, or
silently moving private spatial data to a hosted service.

## Decision

The sensing server advertises one bounded `_ruview._tcp.local.` service when it
is bound to a routable interface. Loopback-only instances do not advertise,
and operators can disable advertisement with `--no-mdns`.

The TXT contract is deliberately small:

| Key | Required | Meaning |
|---|---:|---|
| `schema=ruview.installation.v1` | yes | Fail-closed protocol discriminator |
| `tls=0|1` | yes | HTTP or HTTPS origin construction |
| `installation=<opaque id>` | no | Bounded routing hint, never authentication |

The advertisement contains no node inventory, room identifier, SSID,
credentials, CSI, vital estimates, pose, identity, or learning data. The
hostname is bounded and sanitized before registration. Advertisement failure
is recoverable and does not stop sensing.

RuView Mobile resolves only the matching service and schema, validates the
resulting origin under its private-LAN/HTTPS policy, and requires an application
health probe before selection. Its adaptive broker retains the configured
origin preference and requires two failed configured probes plus two healthy
fallback probes before switching. Credentials remain scoped to their saved
origin.

The recovery ladder is:

1. configured private-LAN or HTTPS origin;
2. verified Bonjour local origin;
3. physically qualified Apple peer-to-peer local path;
4. explicit, authenticated HTTPS relay for bounded derived frames only;
5. optional administrator-managed WireGuard/Tailscale access.

Only levels 1 and 2 are implemented and software-validated by this decision.
Peer-to-peer browsing is enabled on Apple platforms, but it is not a qualified
peer-link data plane. This repository does not provide a hosted relay and must
fail closed when no local endpoint is healthy.

## Security and privacy consequences

- Service discovery is routing evidence, not installation authentication.
- Public HTTP origins, credentials in URLs, malformed records, and records with
  the wrong schema are rejected before use.
- Raw CSI, RSSI streams, camera/LiDAR frames, room geometry, pose labels,
  identity data, and training examples remain local.
- Future relay work requires a separate consent, authentication, revocation,
  minimization, and physical evidence review.
- ESP32-S3/C6 nodes remain provisioned to the sensing installation. This ADR
  does not claim direct phone-to-node discovery or invent a firmware protocol.

## Validation

Software acceptance requires:

- unit tests for bounded advertisement construction and hostname sanitation;
- mobile parser rejection, route scoring, anti-flapping, and Settings UI tests;
- Rust, TypeScript, lint, security, metaharness, Expo, and native compile gates;
- a real Bonjour resolve of the TXT contract followed by a successful
  `/api/v1/status` probe.

Physical qualification additionally requires installation discovery on an
iPhone, live frame and node-inventory receipt, DHCP-change recovery without
flapping, and a five-to-ten-minute zero-fusion-error burn-in. Simulator and
host-only results remain `MEASURED_SOFTWARE`; peer-link, relay, and physical
reconnection claims remain `NOT_MEASURED` until those captures exist.

## Implementation references

- `v2/crates/wifi-densepose-sensing-server/src/discovery.rs`
- `v2/crates/wifi-densepose-sensing-server/src/main.rs`
- Mobile companion decision: `cognitum-one/ruview-mobile`,
  `docs/adr/ADR-026-adaptive-local-installation-discovery-and-transport-recovery.md`
- Related decisions: ADR-034, ADR-054, ADR-296
