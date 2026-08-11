# ADR-293: Sensor data-plane hardening — UDP bind control and source allowlist (step one)

- **Status**: Accepted — initial implementation (this PR)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: security, udp, sensor-ingest, sensing-server

## Context

The CSI UDP receiver binds `0.0.0.0:{udp_port}` unconditionally
(`main.rs:5706`), with no equivalent of the HTTP `--bind-addr` flag (which
correctly defaults to `127.0.0.1`), no source allowlist, no message
authentication, no device identity, and no replay defense. Any host that can
reach the UDP port can inject a valid-shaped frame, flip an auto-detecting
server into a live source state, and influence presence/vital/automation
outputs (issue 1394).

An IP allowlist does not stop LAN spoofing, but bind control plus an allowlist
is the correct, shippable first step; per-device keys + authenticated
encryption + monotonic sequence + freshness window + replay rejection is the
full fix and is larger.

## Decision

**This PR (step one):**

- Add `--udp-bind` (env `RUVIEW_UDP_BIND`), **defaulting to `127.0.0.1`**.
  Binding to a routable address is now an explicit operator choice, mirroring
  the HTTP path. Desktop/appliance defaults stay loopback.
- Add an optional source IP/CIDR allowlist (`--udp-allow`); when set, frames
  from other sources are dropped and counted. Loopback is always allowed.
- Emit a startup security log line stating the bind scope and whether an
  allowlist is active; refuse a routable bind without an allowlist unless an
  explicit `--udp-insecure-lan` override is passed (parallel to the existing
  Docker HTTP refusal).
- Publish a `SECURITY.md`/advisory note describing the threat model and safe
  deployment.

**Explicitly deferred to a follow-up ADR (step two):** per-device provisioned
keys, MAC/AEAD, device identifiers, monotonic sequence numbers, freshness
window, and replay rejection. This ADR documents that gap rather than
implying the data plane is authenticated.

## Consequences

- Removes the default open-to-LAN exposure with a one-line-safe default.
- Not spoof-proof on a trusted LAN — the advisory says so plainly, and the
  override name (`--udp-insecure-lan`) makes the residual risk legible.
- A behavior change for anyone relying on the old implicit `0.0.0.0` default;
  called out in the changelog and the startup log.

## Validation

- Unit tests: default bind is loopback; routable bind without allowlist is
  refused unless overridden; allowlist accept/drop with counting; loopback
  always allowed.
- `cargo test -p wifi-densepose-sensing-server`.
- Real-silicon validation of the LAN path remains required before any
  deployment claim.
