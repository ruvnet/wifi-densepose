# homecore-hap

`homecore-hap` is HOMECORE's bounded, fail-closed HAP IP accessory server
(ADR-125). It implements the HAP R2 cryptographic pairing and transport
boundary without relying on the broken `hap` 0.1 pre-release crate.

## Security and protocol coverage

- Pair-Setup M1-M6 uses the RFC 5054 3072-bit group with SHA-512, the HAP
  compatibility proof construction, HKDF-SHA512, ChaCha20-Poly1305, and
  Ed25519 long-term keys.
- Pair-Verify M1-M4 uses ephemeral X25519, strict Ed25519 transcript
  verification, HKDF-SHA512, and authenticated encrypted sub-TLVs.
- After successful M4, all HTTP and `EVENT/1.0` traffic uses HAP records:
  two-byte little-endian authenticated lengths, at most 1024 plaintext bytes,
  independent directional keys, and monotonic 64-bit nonces. Authentication,
  replay, truncation, and oversize failures close the connection without an
  oracle response.
- `/accessories`, `/characteristics`, and `/pairings` are inaccessible until
  Pair-Verify succeeds on that TCP connection. Pairings add/remove/list is
  restricted to a currently persisted administrator.
- Accessory identity, Ed25519 seed, SRP salt/verifier, and controller records
  are stored in a versioned file using same-directory atomic replacement.
  Created Unix directories use mode `0700`, files use `0600`, and permissive,
  oversized, symlinked, legacy, or malformed stores fail closed.
- The raw setup code is returned only during first provisioning. Only its SRP
  verifier is persisted; `SetupCode` redacts `Debug` output and zeroizes on
  drop.
- Removing the last administrator atomically clears every pairing. Live
  sessions observe pairing revisions and are revoked, while the removal
  response is delivered before the requesting session closes.

The server also bounds connections, headers, bodies, request time, shutdown,
TLV sizes, controller counts, setup attempts, and concurrent Pair-Setup.
mDNS advertises the persisted accessory identifier and updates `sf` after
pairing or unpairing.

## Provisioning and server integration

```rust,no_run
use std::{net::IpAddr, sync::Arc};
use homecore_hap::{
    start_server, HapBridge, HapServerConfig, HapServiceRecord,
    MdnsSdAdvertiser, PairingStore,
};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let provisioned = PairingStore::load_or_create(
    "/var/lib/homecore-hap/security.json",
)?;
if let Some(setup_code) = provisioned.setup_code.as_ref() {
    // Send this once to a trusted local display or provisioning boundary.
    println!("HAP setup code: {}", setup_code.expose());
}
let pairings = Arc::new(provisioned.store);
let record = HapServiceRecord::bridge(
    "HOMECORE Bridge",
    51826,
    pairings.accessory_id()?,
);
let bridge = HapBridge::new(record);
let advertiser = Arc::new(MdnsSdAdvertiser::new(
    "homecore",
    "192.168.1.50".parse::<IpAddr>()?,
)?);

let server = start_server(
    HapServerConfig::default(),
    bridge.clone(),
    pairings,
    advertiser,
).await?;

// Feed HOMECORE StateChanged events through bridge.update_accessory(...).
server.shutdown().await?;
# Ok(())
# }
```

The mDNS device ID must equal the persisted accessory ID; startup rejects a
mismatch. Real mDNS also requires a LAN-routable advertised address and
multicast access.

## Validation and remaining interoperability limits

The deterministic suite covers the HAP SRP session-key vector, complete
Pair-Setup and Pair-Verify ceremonies, transcript tampering, wrong proofs,
malformed/replayed/oversized records, atomic restart, last-admin removal, and
a real TCP lifecycle from Pair-Verify through encrypted `/accessories`.

This is protocol-level HAP R2 coverage, not a claim of Apple certification:

- It has not yet been exercised against a current Apple Home controller or
  the current commercial MFi specification.
- Transient and split Pair-Setup flags are rejected as `Unavailable`.
- Writable characteristic service calls, timed writes, resource endpoints,
  and stable persisted AID/IID allocation are not implemented. The present
  characteristic surface is read and event subscription only.
- Operational hardening still depends on protecting the host and the
  `0600` security file; no hardware-backed key store is integrated.

Build and test:

```bash
cargo test -p homecore-hap --no-default-features
cargo test -p homecore-hap --features hap-server
cargo clippy -p homecore-hap --all-targets --features hap-server -- -D warnings
```

## Decisions

- [ADR-125 — native Apple Home HAP bridge](../../../docs/adr/ADR-125-ruview-apple-home-native-hap-bridge.md)
- [ADR-130 — bounded async REST/WebSocket server patterns](../../../docs/adr/ADR-130-homecore-rest-websocket-api.md)
- [ADR-161 — server-layer security and honest labeling](../../../docs/adr/ADR-161-homecore-server-layer-security.md)
