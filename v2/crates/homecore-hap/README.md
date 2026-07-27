# homecore-hap

`homecore-hap` is the fail-closed network foundation for HOMECORE's Apple
HomeKit Accessory Protocol bridge (ADR-125). It maps HOMECORE entities to HAP
services and provides the bounded server, persistence, discovery, and request
gating needed by a complete HAP implementation.

It does **not currently complete Apple Home pairing**. Pairing requests receive
a valid TLV8 `Unavailable` error, and accessory/characteristic endpoints return
HTTP 470 until an authenticated Pair-Verify session exists. There is no
plaintext header, bearer-token, or test credential bypass.

## Implemented

- Bounded Tokio TCP lifecycle with connection, header, body, request-time, and
  shutdown limits.
- Incremental HTTP/1.1 parsing through `httparse`; duplicate
  `Content-Length`, transfer encoding, truncated input, and oversized input
  fail closed.
- Versioned controller pairing records with bounded parsing, atomic same-
  directory replacement, Unix `0600` files/`0700` created directories, and
  refusal to load permissive or symlinked files.
- Controller identifiers, administrator invariants, and Ed25519 public keys
  validated through `ed25519-dalek`.
- Session state machine for Connected, Pair-Setup, Pair-Verify, Authenticated,
  and Closing. Authentication requires a valid signature from a persisted
  controller over the Pair-Verify transcript supplied by the future protocol
  phase.
- Real `_hap._tcp.local.` advertisement through `mdns-sd` when
  `hap-server` is enabled. `NullAdvertiser` provides deterministic,
  network-free tests and deployments.
- HAP-shaped `/accessories`, `/characteristics`, event subscription, and
  `EVENT/1.0` flow backed by `HapBridge` snapshots and bounded broadcasts.
  These handlers are structurally present but network-inaccessible until
  encrypted Pair-Verify is complete.

## Deliberately incomplete protocol phases

The following must land together before this crate may claim Apple Home
interoperability:

1. Pair-Setup M1-M6: SRP-6a proof exchange, setup-code policy, accessory
   Ed25519 identity persistence, HKDF derivation, and ChaCha20-Poly1305
   encrypted sub-TLVs.
2. Pair-Verify M1-M4: ephemeral X25519 exchange, accessory/controller Ed25519
   transcript signatures, HKDF session derivation, and encrypted sub-TLVs.
3. Encrypted HAP transport: length-prefixed frames, independent read/write
   ChaCha20-Poly1305 keys and monotonically increasing nonces, with strict
   frame limits and connection teardown on authentication failure.
4. Authenticated `/pairings` add/remove/list semantics and live mDNS `sf`
   updates.
5. Stable persisted AID/IID allocation and a HOMECORE service-call adapter for
   writable characteristics. The present endpoint is read/event-only.
6. Validation against Apple Home or a known-conformant HAP controller,
   including pair, restart, event delivery, write, unpair, and re-pair.

No cryptographic primitive should be implemented locally. The remaining work
must use reviewed RustCrypto/PAKE crates and protocol test vectors.

## Server integration

```rust,no_run
use std::{net::IpAddr, sync::Arc};
use homecore_hap::{
    start_server, HapBridge, HapServerConfig, HapServiceRecord,
    MdnsSdAdvertiser, PairingStore,
};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let record = HapServiceRecord::bridge(
    "HOMECORE Bridge",
    51826,
    "AA:BB:CC:DD:EE:FF",
);
let bridge = HapBridge::new(record);
let pairings = Arc::new(PairingStore::open(
    "/var/lib/homecore-hap/pairings.json",
)?);
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
// On process shutdown:
server.shutdown().await?;
# Ok(())
# }
```

Build and test:

```bash
cargo test -p homecore-hap --no-default-features
cargo test -p homecore-hap --features hap-server
```

Real mDNS requires the advertised address to be LAN-routable and the runtime to
have multicast access. Containers normally need host networking or macvlan.

## Decisions

- [ADR-125 — native Apple Home HAP bridge](../../docs/adr/ADR-125-ruview-apple-home-native-hap-bridge.md)
- [ADR-130 — bounded async REST/WebSocket server patterns](../../docs/adr/ADR-130-homecore-rest-websocket-api.md)
- [ADR-161 — server-layer security and explicit HAP deferral](../../docs/adr/ADR-161-homecore-server-layer-security.md)
