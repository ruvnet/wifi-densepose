# ADR-268: Qualcomm Atheros CSI Platform Strategy

- **Status**: accepted
- **Date**: 2026-07-18
- **Tags**: qualcomm, atheros, csi, ath9k, ath11k, ath12k, simulator

## Context

RuView needs a Qualcomm path that is useful before vendor hardware access while
remaining honest about firmware boundaries. QCA9300 has demonstrated CSI tooling
through ath9k/PicoScenes-class systems. QCN9074 and QCN9274 have upstream Linux
connectivity drivers, but upstream ath11k/ath12k support does not by itself prove
that raw per-packet complex CSI is exported by public firmware.

## Decision

1. Use QCA9300 as the first physical baseline: 802.11n, up to 3x3 MIMO and
   20/40 MHz. Accept translated captures from established research tooling.
2. Model QCN9074 (Wi-Fi 6/6E, 4x4, up to 160 MHz) and QCN9274 (Wi-Fi 7, 4x4,
   up to 160 MHz in protocol v1) as explicitly experimental simulator profiles.
3. Keep firmware/kernel formats behind a Rust adapter. RuView ingests only the
   validated QCS1 application envelope defined by ADR-269.
4. Never label simulated frames as hardware. Physical support requires captured
   fixtures, firmware provenance, antenna ordering, scaling and repeatability tests.
5. Prefer an upstream-reviewed Generic Netlink or relay-style export if modern
   Qualcomm firmware exposes CFR/CSI; do not depend on undisclosed structs.

## Consequences

- Development, APIs and downstream sensing can be tested immediately.
- QCA9300 offers the shortest path to real Qualcomm data.
- Modern profiles may remain simulator-only until firmware cooperation exists.
- A translation copy is accepted in exchange for a stable, fuzzable boundary.

## Links

- [ADR-269: QCS1 wire protocol](ADR-269-qualcomm-csi-wire-protocol.md)
- [Linux ath11k supported devices](https://wireless.docs.kernel.org/en/latest/en/users/drivers/ath11k.html)
- [PicoScenes supported hardware](https://ps.zpj.io/manual/hardware.html)
- [ADR-270: vendor integration portfolio and acceptance gates](ADR-270-vendor-rf-sensing-integration-program.md)

