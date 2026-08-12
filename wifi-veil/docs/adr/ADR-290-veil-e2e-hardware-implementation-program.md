# ADR-290: VEIL end-to-end hardware implementation program (multi-provider firmware)

| Field | Value |
|-------|-------|
| **Status** | Proposed — P4 scaffolding (build-only); portable core validated on host |
| **Date** | 2026-08-09 |
| **Parent** | ADR-288 (VEIL shield), ADR-289 (harness), ADR-282 (L0–L5 evidence ladder) |
| **Location** | `firmware/privshield/` |
| **Relates to** | `firmware/esp32-csi-node/` (the CSI sensor/attacker node), ADR-280 (governed actuation), ADR-141 (attestation) |

## 0. PROOF discipline

The **only** artifact validated here is the portable C core
(`firmware/privshield/core/`): a host test (`make test`) checks energy
conservation, reversibility, wrong-key failure, and — pinned — that its
SplitMix64 key schedule is **byte-identical to the Rust crate's** PRNG. That is
`build`/host-level evidence, not silicon. Every per-provider adapter is a
**build-only scaffold** with `TODO(hw)` markers: `SYNTHETIC / L0`, no captured
log, no `MEASURED` claim. Nothing in this ADR asserts VEIL works on real
hardware; it asserts a *plan and a shared core* to get there (P5).

## 1. Context

ADR-288 shipped VEIL as a deterministic, no-radio Rust model, and the 2025–2026
SOTA sweep (ADR-288 §sota) confirmed the mechanism's family is real and
standard-permitted. The open question left was **"does this run on real WiFi
hardware, and on which?"** — including the user asks: *can OpenWRT / open WiFi
software implement it, and can ESP32 help scramble signals?* Answering requires
committing to the platform reality rather than assuming a uniform "firmware"
target.

## 2. Decision

Stand up `firmware/privshield/` as a **multi-provider E2E program** around one
shared, validated core:

1. **A portable C shield core** (`core/veil_shield.{h,c}`) — the keyed
   Givens-rotation obfuscation, `no_std`-friendly C99 (no malloc/libc I/O), with
   a SplitMix64 key schedule matching the Rust crate so on-air behavior is
   identical everywhere and every adapter links the *same* math. Host-tested.
2. **Per-provider adapters**, each built and graded by a hardware research
   agent, honest about what its stack can actually touch:
   - **`openwifi/`** (open PHY/MAC on SDR/FPGA) — the highest-capability path and
     the one that can host the **keyed-reversible** design end-to-end
     (protector + AP-side compensation). Carries the **P5 measurement protocol**
     (`MEASUREMENT.md`) that yields the first `MEASURED` result with a witness.
   - **`openwrt/`** (Linux `mac80211`, mt76/ath9k…) — the commodity path.
     Sounding-cadence randomization, MU-group and stream-mapping control are
     feasible from the driver/hostapd; the per-packet unitary on the LTF spatial
     mapping is firmware-deep on most parts. Partial.
   - **`nexmon/`** (Broadcom/Cypress C firmware patches) — the commodity
     C-firmware route; the read path is proven (Wi-BFI/nexmon_csi), the transmit
     report-shaping path is research-grade/partial.
   - **`esp32/`** (ESP-IDF) — **not** a feedback protector (the beamforming path
     is a closed blob): ESP32 shapes CSI *read*, not transmitted feedback. Its
     legitimate roles are a **sensing detector** (trigger the AP-side shield) and
     an **RIS controller** (drive an external reconfigurable surface to scramble
     the sensing direction — the honest way ESP32 "helps scramble", via an
     external surface, not its own PHY).
3. **Compliance stance carried into hardware:** every control shapes the node's
   own standards-conformant emission and preserves energy; the ESP32
   decoy/cover-traffic idea is documented as *legally sensitive / not
   recommended* precisely because it edges toward the interference line.

Per-provider feasibility grades live in each subdir README and the top-level
feasibility matrix; they are the answer to the "which hardware" question.

## 3. What this explicitly is NOT

- **Not validated firmware.** No adapter has run on silicon; there is no witness.
  The scaffolds compile-*shaped*, not compile-*guaranteed* on their toolchains
  (which are absent in this environment).
- **Not a claim that ESP32 can shield beamforming feedback** — it cannot; it is a
  detector/RIS-controller only.
- **Not jamming, on any platform.** Compliant waveform shaping only.
- **Not a MEASURED result.** That is P5, gated on a captured log.

## 4. Consequences

- One validated core, four honest provider scaffolds, and a concrete P5
  measurement plan — a real path from model to silicon, with the effort/blocker
  reality made explicit per platform.
- The shared core keeps every future hardware result consistent with the crate
  and with each other.
- Scope stays inside `firmware/privshield/`; no other crate/firmware is touched
  (the existing `esp32-csi-node` remains the sensor/attacker node).

## 5. Validation

```bash
cd firmware/privshield/core && make test    # host: energy/reversibility/PRNG parity
# per-provider builds require their toolchains (ESP-IDF, OpenWRT SDK, Nexmon,
# Vivado) and real hardware — see each subdir's BUILD/INTEGRATION notes.
```
