# VEIL — OpenWRT / Linux `mac80211` adapter

> **STATUS: `SYNTHETIC / L0` — BUILD-ONLY, UNTESTED ON HARDWARE.**
> No radio was driven, no CSI captured, no log produced on silicon. Every
> claim below is a design/feasibility statement, not a `MEASURED` result. This
> adapter uses **compliant waveform controls only** — it never jams and emits
> no denial energy.

This directory is the OpenWRT/`mac80211` platform adapter for the VEIL privacy
shield. It links the validated portable core
(`../core/veil_shield.{h,c}` — the keyed Givens rotation over the identity-bearing
"fine" subspace of 802.11 compressed beamforming feedback) and drives the subset
of controls that Linux userspace/`mac80211` can actually reach on commodity APs.

---

## Feasibility grade: **C** (partial — coarse compliant controls only)

**Why C, not higher.** VEIL's defining action is a *per-packet keyed unitary* on
the compressed beamforming-feedback angles (equivalently, a keyed Q on the LTF
spatial mapping / precoder). On every mainstream OpenWRT AP chipset
(Qualcomm ath10k/ath11k/ath12k, MediaTek mt76 / mt7915), that report is generated
and the precoder applied **inside the WiFi MCU firmware blob** — userspace and the
open driver never touch the pre-transmit V matrix. So the full keyed-rotation path
is **blob-blocked** from OpenWRT. What remains reachable is a set of *coarse*
compliant knobs that perturb, but do not cryptographically obfuscate, the CSI a
sensor observes. That is a real, honest defense-in-depth layer — hence C, not D —
but it is not the full VEIL transform.

**Why not D.** Some controls genuinely work from userspace (TX antenna map;
hostapd-mediated sounding/beamformer capability), and one chipset family
(**ath9k**) is open enough at the register level that a *driver patch* could reach
the static spatial-mapping matrix — a credible route to B on that specific,
older hardware. openwifi (FPGA) and Nexmon (Broadcom) are the routes to the full
A-grade keyed rotation, but those are **separate adapters**, not OpenWRT.

---

## What is FEASIBLE vs. BLOB-BLOCKED from OpenWRT

| VEIL control | Reachable from OpenWRT? | Mechanism (real API / knob) | Notes |
|---|---|---|---|
| **TX antenna-map perturbation** | ✅ Feasible | `NL80211_CMD_SET_WIPHY` + `NL80211_ATTR_WIPHY_ANTENNA_TX` / `_RX` | Coarse static spatial-mapping change. Many drivers require phy DOWN and symmetric masks. Compliant. |
| **NDP sounding-cadence jitter** | 🟡 Indirect | hostapd `ctrl_iface` (rewrite `SOUNDING-DIMENSION`, toggle `[SU-BEAMFORMER]`, `RECONFIGURE`) | No `nl80211` "set sounding interval" exists; the per-NDP timer lives in driver/firmware. We can only jitter the *offered* capability. |
| **Beamformer/beamformee capability toggle** | ✅ Feasible | hostapd `vht_capab` / `he_su_beamformer` etc. | Standards-compliant advertisement. Coarse on/off, not per-packet. |
| **Spatial-stream → antenna mapping (static Q)** | 🟡 Driver-patch (ath9k only) | ath9k PHY spatial-mapping registers (`AR_PHY_*`) | Open enough to patch on ath9k; opaque/firmware on ath10k+/mt76. Not a stock userspace knob. |
| **MU-MIMO group shuffling** | ❌ Blob-blocked | would need `NL80211_CMD_VENDOR` subcmd that upstream mt76/ath do **not** expose | Group formation + steering matrices computed in MCU firmware. |
| **Per-packet keyed unitary on LTF / precoder** | ❌ Blob-blocked | — | The core VEIL transform. Lives in firmware on all commodity AP parts. Requires firmware patch, or use openwifi / Nexmon adapters. |
| **Compressed-BF-report angle edit (φ/ψ)** | ❌ Blob-blocked | — | Report is generated in firmware/PHY; not exposed pre-TX on OpenWRT. |
| **External sensing-solicitation detection (NDPA cadence)** | 🟡 Partial | `NL80211_CMD_FRAME` + `NL80211_CMD_REGISTER_FRAME`, or monitor-mode capture | Commodity drivers do not forward raw NDPA to userspace by default. |

---

## Best candidate chipsets / drivers

- **ath9k (Atheros 802.11n)** — *best open target for a driver-side patch.* The
  most transparent open driver (no per-packet firmware for the datapath), with a
  long history of PHY register access and the Atheros CSI Tool ecosystem. A
  static spatial-mapping perturbation and CSI observation are realistic here;
  full HT beamforming-feedback editing still is not in open code. 802.11n-only.
- **mt76 (MediaTek mt7915 / mt7622-mt7615)** — *best-maintained modern open
  driver* and the most likely place upstream would eventually accept a vendor
  hook, but beamforming/sounding/MU grouping run in the MCU firmware today, so
  the keyed path needs a firmware patch (blob-blocked out of the box).
- **ath10k / ath11k / ath12k (Qualcomm)** — most capable radios but the most
  closed: regulatory + beamforming + sounding all firmware-side. ath11k/ath12k
  have **no open firmware** at all. Worst target for the keyed path.
- **openwifi (FPGA SDR) / Nexmon (Broadcom)** — the only routes to the full
  A-grade keyed rotation; handled by the sibling `../openwifi/` and `../nexmon/`
  adapters, **not** this OpenWRT one.

**Recommendation:** for OpenWRT specifically, target **ath9k** for a
driver-patch proof-of-concept (spatial-mapping + CSI), and **mt76/mt7915** as the
strategic modern platform pending a firmware/vendor-subcmd hook.

---

## Build (host, build-only)

```bash
make core      # always works: compiles+links the portable core, no libnl needed
make daemon    # builds veil_shieldd IF libnl-genl-3.0 dev headers are present
make clean
```

`make daemon` cleanly **skips** (does not fail) when `libnl-genl-3.0` is absent,
printing the required dev packages. On an OpenWRT buildroot use `openwrt.mk`
(rename to `Makefile` under `package/utils/veil-shieldd/`), which builds against
`libnl-tiny`. See `INTEGRATION.md` for the per-control hook points and exactly
what a driver/firmware patch would need to touch.

---

## Sources

- Linux `nl80211.h` (in-tree, this host): `NL80211_CMD_SET_WIPHY`,
  `NL80211_ATTR_WIPHY_ANTENNA_TX` / `_RX`, `NL80211_CMD_VENDOR`,
  `NL80211_CMD_FRAME` / `NL80211_CMD_REGISTER_FRAME`.
- ath10k configuration (beamforming only via hostapd `vht_capab`, no debugfs
  sounding control): <https://wireless.docs.kernel.org/en/latest/en/users/drivers/ath10k/configuration.html>
- hostapd beamforming/sounding knobs (`[SU-BEAMFORMER]`, `[MU-BEAMFORMER]`,
  `[SOUNDING-DIMENSION-4]`, `he_su_beamformer`):
  <https://w1.fi/cgit/hostap/tree/hostapd/hostapd.conf> and
  <https://github.com/morrownr/USB-WiFi/blob/main/home/AP_Mode/hostapd-WiFi6.conf>
- mt76 beamforming lives in firmware (mt7622/mt7615 performance/beamforming
  discussion): <https://github.com/openwrt/mt76/issues/863>
- Qualcomm firmware closedness (ath11k/ath12k no open firmware; regulatory +
  features firmware-enforced): ath10k mailing-list thread
  <https://ath10k.infradead.narkive.com/6bdEJZih/qca99xx-with-mu-mimo-and-beamforming>
  and CodeLinaro ath firmware <https://git.codelinaro.org/clo/ath-firmware/ath11k-firmware>
- ath11k reports VHT beamformee spatial streams *from firmware*:
  <https://lkml.iu.edu/2210.2/09619.html>
