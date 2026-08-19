# WiFi Veil ↔ `mac80211` / driver integration map

> **`SYNTHETIC / L0` — BUILD-ONLY, UNTESTED ON HARDWARE.** These are hook-point
> designs derived from public API/source, not validated on silicon. Function and
> attribute names are real (verified against in-tree `linux/nl80211.h` and public
> hostapd/driver docs); where a hook does **not** exist upstream it is marked
> `TODO(hw)` with what a patch would have to add. Compliant controls only.

Legend: **US** = userspace-reachable today · **DP** = needs driver patch ·
**FW** = needs firmware patch (blob-blocked).

---

## 1. TX antenna-map perturbation — **US** (feasible)

- **Daemon:** `veil_set_tx_antenna_mask()` in `veil_shieldd.c`.
- **Kernel path:** `nl80211` → `cfg80211_ops.set_antenna()` → driver
  `.set_antenna` (e.g. `mt7915_set_antenna`, `ath9k` `set_antenna`).
- **Attributes:** `NL80211_CMD_SET_WIPHY`, `NL80211_ATTR_WIPHY_ANTENNA_TX`,
  `NL80211_ATTR_WIPHY_ANTENNA_RX`.
- **Constraints:** many drivers require the phy DOWN and accept only symmetric
  masks; validate per driver. Coarse static spatial-mapping change, not the keyed
  rotation. Fully standards-compliant.

## 2. NDP sounding-cadence jitter — **US (indirect)**

- **Daemon:** `veil_randomize_sounding_cadence()` / `veil_next_cadence_ms()`.
  The schedule is derived from the session key via the core SplitMix64 so the
  paired receiver can anticipate it (not random spraying).
- **Real lever:** hostapd `ctrl_iface` (UNIX socket `/var/run/hostapd/<iface>`):
  `SET he_su_beamformer …` / rewrite `vht_capab` `[SOUNDING-DIMENSION-n]` /
  toggle `[SU-BEAMFORMER]`, then `RECONFIGURE`. Config keys documented in
  `hostapd.conf`.
- **`TODO(hw)`:** there is **no** `nl80211` "set sounding interval" command; the
  per-NDP timer is in driver/firmware. We can only jitter the *offered* cadence.
  The `ctrl_iface` write itself is not yet wired (function currently only
  computes `ms`).

## 3. MU-MIMO group shuffling — **FW** (blob-blocked)

- **Daemon:** `veil_shuffle_mumimo_groups()` — explicit `-ENOTSUP` no-op.
- **Where it lives:** MU group formation + per-group steering matrices are
  computed in the WiFi MCU firmware on mt76 (mt7915) and all ath1x parts.
- **`TODO(hw)`:** would require `NL80211_CMD_VENDOR` with a driver-specific
  `NL80211_ATTR_VENDOR_ID` / `NL80211_ATTR_VENDOR_SUBCMD` /
  `NL80211_ATTR_VENDOR_DATA` that upstream mt76/ath do **not** define, plus a
  firmware change to honor an externally supplied grouping. Not reachable without
  both a driver and firmware patch.

## 4. Per-packet keyed unitary (the core WiFi Veil transform) — **FW** (blob-blocked)

- **Daemon:** `veil_apply_keyed_rotation()` → `veil_shield_apply(fine, n, key,
  passes)` from the portable core. Orthogonal / energy-preserving (the
  "not jamming" invariant, checked via `veil_l2_norm` before/after).
- **What a full path must touch:**
  - **mt76 (mt7915):** the MCU firmware stage that builds the compressed
    beamforming report (φ/ψ angles) or applies the steering/precoder Q to the
    LTF spatial mapping. A firmware patch would call the rotation on the fine
    subspace *before* the report is emitted / precoder applied. The driver
    (`mt7915/mcu.c`) would ferry the key/passes down via a new MCU command.
  - **ath9k (DP, best open case):** the static spatial-mapping matrix is set via
    `AR_PHY_*` registers in the open PHY init; a driver patch could apply a keyed
    *static* Q there. This is coarser than a true per-packet report edit but is
    the most credible OpenWRT-adjacent route (older 802.11n hardware only).
  - **ath10k/ath11k/ath12k:** report generation + precoder are entirely
    firmware-side with no open firmware (ath11k/ath12k) — not patchable.
- **`TODO(hw)`:** on OpenWRT there is **no** userspace/`mac80211` hook that hands
  the pre-precoder V/steering buffer to the daemon before TX. Reaching it needs
  the driver+firmware patch above, or use the **openwifi (FPGA)** / **Nexmon
  (Broadcom)** adapters, which expose the datapath. The daemon only proves the
  math is invariant; nothing goes on air.

## 5. Sensing-solicitation (NDPA) detection — **US/DP** (partial)

- **Daemon:** `veil_event_cb()` on `NL80211_CMD_FRAME`.
- **Real path:** `NL80211_CMD_REGISTER_FRAME` to subscribe to specific
  management action categories, delivered as `NL80211_CMD_FRAME` with
  `NL80211_ATTR_FRAME`. Classify VHT/HE compressed beamforming action
  (categories 21 / 30) and NDP Announcement to measure cadence.
- **`TODO(hw)`:** commodity drivers do **not** forward raw NDPA to userspace by
  default; honest external-solicitation detection needs monitor-mode capture or a
  driver notification that is not guaranteed upstream. Frame parsing is stubbed.

---

## Summary of the effort boundary

| Control | Effort to reach full WiFi Veil fidelity |
|---|---|
| TX antenna map | Ready now (US), coarse only |
| Sounding cadence jitter | Wire hostapd `ctrl_iface` (US), coarse only |
| Static spatial Q | ath9k driver patch (DP) |
| MU grouping | driver vendor subcmd + firmware (FW) |
| Per-packet keyed rotation | mt76/ath **firmware** patch, or openwifi/Nexmon adapter (FW) |
| NDPA detection | frame registration + likely driver patch (US/DP) |
