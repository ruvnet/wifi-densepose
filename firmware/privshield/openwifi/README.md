# WiFi Veil protector — openwifi (Xilinx Zynq + AD9361, open PHY/MAC)

> **STATUS: SYNTHETIC / L0 — build-only scaffold. No hardware, no flash, no
> capture. Nothing here has run on silicon.** Per CLAUDE.md, none of this is a
> `MEASURED` result and none may be claimed as working. Files are honest
> skeletons with real openwifi idioms plus `TODO(hw)` / `TODO(hdl)` markers, not
> validated firmware or complete HDL. **Compliant waveform controls only — the
> keyed rotation is orthogonal (energy-preserving) and shapes only this node's
> own standards-conformant emission. Never jamming.**

## Feasibility grade: **B (capability ceiling A; effort D)**

openwifi is the **only** platform in this tree where a true end-to-end keyed
rotation *and its inverse* are physically reachable, because it is the only one
that exposes the full open PHY/MAC on FPGA: `openofdm_tx`/`openofdm_rx`,
`tx_intf`/`rx_intf`, and `side_ch`, all AXI-Lite-programmable from a Linux
driver ([FPGA module design][fmd], [openwifi overview][ov]). That is the **A**
capability ceiling.

It is graded **B**, not A, for two honest reasons that make it the
highest-*effort* path:

1. **openwifi has no native explicit transmit beamforming.** It ships as an
   802.11a/g/n **single-spatial-stream (SISO)** design. It does not run NDP
   sounding, does not compute an SVD `V` matrix, and does not emit a compressed
   beamforming report. The two-antenna app note is **RX-only** coherent capture
   (`side_ch_ctl wh3h11`), not a MIMO transmit spatial mapper ([iq_2ant][2ant]).
   So there is no shipped compressed-BF-report to obfuscate and no shipped
   spatial-mapping matrix `Q` to left-multiply — both must be **added in HDL**.
2. Reaching a true two-stream demo needs a **second TX chain** (the AD9361 on
   fmcomms2/3 has two DACs) plus a new spatial-mapping RTL stage and a Vivado
   rebuild — days-to-weeks of FPGA work, not a driver patch.

Because of (1), on openwifi WiFi Veil is realized as the **client-transparent
per-packet keyed unitary** (LeakyBeam family) applied at the TX spatial-mapping
stage, with the legitimate STA (a second openwifi node sharing the key)
inverting it — **not** as obfuscation of a compressed-BF report the hardware
never produces. This keeps the claim honest: we rotate the *transmitted spatial
mapping* so a sniffer's per-subcarrier channel estimate `H·Q(key)` is scrambled,
and the keyed receiver applies `Q(key)^H` before channel estimation.

## Exact insertion points

The rotation is a keyed orthogonal (unitary) matrix `Q(key, session)` computed
by the portable core (`../core/veil_shield.{h,c}`), the same SplitMix64 schedule
used everywhere, so both ends derive the identical `Q` from the shared key.

**TX (protector) — FPGA, new block `veil_rot`:**
Insert on the baseband IQ AXI-Stream path **between `openofdm_tx` (post-IFFT,
post-CP) and `tx_intf`** (which feeds the AD9361 DAC). `veil_rot` left-multiplies
the per-subcarrier / per-stream sample vector by `Q(key)`. Its coefficients (or a
key seed + on-FPGA schedule) are written over **AXI-Lite** from the driver shim
using the standard openwifi `iowrite32(value, base_addr + reg)` idiom
([tx_intf driver][txintf]). See `HDL_NOTES.md`.

**RX (legitimate STA) — FPGA, new block `veil_unrot`:**
Insert **between `rx_intf` (AD9361 ADC) and `openofdm_rx`**, or in the frequency
domain immediately after the FFT and **before channel estimation**, applying
`Q(key)^H`. Same AXI-Lite programming path.

**Driver / control plane:** the C shim `veil_openwifi.c` computes the session
key schedule via the core and programs the blocks. Real openwifi control idioms:
AXI-Lite MMIO from the kernel driver, and the `sdrctl` nl80211-testmode tool /
`side_ch_ctl` register pokes for bring-up ([sdrctl/side_ch][ov], [frequent
tricks][ft]). Where the exact offsets/bitfields are not yet fixed, the shim
marks `TODO(hw)`; RTL specifics are `TODO(hdl)`.

Doing the rotation in HDL (not the DMA'd payload) is deliberate: it keeps the
frame **standards-conformant on the wire** and preserves transmit energy — the
"not jamming" invariant the core guarantees by construction (orthogonal `Q`).

## Two-node measurement plan (the P5 path)

Three roles produce the first `MEASURED` / P5 result (full protocol +
required witness log in `MEASUREMENT.md`):

- **Protector AP** — openwifi node A, `veil_rot` engaged, TX spatial mapping
  keyed with the session key.
- **Legitimate STA** — openwifi node B, shares the key, `veil_unrot` engaged;
  should see **near-baseline throughput** (rotation cancels).
- **Attacker sniffer** — a commodity Wi-Fi NIC running **Wi-BFI** / monitor
  capture, extracting the per-subcarrier CSI / beamforming feedback and running
  the re-ID model ([Wi-BFI][wibfi]).

Headline metric: **re-identification accuracy off vs. on** at the attacker
(target: collapse toward chance) **while** iperf throughput A↔B stays near
baseline and per-frame energy is unchanged. No number here is real until a
captured on-silicon log exists.

## Bill of materials (target, not procured)

- 2× Xilinx Zynq-7000 board with AD9361 FMC (e.g. ZC706 + fmcomms2/3, or
  ADRV9361-Z7035 / Antenna-SDR), openwifi image per the openwifi build docs.
- 1× attacker host + Wi-BFI-capable NIC (per Wi-BFI's supported list).
- Vivado for the FPGA rebuild that adds `veil_rot` / `veil_unrot`.

## Files here

| File | What it is |
|---|---|
| `README.md` | this — feasibility, insertion points, measurement plan |
| `veil_openwifi.c` | driver-side C shim: core → session `Q` → AXI-Lite program (scaffold, `TODO(hw)`) |
| `HDL_NOTES.md` | the `veil_rot` / `veil_unrot` Verilog blocks (design notes, `TODO(hdl)`) |
| `MEASUREMENT.md` | exact P5 protocol, metrics, and the required witness artifact |

## Sources

- FPGA module design — [deepwiki.com/open-sdr/openwifi/2.2-fpga-module-design][fmd]
- openwifi overview (sdrctl, side_ch, nl80211 testmode) — [deepwiki.com/open-sdr/openwifi/1-openwifi-overview][ov]
- Two-antenna IQ (RX-only) app note — [github.com/open-sdr/openwifi .../iq_2ant.md][2ant]
- tx_intf driver register idioms (`iowrite32`/`ioread32`) — [github.com/open-sdr/openwifi .../tx_intf.c][txintf]
- Frequent tricks / register pokes — [github.com/open-sdr/openwifi .../frequent_trick.md][ft]
- openwifi paper (SDR 802.11 on SoC) — [researchgate .../342582824][paper]
- Wi-BFI (attacker BF-feedback extraction) — [arxiv.org/pdf/2309.04408][wibfi]

[fmd]: https://deepwiki.com/open-sdr/openwifi/2.2-fpga-module-design
[ov]: https://deepwiki.com/open-sdr/openwifi/1-openwifi-overview
[2ant]: https://github.com/open-sdr/openwifi/blob/master/doc/app_notes/iq_2ant.md
[txintf]: https://github.com/open-sdr/openwifi/blob/master/driver/tx_intf/tx_intf.c
[ft]: https://github.com/open-sdr/openwifi/blob/master/doc/app_notes/frequent_trick.md
[paper]: https://www.researchgate.net/publication/342582824_openwifi_a_free_and_open-source_IEEE80211_SDR_implementation_on_SoC
[wibfi]: https://arxiv.org/pdf/2309.04408
