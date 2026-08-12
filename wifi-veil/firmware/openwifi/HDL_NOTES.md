# HDL notes — `veil_rot` (TX) / `veil_unrot` (RX)

> **STATUS: SYNTHETIC / L0 — design notes only. No RTL is shipped here, none has
> been synthesized, placed, routed, or run on an FPGA.** This describes the
> Verilog blocks that *would* apply the keyed unitary in the openwifi datapath.
> Every concrete number (offsets, latency, resource use) is `TODO(hdl)` until a
> real build exists. **Orthogonal transform ⇒ transmit energy preserved:
> compliant, never jamming.**

## Where the blocks sit

openwifi's baseband IQ moves as **AXI-Stream** between blocks and its control is
**AXI-Lite** ([FPGA module design][fmd]). The two new blocks are AXI-Stream
pass-through filters with an AXI-Lite slave for the key schedule.

```
TX (protector):
  openofdm_tx  ──AXI-S(IQ)──►  [ veil_rot ]  ──AXI-S(IQ)──►  tx_intf ──► AD9361 DAC
                                    ▲ AXI-Lite (key, coeff RAM)
                                    └── veil_openwifi.c

RX (legitimate STA, shares key):
  AD9361 ADC ──► rx_intf ──AXI-S──► [ veil_unrot ] ──AXI-S──► openofdm_rx (FFT → chan est)
                                        ▲ AXI-Lite
                                        └── veil_openwifi.c
```

`veil_unrot` may equivalently sit **in the frequency domain**, right after the
FFT and **before channel estimation**, if a per-subcarrier `Q^H` is cheaper to
apply there. Same AXI-Lite contract either way.

## Why a *new* block is required (honesty)

openwifi is **SISO 802.11a/g/n** and has **no explicit-beamforming / spatial-
mapping stage** and **no compressed-BF-report generation** — the two-antenna app
note is RX-only capture, not a TX spatial mapper ([iq_2ant][2ant]). So there is
no existing `Q` matrix to modify; `veil_rot`/`veil_unrot` **introduce** the
spatial-mapping stage. Two realizable RTL scopes:

- **Scope A — 1×1 per-subcarrier phase/rotation (lower effort).** Treat the
  rotation as operating over a **synthetic vector** formed from the fine
  subspace of the per-packet subcarrier response (a stream of `N` IQ elements
  the block buffers), applying the core's Givens schedule across those elements.
  Single TX chain; no board change. This is enough to *scramble the CSI a
  sniffer estimates* and to demonstrate keyed invert at RX. It is **not** true
  spatial MIMO.
- **Scope B — 2×2 true spatial mapping (higher effort, the A-capability demo).**
  Enable the **second TX chain** (AD9361 has 2 DACs on fmcomms2/3) and apply a
  keyed 2×2 unitary across the two streams — a genuine transmit spatial mapping
  the standard marks "not restricted." Needs a Vivado top-level rebuild wiring
  the 2nd DAC and the extra AXI-S lane. `TODO(hdl)`.

## `veil_rot` datapath

The core applies `passes` **Givens rotations** `G(i,j,θ)` composed into `Q`
(`../core/veil_shield.c`). In hardware we apply the *same schedule* to the on-air
sample vector, so both ends derive identical coefficients from the shared key —
no matrix is transmitted.

Per Givens op on elements `(i, j)` with programmed `(cos, sin)` in Q1.15:
```
  v_i' =  cos*v_i - sin*v_j
  v_j' =  sin*v_i + cos*v_j        // complex IQ: apply to I and Q lanes
```
- Coefficients arrive from `veil_openwifi.c` as the packed `(i, j, cos, sin)`
  schedule (2 AXI-Lite words per pass; packing defined in `veil_openwifi.c`).
- `veil_unrot` applies the schedule **in reverse with negated sin** (`sin → -sin`,
  i.e. `Gᵀ`), matching `veil_shield_recover`. A `CTRL.inverse` bit selects it.
- Fixed point: openwifi baseband IQ is 16-bit I / 16-bit Q; coeffs are signed
  Q1.15. `TODO(hdl)`: guard-bit / rounding so the composed rotation stays
  norm-preserving to spec and never clips (clipping would break the
  energy-preservation invariant — must be verified, not assumed).

## AXI-Lite register map (must match `veil_openwifi.c`)

| Offset | Name | Meaning |
|---|---|---|
| `0x00` | `CTRL` | bit0 enable, bit1 inverse (`veil_unrot`), bit2 load |
| `0x04` | `KEY_LO` | session key [31:0] |
| `0x08` | `KEY_HI` | session key [63:32] |
| `0x0C` | `NDIM` | on-air fine-block dimension `N` (≤ 64) |
| `0x10` | `PASSES` | number of Givens passes (default 96) |
| `0x14` | `COEFF_ADDR` | write index into coeff RAM |
| `0x18` | `COEFF_DATA` | packed `{j,i}` then `{sin,cos}` (2 words/pass) |
| `0x1C` | `STATUS` | bit0 ready, bit1 applied, bit2 err |

`TODO(hdl)`: regenerate this from the block's `*_s_axi.v` once written (cf.
`openofdm_tx`'s 6 AXI-Lite registers at `ip/openofdm_tx/src/openofdm_tx_s_axi.v`)
and reconcile any offset changes back into `veil_openwifi.c`.

## Timing / integration risks (call them out, don't hide them)

- **802.11 SIFS budget.** The block adds pipeline latency between IFFT and DAC;
  it must not violate the tight TX timing openwifi maintains in `tx_intf`.
  `TODO(hdl)`: measure added cycles; keep within budget or absorb in existing
  FIFO slack.
- **On-FPGA schedule vs. per-packet coeff load.** For per-*packet* keying, either
  compute the SplitMix64 schedule on-FPGA from `(key, packet_counter)` or
  double-buffer the coeff RAM. `TODO(hdl)`.
- **Bit-exactness with the core.** The on-FPGA (or shim-fed) `(cos,sin)` must
  reproduce the core's schedule so `veil_unrot` inverts exactly. First gate is a
  **self-loopback** IQ test (`veil_rot → veil_unrot`, assert recovered == input
  within Q1.15 round-off) using openwifi's existing packet/IQ self-loopback
  facility ([self-loopback app note][loop]). Passing loopback is a correctness
  gate, **not** a defense `MEASURED` claim.

## Build

`TODO(hdl)`: add `veil_rot`/`veil_unrot` as `openwifi-hw` IP, instantiate in the
board block design, and rebuild the bitstream with Vivado per the openwifi-hw
build flow ([openwifi-hw][hw]). No bitstream is produced from this directory.

## Sources

- FPGA module design (AXI-S / AXI-Lite, block roles) — [deepwiki][fmd]
- openwifi-hw (FPGA IP + build flow) — [github.com/open-sdr/openwifi-hw][hw]
- Two-antenna IQ (RX-only; confirms no TX spatial mapper ships) — [iq_2ant][2ant]
- Packet/IQ self-loopback test — [self-loopback app note][loop]

[fmd]: https://deepwiki.com/open-sdr/openwifi/2.2-fpga-module-design
[hw]: https://github.com/open-sdr/openwifi-hw
[2ant]: https://github.com/open-sdr/openwifi/blob/master/doc/app_notes/iq_2ant.md
[loop]: https://github.com/open-sdr/openwifi/blob/master/doc/app_notes/packet-iq-self-loopback-test.md
