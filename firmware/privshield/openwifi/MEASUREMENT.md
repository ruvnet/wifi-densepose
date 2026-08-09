# P5 measurement protocol — openwifi WiFi Veil end-to-end

> **STATUS: SYNTHETIC / L0 — this is a PLAN, not a result. No hardware has been
> run; no capture, log, or number in this repo is real.** This document defines
> exactly what must be executed and captured to earn the first `MEASURED` claim
> under CLAUDE.md's hardware-evidence rule. Until the witness artifact below
> exists, every accuracy/throughput/energy statement about openwifi WiFi Veil is
> `SYNTHETIC` and must be labelled so. **Compliant waveform controls only —
> orthogonal, energy-preserving; never jamming.**

## Roadmap position

This is roadmap **P5**: the two-node hardware measurement that turns the P4
build scaffolds into a `MEASURED` defense result. Prerequisite gates (all on real
silicon, all currently unmet): a bitstream with `veil_rot`/`veil_unrot`
(`HDL_NOTES.md`), a driver loading `veil_openwifi.c`, and a passing on-FPGA
**self-loopback** correctness test.

## Topology

```
   [ Protector AP ]              over the air              [ Legitimate STA ]
   openwifi node A   ───────────────────────────────────► openwifi node B
   veil_rot: Q(key) engaged      │                        veil_unrot: Q^H(key)
                                 │                        (shares key with A)
                                 ▼
                        [ Attacker sniffer ]
                        commodity NIC, monitor mode
                        Wi-BFI CSI/BF-feedback extraction
                        + re-ID model
```

The attacker is **passive** (monitor capture only). Nothing in this test
transmits to interfere with any station.

## Hardware list

| Role | Hardware | Software |
|---|---|---|
| Protector AP (A) | Zynq-7000 + AD9361 FMC (ZC706+fmcomms2/3, or ADRV9361-Z7035) | openwifi image + `veil_rot` bitstream + `veil_openwifi.c` |
| Legitimate STA (B) | second identical openwifi node | openwifi image + `veil_unrot` bitstream + `veil_openwifi.c`, same key as A |
| Attacker | host + Wi-BFI-supported Wi-Fi NIC in monitor mode | Wi-BFI ([arxiv 2309.04408][wibfi]) + re-ID model |
| Bench | shielded room or wired attenuator path preferred | `iperf3`, power meter / board rail sense |

Key agreement A↔B is out-of-band for the demo (pre-shared session key);
per-packet keying uses `(key, packet_counter)` as in `HDL_NOTES.md`.

## Procedure

Run every condition **twice**: WiFi Veil **OFF** (baseline) and **ON**. Same
positions, same MCS, same duration, same seed for the attacker model.

1. **Correctness precondition (not a defense claim).** Confirm on-FPGA
   self-loopback recovers IQ within Q1.15 round-off, and A→B link works with
   `veil_unrot` engaged. Capture the console log.
2. **Attacker capture.** Sniffer records CSI / beamforming-feedback for a fixed
   traffic pattern A→B, OFF then ON. Save raw captures (pcap + Wi-BFI output).
3. **Re-ID metric.** Run the same re-identification / fingerprinting model on the
   OFF and ON captures. Report accuracy and confusion vs. the **chance / mean
   baseline** (per CLAUDE.md, a defense claim needs the baseline and a
   leakage-free held-out split — never report bare accuracy).
4. **Throughput (near-free check).** `iperf3` A↔B, OFF vs. ON, both directions.
   Expected: ON ≈ OFF (the receiver inverts the rotation). Save `iperf3 --json`.
5. **Energy / compliance.** Record per-frame TX energy OFF vs. ON (rail sense or
   power meter) to substantiate the "energy-preserving / not jamming" claim, and
   spectrum/mask conformance if a spectrum analyzer is available.

## Metrics reported

| Metric | OFF | ON | Requirement for a pass |
|---|---|---|---|
| Attacker re-ID accuracy vs. chance | baseline | — | collapses toward chance ON |
| iperf3 throughput A↔B | baseline | — | ON within a few % of OFF |
| Per-frame TX energy | baseline | — | ON ≈ OFF (orthogonality holds on-air) |
| Spectral mask conformance | pass | — | still conformant ON |

## Required witness artifact (CLAUDE.md gate)

Before **any** `MEASURED` claim, this directory (or the P5 evidence path) must
contain a **captured real-silicon log**, not a build or simulator output:

- Boot/runtime console log of both openwifi nodes showing the `veil_rot` /
  `veil_unrot` bitstream loaded and `veil_openwifi.c` programming the session
  (register writes / STATUS ready), with timestamps and board identifiers.
- The self-loopback correctness log (step 1).
- Raw attacker captures (pcap + Wi-BFI output) for OFF and ON, plus the exact
  re-ID reproducer command and its output.
- `iperf3 --json` for OFF and ON; energy trace for OFF and ON.
- A manifest tying each artifact to the git commit of the RTL, driver, and shim
  used, so the result is reproducible.

Label the result `MEASURED` **only** with all of the above captured from real
hardware. A successful Vivado build, a Verilator/QEMU run, or the host
`veil_openwifi.c` self-test is **not** hardware evidence and must stay
`SYNTHETIC`. No log in this repo today — do not fabricate one.

## Sources

- Wi-BFI (attacker BF-feedback extraction) — [arxiv.org/pdf/2309.04408][wibfi]
- Packet/IQ self-loopback test — [openwifi self-loopback app note][loop]

[wibfi]: https://arxiv.org/pdf/2309.04408
[loop]: https://github.com/open-sdr/openwifi/blob/master/doc/app_notes/packet-iq-self-loopback-test.md
