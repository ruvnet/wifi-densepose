# Building the WiFi Veil Nexmon patch — **UNTESTED**

> **This procedure has never been run.** It has not been built with the Nexmon
> toolchain, not flashed, and not captured on air. Addresses/symbols in
> `patch/veil_patch.c` are placeholders (one is intentionally invalid,
> `0xDEAD0000`) so it will **not** produce a flashable image as-is. This file
> documents *how it would build* so a hardware operator with real silicon can
> take it forward. `SYNTHETIC / L0`, per CLAUDE.md.

## Prerequisites (host, not in this repo)

- A Linux host (Nexmon expects an x86_64 Ubuntu-like build host) with the
  Broadcom-flavored ARM toolchain Nexmon downloads/uses, plus `git`, `make`,
  `gcc-arm-none-eabi`, `flex`, `bison`, `libisl`, `automake`.
- Nexmon checked out **outside** this repo (do not vendor it here):
  ```bash
  git clone https://github.com/seemoo-lab/nexmon.git
  cd nexmon
  source setup_env.sh        # sets NEXMON_ROOT, toolchain paths
  make                       # builds libISL / firmwares tooling
  ```
- The target firmware blob present on the device: BCM43455c0
  (`brcmfmac43455-sdio.bin`), version **7_45_189** (Cypress) or 7_45_154
  (Raspbian). Do **not** commit the blob or any extracted symbols/ROM to RuView.

## Where this patch would live in the Nexmon tree

Nexmon builds per chip/firmware under `patches/<chip>/<fwver>/<name>/`. This
adapter would be a Nexmon project, e.g.:

```
$NEXMON_ROOT/patches/bcm43455c0/7_45_189/veil/
├── Makefile            # copy of an existing nexmon patch Makefile (e.g. nexmon_csi's)
├── src/
│   ├── veil_patch.c    # <- symlink/copy of firmware/privshield/nexmon/patch/veil_patch.c
│   ├── veil_shield.c   # <- from firmware/privshield/core/  (compiled into the patch)
│   └── veil_shield.h   # <- from firmware/privshield/core/
└── ...
```

Keep the RuView copies canonical; the Nexmon tree gets copies/symlinks so the
core stays byte-identical to `../core/`.

## Linking the portable core (MCU-friendly)

The core is `no_std`-style C99: no malloc, no libc I/O, only `<math.h>`
(`sinf`/`cosf`/`sqrtf`/`sqrt`). To build it into the patch:

1. Add `veil_shield.c` to the patch `Makefile`'s object list (alongside
   `patch.o`/`wrapper.o`), so it compiles with the same ARM flags.
2. Ensure the firmware provides `sinf`/`cosf`/`sqrtf`. **TODO(hw):** Broadcom
   firmware may not export libm. Options, in order of preference:
   - link a small `libm`/`compiler-rt` for `arm-none-eabi`;
   - or replace the trig with a fixed-point / CORDIC Givens rotation
     (`TODO(reverse-engineer)`), which also avoids float on parts without an FPU.
3. All WiFi Veil working storage is stack-bounded (`VEIL_MAX_FINE`, `CACHE` in the
   core) — no heap is introduced on-chip.

## Build

```bash
cd $NEXMON_ROOT/patches/bcm43455c0/7_45_189/veil
make            # produces the patched brcmfmac43455-sdio.bin
```

Before `make` can succeed you must first resolve every `TODO(reverse-engineer)`
in `veil_patch.c`:

- replace `0xDEAD0000` and the `wlc_sendmgmt_veil_target` symbol with the real,
  disassembled target address/symbol for 7_45_189;
- implement `veil_bfr_unpack_fine` / `veil_bfr_pack_fine` (the angle bit-field
  codec) and the report-body offset/length;
- confirm the compressed-beamforming report is assembled in ARM on this chip
  (else move to hook candidate #2/#3 — see README).

## Flash (Raspberry Pi, on-device)

**TODO(hw) — untested.** Typical Nexmon flow on the Pi:

```bash
# back up stock firmware first!
sudo cp /lib/firmware/brcm/brcmfmac43455-sdio.bin ~/brcmfmac43455-sdio.bin.orig

sudo cp brcmfmac43455-sdio.bin /lib/firmware/brcm/brcmfmac43455-sdio.bin
# (some setups also need the matching *.clm_blob / nexmon's own copy path)

sudo rmmod brcmfmac && sudo modprobe brcmfmac   # reload driver with new firmware
dmesg | tail                                    # confirm firmware loaded
```

Push the session key at runtime (matches the IOCTL stub in `veil_patch.c`):

```bash
# TODO(hw): nexutil vendor-IOCTL id and payload format are placeholders
nexutil -s<VEIL_IOCTL_SET_KEY> -b -l8 -v<base64-8-byte-key>
```

**Recovery:** if WiFi breaks, restore the backup blob and reload the driver.
A bad flashpatch offset can knock out WiFi until you reflash stock firmware.

## Validation you can honestly do (still not `MEASURED` firmware)

1. **Host unit test of the math** (already green in this repo):
   `cd ../../core && make test`.
2. **Read-back on hardware** with `nexmon_csi`/Wi-BFI: capture the report with
   and without the patch and check the fine subspace changed while SNR/norm is
   preserved. This validates the transform end-to-end but is a *receiver*
   observation, not proof the TX hook is robust.
3. Only a captured device runtime log showing the shaped report leaving *this*
   node, plus receiver-side recovery with the shared key, would move any claim
   from `SYNTHETIC`/`CLAIMED` toward `MEASURED` (roadmap P5).

## References

See `README.md` for sources (Nexmon, nexmon_csi, Wi-BFI, D11 reverse
engineering).
