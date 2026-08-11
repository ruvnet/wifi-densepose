/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * veil_openwifi — driver-side shim that binds the portable VEIL core
 * (../core/veil_shield.{h,c}) to the openwifi FPGA TX/RX datapath.
 *
 * STATUS: SYNTHETIC / L0. Build-only scaffold. Never compiled into the openwifi
 * kernel module on real silicon, never flashed, never captured. Do NOT claim
 * runtime behavior without a captured hardware log (CLAUDE.md hardware-evidence
 * rule). Register offsets, bitfields, and the FPGA blocks it programs
 * (veil_rot / veil_unrot) do NOT exist in upstream openwifi yet — every place
 * that depends on real hardware is marked TODO(hw); RTL specifics live in
 * HDL_NOTES.md and are marked TODO(hdl) there.
 *
 * ROLE (honest): this shim runs on the protector AP and on the legitimate STA.
 *   - Protector: derive the per-session keyed unitary Q(key) from the core and
 *     program the veil_rot block that left-multiplies the transmit spatial
 *     mapping (inserted between openofdm_tx and tx_intf; see HDL_NOTES.md).
 *   - Legitimate STA: derive the same Q(key) and program veil_unrot to apply
 *     Q^H before channel estimation, cancelling the rotation (near-free tput).
 * The transform is orthogonal, so transmit energy is preserved: compliant,
 * NOT jamming. openwifi ships SISO with no explicit beamforming, so this is the
 * client-transparent per-packet unitary route, not obfuscation of a compressed
 * beamforming report (openwifi never generates one) — see README.md.
 *
 * openwifi idioms used where known:
 *   - AXI-Lite MMIO from the driver: iowrite32(value, base + reg) /
 *     ioread32(base + reg), matching driver/tx_intf/tx_intf.c reg_write/reg_read.
 *   - Coefficients are quantized to the fixed-point width the datapath uses
 *     (openwifi baseband IQ is 16-bit I / 16-bit Q); see VEIL_ROT_FRAC below.
 *
 * This file is written to compile in two modes:
 *   - Host/CI (default): __KERNEL__ undefined -> MMIO is stubbed to a local
 *     shadow buffer so the key-schedule + quantization logic is unit-testable
 *     with no hardware. This is the ONLY path exercised today.
 *   - In-tree kernel build: define VEIL_OPENWIFI_KERNEL to pull the real
 *     linux/io.h accessors. Untested. TODO(hw).
 */

#include "../core/veil_shield.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * MMIO layer. Real openwifi drivers keep a per-block __iomem base and use
 * iowrite32/ioread32. We isolate that here so host/CI builds need no kernel.
 * ------------------------------------------------------------------------- */
#if defined(VEIL_OPENWIFI_KERNEL)
#include <linux/io.h>
typedef void __iomem *veil_mmio_base;
static inline void veil_reg_write(veil_mmio_base b, uint32_t reg, uint32_t v) {
    iowrite32(v, (uint8_t __iomem *)b + reg);
}
static inline uint32_t veil_reg_read(veil_mmio_base b, uint32_t reg) {
    return ioread32((uint8_t __iomem *)b + reg);
}
#else
/* Host/CI shadow: a small register file so logic is testable with no radio. */
#define VEIL_SHADOW_REGS 256
typedef struct {
    uint32_t regs[VEIL_SHADOW_REGS];
} veil_mmio_shadow;
typedef veil_mmio_shadow *veil_mmio_base;
static inline void veil_reg_write(veil_mmio_base b, uint32_t reg, uint32_t v) {
    if (b && (reg >> 2) < VEIL_SHADOW_REGS) {
        b->regs[reg >> 2] = v;
    }
}
static inline uint32_t veil_reg_read(veil_mmio_base b, uint32_t reg) {
    if (b && (reg >> 2) < VEIL_SHADOW_REGS) {
        return b->regs[reg >> 2];
    }
    return 0;
}
#endif

/* -------------------------------------------------------------------------
 * Register map for the (not-yet-existing) veil_rot / veil_unrot AXI-Lite
 * slaves. Offsets are PLACEHOLDERS chosen to be word-aligned; the real map is
 * fixed when the RTL lands. TODO(hw): confirm against the generated
 * *_s_axi.v once veil_rot exists (cf. openofdm_tx's 6 AXI-Lite regs).
 * ------------------------------------------------------------------------- */
#define VEIL_ROT_REG_CTRL        0x00u  /* bit0 enable, bit1 inverse, bit2 load */
#define VEIL_ROT_REG_KEY_LO      0x04u  /* session key [31:0]  */
#define VEIL_ROT_REG_KEY_HI      0x08u  /* session key [63:32] */
#define VEIL_ROT_REG_NDIM        0x0Cu  /* fine-block dimension N applied on-air */
#define VEIL_ROT_REG_PASSES      0x10u  /* number of Givens passes */
#define VEIL_ROT_REG_COEFF_ADDR  0x14u  /* write index into the coeff RAM */
#define VEIL_ROT_REG_COEFF_DATA  0x18u  /* {Q16.15 sin, Q16.15 cos} packed */
#define VEIL_ROT_REG_STATUS      0x1Cu  /* bit0 ready, bit1 applied, bit2 err */

#define VEIL_ROT_CTRL_ENABLE     (1u << 0)
#define VEIL_ROT_CTRL_INVERSE    (1u << 1)
#define VEIL_ROT_CTRL_LOAD       (1u << 2)

#define VEIL_ROT_STATUS_READY    (1u << 0)

/* Fixed-point: openwifi baseband IQ is 16-bit. We program rotation coeffs as
 * signed Q1.15 (fractional bits = 15). cos/sin in [-1,1] map cleanly. */
#define VEIL_ROT_FRAC 15

/* Default schedule parameters — kept byte-consistent with the core/Rust crate
 * defaults. N is the on-air fine-block dimension the datapath vectorizes over;
 * for the SISO-plus-synthetic-stream demo this is small (see HDL_NOTES.md). */
#define VEIL_OW_DEFAULT_PASSES 96u
#define VEIL_OW_MAX_NDIM       64u   /* bounded coeff RAM; keeps it malloc-free */

typedef enum {
    VEIL_OW_ROLE_PROTECTOR = 0, /* TX veil_rot, forward rotation Q      */
    VEIL_OW_ROLE_LEGIT_RX  = 1, /* RX veil_unrot, inverse rotation Q^H  */
} veil_ow_role;

typedef struct {
    veil_mmio_base base;   /* AXI-Lite base of veil_rot / veil_unrot slave */
    uint64_t       key;    /* shared session key (both ends must match)    */
    uint32_t       ndim;   /* fine-block dimension, <= VEIL_OW_MAX_NDIM    */
    uint32_t       passes; /* Givens passes                                */
    veil_ow_role   role;
} veil_ow_ctx;

/* Saturating float -> signed Q1.15. */
static int16_t veil_q15(float x) {
    float scaled = x * (float)(1 << VEIL_ROT_FRAC);
    if (scaled >  32767.0f) return  32767;
    if (scaled < -32768.0f) return -32768;
    return (int16_t)lrintf(scaled);
}

/* -------------------------------------------------------------------------
 * Coefficient generation. The core's schedule is (i, j, theta) Givens ops
 * derived from SplitMix64(key). The FPGA applies the SAME schedule to on-air
 * samples, so we hand it the per-pass (i, j, cos, sin). We regenerate the
 * schedule here with the identical draw order as veil_shield.c so the shim and
 * the (future) RTL agree bit-for-bit with the reference crate.
 *
 * NOTE: this mirrors veil_shield.c's private schedule. It is duplicated (not
 * exported) on purpose — the core stays a pure in-memory transform with a
 * stable ABI; the adapter owns the hardware-facing serialization. If the core
 * later exports its schedule, collapse this. TODO(hw): validate equality with a
 * captured on-FPGA coeff dump before any MEASURED claim.
 * ------------------------------------------------------------------------- */
typedef struct {
    uint16_t i;
    uint16_t j;
    int16_t  cos_q15;
    int16_t  sin_q15;
} veil_ow_givens;

/* TAU matches VEIL_TAU in veil_shield.c / Rust core::f32::consts::TAU. */
#define VEIL_OW_TAU 6.28318530717958647692f

static void veil_ow_build_schedule(uint64_t key, uint32_t n, uint32_t passes,
                                   veil_ow_givens *out /* [passes] */) {
    veil_rng r;
    uint32_t p;
    if (n < 2) {
        for (p = 0; p < passes; p++) {
            out[p].i = 0; out[p].j = 0;
            out[p].cos_q15 = veil_q15(1.0f); out[p].sin_q15 = 0;
        }
        return;
    }
    veil_rng_seed(&r, key);
    for (p = 0; p < passes; p++) {
        uint32_t i = (uint32_t)(veil_rng_next_u64(&r) % (uint64_t)n);
        uint32_t j = (uint32_t)(veil_rng_next_u64(&r) % (uint64_t)n);
        float theta;
        if (j == i) {
            j = (j + 1) % n;
        }
        theta = veil_rng_next_f32(&r) * VEIL_OW_TAU;
        out[p].i = (uint16_t)i;
        out[p].j = (uint16_t)j;
        out[p].cos_q15 = veil_q15(cosf(theta));
        out[p].sin_q15 = veil_q15(sinf(theta));
    }
}

/* -------------------------------------------------------------------------
 * Public API.
 * ------------------------------------------------------------------------- */

/* Program a session key into the veil_rot/veil_unrot block. Returns 0 on the
 * host shadow path; on real hardware it must poll STATUS_READY. */
int veil_ow_program_session(veil_ow_ctx *ctx) {
    veil_ow_givens sched[VEIL_OW_DEFAULT_PASSES];
    uint32_t ctrl = VEIL_ROT_CTRL_LOAD;
    uint32_t p, passes, n;

    if (!ctx || ctx->ndim < 2 || ctx->ndim > VEIL_OW_MAX_NDIM) {
        return -1; /* bounds check the on-air dimension (least authority) */
    }
    passes = ctx->passes ? ctx->passes : VEIL_OW_DEFAULT_PASSES;
    if (passes > VEIL_OW_DEFAULT_PASSES) {
        passes = VEIL_OW_DEFAULT_PASSES; /* bounded, stack-only schedule */
    }
    n = ctx->ndim;

    veil_ow_build_schedule(ctx->key, n, passes, sched);

    /* Program header registers. */
    veil_reg_write(ctx->base, VEIL_ROT_REG_KEY_LO, (uint32_t)(ctx->key));
    veil_reg_write(ctx->base, VEIL_ROT_REG_KEY_HI, (uint32_t)(ctx->key >> 32));
    veil_reg_write(ctx->base, VEIL_ROT_REG_NDIM,   n);
    veil_reg_write(ctx->base, VEIL_ROT_REG_PASSES, passes);

    /* Stream the (i, j, cos, sin) schedule into the coeff RAM. Packing:
     * COEFF_DATA = {i[15:0]... } is too wide for one 32-bit word, so we use a
     * 2-word-per-pass convention: word A = {j[15:0], i[15:0]}, word B =
     * {sin_q15[15:0], cos_q15[15:0]}. TODO(hdl): the veil_rot coeff-RAM write
     * FSM must match this exact packing. TODO(hw): confirm endianness of the
     * AXI-Lite slave. */
    for (p = 0; p < passes; p++) {
        uint32_t wa = ((uint32_t)(uint16_t)sched[p].j << 16) |
                       (uint32_t)(uint16_t)sched[p].i;
        uint32_t wb = ((uint32_t)(uint16_t)sched[p].sin_q15 << 16) |
                       (uint32_t)(uint16_t)sched[p].cos_q15;
        veil_reg_write(ctx->base, VEIL_ROT_REG_COEFF_ADDR, p * 2u);
        veil_reg_write(ctx->base, VEIL_ROT_REG_COEFF_DATA, wa);
        veil_reg_write(ctx->base, VEIL_ROT_REG_COEFF_ADDR, p * 2u + 1u);
        veil_reg_write(ctx->base, VEIL_ROT_REG_COEFF_DATA, wb);
    }

    if (ctx->role == VEIL_OW_ROLE_LEGIT_RX) {
        ctrl |= VEIL_ROT_CTRL_INVERSE; /* veil_unrot applies Q^H */
    }
    veil_reg_write(ctx->base, VEIL_ROT_REG_CTRL, ctrl);

    /* TODO(hw): on real silicon, poll VEIL_ROT_REG_STATUS for READY here and
     * time out. The host shadow has no FSM, so we return success directly and
     * DO NOT claim the hardware accepted it. */
#if defined(VEIL_OPENWIFI_KERNEL)
    {
        int spins = 100000; /* TODO(hw): calibrate against real ready latency */
        while (spins-- > 0) {
            if (veil_reg_read(ctx->base, VEIL_ROT_REG_STATUS) &
                VEIL_ROT_STATUS_READY) {
                break;
            }
        }
        if (spins <= 0) {
            return -2; /* not ready — never treat as success */
        }
    }
#endif
    return 0;
}

/* Engage / disengage the block (bit0 of CTRL), preserving the inverse bit. */
int veil_ow_set_enabled(veil_ow_ctx *ctx, int enable) {
    uint32_t ctrl;
    if (!ctx) {
        return -1;
    }
    ctrl = veil_reg_read(ctx->base, VEIL_ROT_REG_CTRL);
    if (enable) {
        ctrl |= VEIL_ROT_CTRL_ENABLE;
    } else {
        ctrl &= ~VEIL_ROT_CTRL_ENABLE;
    }
    veil_reg_write(ctx->base, VEIL_ROT_REG_CTRL, ctrl);
    return 0;
}

/*
 * Control-plane bring-up alternatives (documented idioms, not wired here):
 *   - sdrctl (nl80211 testmode) for driver-level toggles once a testmode verb
 *     is added, e.g. a "veil" subcommand mirroring existing sdrctl reg pokes.
 *   - side_ch_ctl-style hex register pokes during bench bring-up, e.g. the
 *     side_ch app note's `./side_ch_ctl whXXdY` write convention, retargeted at
 *     the veil_rot slave. TODO(hw): pick and document the actual verb.
 *
 * Self-loopback validation (before over-the-air): openwifi supports a
 * packet/IQ self-loopback test. Route veil_rot -> veil_unrot in loopback and
 * assert recovered IQ == original within Q1.15 round-off. That is the first
 * on-FPGA correctness gate (still not a defense MEASURED claim). TODO(hw).
 */

#if defined(VEIL_OPENWIFI_SELFTEST)
/* Host-only smoke test of the schedule/quantization path — NO hardware.
 * Verifies the shadow register file receives a plausible, bounded program.
 * Build: cc -DVEIL_OPENWIFI_SELFTEST veil_openwifi.c ../core/veil_shield.c -lm */
#include <stdio.h>
int main(void) {
    veil_mmio_shadow shadow;
    veil_ow_ctx ctx;
    memset(&shadow, 0, sizeof(shadow));
    ctx.base = &shadow;
    ctx.key = 0x0123456789ABCDEFull;
    ctx.ndim = 16;
    ctx.passes = VEIL_OW_DEFAULT_PASSES;
    ctx.role = VEIL_OW_ROLE_PROTECTOR;

    if (veil_ow_program_session(&ctx) != 0) {
        printf("FAIL: program_session\n");
        return 1;
    }
    if (veil_ow_set_enabled(&ctx, 1) != 0) {
        printf("FAIL: set_enabled\n");
        return 1;
    }
    if (veil_reg_read(&shadow, VEIL_ROT_REG_NDIM) != 16u) {
        printf("FAIL: ndim not programmed\n");
        return 1;
    }
    if (!(veil_reg_read(&shadow, VEIL_ROT_REG_CTRL) & VEIL_ROT_CTRL_ENABLE)) {
        printf("FAIL: enable bit\n");
        return 1;
    }
    printf("OK (SYNTHETIC/L0 host shadow only — NOT hardware-validated)\n");
    return 0;
}
#endif
