/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * veil_patch.c — VEIL protector, Nexmon (Broadcom/Cypress) path.
 *
 * ============================ HONESTY BANNER ============================
 * SYNTHETIC / L0 / BUILD-ONLY. This file is an HONEST SKELETON in Nexmon
 * style. It has NOT been built with the Nexmon toolchain, NOT flashed to a
 * chip, and NOT captured on air. Every __attribute__((at(...))) address and
 * every firmware symbol below is a PLACEHOLDER. Do not treat this as working
 * firmware. See ../README.md for the feasibility grade (C, research-grade).
 *
 * Goal: call the portable VEIL core (../../core/veil_shield.c)
 * `veil_shield_apply()` on the compressed-beamforming-feedback FINE ANGLES in
 * the transmitted VHT/HE compressed beamforming report, so the identity-bearing
 * fine subspace is obfuscated by a keyed, ENERGY-PRESERVING (orthogonal)
 * Givens rotation before the frame leaves the radio. Compliant only, never
 * jamming: the transform preserves the report's L2 norm.
 *
 * Target: BCM43455c0 (Raspberry Pi 3B+/4B), firmware 7_45_189. Others TODO.
 * =======================================================================
 */

#pragma NEXMON targetregion "patch"

#include <firmware_version.h>   /* FW_VER_7_45_189, CHIP_VER_BCM43455c0 (Nexmon) */
#include <patcher.h>            /* BPatch / GPatch / __attribute__((at(...)))    */
#include <structs.h>            /* struct sk_buff, struct wlc_info, etc.         */
#include <wrapper.h>            /* Nexmon wrappers for ROM/firmware functions    */

/* --- Portable VEIL core, linked/inlined for the MCU -------------------------
 * The core is pure C99: no malloc, no libc I/O, only <math.h> (sinf/cosf/sqrtf).
 * On the Nexmon ARM target we compile ../../core/veil_shield.c into this patch
 * object (see ../BUILD.md) and pull in only the declarations here. Everything
 * operates on a caller-provided fixed buffer — no dynamic allocation on-chip.  */
#include "veil_shield.h"

/* ------------------------------------------------------------------------- */
/* Configuration (compile-time; no on-chip allocation)                       */
/* ------------------------------------------------------------------------- */

/* Max fine-angle count we will touch in one report. Sized for a VHT SU report
 * fine block; bound it so all working storage is on the stack, malloc-free. */
#define VEIL_MAX_FINE   64u

/* Rotation passes — MUST match the associated receiver and the Rust reference
 * crate default so recover() inverts exactly. TODO(hw): confirm against the
 * receiver config actually deployed. */
#define VEIL_PASSES     96u

/* Session key. TODO(hw): DO NOT hardcode a real key in flashed firmware. Inject
 * via nexutil IOCTL (see veil_ioctl_set_key stub) or a provisioning step; this
 * placeholder exists only so the skeleton type-checks. */
static uint64_t g_veil_key = 0x0000000000000000ULL;

/* ------------------------------------------------------------------------- */
/* Bridge: decode angles -> rotate -> re-encode, in place                    */
/* ------------------------------------------------------------------------- */
/*
 * TODO(reverse-engineer): The compressed beamforming report packs the phi/psi
 * angles as bit-fields whose widths depend on the codebook (VHT: (7,5) or (9,7);
 * HE differs) and on Nc/Nr. The bytes handed to us are NOT plain floats. This
 * bridge must:
 *   (1) parse the fine-angle bit-fields from `report` into `fine[]` as floats
 *       in the same units/order the receiver + Rust reference expect,
 *   (2) call veil_shield_apply() on that flat vector,
 *   (3) re-quantize and repack the rotated angles back into `report`,
 *       preserving all coarse/header fields and the frame length.
 * Steps (1)/(3) are the real work and are UNIMPLEMENTED here.
 */
static void veil_shape_report_inplace(uint8_t *report, uint32_t report_len)
{
    if (report == 0 || report_len == 0)
        return;

    float fine[VEIL_MAX_FINE];
    uint32_t n = 0;

    /* TODO(reverse-engineer): unpack fine-angle bit-fields -> fine[0..n) */
    /* n = veil_bfr_unpack_fine(report, report_len, fine, VEIL_MAX_FINE);  */
    if (n < 2 || n > VEIL_MAX_FINE)
        return; /* nothing safely shapeable; leave frame untouched (fail-open) */

    /* Orthogonal, energy-preserving, keyed. This is the ONLY validated step. */
    veil_shield_apply(fine, (size_t)n, g_veil_key, VEIL_PASSES);

    /* TODO(reverse-engineer): repack fine[0..n) back into `report` bit-fields,
     * keeping report_len and all non-fine fields byte-identical.            */
    /* veil_bfr_pack_fine(report, report_len, fine, n);                      */
    (void)report_len;
}

/* ------------------------------------------------------------------------- */
/* Hook candidate #1 (see README): ARM action-frame TX assembly             */
/* ------------------------------------------------------------------------- */
/*
 * We hook the point where the "wl" driver has assembled the VHT Compressed
 * Beamforming Report action frame in an sk_buff, just before it is queued to
 * the D11 for transmission, locate the report body, and shape it.
 *
 * TODO(reverse-engineer): the symbol/address below is a PLACEHOLDER. The real
 * target must be found by disassembling 7_45_189 (IDA + Nexmon's wl_ram.elf
 * symbol map) and confirming: (a) the report body is assembled in ARM (not
 * only in D11 ucode), (b) `p` really carries a compressed-beamforming action
 * frame, and (c) the offset of the report body within the frame.
 *
 * If (a) is false on this chip, candidate #1 is dead and we fall to #2/#3
 * (TX template-RAM rewrite / D11 ucode patch) — both documented in README,
 * neither implemented here.
 */

/* Original firmware function prototype (PLACEHOLDER signature). */
extern int wlc_sendmgmt_veil_target(struct wlc_info *wlc, void *p, void *scb);

/* Our replacement. GPatch/BPatch below redirects the target to this. */
int wlc_sendmgmt_veil_hook(struct wlc_info *wlc, void *p, void *scb)
{
    /* TODO(reverse-engineer): confirm `p` is a struct sk_buff* and that this
     * frame is a VHT/HE compressed beamforming action frame (category 21
     * VHT / 30 HE, action = Compressed Beamforming). Guard hard so we never
     * mangle unrelated management frames. */
    struct sk_buff *skb = (struct sk_buff *)p;
    if (skb != 0 /* && veil_is_bf_report_action(skb) */) {
        /* TODO(reverse-engineer): compute report body pointer + length from the
         * action-frame layout. PLACEHOLDER offsets: */
        uint8_t  *report     = 0; /* skb->data + VEIL_BFR_BODY_OFFSET; */
        uint32_t  report_len = 0; /* skb->len  - VEIL_BFR_BODY_OFFSET; */
        veil_shape_report_inplace(report, report_len);
    }

    /* Always fall through to the real firmware routine so normal TX proceeds. */
    return wlc_sendmgmt_veil_target(wlc, p, scb);
}

/*
 * Redirect the firmware's mgmt/action TX routine to our hook.
 * PLACEHOLDER ADDRESS — 0xDEAD0000 is intentionally invalid so nobody mistakes
 * this for a real, flashable patch. TODO(reverse-engineer): replace with the
 * verified address for CHIP_VER_BCM43455c0 / FW_VER_7_45_189.
 *
 * Nexmon idiom: a branch patch that overwrites the target's prologue with a
 * branch to our replacement (which tail-calls the saved original).
 */
__attribute__((at(0xDEAD0000, "flashpatch", CHIP_VER_BCM43455c0, FW_VER_7_45_189)))
BPatch(veil_sendmgmt_hook, wlc_sendmgmt_veil_hook);

/* ------------------------------------------------------------------------- */
/* Key provisioning via nexutil IOCTL (stub)                                 */
/* ------------------------------------------------------------------------- */
/*
 * TODO(hw): register a custom IOCTL so `nexutil` can push the 64-bit session
 * key at runtime instead of baking it into flash. Hook the driver's ioctl
 * dispatch (wlc_ioctl) the same way nexmon_csi installs its config IOCTLs.
 * Left as a stub: the dispatch address and the nexmon_ioctl plumbing are
 * PLACEHOLDERS.
 */
#define VEIL_IOCTL_SET_KEY  0x7EIL /* TODO(hw): pick a free vendor IOCTL id */

int veil_ioctl_set_key(struct wlc_info *wlc, const uint8_t *buf, uint32_t len)
{
    (void)wlc;
    if (buf == 0 || len < sizeof(uint64_t))
        return -1;
    uint64_t k = 0;
    for (uint32_t i = 0; i < sizeof(uint64_t); i++)
        k |= ((uint64_t)buf[i]) << (8u * i);
    g_veil_key = k;
    return 0;
}

/*
 * ---------------------------------------------------------------------------
 * Candidate #2 (TX template-RAM rewrite) and #3 (D11 ucode patch) are NOT
 * implemented. See ../README.md "Hook-point candidates". #3 would require the
 * D11 assembler and the PHY/SHM angle-staging map — deepest and most fragile.
 * ---------------------------------------------------------------------------
 */
