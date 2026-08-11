/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * veil_shieldd — OpenWRT / Linux mac80211 userspace adapter for the VEIL
 * compliant-waveform privacy shield (ADR-288 / ADR-290).
 *
 * ============================= HONESTY BANNER ==============================
 * STATUS: SYNTHETIC / L0 — BUILD-ONLY SCAFFOLD, UNTESTED ON HARDWARE.
 *
 * This daemon compiles and links the portable veil_shield core, and it issues
 * REAL nl80211/libnl calls for the small set of controls that Linux actually
 * exposes to userspace (antenna TX mask, station/BSS observation). Everything
 * that would edit the per-packet spatial mapping / precoder or the compressed
 * beamforming-feedback angles is BLOB-BLOCKED on commodity Qualcomm/MediaTek
 * parts and is marked `TODO(hw)` at the exact call site — see README.md and
 * INTEGRATION.md. Nothing here has been run against a radio. Do not read any
 * comment in this file as evidence that VEIL obfuscation reaches the air.
 *
 * COMPLIANCE: every control below is a standards-compliant configuration or
 * observation action. This daemon never transmits energy to deny a channel;
 * it only shapes/observes our own compliant frames. It is NOT a jammer.
 * ==========================================================================
 *
 * Build deps (OpenWRT: libnl-tiny; desktop: libnl-3 + libnl-genl-3):
 *   pkg-config --cflags --libs libnl-genl-3.0
 * See Makefile (host build-check) and openwrt.mk (package stub).
 */

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Real libnl / nl80211 headers. On OpenWRT these resolve to libnl-tiny; on a
 * desktop to libnl-3. If the toolchain lacks them the host Makefile still
 * builds the core object so the rotation math is validated in isolation. */
#include <netlink/netlink.h>
#include <netlink/genl/genl.h>
#include <netlink/genl/ctrl.h>
#include <linux/nl80211.h>

#include "veil_shield.h"

/* ---- Tunables (compliant, conservative defaults) ---------------------- */
#define VEIL_DEFAULT_PASSES 96u          /* matches core default (ADR-290) */
#define VEIL_CADENCE_JITTER_MIN_MS 20    /* NDP sounding cadence jitter floor */
#define VEIL_CADENCE_JITTER_MAX_MS 400   /* ... and ceiling (stays in-spec)   */

/* ---- Daemon context --------------------------------------------------- */
struct veil_ctx {
    struct nl_sock *sock;   /* generic-netlink socket to nl80211            */
    int             family; /* resolved "nl80211" genl family id           */
    int             ifindex;/* target AP interface (e.g. phy0-ap0)         */
    uint64_t        key;    /* shared session key for the keyed rotation    */
    size_t          passes; /* Givens passes                                */
    volatile sig_atomic_t running;
};

static struct veil_ctx g_ctx;

static void on_signal(int sig) { (void)sig; g_ctx.running = 0; }

/* ---------------------------------------------------------------------- */
/* nl80211 bring-up — all REAL libnl-genl-3 API names.                     */
/* ---------------------------------------------------------------------- */
static int veil_nl_connect(struct veil_ctx *c) {
    c->sock = nl_socket_alloc();
    if (!c->sock) {
        fprintf(stderr, "veil: nl_socket_alloc failed\n");
        return -ENOMEM;
    }
    if (genl_connect(c->sock)) {
        fprintf(stderr, "veil: genl_connect failed\n");
        return -EIO;
    }
    c->family = genl_ctrl_resolve(c->sock, "nl80211");
    if (c->family < 0) {
        fprintf(stderr, "veil: genl_ctrl_resolve(nl80211) failed: %d\n",
                c->family);
        return c->family;
    }
    /* Observe MLME events (auth/assoc, and — where the driver forwards them —
     * action-frame notifications). Real multicast group name is "mlme". */
    int grp = genl_ctrl_resolve_grp(c->sock, "nl80211", "mlme");
    if (grp >= 0) {
        (void)nl_socket_add_membership(c->sock, grp);
    }
    return 0;
}

/* ---------------------------------------------------------------------- */
/* CONTROL 1 (FEASIBLE): TX antenna-map perturbation.                      */
/* Rotating the allowed TX antenna bitmap changes the static spatial       */
/* mapping the PHY uses, coarsely perturbing the CSI a sensor observes.    */
/* This is a genuinely userspace-reachable, compliant knob.                */
/*   NL80211_CMD_SET_WIPHY + NL80211_ATTR_WIPHY_ANTENNA_TX / _RX           */
/* NOTE: many drivers only accept this while the phy is DOWN, and only on  */
/* symmetric masks — validate per driver. Coarse, not the keyed rotation.  */
/* ---------------------------------------------------------------------- */
static int veil_set_tx_antenna_mask(struct veil_ctx *c,
                                    uint32_t tx_mask, uint32_t rx_mask) {
    struct nl_msg *msg = nlmsg_alloc();
    if (!msg) return -ENOMEM;
    genlmsg_put(msg, NL_AUTO_PORT, NL_AUTO_SEQ, c->family, 0, 0,
                NL80211_CMD_SET_WIPHY, 0);
    /* wiphy is addressed via the interface index on most drivers. */
    NLA_PUT_U32(msg, NL80211_ATTR_IFINDEX, (uint32_t)c->ifindex);
    NLA_PUT_U32(msg, NL80211_ATTR_WIPHY_ANTENNA_TX, tx_mask);
    NLA_PUT_U32(msg, NL80211_ATTR_WIPHY_ANTENNA_RX, rx_mask);
    int ret = nl_send_auto(c->sock, msg);
    nlmsg_free(msg);
    if (ret < 0) return ret;
    return nl_recvmsgs_default(c->sock); /* consume ACK/ERR */
nla_put_failure:
    nlmsg_free(msg);
    return -EMSGSIZE;
}

/* ---------------------------------------------------------------------- */
/* CONTROL 2 (FEASIBLE, indirect): NDP sounding-cadence randomization.     */
/* mac80211/driver decides when to send NDP Announcement + NDP. There is   */
/* NO stable nl80211 attribute to set the sounding period directly, so the */
/* compliant lever from userspace is hostapd's advertised sounding         */
/* capability and dimensions, toggled/rewritten over the hostapd ctrl      */
/* interface (RECONFIGURE / SET). We jitter the *offered* cadence.         */
/*                                                                          */
/* TODO(hw): there is no nl80211 "set sounding interval" command. Confirm  */
/* against hostapd ctrl_iface docs; the direct per-NDP timer lives in      */
/* driver/firmware. See INTEGRATION.md §2. Cite:                           */
/*   https://w1.fi/cgit/hostap/tree/hostapd/hostapd.conf                    */
/* ---------------------------------------------------------------------- */
static unsigned veil_next_cadence_ms(struct veil_ctx *c) {
    /* Derive jitter deterministically from the session key stream so the
     * paired receiver can anticipate the schedule (compliant, not random
     * spraying). Reuses the core SplitMix64 for byte-identical behavior. */
    static veil_rng r;
    static int seeded = 0;
    if (!seeded) { veil_rng_seed(&r, c->key ^ 0xCADE11CEULL); seeded = 1; }
    unsigned span = VEIL_CADENCE_JITTER_MAX_MS - VEIL_CADENCE_JITTER_MIN_MS;
    return VEIL_CADENCE_JITTER_MIN_MS +
           (unsigned)(veil_rng_next_f32(&r) * (float)span);
}

static int veil_randomize_sounding_cadence(struct veil_ctx *c) {
    unsigned ms = veil_next_cadence_ms(c);
    /* TODO(hw): push `ms` into the offered sounding cadence. On OpenWRT the
     * realistic path is the hostapd ctrl_iface (UNIX socket at
     * /var/run/hostapd/<iface>): rewrite he/vht sounding-dimension or toggle
     * beamformer capability and RECONFIGURE. mac80211 has no direct knob.
     * This function currently only computes the schedule. */
    fprintf(stderr, "veil: [feasible/indirect] next sounding jitter = %u ms "
                    "(TODO(hw): apply via hostapd ctrl_iface)\n", ms);
    return 0;
}

/* ---------------------------------------------------------------------- */
/* CONTROL 3 (MOSTLY BLOB-BLOCKED): MU-MIMO group shuffling.               */
/* The MU group definition + steering matrices are computed and applied in */
/* the WiFi MCU firmware on mt76 (mt7915) and all ath1x parts. There is no */
/* generic nl80211 command to reshuffle MU groups. Only a vendor subcmd    */
/* (NL80211_CMD_VENDOR) on a driver that chose to expose one could do it.  */
/* ---------------------------------------------------------------------- */
static int veil_shuffle_mumimo_groups(struct veil_ctx *c) {
    (void)c;
    /* TODO(hw): requires NL80211_CMD_VENDOR + a driver-specific
     * NL80211_ATTR_VENDOR_ID / _SUBCMD / _DATA that does not exist upstream
     * for mt76/ath. Without a driver+firmware patch this is unreachable.
     * See INTEGRATION.md §3. Left as an explicit no-op, not a fake success. */
    fprintf(stderr, "veil: [blob-blocked] MU-MIMO group shuffle needs a "
                    "vendor subcmd / firmware patch (TODO(hw))\n");
    return -ENOTSUP;
}

/* ---------------------------------------------------------------------- */
/* CONTROL 4 (BLOB-BLOCKED on commodity AP silicon): the keyed rotation.   */
/* This is the actual VEIL transform — a keyed Givens rotation on the fine */
/* subspace of the compressed beamforming feedback (the phi/psi angles),   */
/* or equivalently a unitary Q on the LTF spatial mapping. On mt76/ath the */
/* feedback report is generated and the precoder applied inside firmware,  */
/* so userspace cannot edit it. This function shows WHERE the core plugs   */
/* in for the platforms that CAN reach the buffer (openwifi FPGA datapath, */
/* Nexmon Broadcom patch) — it operates on a caller-supplied fine block.   */
/* ---------------------------------------------------------------------- */
static int veil_apply_keyed_rotation(struct veil_ctx *c,
                                     float *fine, size_t n) {
    if (!fine || n < 2) return -EINVAL;
    /* Pure, orthogonal, energy-preserving (the "not jamming" invariant). */
    float before = veil_l2_norm(fine, n);
    veil_shield_apply(fine, n, c->key, c->passes);
    float after = veil_l2_norm(fine, n);
    /* TODO(hw): on OpenWRT there is NO userspace/mac80211 hook that hands us
     * this buffer before TX. Reaching it requires a driver+firmware patch
     * (mt76 MCU / ath) to expose the pre-precoder V/steering matrix, OR use
     * the openwifi (FPGA) or Nexmon adapters. See INTEGRATION.md §4.
     * We only prove the math is invariant here; nothing goes on air. */
    fprintf(stderr, "veil: [blob-blocked path] rotated %zu coeffs, "
                    "L2 %.6f -> %.6f (delta %.2e; must be ~0)\n",
            n, before, after, (double)(after - before));
    return 0;
}

/* ---------------------------------------------------------------------- */
/* Event loop: watch for sensing-solicitation cadence.                     */
/* We register interest in MLME/frame events. On commodity drivers the raw */
/* NDP Announcement is NOT forwarded to userspace, so honest detection of  */
/* an *external* sensing solicitation needs monitor-mode capture or a      */
/* driver notification that does not exist upstream — marked TODO(hw).     */
/* ---------------------------------------------------------------------- */
static int veil_event_cb(struct nl_msg *msg, void *arg) {
    struct veil_ctx *c = (struct veil_ctx *)arg;
    struct genlmsghdr *gnlh = nlmsg_data(nlmsg_hdr(msg));
    switch (gnlh->cmd) {
    case NL80211_CMD_FRAME:
        /* TODO(hw): parse NL80211_ATTR_FRAME; classify VHT/HE compressed
         * beamforming action (category 21/30) or NDPA to measure solicitation
         * cadence. Requires the driver to forward these frames (registered via
         * NL80211_CMD_REGISTER_FRAME / monitor). Not guaranteed upstream. */
        (void)veil_randomize_sounding_cadence(c);
        break;
    case NL80211_CMD_NEW_STATION:
    case NL80211_CMD_DEL_STATION:
        /* Membership churn changes MU grouping surface. */
        (void)veil_shuffle_mumimo_groups(c);
        break;
    default:
        break;
    }
    return NL_SKIP;
}

static void usage(const char *p) {
    fprintf(stderr,
        "Usage: %s -i <ifindex> [-k <key_hex>] [-p <passes>]\n"
        "  BUILD-ONLY / UNTESTED-ON-HARDWARE. See README.md.\n", p);
}

int main(int argc, char **argv) {
    memset(&g_ctx, 0, sizeof(g_ctx));
    g_ctx.key = 0xA5A5A5A5A5A5A5A5ULL; /* placeholder; real key from keystore */
    g_ctx.passes = VEIL_DEFAULT_PASSES;
    g_ctx.ifindex = -1;
    g_ctx.running = 1;

    int opt;
    while ((opt = getopt(argc, argv, "i:k:p:h")) != -1) {
        switch (opt) {
        case 'i': g_ctx.ifindex = atoi(optarg); break;
        case 'k': g_ctx.key = strtoull(optarg, NULL, 16); break;
        case 'p': g_ctx.passes = (size_t)strtoul(optarg, NULL, 10); break;
        case 'h': default: usage(argv[0]); return (opt == 'h') ? 0 : 2;
        }
    }
    if (g_ctx.ifindex < 0) { usage(argv[0]); return 2; }

    fprintf(stderr, "veil_shieldd: SYNTHETIC/L0 build-only scaffold — "
                    "no RF is emitted, nothing is validated on silicon.\n");

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    if (veil_nl_connect(&g_ctx)) return 1;

    /* Install the event callback (valid-message path). */
    nl_socket_modify_cb(g_ctx.sock, NL_CB_VALID, NL_CB_CUSTOM,
                        veil_event_cb, &g_ctx);
    nl_socket_disable_seq_check(g_ctx.sock); /* required for multicast events */

    /* Self-check the one genuinely feasible active control at startup. Comment
     * this out on a live AP; it may bounce the radio depending on the driver.
     * (void)veil_set_tx_antenna_mask(&g_ctx, 0x3, 0x3); */
    (void)veil_set_tx_antenna_mask;

    /* Prove the linked core is byte-consistent (no radio involved). */
    {
        float demo[8] = {1,0,0,0,0,0,0,0};
        (void)veil_apply_keyed_rotation(&g_ctx, demo, 8);
        veil_shield_recover(demo, 8, g_ctx.key, g_ctx.passes);
        fprintf(stderr, "veil: recover round-trip demo[0]=%.6f (expect ~1.0)\n",
                (double)demo[0]);
    }

    while (g_ctx.running) {
        int r = nl_recvmsgs_default(g_ctx.sock);
        if (r < 0 && r != -NLE_AGAIN) {
            fprintf(stderr, "veil: nl_recvmsgs_default: %d\n", r);
            break;
        }
    }

    nl_socket_free(g_ctx.sock);
    return 0;
}
