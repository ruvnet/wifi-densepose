/* SPDX-License-Identifier: MIT OR Apache-2.0
 * veil_shield core — see veil_shield.h. Pure computation; no radio, no I/O. */
#include "veil_shield.h"
#include <math.h>

/* Two-pi constant matching Rust core::f32::consts::TAU. */
#define VEIL_TAU 6.28318530717958647692f

void veil_rng_seed(veil_rng *r, uint64_t seed) {
    /* Rust: state = seed ^ 0x9E3779B97F4A7C15 */
    r->state = seed ^ 0x9E3779B97F4A7C15ULL;
}

uint64_t veil_rng_next_u64(veil_rng *r) {
    /* SplitMix64, identical constants to the Rust crate. */
    r->state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = r->state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

float veil_rng_next_f32(veil_rng *r) {
    /* (next_u64 >> 40) / 2^24 — 24 mantissa bits, matches Rust `next_f32`. */
    uint64_t bits = veil_rng_next_u64(r) >> 40;
    return (float)bits / (float)(1u << 24);
}

/* Apply one Givens rotation on coordinates (i, j) by angle theta. Orthogonal. */
static void givens(float *v, size_t i, size_t j, float theta) {
    float c = cosf(theta), s = sinf(theta);
    float vi = v[i], vj = v[j];
    v[i] = c * vi - s * vj;
    v[j] = s * vi + c * vj;
}

/* Build the (i, j, theta) schedule deterministically from the key. The order
 * and draws mirror `protector.rs::session_rotation`. */
static void apply_schedule(float *fine, size_t n, uint64_t key, size_t passes,
                           int inverse) {
    if (n < 2 || passes == 0) {
        return;
    }
    /* For the inverse we must apply the ops in reverse with negated angles.
     * Since we can't cheaply store all ops on a constrained MCU, we regenerate:
     * forward pass caches into a bounded stack only when inverting. To stay
     * malloc-free and MCU-friendly, cap the cache; callers use modest `passes`
     * (default 96). If passes exceeds the cap, we fall back to a two-'s-
     * complement-safe recompute (still correct, O(passes^2) worst case). */
    enum { CACHE = 256 };
    if (!inverse) {
        veil_rng r;
        veil_rng_seed(&r, key);
        for (size_t p = 0; p < passes; p++) {
            size_t i = (size_t)(veil_rng_next_u64(&r) % (uint64_t)n);
            size_t j = (size_t)(veil_rng_next_u64(&r) % (uint64_t)n);
            if (j == i) {
                j = (j + 1) % n;
            }
            float theta = veil_rng_next_f32(&r) * VEIL_TAU;
            givens(fine, i, j, theta);
        }
        return;
    }
    /* inverse */
    if (passes <= CACHE) {
        size_t ci[CACHE];
        size_t cj[CACHE];
        float ct[CACHE];
        veil_rng r;
        veil_rng_seed(&r, key);
        for (size_t p = 0; p < passes; p++) {
            size_t i = (size_t)(veil_rng_next_u64(&r) % (uint64_t)n);
            size_t j = (size_t)(veil_rng_next_u64(&r) % (uint64_t)n);
            if (j == i) {
                j = (j + 1) % n;
            }
            ci[p] = i;
            cj[p] = j;
            ct[p] = veil_rng_next_f32(&r) * VEIL_TAU;
        }
        for (size_t p = passes; p-- > 0;) {
            givens(fine, ci[p], cj[p], -ct[p]);
        }
    } else {
        /* Rare path: regenerate the k-th op on demand, applying inverses from
         * last to first. O(passes^2) but malloc-free and correct. */
        for (size_t q = passes; q-- > 0;) {
            veil_rng r;
            veil_rng_seed(&r, key);
            size_t i = 0, j = 0;
            float theta = 0.0f;
            for (size_t p = 0; p <= q; p++) {
                i = (size_t)(veil_rng_next_u64(&r) % (uint64_t)n);
                j = (size_t)(veil_rng_next_u64(&r) % (uint64_t)n);
                if (j == i) {
                    j = (j + 1) % n;
                }
                theta = veil_rng_next_f32(&r) * VEIL_TAU;
            }
            givens(fine, i, j, -theta);
        }
    }
}

void veil_shield_apply(float *fine, size_t n, uint64_t key, size_t passes) {
    apply_schedule(fine, n, key, passes, 0);
}

void veil_shield_recover(float *fine, size_t n, uint64_t key, size_t passes) {
    apply_schedule(fine, n, key, passes, 1);
}

float veil_l2_norm(const float *v, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        acc += (double)v[i] * (double)v[i];
    }
    return (float)sqrt(acc);
}
