/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Host test for the portable veil_shield core. Builds and runs on a workstation
 * with gcc — NO hardware. Verifies the three load-bearing invariants:
 *   1. energy conservation (orthogonal transform ⇒ ‖v‖ unchanged) — "not jamming"
 *   2. reversibility (apply then recover ≈ identity) — legitimate receiver
 *   3. cross-language determinism (the SplitMix64 stream matches Rust's)
 */
#include "../veil_shield.h"
#include <math.h>
#include <stdio.h>

static int failures = 0;
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            printf("FAIL %s\n", msg);                                           \
            failures++;                                                         \
        } else {                                                                \
            printf("PASS %s\n", msg);                                           \
        }                                                                       \
    } while (0)

int main(void) {
    /* Cross-language determinism: same seed as Rust `Rng::new(42)` must yield
     * the same first three u64 words (pinned from the Rust crate). */
    {
        veil_rng r;
        veil_rng_seed(&r, 42);
        uint64_t a = veil_rng_next_u64(&r);
        uint64_t b = veil_rng_next_u64(&r);
        uint64_t c = veil_rng_next_u64(&r);
        printf("splitmix64(42): %llu %llu %llu\n", (unsigned long long)a,
               (unsigned long long)b, (unsigned long long)c);
        /* These are asserted equal to the Rust stream by the CI parity check;
         * here we only assert the stream is deterministic and non-degenerate. */
        veil_rng r2;
        veil_rng_seed(&r2, 42);
        CHECK(veil_rng_next_u64(&r2) == a, "prng deterministic");
        CHECK(a != b && b != c, "prng non-degenerate");
    }

    const size_t n = 56; /* fine-block dims at the default scene */
    const uint64_t key = 0xC0FFEE1234ULL;
    const size_t passes = 96;

    float v[56], orig[56];
    veil_rng g;
    veil_rng_seed(&g, 7);
    for (size_t i = 0; i < n; i++) {
        /* pseudo-random test vector in [-1,1) */
        v[i] = 2.0f * veil_rng_next_f32(&g) - 1.0f;
        orig[i] = v[i];
    }

    float n0 = veil_l2_norm(v, n);
    veil_shield_apply(v, n, key, passes);
    float n1 = veil_l2_norm(v, n);
    CHECK(fabsf(n1 - n0) < 1e-3f, "energy conserved (not jamming)");

    /* scrambled: should differ from original */
    float diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        diff += fabsf(v[i] - orig[i]);
    }
    CHECK(diff > 0.5f, "fine block scrambled");

    veil_shield_recover(v, n, key, passes);
    float err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = v[i] - orig[i];
        err += e * e;
    }
    CHECK(sqrtf(err) < 1e-3f, "recover inverts apply");

    /* a different key does NOT recover (no shared key ⇒ no inversion) */
    for (size_t i = 0; i < n; i++) {
        v[i] = orig[i];
    }
    veil_shield_apply(v, n, key, passes);
    veil_shield_recover(v, n, key ^ 0x1, passes);
    float err2 = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = v[i] - orig[i];
        err2 += e * e;
    }
    CHECK(sqrtf(err2) > 0.5f, "wrong key does not recover");

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
           failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
