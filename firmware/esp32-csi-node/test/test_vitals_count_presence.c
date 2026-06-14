/**
 * @file test_vitals_count_presence.c
 * @brief Host-side unit tests for the issue #998 / #996 vitals logic fixes.
 *
 * Covers two pure decision functions extracted from edge_processing.c:
 *   1. count_distinct_persons()  — issue #998 person over-count gate
 *                                   (energy gate + spatial dedup).
 *   2. person_count_debounce()   — issue #998 count persistence debounce.
 *   3. presence_flag_update()    — issue #996 presence hysteresis + clear
 *                                   debounce (Schmitt trigger).
 *
 * Build (Linux/macOS/Windows with any C99 compiler):
 *   cc -std=c99 -Wall -I../main -o test_vitals \
 *      test_vitals_count_presence.c && ./test_vitals
 *
 * Exits 0 on all-pass, prints which assertion failed otherwise.
 *
 * Why a separate host test file: these are deterministic logic checks for the
 * exact boundary behaviour the issues describe; libFuzzer adds no signal here.
 *
 * IMPORTANT — these three functions are copied VERBATIM from
 * firmware/esp32-csi-node/main/edge_processing.c. They are pure (no globals,
 * no ESP-IDF). If the firmware copy changes, update the copy here and re-run
 * this test before the firmware change merges. The named tuning constants are
 * pulled from the real header so the test and firmware can never disagree on
 * thresholds.
 *
 * HARDWARE-GATED CAVEAT: these tests pin the *logic* (no flicker / no
 * over-count for the synthetic traces). True count accuracy and the exact
 * energy/separation/hysteresis thresholds that best match a real room vs
 * labelled ground truth remain hardware- and data-gated (COM9 ESP32-S3 +
 * labelled occupancy). This is a robustness/logic fix, not a validated
 * accuracy claim.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* Named tuning constants come from the real firmware header so the test can
 * never silently diverge from the constants the firmware compiles with. */
#include "edge_processing.h"

/* ──────────────────────────────────────────────────────────────────────
 *  System under test — copied VERBATIM from edge_processing.c.
 * ────────────────────────────────────────────────────────────────────── */

/* count_distinct_persons() — issue #998 energy gate + spatial dedup. */
static uint8_t count_distinct_persons(const float *energy, const uint8_t *sc_idx,
                                      uint8_t n_groups)
{
    if (n_groups == 0) return 0;

    float max_energy = 0.0f;
    for (uint8_t g = 0; g < n_groups; g++) {
        if (energy[g] > max_energy) max_energy = energy[g];
    }
    if (max_energy <= 0.0f) return 0;

    float min_energy = max_energy * EDGE_PERSON_MIN_ENERGY_RATIO;

    uint8_t counted_sc[EDGE_MAX_PERSONS];
    uint8_t count = 0;

    bool used[EDGE_MAX_PERSONS];
    for (uint8_t g = 0; g < n_groups && g < EDGE_MAX_PERSONS; g++) used[g] = false;

    for (uint8_t iter = 0; iter < n_groups && iter < EDGE_MAX_PERSONS; iter++) {
        int best = -1;
        float best_e = min_energy;
        for (uint8_t g = 0; g < n_groups && g < EDGE_MAX_PERSONS; g++) {
            if (used[g]) continue;
            if (energy[g] >= best_e) { best_e = energy[g]; best = g; }
        }
        if (best < 0) break;
        used[best] = true;

        bool duplicate = false;
        for (uint8_t c = 0; c < count; c++) {
            int sep = (int)sc_idx[best] - (int)counted_sc[c];
            if (sep < 0) sep = -sep;
            if (sep < EDGE_PERSON_MIN_SC_SEP) { duplicate = true; break; }
        }
        if (duplicate) continue;

        counted_sc[count++] = sc_idx[best];
    }

    if (count == 0) count = 1;
    return count;
}

/* person_count_debounce() — issue #998 count persistence. */
static uint8_t person_count_debounce(uint8_t raw, uint8_t *candidate,
                                     uint8_t *streak, uint8_t *stable)
{
    if (raw == *stable) {
        *candidate = raw;
        *streak = 0;
        return *stable;
    }
    if (raw == *candidate) {
        if (*streak < 0xFF) (*streak)++;
    } else {
        *candidate = raw;
        *streak = 1;
    }
    if (*streak >= EDGE_PERSON_PERSIST_FRAMES) {
        *stable = *candidate;
        *streak = 0;
    }
    return *stable;
}

/* presence_flag_update() — issue #996 hysteresis + clear debounce. */
static bool presence_flag_update(bool prev, float score, float threshold,
                                 uint8_t *below_count)
{
    float low_thresh = threshold * EDGE_PRESENCE_HYST_RATIO;

    if (score > threshold) {
        *below_count = 0;
        return true;
    }

    if (score >= low_thresh) {
        *below_count = 0;
        return prev;
    }

    if (*below_count < 0xFF) (*below_count)++;
    if (!prev) {
        return false;
    }
    if (*below_count >= EDGE_PRESENCE_CLEAR_FRAMES) {
        *below_count = 0;
        return false;
    }
    return true;
}

/* ──────────────────────────────────────────────────────────────────────
 *  Test harness
 * ────────────────────────────────────────────────────────────────────── */

static int g_failed = 0;
static int g_passed = 0;

#define CHECK_EQ_U8(label, got, expected) do {                              \
    if ((uint8_t)(got) == (uint8_t)(expected)) { g_passed++; }              \
    else {                                                                  \
        g_failed++;                                                         \
        printf("FAIL: %s — got=%u expected=%u\n",                           \
               (label), (unsigned)(uint8_t)(got),                           \
               (unsigned)(uint8_t)(expected));                              \
    }                                                                       \
} while (0)

#define CHECK_TRUE(label, cond) do {                                        \
    if (cond) { g_passed++; }                                               \
    else { g_failed++; printf("FAIL: %s — expected true\n", (label)); }     \
} while (0)

/* ──────────────────────────────────────────────────────────────────────
 *  #998 — count_distinct_persons: single body must NOT report EDGE_MAX_PERSONS
 * ────────────────────────────────────────────────────────────────────── */

/* One strong signature + weak multipath echoes in adjacent subcarrier groups.
 * This is exactly the field report: one person ~50 cm → persons=4. The energy
 * gate + spatial dedup must collapse this to 1. */
static void test_count_single_strong_signature(void)
{
    /* 4 groups: one dominant, three weak multipath (below the energy gate),
     * representative subcarriers clustered (adjacent → one body). */
    float   energy[EDGE_MAX_PERSONS] = {10.0f, 0.6f, 0.4f, 0.3f};
    uint8_t sc[EDGE_MAX_PERSONS]     = {20, 21, 22, 23};
    CHECK_EQ_U8("single strong signature → 1",
                count_distinct_persons(energy, sc, EDGE_MAX_PERSONS), 1);
}

/* Even if the weak echoes are spatially spread, they're still below the energy
 * gate, so they don't count. */
static void test_count_single_spread_multipath(void)
{
    float   energy[EDGE_MAX_PERSONS] = {10.0f, 1.0f, 0.8f, 0.5f};
    uint8_t sc[EDGE_MAX_PERSONS]     = {10, 40, 70, 100};
    CHECK_EQ_U8("single body spread multipath → 1",
                count_distinct_persons(energy, sc, EDGE_MAX_PERSONS), 1);
}

/* Two genuine, well-separated, comparably-strong signatures → 2. */
static void test_count_two_well_separated(void)
{
    float   energy[EDGE_MAX_PERSONS] = {10.0f, 9.0f, 0.3f, 0.2f};
    uint8_t sc[EDGE_MAX_PERSONS]     = {10, 90, 11, 12};
    CHECK_EQ_U8("two well-separated strong → 2",
                count_distinct_persons(energy, sc, EDGE_MAX_PERSONS), 2);
}

/* Two strong but spatially ADJACENT signatures collapse to 1 (same body):
 * spatial dedup prevents double-counting one person's two strong subcarriers. */
static void test_count_two_strong_adjacent_dedup(void)
{
    float   energy[EDGE_MAX_PERSONS] = {10.0f, 9.0f, 0.3f, 0.2f};
    uint8_t sc[EDGE_MAX_PERSONS]     = {20, 21, 60, 61};  /* 20 & 21 adjacent */
    CHECK_EQ_U8("two strong but adjacent → 1 (dedup)",
                count_distinct_persons(energy, sc, EDGE_MAX_PERSONS), 1);
}

/* No signal at all → 0 persons (empty room). */
static void test_count_no_signal(void)
{
    float   energy[EDGE_MAX_PERSONS] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint8_t sc[EDGE_MAX_PERSONS]     = {10, 30, 50, 70};
    CHECK_EQ_U8("no signal → 0", count_distinct_persons(energy, sc, EDGE_MAX_PERSONS), 0);
}

/* Three genuine well-separated strong signatures → 3 (gate doesn't under-count). */
static void test_count_three_well_separated(void)
{
    float   energy[EDGE_MAX_PERSONS] = {10.0f, 9.0f, 8.0f, 0.2f};
    uint8_t sc[EDGE_MAX_PERSONS]     = {10, 50, 90, 11};
    CHECK_EQ_U8("three well-separated strong → 3",
                count_distinct_persons(energy, sc, EDGE_MAX_PERSONS), 3);
}

/* ──────────────────────────────────────────────────────────────────────
 *  #998 — person_count_debounce: a single noisy frame can't change the count
 * ────────────────────────────────────────────────────────────────────── */

static void test_debounce_rejects_transient_spike(void)
{
    uint8_t candidate = 1, streak = 0, stable = 1;  /* settled on 1 person */

    /* One spurious frame reports 4 — must NOT promote. */
    uint8_t out = person_count_debounce(4, &candidate, &streak, &stable);
    CHECK_EQ_U8("transient spike held at 1", out, 1);

    /* Back to 1 — resets pending change. */
    out = person_count_debounce(1, &candidate, &streak, &stable);
    CHECK_EQ_U8("recovered to 1", out, 1);
    CHECK_EQ_U8("streak reset", streak, 0);
}

static void test_debounce_accepts_sustained_change(void)
{
    uint8_t candidate = 1, streak = 0, stable = 1;

    uint8_t out = 1;
    /* A genuine 2-person arrival must hold EDGE_PERSON_PERSIST_FRAMES frames. */
    for (int i = 0; i < EDGE_PERSON_PERSIST_FRAMES; i++) {
        out = person_count_debounce(2, &candidate, &streak, &stable);
    }
    CHECK_EQ_U8("sustained 2 promoted", out, 2);
    CHECK_EQ_U8("stable now 2", stable, 2);
}

/* A flapping count (2,1,2,1,...) never accumulates a streak → stays at stable. */
static void test_debounce_flapping_stays_stable(void)
{
    uint8_t candidate = 1, streak = 0, stable = 1;
    uint8_t out = 1;
    for (int i = 0; i < 10; i++) {
        out = person_count_debounce((i & 1) ? 1 : 2, &candidate, &streak, &stable);
    }
    CHECK_EQ_U8("flapping count stays at 1", out, 1);
}

/* ──────────────────────────────────────────────────────────────────────
 *  #996 — presence_flag_update: dithering score must NOT flicker the flag
 * ────────────────────────────────────────────────────────────────────── */

/* Field trace dithers around the OLD single threshold while the person is
 * clearly present. With T_high=10, T_low=5, a score sequence that crosses 10
 * up and down must produce a STABLE flag (no per-frame flicker). */
static void test_presence_no_flicker_on_dither(void)
{
    const float threshold = 10.0f;  /* high threshold */
    /* Observed-style trace (issue evidence: 2.6-26.7), but here we model the
     * realistic "person present" case where the score mostly sits in/above the
     * dead band and only briefly dips. */
    float trace[] = {5.6f, 23.0f, 6.8f, 12.0f, 8.0f, 26.7f, 7.0f, 11.0f, 9.0f, 24.0f};
    int n = (int)(sizeof(trace) / sizeof(trace[0]));

    bool flag = false;
    uint8_t below = 0;
    int flips = 0;
    bool prev = flag;
    for (int i = 0; i < n; i++) {
        flag = presence_flag_update(flag, trace[i], threshold, &below);
        if (i > 0 && flag != prev) flips++;
        prev = flag;
    }
    /* First sample (5.6) is below T_low=5? No, 5.6 >= 5 → dead band, holds
     * initial false until 23.0 asserts. After that, dips to 6.8/8.0/7.0/9.0 are
     * all >= T_low (5), so they HOLD true. The only transition is the initial
     * false→true. No flicker. */
    CHECK_TRUE("presence asserted by end", flag);
    CHECK_TRUE("at most one transition (no flicker)", flips <= 1);
}

/* Hard dither straddling T_low must still not flicker frame-to-frame because of
 * the clear debounce: brief sub-T_low dips don't immediately clear. */
static void test_presence_clear_debounce_holds(void)
{
    const float threshold = 10.0f;  /* T_low = 5.0 */
    bool flag = false;
    uint8_t below = 0;

    /* Assert. */
    flag = presence_flag_update(flag, 20.0f, threshold, &below);
    CHECK_TRUE("asserted on strong score", flag);

    /* A few brief dips below T_low (< CLEAR_FRAMES) must NOT clear. */
    for (int i = 0; i < EDGE_PRESENCE_CLEAR_FRAMES - 1; i++) {
        flag = presence_flag_update(flag, 1.0f, threshold, &below);
    }
    CHECK_TRUE("brief dips below T_low still present", flag);

    /* Recovery resets the debounce. */
    flag = presence_flag_update(flag, 20.0f, threshold, &below);
    CHECK_TRUE("recovered", flag);
    CHECK_EQ_U8("below_count reset on recovery", below, 0);
}

/* A genuine departure (score drops and STAYS low) clears within the hold window. */
static void test_presence_genuine_departure_clears(void)
{
    const float threshold = 10.0f;
    bool flag = false;
    uint8_t below = 0;

    flag = presence_flag_update(flag, 20.0f, threshold, &below);
    CHECK_TRUE("asserted", flag);

    /* Person leaves: score stays well below T_low for CLEAR_FRAMES frames. */
    for (int i = 0; i < EDGE_PRESENCE_CLEAR_FRAMES; i++) {
        flag = presence_flag_update(flag, 0.5f, threshold, &below);
    }
    CHECK_TRUE("cleared after sustained low", !flag);
}

/* Schmitt gap: a score in the dead band (between T_low and T_high) holds state,
 * it neither asserts from false nor clears from true. */
static void test_presence_dead_band_holds_state(void)
{
    const float threshold = 10.0f;  /* dead band 5..10 */
    uint8_t below = 0;

    /* From false, a dead-band score does not assert. */
    bool flag = presence_flag_update(false, 7.0f, threshold, &below);
    CHECK_TRUE("dead band does not assert from false", !flag);

    /* From true, a dead-band score does not clear. */
    below = 0;
    flag = presence_flag_update(true, 7.0f, threshold, &below);
    CHECK_TRUE("dead band does not clear from true", flag);
}

/* ──────────────────────────────────────────────────────────────────────
 *  main
 * ────────────────────────────────────────────────────────────────────── */

int main(void)
{
    /* #998 person count gate */
    test_count_single_strong_signature();
    test_count_single_spread_multipath();
    test_count_two_well_separated();
    test_count_two_strong_adjacent_dedup();
    test_count_no_signal();
    test_count_three_well_separated();

    /* #998 count debounce */
    test_debounce_rejects_transient_spike();
    test_debounce_accepts_sustained_change();
    test_debounce_flapping_stays_stable();

    /* #996 presence hysteresis */
    test_presence_no_flicker_on_dither();
    test_presence_clear_debounce_holds();
    test_presence_genuine_departure_clears();
    test_presence_dead_band_holds_state();

    printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
