/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * veil_shield — portable C core of the VEIL compliant-waveform privacy shield
 * (ADR-288 / ADR-290). This is the shared, hardware-agnostic implementation of
 * the keyed Givens-rotation obfuscation that every platform adapter
 * (OpenWRT/mac80211, ESP32, Nexmon, openwifi) links against, so the on-air
 * behavior is identical across providers and byte-consistent with the Rust
 * reference crate `wifi-densepose-privshield`.
 *
 * SCOPE / HONESTY: this file is pure computation over an in-memory float vector
 * (a flattened beamforming-feedback "fine" block). It does NOT touch a radio,
 * emit RF, or read hardware. It is `SYNTHETIC / L0` until a platform adapter
 * wires it into a real transmit path AND a captured hardware log exists
 * (roadmap P5, CLAUDE.md). It is `no_std`-friendly C99: no malloc, no libc I/O,
 * only <math.h> (sinf/cosf/sqrtf).
 *
 * Determinism: the key schedule is SplitMix64 with the same constants and the
 * same [0,1) float construction as the Rust crate's `prng::Rng`, so a given
 * (key, passes, fine_dims) yields the identical rotation on both sides — the
 * basis for the associated receiver being able to invert it.
 */
#ifndef VEIL_SHIELD_H
#define VEIL_SHIELD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Deterministic SplitMix64 stream (matches Rust `prng::Rng`). */
typedef struct {
    uint64_t state;
} veil_rng;

/* Seed a stream. Distinct seeds yield independent streams. */
void veil_rng_seed(veil_rng *r, uint64_t seed);

/* Next raw 64-bit word. */
uint64_t veil_rng_next_u64(veil_rng *r);

/* Uniform float in [0, 1) using the top 24 bits (matches Rust `next_f32`). */
float veil_rng_next_f32(veil_rng *r);

/* Apply the keyed rotation to the fine block `fine[0..n)` in place.
 * `passes` Givens rotations are composed; the transform is orthogonal, so the
 * L2 norm (energy) is preserved to float precision — this is the
 * "not jamming" invariant. */
void veil_shield_apply(float *fine, size_t n, uint64_t key, size_t passes);

/* Invert the keyed rotation (associated receiver, holding the shared key).
 * `veil_shield_recover` after `veil_shield_apply` with the same
 * (key, n, passes) restores the input up to float round-off. */
void veil_shield_recover(float *fine, size_t n, uint64_t key, size_t passes);

/* Convenience: L2 norm of a vector (for the energy-conservation check). */
float veil_l2_norm(const float *v, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* VEIL_SHIELD_H */
