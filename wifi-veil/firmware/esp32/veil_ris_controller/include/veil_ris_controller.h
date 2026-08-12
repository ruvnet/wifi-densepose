/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * veil_ris_controller — drive an EXTERNAL reconfigurable intelligent surface
 * (RIS) to obfuscate the sensing-direction channel.
 *
 * STATUS: SYNTHETIC / L0. Build-only ESP-IDF component skeleton. Never flashed,
 * never captured on silicon. No RIS hardware exists in this repo. Do NOT claim
 * runtime or on-air behavior without a captured hardware log.
 *
 * WHY THIS EXISTS (honest framing): the ESP32 cannot shape its own transmitted
 * beamforming feedback — the precoding / compressed-BF-report path lives in the
 * closed Espressif Wi-Fi PHY blob (esp-phy-lib) and is not modifiable (see
 * README.md). The legitimate, compliant way an ESP32 can "help scramble" a
 * sensing signal is to act as the *controller for a separate passive surface*:
 * a RIS whose per-element phase states are switched over time. Following the
 * PrivISAC pattern (arXiv:2601.04488), each element is toggled between two
 * states chosen so the surface's response is ~identical in the *communication*
 * direction (throughput preserved) but differs sharply in the *sensing*
 * direction (an eavesdropper's channel is perturbed). The ESP32 is a GPIO/SPI
 * pin-driver here; it emits no RF of its own.
 *
 * The state schedule is *keyed* and deterministic: it is drawn from the
 * portable core's `veil_rng` (SplitMix64), so an associated / authorized
 * sensor holding the same key can reconstruct — and thus tolerate — the
 * schedule, while an unauthorized observer cannot.
 */
#ifndef VEIL_RIS_CONTROLLER_H
#define VEIL_RIS_CONTROLLER_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* How the surface's element bits are clocked out. */
typedef enum {
    VEIL_RIS_IFACE_GPIO = 0, /* small surfaces: one GPIO per element / bank */
    VEIL_RIS_IFACE_SPI,      /* larger surfaces: shift-register / driver IC */
} veil_ris_iface_t;

typedef struct {
    veil_ris_iface_t iface;

    /* Number of independently switchable RIS elements (or 1-bit banks). */
    size_t   n_elements;

    /* Keyed, deterministic schedule (shared with the associated receiver). */
    uint64_t key;

    /* Dwell time per configuration, microseconds. Must be short vs. the
     * channel coherence time to spread perturbation across the sensing burst,
     * yet long enough for the surface's switching diodes to settle. */
    uint32_t dwell_us;

    /* GPIO backend: one pin per element (n_elements <= number of pins). */
    const int *gpio_pins;    /* borrowed; length == n_elements               */

    /* SPI backend: bits are packed MSB-first into ceil(n_elements/8) bytes and
     * shifted out per configuration. */
    int      spi_host;       /* e.g. SPI2_HOST                               */
    int      spi_cs_gpio;    /* latch / chip-select                          */
    int      spi_clock_hz;   /* driver-IC clock                             */
} veil_ris_controller_cfg_t;

/* Initialize the chosen interface. Registration only — says nothing about a
 * physical surface actually switching. */
esp_err_t veil_ris_controller_init(const veil_ris_controller_cfg_t *cfg);

/* Compute the next keyed configuration bitmap and clock it to the surface.
 * `out_bits` (optional, may be NULL) receives the packed bitmap for tests.
 * `out_len` is the byte length of `out_bits` on input. The bit pattern is
 * derived purely from `veil_rng` + the PrivISAC two-state assignment, so it is
 * reproducible from (key, step_index). */
esp_err_t veil_ris_controller_step(uint8_t *out_bits, size_t out_len);

/* Start/stop a periodic timer that calls _step() every dwell_us. */
esp_err_t veil_ris_controller_start(void);
esp_err_t veil_ris_controller_stop(void);

/* Monotonic count of configurations applied since init (telemetry/tests). */
uint64_t veil_ris_controller_step_count(void);

#ifdef __cplusplus
}
#endif

#endif /* VEIL_RIS_CONTROLLER_H */
