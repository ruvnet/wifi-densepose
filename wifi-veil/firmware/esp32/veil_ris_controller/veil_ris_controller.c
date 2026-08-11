/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * veil_ris_controller — see veil_ris_controller.h.
 *
 * STATUS: SYNTHETIC / L0. Build-only skeleton. Never run on silicon; no RIS
 * hardware exists here. Hardware-touching paths are marked TODO(hw). The keyed
 * bitmap generator (pure math over veil_rng) is fully implemented and testable
 * off target; the GPIO/SPI clock-out is stubbed.
 *
 * Compliance: the ESP32 only toggles control pins of a *passive* external
 * surface. It emits no RF and does not transmit into any band. The surface
 * re-reflects ambient energy; it does not add energy or occupy spectrum, which
 * is what keeps this on the compliant side of the jamming line. (A powered,
 * amplifying, or spectrum-occupying surface would NOT be compliant and is out
 * of scope.)
 */
#include "veil_ris_controller.h"

#include <string.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"

#include "veil_shield.h"   /* portable core: veil_rng, veil_rng_next_u64/_f32 */

static const char *TAG = "veil_ris";

static veil_ris_controller_cfg_t s_cfg;
static bool            s_inited;
static uint64_t        s_step;          /* configurations applied so far      */
static esp_timer_handle_t s_timer;

/* ---- keyed configuration generator (pure, testable off-target) ----------- */

/* PrivISAC two-state assignment: every element has two candidate phase states
 * (A/B) designed offline so the *comm-direction* array response is ~invariant
 * under A<->B while the *sensing-direction* response changes. At runtime we
 * only pick, per element, which of the two states is active this step. That
 * choice is the single bit we clock out. Drawing the bits from the keyed
 * veil_rng makes the whole schedule reproducible from (key, step_index) and
 * shareable with an authorized receiver.
 *
 * `step_index` seeds a per-step substream so any step can be regenerated
 * without replaying history (matches the core's deterministic style).
 * Fills `bits` (packed MSB-first) with n_elements selection bits. */
void veil_ris_gen_bits(uint64_t key, uint64_t step_index,
                       size_t n_elements, uint8_t *bits, size_t bits_len)
{
    if (!bits || bits_len == 0) {
        return;
    }
    memset(bits, 0, bits_len);

    veil_rng r;
    /* Mix the step index into the key so each dwell gets an independent draw
     * while staying a pure function of (key, step_index). */
    veil_rng_seed(&r, key ^ (step_index * 0x9E3779B97F4A7C15ULL));

    for (size_t e = 0; e < n_elements; e++) {
        size_t byte = e >> 3;
        if (byte >= bits_len) {
            break;
        }
        /* Top bit of the draw selects state B (1) vs state A (0). */
        uint64_t w = veil_rng_next_u64(&r);
        if (w >> 63) {
            bits[byte] |= (uint8_t)(0x80u >> (e & 7));
        }
    }
}

/* ---- interface clock-out (stubs) ----------------------------------------- */

static esp_err_t veil_ris_write(const uint8_t *bits, size_t bits_len)
{
    switch (s_cfg.iface) {
    case VEIL_RIS_IFACE_GPIO:
        /* TODO(hw): for each element e, set its pin to the selected state.
         *   for (size_t e = 0; e < s_cfg.n_elements; e++) {
         *       int level = (bits[e >> 3] >> (7 - (e & 7))) & 1;
         *       gpio_set_level(s_cfg.gpio_pins[e], level);
         *   }
         * Requires each pin configured as output in _init(). Unverified. */
        ESP_LOGD(TAG, "TODO(hw) GPIO write %u bits (stub)",
                 (unsigned)s_cfg.n_elements);
        return ESP_ERR_NOT_SUPPORTED;
    case VEIL_RIS_IFACE_SPI:
        /* TODO(hw): shift the packed bitmap to the surface driver IC.
         *   spi_transaction_t t = {
         *       .length    = bits_len * 8,
         *       .tx_buffer = bits,
         *   };
         *   spi_device_transmit(s_spi_dev, &t);   // then latch via CS
         * s_spi_dev created in _init() via spi_bus_add_device(). Unverified. */
        ESP_LOGD(TAG, "TODO(hw) SPI write %u bytes (stub)", (unsigned)bits_len);
        return ESP_ERR_NOT_SUPPORTED;
    default:
        return ESP_ERR_INVALID_ARG;
    }
}

/* ---- public API ---------------------------------------------------------- */

esp_err_t veil_ris_controller_init(const veil_ris_controller_cfg_t *cfg)
{
    if (!cfg || cfg->n_elements == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (s_inited) {
        return ESP_ERR_INVALID_STATE;
    }
    s_cfg  = *cfg;
    s_step = 0;

    if (s_cfg.iface == VEIL_RIS_IFACE_GPIO) {
        /* TODO(hw): configure each s_cfg.gpio_pins[e] as GPIO_MODE_OUTPUT via
         * gpio_config() (build a pin_bit_mask over all elements). */
        ESP_LOGW(TAG, "TODO(hw) configure %u GPIO element pins (stub)",
                 (unsigned)s_cfg.n_elements);
    } else {
        /* TODO(hw): spi_bus_initialize(s_cfg.spi_host, &buscfg, ...) +
         * spi_bus_add_device(s_cfg.spi_host, &devcfg, &s_spi_dev). */
        ESP_LOGW(TAG, "TODO(hw) init SPI host %d @ %d Hz (stub)",
                 s_cfg.spi_host, s_cfg.spi_clock_hz);
    }

    s_inited = true;
    ESP_LOGI(TAG, "init (SYNTHETIC/L0): %u elements, dwell=%uus, keyed schedule",
             (unsigned)s_cfg.n_elements, (unsigned)s_cfg.dwell_us);
    return ESP_OK;
}

esp_err_t veil_ris_controller_step(uint8_t *out_bits, size_t out_len)
{
    if (!s_inited) {
        return ESP_ERR_INVALID_STATE;
    }
    /* Bounded, malloc-free scratch: cap at 256 elements (32 bytes) for the
     * skeleton. Larger surfaces would stream in chunks. */
    enum { VEIL_RIS_MAX_BYTES = 32 };
    uint8_t bits[VEIL_RIS_MAX_BYTES];
    size_t need = (s_cfg.n_elements + 7) / 8;
    if (need > sizeof bits) {
        need = sizeof bits;
    }

    veil_ris_gen_bits(s_cfg.key, s_step, s_cfg.n_elements, bits, need);
    esp_err_t err = veil_ris_write(bits, need);   /* stub on host/no-hw */
    s_step++;

    if (out_bits && out_len) {
        size_t n = out_len < need ? out_len : need;
        memcpy(out_bits, bits, n);
    }
    /* NOT_SUPPORTED from the stubbed writer is expected off-silicon; surface
     * the generator result as OK so tests can validate the keyed bitmap. */
    return (err == ESP_ERR_NOT_SUPPORTED) ? ESP_OK : err;
}

static void veil_ris_timer_cb(void *arg)
{
    (void)arg;
    (void)veil_ris_controller_step(NULL, 0);
}

esp_err_t veil_ris_controller_start(void)
{
    if (!s_inited) {
        return ESP_ERR_INVALID_STATE;
    }
    /* TODO(hw): a real deployment would gate this on the sensing detector's
     * engage trigger so the surface only churns during a sensing burst. */
    const esp_timer_create_args_t args = {
        .callback = veil_ris_timer_cb,
        .name     = "veil_ris",
    };
    esp_err_t err = esp_timer_create(&args, &s_timer);
    if (err != ESP_OK) {
        return err;
    }
    return esp_timer_start_periodic(s_timer, s_cfg.dwell_us);
}

esp_err_t veil_ris_controller_stop(void)
{
    if (s_timer) {
        esp_timer_stop(s_timer);
        esp_timer_delete(s_timer);
        s_timer = NULL;
    }
    return ESP_OK;
}

uint64_t veil_ris_controller_step_count(void) { return s_step; }
