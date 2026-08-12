/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * SYNTHETIC / L0 — build-only. Exercises veil_ris_controller's public API
 * against a GPIO-backed config so the component compiles and links on a real
 * ESP-IDF toolchain. Never flashed; no physical RIS exists. Per the component
 * README, do not treat a successful build as a runtime or on-air claim.
 */
#include "esp_log.h"
#include "veil_ris_controller.h"

static const char *TAG = "veil_ris_controller_example";
static const int kRisPins[4] = {4, 5, 6, 7};

void app_main(void)
{
    veil_ris_controller_cfg_t cfg = {
        .iface     = VEIL_RIS_IFACE_GPIO,
        .n_elements = 4,
        .key       = 0x5EED5EED5EED5EEDULL,
        .dwell_us  = 500,
        .gpio_pins = kRisPins,
        .spi_host  = -1,
        .spi_cs_gpio = -1,
        .spi_clock_hz = 0,
    };

    ESP_ERROR_CHECK(veil_ris_controller_init(&cfg));
    ESP_ERROR_CHECK(veil_ris_controller_step(NULL, 0));
    ESP_LOGI(TAG, "step_count=%llu (build-only, never flashed)",
             (unsigned long long)veil_ris_controller_step_count());
}
