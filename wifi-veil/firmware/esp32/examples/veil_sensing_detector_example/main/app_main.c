/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * SYNTHETIC / L0 — build-only. Exercises veil_sensing_detector's public API
 * against the GPIO trigger backend so the component compiles and links on a
 * real ESP-IDF toolchain. Never flashed; no CSI is ever captured. Per the
 * component README, do not treat a successful build as a runtime or on-air
 * claim.
 */
#include "esp_log.h"
#include "veil_sensing_detector.h"

static const char *TAG = "veil_sensing_detector_example";

void app_main(void)
{
    veil_sensing_detector_cfg_t cfg = VEIL_SENSING_DETECTOR_DEFAULT_CFG();
    cfg.backend  = VEIL_TRIGGER_GPIO;
    cfg.gpio_num = 8;

    ESP_ERROR_CHECK(veil_sensing_detector_start(&cfg));
    ESP_LOGI(TAG, "rate_hz=%.2f engaged=%d (build-only, never flashed)",
             veil_sensing_detector_rate_hz(), veil_sensing_detector_engaged());
}
