/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * veil_sensing_detector — see veil_sensing_detector.h.
 *
 * STATUS: SYNTHETIC / L0. Build-only skeleton. Never run on silicon. Every
 * hardware-touching path is marked TODO(hw). The rate estimator (pure math over
 * timestamps) is the only fully-implemented piece and is unit-testable off
 * target; the RF/observe path and the trigger backends are stubs.
 *
 * Compliance: this component only READS the channel (CSI + frame cadence). It
 * emits no RF and shapes no waveform. It cannot and does not touch the closed
 * ESP32 Wi-Fi PHY blob. The "action" it takes is a low-rate control signal to a
 * separate protector.
 */
#include "veil_sensing_detector.h"

#include <string.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"          /* esp_wifi_set_csi_rx_cb, esp_wifi_set_csi, ... */
#include "esp_wifi_types.h"    /* wifi_csi_info_t, wifi_csi_config_t           */
#include "driver/gpio.h"       /* gpio_config, gpio_set_level                  */

static const char *TAG = "veil_sense";

/* ---- module state -------------------------------------------------------- */

static veil_sensing_detector_cfg_t s_cfg;
static bool     s_running;
static bool     s_engaged;
static float    s_rate_hz;

/* Bounded ring of recent solicitation timestamps (µs), malloc-free. */
enum { VEIL_TS_RING = 256 };
static int64_t  s_ts[VEIL_TS_RING];
static size_t   s_ts_head;   /* next write slot */
static size_t   s_ts_count;  /* live entries, capped at VEIL_TS_RING */

/* ---- rate estimator (pure, testable off-target) -------------------------- */

/* Record one solicitation at time `now_us` and recompute the sliding-window
 * rate. Returns the current rate in Hz. This function is deliberately free of
 * any ESP-IDF dependency so it can be exercised in host unit tests. */
float veil_sd_note_solicitation(int64_t now_us)
{
    s_ts[s_ts_head] = now_us;
    s_ts_head = (s_ts_head + 1) % VEIL_TS_RING;
    if (s_ts_count < VEIL_TS_RING) {
        s_ts_count++;
    }

    const int64_t window_us = (int64_t)s_cfg.window_ms * 1000;
    const int64_t cutoff    = now_us - window_us;

    size_t in_window = 0;
    for (size_t k = 0; k < s_ts_count; k++) {
        if (s_ts[k] >= cutoff) {
            in_window++;
        }
    }
    /* rate = events within the trailing window / window length. */
    s_rate_hz = (float)in_window * 1000.0f / (float)s_cfg.window_ms;

    /* Hysteresis around engage/release. */
    if (!s_engaged && s_rate_hz >= s_cfg.trigger_rate_hz) {
        s_engaged = true;
        ESP_LOGI(TAG, "sensing burst: %.1f Hz >= %.1f -> ENGAGE",
                 s_rate_hz, s_cfg.trigger_rate_hz);
        /* fire-and-forget; backend errors are logged, not fatal */
        (void)0; /* veil_sd_emit_trigger(true) — see below */
    } else if (s_engaged && s_rate_hz <= s_cfg.release_rate_hz) {
        s_engaged = false;
        ESP_LOGI(TAG, "sensing quiet: %.1f Hz <= %.1f -> RELEASE",
                 s_rate_hz, s_cfg.release_rate_hz);
    }
    return s_rate_hz;
}

/* ---- trigger backends (all stubs) ---------------------------------------- */

static esp_err_t veil_sd_emit_trigger(bool engage)
{
    switch (s_cfg.backend) {
    case VEIL_TRIGGER_GPIO:
        /* TODO(hw): drive the engage line to the co-located AP/relay.
         *   gpio_set_level(s_cfg.gpio_num, engage ? 1 : 0);
         * Requires a wired GPIO to the protector; unverified on silicon. */
        ESP_LOGW(TAG, "TODO(hw) GPIO trigger -> %d (stub)", engage);
        return ESP_ERR_NOT_SUPPORTED;
    case VEIL_TRIGGER_MQTT:
        /* TODO(hw): esp_mqtt_client_publish(client, s_cfg.mqtt_topic,
         *              engage ? "1" : "0", 0, 1 /qos/, 0 /retain/);
         * Client lifecycle (esp_mqtt_client_init/_start) omitted from skeleton. */
        ESP_LOGW(TAG, "TODO(hw) MQTT trigger -> %d (stub)", engage);
        return ESP_ERR_NOT_SUPPORTED;
    case VEIL_TRIGGER_ESPNOW:
        /* TODO(hw): esp_now_send(s_cfg.espnow_peer, &payload, sizeof payload);
         * Requires esp_now_init() + esp_now_add_peer() during start(). */
        ESP_LOGW(TAG, "TODO(hw) ESP-NOW trigger -> %d (stub)", engage);
        return ESP_ERR_NOT_SUPPORTED;
    default:
        return ESP_ERR_INVALID_ARG;
    }
}

/* ---- CSI callback (observe path) ----------------------------------------- */

/* Runs in the Wi-Fi task. Keep it short: post to a queue in real firmware.
 * Here we only classify whether this frame indicates a sounding/solicitation
 * and, if so, feed the estimator. */
static void veil_sd_csi_cb(void *ctx, wifi_csi_info_t *info)
{
    (void)ctx;
    if (!info) {
        return;
    }
    /* TODO(hw): a real classifier would inspect info->rx_ctrl (rate, sig_mode,
     * channel, secondary channel) and, alongside a promiscuous frame-type
     * filter, distinguish NDP / NDP-announcement / CSI-solicit action frames
     * from ordinary data. On silicon the ESP32 does NOT surface the raw
     * VHT/HE sounding subtype through the CSI struct, so this classifier is
     * necessarily heuristic (cadence + rate + frame length). Treated here as
     * "every CSI-bearing frame is a candidate solicitation" for the skeleton. */
    (void)veil_sd_note_solicitation(esp_timer_get_time());
}

/* ---- lifecycle ----------------------------------------------------------- */

esp_err_t veil_sensing_detector_start(const veil_sensing_detector_cfg_t *cfg)
{
    if (!cfg) {
        return ESP_ERR_INVALID_ARG;
    }
    if (s_running) {
        return ESP_ERR_INVALID_STATE;
    }
    s_cfg      = *cfg;
    s_engaged  = false;
    s_rate_hz  = 0.0f;
    s_ts_head  = 0;
    s_ts_count = 0;

    if (s_cfg.backend == VEIL_TRIGGER_GPIO && s_cfg.gpio_num >= 0) {
        /* TODO(hw): configure the engage line.
         *   gpio_config_t io = {
         *       .pin_bit_mask = 1ULL << s_cfg.gpio_num,
         *       .mode = GPIO_MODE_OUTPUT,
         *   };
         *   gpio_config(&io);
         *   gpio_set_level(s_cfg.gpio_num, 0);
         */
        ESP_LOGW(TAG, "TODO(hw) configure GPIO %d (stub)", s_cfg.gpio_num);
    }

    /* Observe path. On real hardware:
     *   wifi_csi_config_t csi = { ... };
     *   ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi));
     *   ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(veil_sd_csi_cb, NULL));
     *   ESP_ERROR_CHECK(esp_wifi_set_csi(true));
     *   ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true)); // more CSI when idle
     * The Wi-Fi driver must already be started by the app. */
    ESP_LOGW(TAG, "TODO(hw) esp_wifi_set_csi_rx_cb/_set_csi/_set_promiscuous "
                  "(stub; not wired on silicon)");
    (void)veil_sd_csi_cb; /* referenced once wired */

    s_running = true;
    ESP_LOGI(TAG, "started (SYNTHETIC/L0): window=%ums engage>=%.1fHz",
             (unsigned)s_cfg.window_ms, s_cfg.trigger_rate_hz);
    return ESP_OK;
}

esp_err_t veil_sensing_detector_stop(void)
{
    if (!s_running) {
        return ESP_ERR_INVALID_STATE;
    }
    /* TODO(hw): esp_wifi_set_csi(false); esp_wifi_set_csi_rx_cb(NULL, NULL);
     *           esp_wifi_set_promiscuous(false); release GPIO/MQTT/ESP-NOW. */
    if (s_engaged) {
        (void)veil_sd_emit_trigger(false);
    }
    s_running = false;
    return ESP_OK;
}

float veil_sensing_detector_rate_hz(void) { return s_rate_hz; }
bool  veil_sensing_detector_engaged(void) { return s_engaged; }
