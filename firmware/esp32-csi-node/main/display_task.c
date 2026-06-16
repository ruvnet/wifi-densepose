/**
 * @file display_task.c
 * @brief ADR-045: FreeRTOS display heartbeat task - live graph loop.
 *
 * Gracefully skips if the active target's LCD hardware is absent.
 */

#include "display_task.h"
#include "sdkconfig.h"

/* Set true once an AMOLED panel is detected and the display task starts.
 * Defined outside the CONFIG_DISPLAY_ENABLE guard so display_is_active()
 * exists on headless builds too (where it stays false → CSI captures DATA
 * frames; see RuView#893). */
static bool s_display_active = false;

bool display_is_active(void) { return s_display_active; }

#if CONFIG_DISPLAY_ENABLE

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "battery_monitor.h"
#include "csi_collector.h"
#include "edge_processing.h"
#include "display_hal.h"

static const char *TAG = "disp_task";

/* ---- Config ---- */
#define DISP_HEARTBEAT_PERIOD_MS  250

#define DISP_TASK_STACK      (12 * 1024)
#define DISP_TASK_PRIORITY   6
#define DISP_TASK_CORE       1

/* A live graph keeps the display useful without falling back to the full UI. */
#define DISP_GRAPH_TOP_BAND_H     8
#define DISP_GRAPH_BOTTOM_BAND_H  8
#define DISP_GRAPH_MIN_VISIBLE     2

static uint16_t s_row[DISPLAY_PANEL_H_RES];
static uint8_t s_motion_history[DISPLAY_PANEL_H_RES];
static uint16_t s_history_head;

static int clamp_int(int value, int lo, int hi)
{
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static int motion_sample_from_vitals(const edge_vitals_pkt_t *vitals, bool has_vitals)
{
    if (!has_vitals) {
        return 0;
    }

    int motion = (int)(vitals->motion_energy * 18.0f);
    if (motion < 0) motion = 0;
    if (motion > 100) motion = 100;

    /* Presence gives a little more lift when motion is weak. */
    if (motion < DISP_GRAPH_MIN_VISIBLE && vitals->presence_score > 0.0f) {
        motion = (int)(vitals->presence_score * 6.0f);
    }

    return clamp_int(motion, 0, 100);
}

static int rssi_fill_from_vitals(const edge_vitals_pkt_t *vitals, bool has_vitals)
{
    if (!has_vitals) {
        return 0;
    }

    /* Map -100..-40 dBm to 0..100% so the bar stays readable. */
    int fill = ((int)vitals->rssi + 100) * 100 / 60;
    return clamp_int(fill, 0, 100);
}

static void render_graph_frame(int graph_sample, int battery_fill, int rssi_fill)
{
    const int width = DISPLAY_PANEL_H_RES;
    const int height = DISPLAY_PANEL_V_RES;
    const int plot_top = DISP_GRAPH_TOP_BAND_H;
    const int plot_bottom = height - DISP_GRAPH_BOTTOM_BAND_H - 1;
    const int plot_height = plot_bottom - plot_top + 1;
    s_motion_history[s_history_head] = (uint8_t)graph_sample;
    s_history_head = (uint16_t)((s_history_head + 1U) % DISPLAY_PANEL_H_RES);
    const uint16_t history_base = s_history_head;

    for (int y = 0; y < height; y++) {
        memset(s_row, 0, sizeof(s_row));

        if (y < DISP_GRAPH_TOP_BAND_H) {
            int fill = (battery_fill * width) / 100;
            fill = clamp_int(fill, 0, width);
            for (int x = 0; x < fill; x++) {
                s_row[x] = 0xFFFF;
            }
        } else if (y >= height - DISP_GRAPH_BOTTOM_BAND_H) {
            int fill = (rssi_fill * width) / 100;
            fill = clamp_int(fill, 0, width);
            for (int x = 0; x < fill; x++) {
                s_row[x] = 0xFFFF;
            }
        } else {
            const int threshold = plot_bottom - y + 1;
            for (int x = 0; x < width; x++) {
                int idx = (history_base + x) % width;
                int sample_height = (s_motion_history[idx] * plot_height) / 100;
                if (sample_height >= threshold) {
                    s_row[x] = 0xFFFF;
                }
            }
        }

        display_hal_draw(0, y, width, y + 1, s_row);
    }
}

/* ---- Display task ---- */
static void display_task(void *arg)
{
    (void)arg;

    const TickType_t frame_period = pdMS_TO_TICKS(DISP_HEARTBEAT_PERIOD_MS);
    TickType_t last_wake = xTaskGetTickCount();
    uint32_t frame = 0;

    ESP_LOGI(TAG, "Display graph task running on Core %d, %u ms period",
             xPortGetCoreID(), (unsigned)DISP_HEARTBEAT_PERIOD_MS);

    while (1) {
        frame++;

        edge_vitals_pkt_t vitals;
        bool has_vitals = edge_get_vitals(&vitals);

        battery_status_t battery;
        bool has_battery = (battery_monitor_read(&battery) == ESP_OK && battery.valid);

        int motion_sample = motion_sample_from_vitals(&vitals, has_vitals);
        int battery_fill = has_battery ? (int)battery.percent : 0;
        int rssi_fill = has_vitals ? rssi_fill_from_vitals(&vitals, true) : 0;
        int graph_sample = motion_sample;
        if (graph_sample < DISP_GRAPH_MIN_VISIBLE) {
            graph_sample = rssi_fill / 2;
        }

        render_graph_frame(graph_sample, battery_fill, rssi_fill);

        ESP_LOGI(TAG, "Display frame %lu before present (graph=%d motion=%d battery=%u%% rssi=%d node=%u)",
                 (unsigned long)frame,
                 graph_sample,
                 motion_sample,
                 has_battery ? (unsigned)battery.percent : 0U,
                 has_vitals ? (int)vitals.rssi : 0,
                 (unsigned)csi_collector_get_node_id());
        display_hal_present();
        ESP_LOGI(TAG, "Display frame %lu after present", (unsigned long)frame);

        if (frame == 1) {
            ESP_LOGI(TAG, "Display first graph frame rendered");
        } else if ((frame % 100U) == 0U) {
            ESP_LOGI(TAG, "Display graph heartbeat frame %lu graph=%d motion=%d battery=%u%% rssi=%d",
                     (unsigned long)frame, graph_sample, motion_sample,
                     has_battery ? (unsigned)battery.percent : 0U,
                     has_vitals ? (int)vitals.rssi : 0);
        }

        vTaskDelayUntil(&last_wake, frame_period);
    }
}

/* ---- Public API ---- */

esp_err_t display_task_start(void)
{
    ESP_LOGI(TAG, "Initializing display subsystem...");

    /* Probe display hardware */
    esp_err_t ret = display_hal_init_panel();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Display not available - running headless");
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Display panel initialized, starting graph loop");

    BaseType_t xret = xTaskCreatePinnedToCore(
        display_task, "display", DISP_TASK_STACK,
        NULL, DISP_TASK_PRIORITY, NULL, DISP_TASK_CORE);

    if (xret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create display task");
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "Display task started (Core %d, priority %d, %u ms heartbeat, %d fps)",
             DISP_TASK_CORE, DISP_TASK_PRIORITY,
             (unsigned)DISP_HEARTBEAT_PERIOD_MS, DISP_FPS_LIMIT);
    s_display_active = true;
    return ESP_OK;
}

#else /* !CONFIG_DISPLAY_ENABLE */

esp_err_t display_task_start(void)
{
    return ESP_OK;
}

#endif /* CONFIG_DISPLAY_ENABLE */
