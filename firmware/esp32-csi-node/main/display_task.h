/**
 * @file display_task.h
 * @brief ADR-045: FreeRTOS display task — LVGL pump on Core 0.
 */

#ifndef DISPLAY_TASK_H
#define DISPLAY_TASK_H

#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Start the display task on Core 0, priority 1.
 *
 * Probes for RM67162 panel and SPIRAM. If either is absent,
 * logs a warning and returns ESP_OK (graceful skip).
 *
 * @return ESP_OK always (display is optional).
 */
esp_err_t display_task_start(void);

/**
 * @return true once an AMOLED panel has been detected and the display task
 * is running; false on headless boards (no panel, or built without display
 * support). Used to choose the CSI promiscuous filter (RuView#893): a board
 * with no display has no QSPI/SPI-flash contention, so it can safely capture
 * DATA frames for proper CSI yield instead of starving on MGMT-only.
 */
bool display_is_active(void);

#ifdef __cplusplus
}
#endif

#endif /* DISPLAY_TASK_H */
