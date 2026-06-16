/**
 * @file display_task.h
 * @brief ADR-045: FreeRTOS display task — live graph loop.
 */

#ifndef DISPLAY_TASK_H
#define DISPLAY_TASK_H

#include "esp_err.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Start the live graph display task on Core 1, priority 6.
 *
 * Probes for the active target's LCD hardware. If the LCD is absent, logs a
 * warning and returns ESP_OK (graceful skip). If display init succeeds but
 * the raw heartbeat task cannot be created, returns an error so the caller
 * can log the real fault.
 *
 * @return ESP_OK on skip or success; error on display init/task failure.
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
