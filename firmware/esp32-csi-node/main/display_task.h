/**
 * @file display_task.h
 * @brief ADR-045: FreeRTOS display task — live graph loop.
 */

#ifndef DISPLAY_TASK_H
#define DISPLAY_TASK_H

#include "esp_err.h"

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

#ifdef __cplusplus
}
#endif

#endif /* DISPLAY_TASK_H */
