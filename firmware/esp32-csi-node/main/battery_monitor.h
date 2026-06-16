#ifndef BATTERY_MONITOR_H
#define BATTERY_MONITOR_H

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

typedef enum {
    BATTERY_POWER_UNKNOWN = 0,
    BATTERY_POWER_BATTERY = 1,
    BATTERY_POWER_CHARGING = 2,
} battery_power_status_t;

typedef struct {
    bool valid;
    uint16_t millivolts;
    uint8_t percent;
    bool charging;
    battery_power_status_t status;
} battery_status_t;

esp_err_t battery_monitor_init(void);
esp_err_t battery_monitor_read(battery_status_t *out);
const char *battery_monitor_status_name(battery_power_status_t status);

#endif /* BATTERY_MONITOR_H */
