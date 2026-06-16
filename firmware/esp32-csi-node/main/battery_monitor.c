#include "battery_monitor.h"
#include "sdkconfig.h"

#if CONFIG_BATTERY_MONITOR_ENABLE

#include <string.h>
#include "driver/gpio.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"

static const char *TAG = "battery";

static bool s_init_attempted;
static bool s_ready;
static adc_oneshot_unit_handle_t s_adc_unit;
static adc_cali_handle_t s_cali;
static bool s_cali_ready;
static adc_channel_t s_channel;

static uint8_t percent_from_mv(uint16_t mv)
{
    if (mv >= 4200) return 100;
    if (mv <= 3300) return 0;
    if (mv >= 3900) return (uint8_t)(70 + ((uint32_t)(mv - 3900) * 30U) / 300U);
    if (mv >= 3700) return (uint8_t)(35 + ((uint32_t)(mv - 3700) * 35U) / 200U);
    if (mv >= 3500) return (uint8_t)(10 + ((uint32_t)(mv - 3500) * 25U) / 200U);
    return (uint8_t)(((uint32_t)(mv - 3300) * 10U) / 200U);
}

const char *battery_monitor_status_name(battery_power_status_t status)
{
    switch (status) {
    case BATTERY_POWER_BATTERY:
        return "BATTERY";
    case BATTERY_POWER_CHARGING:
        return "CHARGING";
    case BATTERY_POWER_UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

esp_err_t battery_monitor_init(void)
{
    if (s_ready) return ESP_OK;
    if (s_init_attempted) return ESP_ERR_INVALID_STATE;
    s_init_attempted = true;

    adc_unit_t unit_id;
    esp_err_t err = adc_oneshot_io_to_channel(CONFIG_BATTERY_ADC_GPIO, &unit_id, &s_channel);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "GPIO%d is not ADC-capable: %s", CONFIG_BATTERY_ADC_GPIO, esp_err_to_name(err));
        return err;
    }

    adc_oneshot_unit_init_cfg_t unit_cfg = {
        .unit_id = unit_id,
    };
    err = adc_oneshot_new_unit(&unit_cfg, &s_adc_unit);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "ADC unit init failed: %s", esp_err_to_name(err));
        return err;
    }

    adc_oneshot_chan_cfg_t chan_cfg = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten = ADC_ATTEN_DB_12,
    };
    err = adc_oneshot_config_channel(s_adc_unit, s_channel, &chan_cfg);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "ADC channel init failed: %s", esp_err_to_name(err));
        return err;
    }

#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    adc_cali_curve_fitting_config_t cali_cfg = {
        .unit_id = unit_id,
        .chan = s_channel,
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    s_cali_ready = (adc_cali_create_scheme_curve_fitting(&cali_cfg, &s_cali) == ESP_OK);
#endif

#if CONFIG_BATTERY_CHARGE_GPIO >= 0
    gpio_config_t gpio_cfg = {
        .pin_bit_mask = 1ULL << CONFIG_BATTERY_CHARGE_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&gpio_cfg);
#endif

    s_ready = true;
    ESP_LOGI(TAG, "battery monitor on GPIO%d ADC unit=%d channel=%d divider=%d/1000",
             CONFIG_BATTERY_ADC_GPIO, (int)unit_id, (int)s_channel,
             CONFIG_BATTERY_ADC_DIVIDER_MILLI);
    return ESP_OK;
}

esp_err_t battery_monitor_read(battery_status_t *out)
{
    if (!out) return ESP_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    out->status = BATTERY_POWER_UNKNOWN;

    if (!s_ready) {
        esp_err_t init_err = battery_monitor_init();
        if (init_err != ESP_OK && !s_ready) return init_err;
    }

    int raw = 0;
    esp_err_t err = adc_oneshot_read(s_adc_unit, s_channel, &raw);
    if (err != ESP_OK) return err;

    int adc_mv = 0;
    if (s_cali_ready) {
        err = adc_cali_raw_to_voltage(s_cali, raw, &adc_mv);
        if (err != ESP_OK) return err;
    } else {
        adc_mv = (raw * 3300) / 4095;
    }

    uint32_t pack_mv = ((uint32_t)adc_mv * (uint32_t)CONFIG_BATTERY_ADC_DIVIDER_MILLI) / 1000U;
    if (pack_mv > UINT16_MAX) pack_mv = UINT16_MAX;

    out->valid = true;
    out->millivolts = (uint16_t)pack_mv;
    out->percent = percent_from_mv(out->millivolts);

#if CONFIG_BATTERY_CHARGE_GPIO >= 0
    out->charging = gpio_get_level(CONFIG_BATTERY_CHARGE_GPIO) == CONFIG_BATTERY_CHARGE_ACTIVE_LEVEL;
#else
    out->charging = false;
#endif
    out->status = out->charging ? BATTERY_POWER_CHARGING : BATTERY_POWER_BATTERY;
    return ESP_OK;
}

#else

const char *battery_monitor_status_name(battery_power_status_t status)
{
    (void)status;
    return "UNKNOWN";
}

esp_err_t battery_monitor_init(void)
{
    return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t battery_monitor_read(battery_status_t *out)
{
    if (!out) return ESP_ERR_INVALID_ARG;
    out->valid = false;
    out->millivolts = 0;
    out->percent = 0;
    out->charging = false;
    out->status = BATTERY_POWER_UNKNOWN;
    return ESP_ERR_NOT_SUPPORTED;
}

#endif /* CONFIG_BATTERY_MONITOR_ENABLE */
