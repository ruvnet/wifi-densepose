/** @file channel_sounding_ingress.c */

#include "channel_sounding_ingress.h"

#include "sdkconfig.h"

#if defined(CONFIG_CHANNEL_SOUNDING_INGRESS_ENABLE)

#include <stdbool.h>
#include <string.h>

#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "channel_sounding_protocol.h"
#include "mbedtls/md.h"
#include "nvs_config.h"
#include "radio_gateway_protocol.h"
#include "radio_gateway_sender.h"

extern nvs_config_t g_nvs_config;

static const char *TAG = "cs_ingress";

typedef struct {
    uint32_t source_id;
    uint32_t source_session_id;
    uint32_t sequence;
    uint64_t last_seen_us;
    bool initialized;
} source_sequence_t;

#define SOURCE_SEQUENCE_SLOTS 8u
static source_sequence_t s_sequences[SOURCE_SEQUENCE_SLOTS];
static uint64_t s_rate_window_start_us;
static uint32_t s_rate_window_count;
static uint32_t s_rate_drops;
static uint32_t s_invalid_drops;
static uint32_t s_auth_drops;
static uint32_t s_replay_drops;
static uint32_t s_queue_drops;

static bool authenticate_frame(const uint8_t frame[RV_CS_FRAME_SIZE])
{
    if (!g_nvs_config.cs_secret_valid || frame[6] != g_nvs_config.cs_key_id) return false;
    uint8_t covered[RV_CS_MAC_INPUT_SIZE];
    uint8_t digest[32];
    rv_cs_mac_input(frame, covered);
    const mbedtls_md_info_t *sha256 = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    if (sha256 == NULL
        || mbedtls_md_hmac(sha256, g_nvs_config.cs_secret,
                           sizeof(g_nvs_config.cs_secret), covered,
                           sizeof(covered), digest) != 0) return false;
    return rv_cs_auth_tag_equal(&frame[RV_CS_SIGNED_PREFIX_SIZE], digest);
}

static bool accept_sequence(uint32_t source_id, uint32_t source_session_id,
                            uint32_t sequence, uint64_t received_at_us)
{
    size_t empty = SOURCE_SEQUENCE_SLOTS;
    size_t oldest = 0u;
    for (size_t i = 0u; i < SOURCE_SEQUENCE_SLOTS; i++) {
        if (s_sequences[i].initialized
            && s_sequences[i].source_id == source_id
            && s_sequences[i].source_session_id == source_session_id) {
            if (!rv_cs_sequence_is_newer(sequence, s_sequences[i].sequence)) {
                return false;
            }
            s_sequences[i].sequence = sequence;
            s_sequences[i].last_seen_us = received_at_us;
            return true;
        }
        if (!s_sequences[i].initialized && empty == SOURCE_SEQUENCE_SLOTS) empty = i;
        if (s_sequences[i].initialized
            && s_sequences[i].last_seen_us < s_sequences[oldest].last_seen_us) {
            oldest = i;
        }
    }
    if (empty == SOURCE_SEQUENCE_SLOTS) {
        uint64_t retire_us =
            (uint64_t)CONFIG_CHANNEL_SOUNDING_SESSION_RETIRE_MS * 1000u;
        if (sequence != 1u
            || received_at_us < s_sequences[oldest].last_seen_us
            || received_at_us - s_sequences[oldest].last_seen_us < retire_us) {
            return false;
        }
        empty = oldest;
    }
    s_sequences[empty].source_id = source_id;
    s_sequences[empty].source_session_id = source_session_id;
    s_sequences[empty].sequence = sequence;
    s_sequences[empty].last_seen_us = received_at_us;
    s_sequences[empty].initialized = true;
    return true;
}

static bool admit_frame(uint64_t received_at_us)
{
    if (s_rate_window_start_us == 0u
        || received_at_us - s_rate_window_start_us >= 1000000u) {
        s_rate_window_start_us = received_at_us;
        s_rate_window_count = 0u;
    }
    if (s_rate_window_count >= CONFIG_CHANNEL_SOUNDING_MAX_FRAMES_PER_SEC) {
        s_rate_drops++;
        if ((s_rate_drops & 63u) == 1u) {
            ESP_LOGW(TAG, "companion admission rate limited (drops=%lu)",
                     (unsigned long)s_rate_drops);
        }
        return false;
    }
    s_rate_window_count++;
    return true;
}

static void process_frame(const uint8_t frame[RV_CS_FRAME_SIZE],
                          uint64_t received_at_us)
{
    rv_cs_measurement_t measurement;
    rv_cs_parse_result_t result = rv_cs_parse_frame(
        frame, RV_CS_FRAME_SIZE,
        (uint32_t)CONFIG_CHANNEL_SOUNDING_MAX_AGE_MS * 1000u,
        (uint16_t)CONFIG_CHANNEL_SOUNDING_MIN_QUALITY_PERMILLE,
        &measurement);
    if (result != RV_CS_PARSE_OK
        || measurement.source_id != g_nvs_config.cs_enrolled_source_id) {
        s_invalid_drops++;
        if ((s_invalid_drops & 63u) == 1u) {
            ESP_LOGW(TAG, "invalid or unenrolled companion frame (reason=%d drops=%lu)",
                     (int)result, (unsigned long)s_invalid_drops);
        }
        return;
    }
    if (!admit_frame(received_at_us)) return;
    if (!authenticate_frame(frame)) {
        s_auth_drops++;
        if ((s_auth_drops & 63u) == 1u) {
            ESP_LOGW(TAG, "unauthenticated companion frame (drops=%lu)",
                     (unsigned long)s_auth_drops);
        }
        return;
    }
    if (!accept_sequence(measurement.source_id,
                         measurement.source_session_id,
                         measurement.sequence, received_at_us)) {
        s_replay_drops++;
        if ((s_replay_drops & 63u) == 1u) {
            ESP_LOGW(TAG, "companion replay or state capacity drop (drops=%lu)",
                     (unsigned long)s_replay_drops);
        }
        return;
    }

    uint32_t uart_uncertainty_us =
        (uint32_t)((RV_CS_FRAME_SIZE * 10u * 1000000u
                    + CONFIG_CHANNEL_SOUNDING_UART_BAUD - 1u)
                   / CONFIG_CHANNEL_SOUNDING_UART_BAUD);
    uint32_t uncertainty_us =
        (uint32_t)measurement.timing_uncertainty_us + uart_uncertainty_us;
    esp_err_t rc = radio_gateway_sender_enqueue(
        RV_GATEWAY_PAYLOAD_CHANNEL_SOUNDING, frame, RV_CS_FRAME_SIZE,
        received_at_us, uncertainty_us);
    if (rc != ESP_OK) {
        s_queue_drops++;
        if ((s_queue_drops & 63u) == 1u) {
            ESP_LOGW(TAG, "Channel Sounding gateway queue drop (drops=%lu)",
                     (unsigned long)s_queue_drops);
        }
    }
}

static void uart_task(void *arg)
{
    (void)arg;
    static const uint8_t magic[4] = { 'R', 'V', 'C', 'S' };
    uint8_t frame[RV_CS_FRAME_SIZE];
    uint8_t chunk[128];
    size_t used = 0u;

    for (;;) {
        int count = uart_read_bytes(CONFIG_CHANNEL_SOUNDING_UART_NUM,
                                    chunk, sizeof(chunk), pdMS_TO_TICKS(100));
        if (count <= 0) continue;
        for (int index = 0; index < count; index++) {
            uint8_t byte = chunk[index];
            if (used < sizeof(magic)) {
                if (byte == magic[used]) {
                    frame[used++] = byte;
                } else {
                    used = byte == magic[0] ? 1u : 0u;
                    if (used == 1u) frame[0] = byte;
                }
                continue;
            }
            frame[used++] = byte;
            if (used == sizeof(frame)) {
                process_frame(frame, (uint64_t)esp_timer_get_time());
                used = 0u;
            }
        }
    }
}

esp_err_t channel_sounding_ingress_init(void)
{
    if (!g_nvs_config.cs_ingress_enabled) {
        ESP_LOGI(TAG, "disabled by NVS (cs_enable=0)");
        return ESP_ERR_NOT_SUPPORTED;
    }
    if (!g_nvs_config.cs_secret_valid) {
        ESP_LOGE(TAG, "enabled without a 32-byte cs_secret; refusing UART ingress");
        return ESP_ERR_INVALID_STATE;
    }
    if (g_nvs_config.cs_enrolled_source_id == 0u) {
        ESP_LOGE(TAG, "enabled without a nonzero enrolled cs_source_id");
        return ESP_ERR_INVALID_STATE;
    }
    if (!radio_gateway_sender_is_ready()) {
        ESP_LOGE(TAG, "authenticated gateway envelope is unavailable");
        return ESP_ERR_INVALID_STATE;
    }
    if (CONFIG_CHANNEL_SOUNDING_UART_NUM == UART_NUM_0) {
        ESP_LOGE(TAG, "UART0 is reserved for console and provisioning");
        return ESP_ERR_INVALID_ARG;
    }
    memset(s_sequences, 0, sizeof(s_sequences));
    uart_config_t config = {
        .baud_rate = CONFIG_CHANNEL_SOUNDING_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    esp_err_t rc = uart_driver_install(CONFIG_CHANNEL_SOUNDING_UART_NUM,
                                       RV_CS_FRAME_SIZE * 8u, 0, 0, NULL, 0);
    if (rc != ESP_OK) return rc;
    rc = uart_param_config(CONFIG_CHANNEL_SOUNDING_UART_NUM, &config);
    if (rc != ESP_OK) {
        uart_driver_delete(CONFIG_CHANNEL_SOUNDING_UART_NUM);
        return rc;
    }
    rc = uart_set_pin(CONFIG_CHANNEL_SOUNDING_UART_NUM,
                      CONFIG_CHANNEL_SOUNDING_UART_TX_GPIO,
                      CONFIG_CHANNEL_SOUNDING_UART_RX_GPIO,
                      UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    if (rc != ESP_OK) {
        uart_driver_delete(CONFIG_CHANNEL_SOUNDING_UART_NUM);
        return rc;
    }

    BaseType_t task = xTaskCreate(uart_task, "cs_uart", 4096,
                                  NULL, 5, NULL);
    if (task != pdPASS) {
        uart_driver_delete(CONFIG_CHANNEL_SOUNDING_UART_NUM);
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "external Channel Sounding ingress active on UART%d; data is unvalidated by hardware evidence",
             CONFIG_CHANNEL_SOUNDING_UART_NUM);
    return ESP_OK;
}

#else

esp_err_t channel_sounding_ingress_init(void)
{
    return ESP_ERR_NOT_SUPPORTED;
}

#endif /* CONFIG_CHANNEL_SOUNDING_INGRESS_ENABLE */
