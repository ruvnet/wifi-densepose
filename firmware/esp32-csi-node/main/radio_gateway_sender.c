/** @file radio_gateway_sender.c */

#include "radio_gateway_sender.h"

#include "sdkconfig.h"

#if defined(CONFIG_BLE_IDENTITY_SCAN_ENABLE) || \
    defined(CONFIG_CHANNEL_SOUNDING_INGRESS_ENABLE)

#include <limits.h>
#include <stdatomic.h>
#include <string.h>

#include "esp_log.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "mbedtls/md.h"

#include "csi_collector.h"
#include "nvs_config.h"
#include "radio_gateway_protocol.h"
#include "stream_sender.h"

extern nvs_config_t g_nvs_config;

static const char *TAG = "radio_gateway";

typedef struct {
    uint8_t payload_type;
    uint8_t payload_len;
    uint8_t payload[RV_GATEWAY_MAX_PAYLOAD_SIZE];
    uint64_t received_at_boot_us;
    uint32_t timing_uncertainty_us;
} gateway_queue_item_t;

static QueueHandle_t s_queue;
static uint64_t s_boot_nonce;
static uint32_t s_sequence;
static atomic_bool s_ready;
static uint32_t s_delivery_drops;

static void sender_task(void *arg)
{
    (void)arg;
    gateway_queue_item_t item;
    uint8_t frame[RV_GATEWAY_MAX_FRAME_SIZE];
    uint8_t covered[RV_GATEWAY_MAC_DOMAIN_SIZE + RV_GATEWAY_MAX_SIGNED_SIZE];
    uint8_t digest[32];

    for (;;) {
        if (xQueueReceive(s_queue, &item, portMAX_DELAY) != pdTRUE) continue;
        if (s_sequence == UINT32_MAX) {
            ESP_LOGE(TAG, "gateway sequence exhausted; reboot or rekey required");
            atomic_store_explicit(&s_ready, false, memory_order_release);
            continue;
        }

        rv_gateway_metadata_t metadata = {
            .payload_type = item.payload_type,
            .flags = RV_GATEWAY_FLAG_RX_MONOTONIC,
            .key_id = g_nvs_config.radio_envelope_key_id,
            .node_id = csi_collector_get_node_id(),
            .sequence = ++s_sequence,
            .boot_nonce = s_boot_nonce,
            .received_at_boot_us = item.received_at_boot_us,
            .timing_uncertainty_us = item.timing_uncertainty_us,
        };
        size_t signed_len = 0u;
        size_t frame_len = 0u;
        if (!rv_gateway_build_unsigned(&metadata, item.payload, item.payload_len,
                                       frame, sizeof(frame), &signed_len,
                                       &frame_len)) {
            ESP_LOGE(TAG, "internal gateway envelope construction failed");
            continue;
        }
        memcpy(covered, RV_GATEWAY_MAC_DOMAIN, RV_GATEWAY_MAC_DOMAIN_SIZE);
        memcpy(&covered[RV_GATEWAY_MAC_DOMAIN_SIZE], frame, signed_len);
        const mbedtls_md_info_t *sha256 =
            mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
        if (sha256 == NULL
            || mbedtls_md_hmac(sha256, g_nvs_config.radio_envelope_secret,
                               sizeof(g_nvs_config.radio_envelope_secret),
                               covered,
                               RV_GATEWAY_MAC_DOMAIN_SIZE + signed_len,
                               digest) != 0) {
            ESP_LOGE(TAG, "gateway envelope HMAC failed");
            continue;
        }
        memcpy(&frame[signed_len], digest, RV_GATEWAY_AUTH_TAG_SIZE);

        /* Radio evidence is bulk data, not the <=48-byte <=1 Hz priority
         * control path. Normal ENOMEM backpressure applies. */
        if (stream_sender_send(frame, frame_len) != (int)frame_len) {
            s_delivery_drops++;
            if ((s_delivery_drops & 63u) == 1u) {
                ESP_LOGW(TAG, "radio envelope UDP delivery dropped (%lu total)",
                         (unsigned long)s_delivery_drops);
            }
        }
    }
}

esp_err_t radio_gateway_sender_init(void)
{
    if (!g_nvs_config.ble_identity_enabled && !g_nvs_config.cs_ingress_enabled) {
        return ESP_ERR_NOT_SUPPORTED;
    }
    if (!g_nvs_config.radio_envelope_secret_valid) {
        ESP_LOGE(TAG, "radio evidence enabled without gateway envelope key");
        return ESP_ERR_INVALID_STATE;
    }
    if (atomic_load_explicit(&s_ready, memory_order_acquire)) return ESP_OK;

    do {
        esp_fill_random(&s_boot_nonce, sizeof(s_boot_nonce));
    } while (s_boot_nonce == 0u);
    s_sequence = 0u;
    s_queue = xQueueCreate(CONFIG_RADIO_GATEWAY_QUEUE_DEPTH,
                           sizeof(gateway_queue_item_t));
    if (s_queue == NULL) return ESP_ERR_NO_MEM;
    if (xTaskCreate(sender_task, "radio_gateway", 4096, NULL, 5, NULL)
        != pdPASS) {
        vQueueDelete(s_queue);
        s_queue = NULL;
        return ESP_ERR_NO_MEM;
    }
    atomic_store_explicit(&s_ready, true, memory_order_release);
    ESP_LOGI(TAG, "authenticated radio envelope ready (queue=%d)",
             CONFIG_RADIO_GATEWAY_QUEUE_DEPTH);
    return ESP_OK;
}

bool radio_gateway_sender_is_ready(void)
{
    return atomic_load_explicit(&s_ready, memory_order_acquire);
}

esp_err_t radio_gateway_sender_enqueue(uint8_t payload_type,
                                       const uint8_t *payload,
                                       size_t payload_len,
                                       uint64_t received_at_boot_us,
                                       uint32_t timing_uncertainty_us)
{
    if (!atomic_load_explicit(&s_ready, memory_order_acquire)
        || s_queue == NULL) return ESP_ERR_INVALID_STATE;
    if (payload == NULL || payload_len > RV_GATEWAY_MAX_PAYLOAD_SIZE) {
        return ESP_ERR_INVALID_ARG;
    }
    gateway_queue_item_t item = {
        .payload_type = payload_type,
        .payload_len = (uint8_t)payload_len,
        .received_at_boot_us = received_at_boot_us,
        .timing_uncertainty_us = timing_uncertainty_us,
    };
    memcpy(item.payload, payload, payload_len);
    return xQueueSend(s_queue, &item, 0) == pdTRUE ? ESP_OK : ESP_ERR_TIMEOUT;
}

#else

esp_err_t radio_gateway_sender_init(void)
{
    return ESP_ERR_NOT_SUPPORTED;
}

bool radio_gateway_sender_is_ready(void)
{
    return false;
}

esp_err_t radio_gateway_sender_enqueue(uint8_t payload_type,
                                       const uint8_t *payload,
                                       size_t payload_len,
                                       uint64_t received_at_boot_us,
                                       uint32_t timing_uncertainty_us)
{
    (void)payload_type;
    (void)payload;
    (void)payload_len;
    (void)received_at_boot_us;
    (void)timing_uncertainty_us;
    return ESP_ERR_NOT_SUPPORTED;
}

#endif
