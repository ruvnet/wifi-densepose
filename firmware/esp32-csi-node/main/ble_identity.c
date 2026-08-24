/** @file ble_identity.c */

#include "ble_identity.h"

#include "sdkconfig.h"

#if defined(CONFIG_BLE_IDENTITY_SCAN_ENABLE)

#include <stdatomic.h>
#include <string.h>
#include <time.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "host/ble_gap.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "mbedtls/md.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"

#include "ble_identity_protocol.h"
#include "csi_collector.h"
#include "nvs_config.h"
#include "radio_gateway_protocol.h"
#include "radio_gateway_sender.h"

extern nvs_config_t g_nvs_config;

static const char *TAG = "ble_identity";
static uint8_t s_own_addr_type;
static uint32_t s_sequence;
static uint64_t s_rate_window_start_us;
static uint32_t s_rate_window_count;
static uint32_t s_rate_drops;
static uint32_t s_queue_drops;
static EventGroupHandle_t s_start_events;
static atomic_bool s_scan_healthy;
static atomic_bool s_host_running;
static int start_scan(void);

#define BLE_START_READY_BIT  (1u << 0)
#define BLE_START_FAILED_BIT (1u << 1)

static bool admit_report(uint64_t observed_at_us)
{
    if (s_rate_window_start_us == 0u
        || observed_at_us - s_rate_window_start_us >= 1000000u) {
        s_rate_window_start_us = observed_at_us;
        s_rate_window_count = 0u;
    }
    if (s_rate_window_count >= CONFIG_BLE_IDENTITY_MAX_REPORTS_PER_SEC) {
        s_rate_drops++;
        if ((s_rate_drops & 63u) == 1u) {
            ESP_LOGW(TAG, "BLE token admission rate limited (drops=%lu)",
                     (unsigned long)s_rate_drops);
        }
        return false;
    }
    s_rate_window_count++;
    return true;
}

static bool authenticate_token(const rv_ble_token_t *token)
{
    if (token == NULL || !g_nvs_config.ble_secret_valid
        || token->key_id != g_nvs_config.ble_key_id) {
        return false;
    }

    uint8_t covered[34];
    uint8_t digest[32];
    rv_ble_token_mac_input(token, covered);
    const mbedtls_md_info_t *sha256 = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    if (sha256 == NULL
        || mbedtls_md_hmac(sha256,
                           g_nvs_config.ble_secret,
                           sizeof(g_nvs_config.ble_secret),
                           covered,
                           sizeof(covered),
                           digest) != 0) {
        return false;
    }
    return rv_ble_auth_tag_equal(token->auth_tag, digest);
}

static bool token_time_is_fresh(const rv_ble_token_t *token, bool *clock_valid)
{
    time_t now = time(NULL);
    *clock_valid = now >= 1700000000;
    if (!*clock_valid) {
        return true; /* Host must validate token_epoch_min before association. */
    }
    uint32_t now_min = (uint32_t)((uint64_t)now / 60u);
    uint32_t delta = now_min > token->epoch_min
                   ? now_min - token->epoch_min
                   : token->epoch_min - now_min;
    return delta <= (uint32_t)CONFIG_BLE_IDENTITY_TOKEN_SKEW_MIN;
}

static uint16_t observation_confidence(int8_t rssi, bool clock_valid)
{
    /* This is evidence quality, not probability that a tag is a person. */
    int quality = 900;
    if (!clock_valid) quality = 650;
    if (rssi < -95) quality -= 250;
    else if (rssi < -85) quality -= 120;
    if (quality < 100) quality = 100;
    return (uint16_t)quality;
}

static void process_advertisement(const uint8_t *data, uint16_t length_data,
                                  int8_t rssi, int8_t tx_power,
                                  bool extended)
{
    if (data == NULL || length_data > 200u || rssi == 127) return;
    rv_ble_token_t token;
    if (!rv_ble_parse_advertisement(data, length_data, &token)) {
        return;
    }
    uint64_t observed_at_us = (uint64_t)esp_timer_get_time();
    if (!admit_report(observed_at_us) || !authenticate_token(&token)) {
        return;
    }

    bool clock_valid = false;
    if (!token_time_is_fresh(&token, &clock_valid)) {
        ESP_LOGW(TAG, "dropping authenticated but stale BLE token (key=%u)",
                 (unsigned)token.key_id);
        return;
    }

    if (s_sequence == UINT32_MAX) {
        ESP_LOGE(TAG, "BLE telemetry sequence exhausted; reboot or rekey required");
        return;
    }
    rv_ble_telemetry_t telemetry = {
        .node_id = csi_collector_get_node_id(),
        .flags = RV_BLE_FLAG_AUTHENTICATED,
        .key_id = token.key_id,
        .sequence = ++s_sequence,
        .observed_at_ms = (uint32_t)(observed_at_us / 1000u),
        .ttl_ms = (uint16_t)CONFIG_BLE_IDENTITY_TTL_MS,
        .confidence_permille = observation_confidence(rssi, clock_valid),
        .rssi_dbm = rssi,
        .tx_power_dbm = tx_power,
        .token_epoch_min = token.epoch_min,
    };
    if (clock_valid) telemetry.flags |= RV_BLE_FLAG_TIME_VERIFIED;
    if (extended) telemetry.flags |= RV_BLE_FLAG_EXTENDED_ADVERT;
    memcpy(telemetry.ephemeral_id, token.ephemeral_id,
           sizeof(telemetry.ephemeral_id));

    uint8_t packet[RV_BLE_TELEMETRY_SIZE];
    if (rv_ble_serialize_telemetry(&telemetry, packet, sizeof(packet))) {
        esp_err_t rc = radio_gateway_sender_enqueue(
            RV_GATEWAY_PAYLOAD_BLE_IDENTITY, packet, sizeof(packet),
            observed_at_us, 1000u);
        if (rc != ESP_OK) {
            s_queue_drops++;
            if ((s_queue_drops & 63u) == 1u) {
                ESP_LOGW(TAG, "BLE gateway queue full or unavailable (drops=%lu)",
                         (unsigned long)s_queue_drops);
            }
        }
    }
}

static int gap_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    switch (event->type) {
    case BLE_GAP_EVENT_DISC:
        process_advertisement(event->disc.data, event->disc.length_data,
                              event->disc.rssi, 127, false);
        return 0;
#if defined(CONFIG_BT_NIMBLE_EXT_SCAN)
    case BLE_GAP_EVENT_EXT_DISC:
        if (event->ext_disc.data_status
            != BLE_GAP_EXT_ADV_DATA_STATUS_COMPLETE) {
            /* Do not authenticate a prefix. Deployments keep the complete
             * advertiser payload bounded so it fits one 257-byte HCI event. */
            return 0;
        }
        process_advertisement(event->ext_disc.data,
                              event->ext_disc.length_data,
                              event->ext_disc.rssi,
                              event->ext_disc.tx_power,
                              true);
        return 0;
#endif
    case BLE_GAP_EVENT_DISC_COMPLETE:
        atomic_store_explicit(&s_scan_healthy, false, memory_order_release);
        ESP_LOGW(TAG, "BLE scan terminated (reason=%d); restarting",
                 event->disc_complete.reason);
        (void)start_scan();
        return 0;
    default:
        return 0;
    }
}

static int start_scan(void)
{
    struct ble_gap_ext_disc_params params;
    memset(&params, 0, sizeof(params));
    params.passive = 1;
    params.itvl = BLE_GAP_SCAN_ITVL_MS(CONFIG_BLE_IDENTITY_SCAN_INTERVAL_MS);
    params.window = BLE_GAP_SCAN_WIN_MS(CONFIG_BLE_IDENTITY_SCAN_WINDOW_MS);

    /*
     * The authenticated service record is 50 bytes and therefore cannot be
     * received through legacy discovery.  duration=0 and period=0 request a
     * continuous extended discovery procedure.  Duplicate filtering is off:
     * otherwise the controller may report a valid token only once for the
     * entire scan session and its three-second host TTL cannot be refreshed.
     * Scan the uncoded primary PHY only so the configured duty ceiling maps
     * to one controller scan window rather than two concurrent PHY windows.
     */
    int rc = ble_gap_ext_disc(s_own_addr_type, 0, 0, 0, 0, 0,
                              &params, NULL, gap_event, NULL);
    if (rc != 0) {
        atomic_store_explicit(&s_scan_healthy, false, memory_order_release);
        if (s_start_events != NULL) {
            xEventGroupSetBits(s_start_events, BLE_START_FAILED_BIT);
        }
        ESP_LOGE(TAG, "passive scan start failed: rc=%d", rc);
    } else {
        atomic_store_explicit(&s_scan_healthy, true, memory_order_release);
        if (s_start_events != NULL) {
            xEventGroupSetBits(s_start_events, BLE_START_READY_BIT);
        }
        ESP_LOGI(TAG, "passive BLE identity scan active: %d/%d ms duty window",
                 CONFIG_BLE_IDENTITY_SCAN_WINDOW_MS,
                 CONFIG_BLE_IDENTITY_SCAN_INTERVAL_MS);
    }
    return rc;
}

static void on_reset(int reason)
{
    atomic_store_explicit(&s_scan_healthy, false, memory_order_release);
    ESP_LOGE(TAG, "NimBLE host reset: reason=%d", reason);
}

static void on_sync(void)
{
    int rc = ble_hs_util_ensure_addr(0);
    if (rc != 0) {
        ESP_LOGE(TAG, "cannot ensure BLE identity address: rc=%d", rc);
        return;
    }
    rc = ble_hs_id_infer_auto(0, &s_own_addr_type);
    if (rc != 0) {
        ESP_LOGE(TAG, "cannot infer BLE address type: rc=%d", rc);
        return;
    }
    (void)start_scan();
}

static void host_task(void *arg)
{
    (void)arg;
    nimble_port_run();
    atomic_store_explicit(&s_scan_healthy, false, memory_order_release);
    atomic_store_explicit(&s_host_running, false, memory_order_release);
    nimble_port_freertos_deinit();
}

esp_err_t ble_identity_init(void)
{
    if (!g_nvs_config.ble_identity_enabled) {
        ESP_LOGI(TAG, "disabled by NVS (ble_enable=0)");
        return ESP_ERR_NOT_SUPPORTED;
    }
    if (!g_nvs_config.ble_secret_valid) {
        ESP_LOGE(TAG, "enabled without a 32-byte ble_secret; refusing to scan");
        return ESP_ERR_INVALID_STATE;
    }
    if (!radio_gateway_sender_is_ready()) {
        ESP_LOGE(TAG, "authenticated gateway envelope is unavailable");
        return ESP_ERR_INVALID_STATE;
    }
    if (CONFIG_BLE_IDENTITY_SCAN_WINDOW_MS > CONFIG_BLE_IDENTITY_SCAN_INTERVAL_MS / 4) {
        ESP_LOGE(TAG, "scan duty exceeds the hard 25%% coexistence ceiling");
        return ESP_ERR_INVALID_ARG;
    }
    if (csi_collector_get_pkt_yield_per_sec()
        < (uint16_t)CONFIG_BLE_IDENTITY_MIN_CSI_PPS) {
        ESP_LOGW(TAG, "CSI yield is below coexistence target at BLE start");
    }
    if (atomic_load_explicit(&s_host_running, memory_order_acquire)) {
        return ble_identity_is_healthy() ? ESP_OK : ESP_ERR_INVALID_STATE;
    }
    if (s_start_events == NULL) {
        s_start_events = xEventGroupCreate();
        if (s_start_events == NULL) return ESP_ERR_NO_MEM;
    }
    xEventGroupClearBits(s_start_events,
                         BLE_START_READY_BIT | BLE_START_FAILED_BIT);

    int rc = nimble_port_init();
    if (rc != 0) {
        ESP_LOGE(TAG, "nimble_port_init failed: rc=%d", rc);
        return ESP_FAIL;
    }
    ble_hs_cfg.sync_cb = on_sync;
    ble_hs_cfg.reset_cb = on_reset;
    atomic_store_explicit(&s_host_running, true, memory_order_release);
    nimble_port_freertos_init(host_task);
    EventBits_t result = xEventGroupWaitBits(
        s_start_events, BLE_START_READY_BIT | BLE_START_FAILED_BIT,
        pdFALSE, pdFALSE, pdMS_TO_TICKS(5000));
    if ((result & BLE_START_READY_BIT) != 0u) return ESP_OK;

    ESP_LOGE(TAG, "BLE scanner did not become healthy within startup deadline");
    (void)nimble_port_stop();
    return (result & BLE_START_FAILED_BIT) != 0u
        ? ESP_FAIL : ESP_ERR_TIMEOUT;
}

bool ble_identity_is_healthy(void)
{
    return atomic_load_explicit(&s_scan_healthy, memory_order_acquire);
}

#else

esp_err_t ble_identity_init(void)
{
    return ESP_ERR_NOT_SUPPORTED;
}

bool ble_identity_is_healthy(void)
{
    return false;
}

#endif /* CONFIG_BLE_IDENTITY_SCAN_ENABLE */
