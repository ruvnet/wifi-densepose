/**
 * @file nvs_config.c
 * @brief Runtime configuration via NVS (Non-Volatile Storage).
 *
 * Checks NVS namespace "csi_cfg" for keys: ssid, password, target_ip,
 * target_port, node_id.  Falls back to Kconfig defaults when absent.
 */

#include "nvs_config.h"

#include <stdbool.h>
#include <string.h>
#include "esp_log.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "sdkconfig.h"

static const char *TAG = "nvs_config";

static bool secret_has_nonzero_byte(const uint8_t *secret, size_t len)
{
    uint8_t combined = 0u;
    for (size_t index = 0u; index < len; index++) combined |= secret[index];
    return combined != 0u;
}

void nvs_config_load(nvs_config_t *cfg)
{
    if (cfg == NULL) {
        ESP_LOGE(TAG, "nvs_config_load: cfg is NULL");
        return;
    }

    /* Start with Kconfig compiled defaults */
    strlcpy(cfg->wifi_ssid, CONFIG_CSI_WIFI_SSID, sizeof(cfg->wifi_ssid));

#ifdef CONFIG_CSI_WIFI_PASSWORD
    strlcpy(cfg->wifi_password, CONFIG_CSI_WIFI_PASSWORD,
            sizeof(cfg->wifi_password));
#else
    cfg->wifi_password[0] = '\0';
#endif

    strlcpy(cfg->target_ip, CONFIG_CSI_TARGET_IP, sizeof(cfg->target_ip));

    cfg->target_port = (uint16_t)CONFIG_CSI_TARGET_PORT;
    cfg->node_id     = (uint8_t)CONFIG_CSI_NODE_ID;

    /* ADR-029: Defaults for channel hopping and TDM.
     * hop_count=1 means single-channel (backward-compatible). */
    cfg->channel_hop_count = 1;
    cfg->channel_list[0]   = (uint8_t)CONFIG_CSI_WIFI_CHANNEL;
    for (uint8_t i = 1; i < NVS_CFG_HOP_MAX; i++) {
        cfg->channel_list[i] = 0;
    }
    cfg->dwell_ms       = 50;
    cfg->tdm_slot_index = 0;
    cfg->tdm_node_count = 1;

    /* ADR-039: Edge intelligence defaults from Kconfig. */
#ifdef CONFIG_EDGE_TIER
    cfg->edge_tier = (uint8_t)CONFIG_EDGE_TIER;
#else
    cfg->edge_tier = 2;
#endif
    cfg->presence_thresh = 0.0f;  /* 0 = auto-calibrate. */
#ifdef CONFIG_EDGE_FALL_THRESH
    cfg->fall_thresh = (float)CONFIG_EDGE_FALL_THRESH / 1000.0f;
#else
    cfg->fall_thresh = 15.0f;  /* Default raised from 2.0 — see issue #263. */
#endif
    cfg->vital_window = 256;
#ifdef CONFIG_EDGE_VITAL_INTERVAL_MS
    cfg->vital_interval_ms = (uint16_t)CONFIG_EDGE_VITAL_INTERVAL_MS;
#else
    cfg->vital_interval_ms = 1000;
#endif
#ifdef CONFIG_EDGE_TOP_K
    cfg->top_k_count = (uint8_t)CONFIG_EDGE_TOP_K;
#else
    cfg->top_k_count = 8;
#endif
#ifdef CONFIG_EDGE_POWER_DUTY
    cfg->power_duty = (uint8_t)CONFIG_EDGE_POWER_DUTY;
#else
    cfg->power_duty = 100;
#endif

    /* ADR-040: WASM programmable sensing defaults from Kconfig. */
#ifdef CONFIG_WASM_MAX_MODULES
    cfg->wasm_max_modules = (uint8_t)CONFIG_WASM_MAX_MODULES;
#else
    cfg->wasm_max_modules = 4;
#endif
    cfg->wasm_verify = 1;  /* Default: verify enabled (secure-by-default). */
#ifndef CONFIG_WASM_VERIFY_SIGNATURE
    cfg->wasm_verify = 0;  /* Kconfig disabled signature verification. */
#endif

    /* ADR-060: Channel override and MAC filter defaults. */
    cfg->csi_channel = 0;  /* 0 = auto-detect from connected AP. */
    cfg->filter_mac_set = 0;
    memset(cfg->filter_mac, 0, 6);

    /* ADR-341: BLE identity is runtime-off even in an enabled build. */
    cfg->ble_identity_enabled = 0;
    cfg->ble_key_id = 0;
    cfg->ble_secret_valid = 0;
    memset(cfg->ble_secret, 0, sizeof(cfg->ble_secret));
    cfg->cs_ingress_enabled = 0;
    cfg->cs_key_id = 0;
    cfg->cs_secret_valid = 0;
    memset(cfg->cs_secret, 0, sizeof(cfg->cs_secret));
    cfg->cs_enrolled_source_id = 0u;
    cfg->radio_envelope_key_id = 0u;
    cfg->radio_envelope_secret_valid = 0u;
    memset(cfg->radio_envelope_secret, 0,
           sizeof(cfg->radio_envelope_secret));

    /* Try to override from NVS */
    nvs_handle_t handle;
    esp_err_t err = nvs_open("csi_cfg", NVS_READONLY, &handle);
    if (err != ESP_OK) {
        ESP_LOGI(TAG, "No NVS config found, using compiled defaults");
        return;
    }

    size_t len;
    char buf[NVS_CFG_PASS_MAX];

    /* WiFi SSID */
    len = sizeof(buf);
    if (nvs_get_str(handle, "ssid", buf, &len) == ESP_OK && len > 1) {
        strlcpy(cfg->wifi_ssid, buf, sizeof(cfg->wifi_ssid));
        ESP_LOGI(TAG, "NVS override: ssid=%s", cfg->wifi_ssid);
    }

    /* WiFi password */
    len = sizeof(buf);
    if (nvs_get_str(handle, "password", buf, &len) == ESP_OK) {
        strlcpy(cfg->wifi_password, buf, sizeof(cfg->wifi_password));
        ESP_LOGI(TAG, "NVS override: password=***");
    }

    /* Target IP */
    len = sizeof(buf);
    if (nvs_get_str(handle, "target_ip", buf, &len) == ESP_OK && len > 1) {
        strlcpy(cfg->target_ip, buf, sizeof(cfg->target_ip));
        ESP_LOGI(TAG, "NVS override: target_ip=%s", cfg->target_ip);
    }

    /* Target port */
    uint16_t port_val;
    if (nvs_get_u16(handle, "target_port", &port_val) == ESP_OK) {
        cfg->target_port = port_val;
        ESP_LOGI(TAG, "NVS override: target_port=%u", cfg->target_port);
    }

    /* Node ID */
    uint8_t node_val;
    if (nvs_get_u8(handle, "node_id", &node_val) == ESP_OK) {
        cfg->node_id = node_val;
        ESP_LOGI(TAG, "NVS override: node_id=%u", cfg->node_id);
    }

    /* ADR-029: Channel hop count */
    uint8_t hop_count_val;
    if (nvs_get_u8(handle, "hop_count", &hop_count_val) == ESP_OK) {
        if (hop_count_val >= 1 && hop_count_val <= NVS_CFG_HOP_MAX) {
            cfg->channel_hop_count = hop_count_val;
            ESP_LOGI(TAG, "NVS override: hop_count=%u", (unsigned)cfg->channel_hop_count);
        } else {
            ESP_LOGW(TAG, "NVS hop_count=%u out of range [1..%u], ignored",
                     (unsigned)hop_count_val, (unsigned)NVS_CFG_HOP_MAX);
        }
    }

    /* ADR-029: Channel list (stored as a blob of up to NVS_CFG_HOP_MAX bytes) */
    len = NVS_CFG_HOP_MAX;
    uint8_t ch_blob[NVS_CFG_HOP_MAX];
    if (nvs_get_blob(handle, "chan_list", ch_blob, &len) == ESP_OK && len > 0) {
        uint8_t count = (len < cfg->channel_hop_count) ? (uint8_t)len : cfg->channel_hop_count;
        for (uint8_t i = 0; i < count; i++) {
            cfg->channel_list[i] = ch_blob[i];
        }
        ESP_LOGI(TAG, "NVS override: chan_list loaded (%u channels)", (unsigned)count);
    }

    /* ADR-029: Dwell time */
    uint32_t dwell_val;
    if (nvs_get_u32(handle, "dwell_ms", &dwell_val) == ESP_OK) {
        if (dwell_val >= 10) {
            cfg->dwell_ms = dwell_val;
            ESP_LOGI(TAG, "NVS override: dwell_ms=%lu", (unsigned long)cfg->dwell_ms);
        } else {
            ESP_LOGW(TAG, "NVS dwell_ms=%lu too small, ignored", (unsigned long)dwell_val);
        }
    }

    /* ADR-029/031: TDM slot index */
    uint8_t slot_val;
    if (nvs_get_u8(handle, "tdm_slot", &slot_val) == ESP_OK) {
        cfg->tdm_slot_index = slot_val;
        ESP_LOGI(TAG, "NVS override: tdm_slot_index=%u", (unsigned)cfg->tdm_slot_index);
    }

    /* ADR-029/031: TDM node count */
    uint8_t tdm_nodes_val;
    if (nvs_get_u8(handle, "tdm_nodes", &tdm_nodes_val) == ESP_OK) {
        if (tdm_nodes_val >= 1) {
            cfg->tdm_node_count = tdm_nodes_val;
            ESP_LOGI(TAG, "NVS override: tdm_node_count=%u", (unsigned)cfg->tdm_node_count);
        } else {
            ESP_LOGW(TAG, "NVS tdm_nodes=%u invalid, ignored", (unsigned)tdm_nodes_val);
        }
    }

    /* ADR-039: Edge intelligence overrides. */
    uint8_t edge_tier_val;
    if (nvs_get_u8(handle, "edge_tier", &edge_tier_val) == ESP_OK) {
        if (edge_tier_val <= 2) {
            cfg->edge_tier = edge_tier_val;
            ESP_LOGI(TAG, "NVS override: edge_tier=%u", (unsigned)cfg->edge_tier);
        }
    }

    /* Presence threshold stored as u16 (value * 1000). */
    uint16_t pres_thresh_val;
    if (nvs_get_u16(handle, "pres_thresh", &pres_thresh_val) == ESP_OK) {
        cfg->presence_thresh = (float)pres_thresh_val / 1000.0f;
        ESP_LOGI(TAG, "NVS override: presence_thresh=%.3f", cfg->presence_thresh);
    }

    /* Fall threshold stored as u16 (value * 1000). */
    uint16_t fall_thresh_val;
    if (nvs_get_u16(handle, "fall_thresh", &fall_thresh_val) == ESP_OK) {
        cfg->fall_thresh = (float)fall_thresh_val / 1000.0f;
        ESP_LOGI(TAG, "NVS override: fall_thresh=%.3f", cfg->fall_thresh);
    }

    uint16_t vital_win_val;
    if (nvs_get_u16(handle, "vital_win", &vital_win_val) == ESP_OK) {
        if (vital_win_val >= 32 && vital_win_val <= 256) {
            cfg->vital_window = vital_win_val;
            ESP_LOGI(TAG, "NVS override: vital_window=%u", cfg->vital_window);
        }
    }

    uint16_t vital_int_val;
    if (nvs_get_u16(handle, "vital_int", &vital_int_val) == ESP_OK) {
        if (vital_int_val >= 100) {
            cfg->vital_interval_ms = vital_int_val;
            ESP_LOGI(TAG, "NVS override: vital_interval_ms=%u", cfg->vital_interval_ms);
        }
    }

    uint8_t topk_val;
    if (nvs_get_u8(handle, "subk_count", &topk_val) == ESP_OK) {
        if (topk_val >= 1 && topk_val <= 32) {
            cfg->top_k_count = topk_val;
            ESP_LOGI(TAG, "NVS override: top_k_count=%u", (unsigned)cfg->top_k_count);
        }
    }

    uint8_t duty_val;
    if (nvs_get_u8(handle, "power_duty", &duty_val) == ESP_OK) {
        if (duty_val >= 10 && duty_val <= 100) {
            cfg->power_duty = duty_val;
            ESP_LOGI(TAG, "NVS override: power_duty=%u%%", (unsigned)cfg->power_duty);
        }
    }

    /* ADR-040: WASM configuration overrides. */
    uint8_t wasm_max_val;
    if (nvs_get_u8(handle, "wasm_max", &wasm_max_val) == ESP_OK) {
        if (wasm_max_val >= 1 && wasm_max_val <= 8) {
            cfg->wasm_max_modules = wasm_max_val;
            ESP_LOGI(TAG, "NVS override: wasm_max_modules=%u", (unsigned)cfg->wasm_max_modules);
        }
    }

    uint8_t wasm_verify_val;
    if (nvs_get_u8(handle, "wasm_verify", &wasm_verify_val) == ESP_OK) {
        cfg->wasm_verify = wasm_verify_val ? 1 : 0;
        ESP_LOGI(TAG, "NVS override: wasm_verify=%u", (unsigned)cfg->wasm_verify);
    }

    /* ADR-040: Load WASM signing public key from NVS (32-byte blob). */
    cfg->wasm_pubkey_valid = 0;
    memset(cfg->wasm_pubkey, 0, 32);
    size_t pubkey_len = 32;
    if (nvs_get_blob(handle, "wasm_pubkey", cfg->wasm_pubkey, &pubkey_len) == ESP_OK
        && pubkey_len == 32)
    {
        cfg->wasm_pubkey_valid = 1;
        ESP_LOGI(TAG, "NVS: wasm_pubkey loaded (%02x%02x...%02x%02x)",
                 cfg->wasm_pubkey[0], cfg->wasm_pubkey[1],
                 cfg->wasm_pubkey[30], cfg->wasm_pubkey[31]);
    } else if (cfg->wasm_verify) {
        ESP_LOGW(TAG, "wasm_verify=1 but no wasm_pubkey in NVS — uploads will be rejected");
    }

    /* ADR-060: CSI channel override. */
    uint8_t csi_ch_val;
    if (nvs_get_u8(handle, "csi_channel", &csi_ch_val) == ESP_OK) {
        if ((csi_ch_val >= 1 && csi_ch_val <= 14) || (csi_ch_val >= 36 && csi_ch_val <= 177)) {
            cfg->csi_channel = csi_ch_val;
            ESP_LOGI(TAG, "NVS override: csi_channel=%u", (unsigned)cfg->csi_channel);
        } else {
            ESP_LOGW(TAG, "NVS csi_channel=%u invalid, ignored", (unsigned)csi_ch_val);
        }
    }

    /* ADR-060: MAC address filter (6-byte blob). */
    size_t mac_len = 6;
    if (nvs_get_blob(handle, "filter_mac", cfg->filter_mac, &mac_len) == ESP_OK && mac_len == 6) {
        cfg->filter_mac_set = 1;
        ESP_LOGI(TAG, "NVS override: filter_mac=%02x:%02x:%02x:%02x:%02x:%02x",
                 cfg->filter_mac[0], cfg->filter_mac[1], cfg->filter_mac[2],
                 cfg->filter_mac[3], cfg->filter_mac[4], cfg->filter_mac[5]);
    }

    /* ADR-066: Swarm bridge */
    len = sizeof(cfg->seed_url);
    if (nvs_get_str(handle, "seed_url", cfg->seed_url, &len) != ESP_OK) {
        cfg->seed_url[0] = '\0';  /* Disabled by default */
    }
    len = sizeof(cfg->seed_token);
    if (nvs_get_str(handle, "seed_token", cfg->seed_token, &len) != ESP_OK) {
        cfg->seed_token[0] = '\0';
    }
    len = sizeof(cfg->zone_name);
    if (nvs_get_str(handle, "zone_name", cfg->zone_name, &len) != ESP_OK) {
        strlcpy(cfg->zone_name, "default", sizeof(cfg->zone_name));
    }
    if (nvs_get_u16(handle, "swarm_hb", &cfg->swarm_heartbeat_sec) != ESP_OK) {
        cfg->swarm_heartbeat_sec = 30;
    }
    if (nvs_get_u16(handle, "swarm_ingest", &cfg->swarm_ingest_sec) != ESP_OK) {
        cfg->swarm_ingest_sec = 5;
    }

    /* ADR-341: BLE identity anchor. The secret is never printed. */
    uint8_t ble_enable_val;
    if (nvs_get_u8(handle, "ble_enable", &ble_enable_val) == ESP_OK) {
        cfg->ble_identity_enabled = ble_enable_val ? 1u : 0u;
        ESP_LOGI(TAG, "NVS override: ble_identity_enabled=%u",
                 (unsigned)cfg->ble_identity_enabled);
    }
    if (nvs_get_u8(handle, "ble_key_id", &cfg->ble_key_id) == ESP_OK) {
        ESP_LOGI(TAG, "NVS override: ble_key_id=%u", (unsigned)cfg->ble_key_id);
    }
    size_t ble_secret_len = sizeof(cfg->ble_secret);
    if (nvs_get_blob(handle, "ble_secret", cfg->ble_secret, &ble_secret_len) == ESP_OK
        && ble_secret_len == sizeof(cfg->ble_secret)
        && secret_has_nonzero_byte(cfg->ble_secret, sizeof(cfg->ble_secret))) {
        cfg->ble_secret_valid = 1u;
        ESP_LOGI(TAG, "NVS: ble_secret loaded (32 bytes)");
    } else if (cfg->ble_identity_enabled) {
        ESP_LOGW(TAG, "ble_enable=1 but exact 32-byte ble_secret is absent; scanner will fail closed");
    }

    uint8_t cs_enable_val;
    if (nvs_get_u8(handle, "cs_enable", &cs_enable_val) == ESP_OK) {
        cfg->cs_ingress_enabled = cs_enable_val ? 1u : 0u;
        ESP_LOGI(TAG, "NVS override: cs_ingress_enabled=%u",
                 (unsigned)cfg->cs_ingress_enabled);
    }
    if (nvs_get_u8(handle, "cs_key_id", &cfg->cs_key_id) == ESP_OK) {
        ESP_LOGI(TAG, "NVS override: cs_key_id=%u", (unsigned)cfg->cs_key_id);
    }
    size_t cs_secret_len = sizeof(cfg->cs_secret);
    if (nvs_get_blob(handle, "cs_secret", cfg->cs_secret, &cs_secret_len) == ESP_OK
        && cs_secret_len == sizeof(cfg->cs_secret)
        && secret_has_nonzero_byte(cfg->cs_secret, sizeof(cfg->cs_secret))) {
        cfg->cs_secret_valid = 1u;
        ESP_LOGI(TAG, "NVS: cs_secret loaded (32 bytes)");
    } else if (cfg->cs_ingress_enabled) {
        ESP_LOGW(TAG, "cs_enable=1 but exact 32-byte cs_secret is absent; ingress will fail closed");
    }
    if (nvs_get_u32(handle, "cs_source_id", &cfg->cs_enrolled_source_id) == ESP_OK) {
        ESP_LOGI(TAG, "NVS: enrolled Channel Sounding source id loaded");
    }

    if (nvs_get_u8(handle, "radio_key_id", &cfg->radio_envelope_key_id) == ESP_OK) {
        ESP_LOGI(TAG, "NVS override: radio envelope key id=%u",
                 (unsigned)cfg->radio_envelope_key_id);
    }
    size_t radio_secret_len = sizeof(cfg->radio_envelope_secret);
    if (nvs_get_blob(handle, "radio_secret", cfg->radio_envelope_secret,
                     &radio_secret_len) == ESP_OK
        && radio_secret_len == sizeof(cfg->radio_envelope_secret)
        && secret_has_nonzero_byte(cfg->radio_envelope_secret,
                                   sizeof(cfg->radio_envelope_secret))) {
        cfg->radio_envelope_secret_valid = 1u;
        ESP_LOGI(TAG, "NVS: radio envelope secret loaded (32 bytes)");
    } else if (cfg->ble_identity_enabled || cfg->cs_ingress_enabled) {
        ESP_LOGW(TAG, "radio evidence enabled but gateway envelope key is absent; both paths fail closed");
    }

    bool reused_radio_key =
        (cfg->ble_secret_valid && cfg->cs_secret_valid
         && memcmp(cfg->ble_secret, cfg->cs_secret,
                   sizeof(cfg->ble_secret)) == 0)
        || (cfg->ble_secret_valid && cfg->radio_envelope_secret_valid
            && memcmp(cfg->ble_secret, cfg->radio_envelope_secret,
                      sizeof(cfg->ble_secret)) == 0)
        || (cfg->cs_secret_valid && cfg->radio_envelope_secret_valid
            && memcmp(cfg->cs_secret, cfg->radio_envelope_secret,
                      sizeof(cfg->cs_secret)) == 0);
    if (reused_radio_key) {
        ESP_LOGE(TAG, "BLE, Channel Sounding and gateway envelope keys must be distinct");
        cfg->ble_secret_valid = 0u;
        cfg->cs_secret_valid = 0u;
        cfg->radio_envelope_secret_valid = 0u;
    }

    /* Validate tdm_slot_index < tdm_node_count */
    if (cfg->tdm_slot_index >= cfg->tdm_node_count) {
        ESP_LOGW(TAG, "tdm_slot_index=%u >= tdm_node_count=%u, clamping to 0",
                 (unsigned)cfg->tdm_slot_index, (unsigned)cfg->tdm_node_count);
        cfg->tdm_slot_index = 0;
    }

    nvs_close(handle);
}
