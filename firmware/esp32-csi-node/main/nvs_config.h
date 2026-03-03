/**
 * @file nvs_config.h
 * @brief Runtime configuration via NVS (Non-Volatile Storage).
 *
 * Reads WiFi credentials and aggregator target from NVS.
 * Falls back to compile-time Kconfig defaults if NVS keys are absent.
 * This allows a single firmware binary to be shipped and configured
 * per-device using the provisioning script.
 */

#ifndef NVS_CONFIG_H
#define NVS_CONFIG_H

#include <stdint.h>

/** Maximum lengths for NVS string fields. */
#define NVS_CFG_SSID_MAX     33
#define NVS_CFG_PASS_MAX     65
#define NVS_CFG_IP_MAX       16

/** Maximum channels in the hop list (must match CSI_HOP_CHANNELS_MAX). */
#define NVS_CFG_HOP_MAX      6

/** Runtime configuration loaded from NVS or Kconfig defaults. */
typedef struct {
    char     wifi_ssid[NVS_CFG_SSID_MAX];
    char     wifi_password[NVS_CFG_PASS_MAX];
    char     target_ip[NVS_CFG_IP_MAX];
    uint16_t target_port;
    uint8_t  node_id;

    /* ADR-029: Channel hopping and TDM configuration */
    uint8_t  channel_hop_count;               /**< Number of channels to hop (1 = no hop). */
    uint8_t  channel_list[NVS_CFG_HOP_MAX];   /**< Channel numbers for hopping. */
    uint32_t dwell_ms;                        /**< Dwell time per channel in ms. */
    uint8_t  tdm_slot_index;                  /**< This node's TDM slot index (0-based). */
    uint8_t  tdm_node_count;                  /**< Total nodes in the TDM schedule. */

    /* MAC address filter for CSI source selection (Issue #98) */
    uint8_t  filter_mac[6];                   /**< Transmitter MAC to accept (all zeros = no filter). */
    uint8_t  filter_mac_enabled;              /**< 1 = filter active, 0 = accept all. */

    /* ADR-039: Edge intelligence configuration */
    uint8_t  edge_tier;                       /**< 0=disabled, 1=phase/stats, 2=vitals, 3=reserved. */
    uint16_t presence_thresh;                 /**< Presence detection threshold (default 50). */
    uint16_t fall_thresh;                     /**< Fall detection threshold (default 500). */
    uint16_t vital_window;                    /**< Vital signs window in frames (default 300). */
    uint16_t vital_interval_ms;               /**< Vitals packet send interval in ms (default 1000). */
    uint8_t  subk_count;                      /**< Top-K subcarrier count (default 32). */
} nvs_config_t;

/**
 * Load configuration from NVS, falling back to Kconfig defaults.
 *
 * Must be called after nvs_flash_init().
 *
 * @param cfg  Output configuration struct.
 */
void nvs_config_load(nvs_config_t *cfg);

#endif /* NVS_CONFIG_H */
