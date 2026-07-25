/**
 * @file swarm_bridge.h
 * @brief ADR-066: ESP32 Swarm Bridge — Cognitum Seed coordinator client.
 *
 * Registers this node with a Cognitum Seed, sends periodic heartbeats,
 * and pushes happiness vectors for cross-zone analytics.
 * Runs as a FreeRTOS task on Core 0.
 */

#ifndef SWARM_BRIDGE_H
#define SWARM_BRIDGE_H

#include <stdint.h>
#include "esp_err.h"
#include "edge_processing.h"

/** Happiness vector dimension. */
#define SWARM_VECTOR_DIM  8

/**
 * ADR-084 Pass 4 defaults — mesh-exchange compression.
 *
 * A happiness vector is only sent in full when its 1-bit-per-dimension
 * sign sketch differs from the last sent sketch by at least this many
 * bits, or when `max_suppress_sec` has elapsed since the last full send
 * (whichever comes first). This keeps a stable room from re-sending the
 * same 8 floats every 5 seconds while still guaranteeing the Seed's view
 * never goes stale for more than `max_suppress_sec`.
 */
#define SWARM_NOVELTY_THRESHOLD_DEFAULT   1
#define SWARM_MAX_SUPPRESS_SEC_DEFAULT    300  /* 5 minutes */

/** Swarm bridge configuration. */
typedef struct {
    char     seed_url[64];     /**< Cognitum Seed base URL (e.g. "http://192.168.1.10:8080"). */
    char     seed_token[64];   /**< Bearer token for Seed WiFi API auth (from pairing). */
    char     zone_name[16];    /**< Zone name for this node (e.g. "bedroom"). */
    uint16_t heartbeat_sec;    /**< Heartbeat interval in seconds (default 30). */
    uint16_t ingest_sec;       /**< Happiness ingest interval in seconds (default 5). */
    uint8_t  enabled;          /**< 1 = bridge active, 0 = disabled. */

    /** ADR-084 Pass 4. Hamming distance (0-8) that forces a full send
     *  regardless of the suppress timer. 0 = use SWARM_NOVELTY_THRESHOLD_DEFAULT. */
    uint8_t  novelty_threshold;
    /** ADR-084 Pass 4. Force a full send after this many seconds even if
     *  novelty stayed below threshold the whole time. 0 = use
     *  SWARM_MAX_SUPPRESS_SEC_DEFAULT. */
    uint16_t max_suppress_sec;
} swarm_config_t;

/**
 * Initialize the swarm bridge and start the background task.
 * Registers this node with the Cognitum Seed on first successful POST.
 *
 * @param cfg      Swarm bridge configuration.
 * @param node_id  This node's identifier (from NVS).
 * @return ESP_OK on success, ESP_ERR_INVALID_ARG if seed_url is empty.
 */
esp_err_t swarm_bridge_init(const swarm_config_t *cfg, uint8_t node_id);

/**
 * Feed the latest vitals packet into the swarm bridge.
 * Called from the main loop whenever new vitals are available.
 *
 * @param vitals  Pointer to the latest vitals packet.
 */
void swarm_bridge_update_vitals(const edge_vitals_pkt_t *vitals);

/**
 * Update the happiness vector to be pushed at the next ingest cycle.
 *
 * @param vector  Float array of happiness values.
 * @param dim     Number of elements (clamped to SWARM_VECTOR_DIM).
 */
void swarm_bridge_update_happiness(const float *vector, uint8_t dim);

/**
 * Get cumulative bridge statistics.
 *
 * @param regs        Output: number of successful registrations.
 * @param heartbeats  Output: number of successful heartbeats sent.
 * @param ingests     Output: number of successful happiness ingests sent.
 * @param errors      Output: number of HTTP errors encountered.
 */
void swarm_bridge_get_stats(uint32_t *regs, uint32_t *heartbeats,
                            uint32_t *ingests, uint32_t *errors);

/**
 * ADR-084 Pass 4. Number of happiness-ingest cycles skipped because the
 * sign-sketch novelty stayed below `novelty_threshold` and the
 * `max_suppress_sec` floor hadn't elapsed yet. A rising count on a stable
 * node is the compression working as intended, not a fault.
 */
uint32_t swarm_bridge_get_suppressed_count(void);

#endif /* SWARM_BRIDGE_H */
