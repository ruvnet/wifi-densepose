/**
 * @file radio_gateway_sender.h
 * @brief Bounded asynchronous sender for authenticated radio envelopes.
 */

#ifndef RADIO_GATEWAY_SENDER_H
#define RADIO_GATEWAY_SENDER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

/** Initialize the boot session, bounded queue, and sender task. */
esp_err_t radio_gateway_sender_init(void);

/** Whether authenticated envelope egress is ready. */
bool radio_gateway_sender_is_ready(void);

/**
 * Nonblocking enqueue of an already sanitized radio payload.
 * Returns `ESP_ERR_TIMEOUT` when backpressure drops the record.
 */
esp_err_t radio_gateway_sender_enqueue(uint8_t payload_type,
                                       const uint8_t *payload,
                                       size_t payload_len,
                                       uint64_t received_at_boot_us,
                                       uint32_t timing_uncertainty_us);

#endif /* RADIO_GATEWAY_SENDER_H */
