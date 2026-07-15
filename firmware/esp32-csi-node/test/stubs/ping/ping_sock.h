/* Host-fuzzing stub for ping/ping_sock.h (ADR-061). The #954 self-ping is
 * unreachable in the fuzz env (esp_netif stub returns no gateway), but the
 * symbols must compile and link. */
#pragma once

#include <stdint.h>

#include "esp_err.h"
#include "lwip/ip_addr.h"

typedef void *esp_ping_handle_t;

typedef void (*esp_ping_cb_t)(esp_ping_handle_t hdl, void *args);

typedef struct {
    uint32_t count;
    uint32_t interval_ms;
    uint32_t timeout_ms;
    uint32_t data_size;
    uint8_t tos;
    int ttl;
    ip_addr_t target_addr;
    uint32_t task_stack_size;
    uint32_t task_prio;
    uint32_t interface;
} esp_ping_config_t;

#define ESP_PING_COUNT_INFINITE (0)

#define ESP_PING_DEFAULT_CONFIG()       \
    {                                   \
        .count = 5,                     \
        .interval_ms = 1000,            \
        .timeout_ms = 1000,             \
        .data_size = 64,                \
        .tos = 0,                       \
        .ttl = 64,                      \
        .target_addr = {0, 0},          \
        .task_stack_size = 2048,        \
        .task_prio = 2,                 \
        .interface = 0,                 \
    }

typedef struct {
    void *cb_args;
    esp_ping_cb_t on_ping_success;
    esp_ping_cb_t on_ping_timeout;
    esp_ping_cb_t on_ping_end;
} esp_ping_callbacks_t;

static inline esp_err_t esp_ping_new_session(const esp_ping_config_t *config,
                                             const esp_ping_callbacks_t *cbs,
                                             esp_ping_handle_t *hdl_out)
{
    (void)config;
    (void)cbs;
    if (hdl_out != NULL) {
        *hdl_out = (void *)0;
    }
    return ESP_FAIL; /* never starts a ping task in the fuzz env */
}

static inline esp_err_t esp_ping_start(esp_ping_handle_t hdl)
{
    (void)hdl;
    return ESP_OK;
}

static inline esp_err_t esp_ping_stop(esp_ping_handle_t hdl)
{
    (void)hdl;
    return ESP_OK;
}

static inline esp_err_t esp_ping_delete_session(esp_ping_handle_t hdl)
{
    (void)hdl;
    return ESP_OK;
}
