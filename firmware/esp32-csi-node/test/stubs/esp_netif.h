/* Host-fuzzing stub for esp_netif.h (ADR-061).
 *
 * csi_collector.c's #954 self-ping needs the STA netif handle + gateway IP.
 * In the fuzz environment there is no network stack: the handle lookup
 * returns NULL, so csi_start_self_ping() takes its no-gateway early-out and
 * the esp_ping path is never exercised (but must compile and link).
 */
#pragma once

#include <stdint.h>
#include <stdio.h>

#include "esp_err.h"

typedef struct esp_netif_obj esp_netif_t;

typedef struct {
    uint32_t addr;
} esp_ip4_addr_t;

typedef struct {
    esp_ip4_addr_t ip;
    esp_ip4_addr_t netmask;
    esp_ip4_addr_t gw;
} esp_netif_ip_info_t;

static inline esp_netif_t *esp_netif_get_handle_from_ifkey(const char *if_key)
{
    (void)if_key;
    return NULL; /* no netif in fuzz env -> self-ping early-out */
}

static inline esp_err_t esp_netif_get_ip_info(esp_netif_t *netif, esp_netif_ip_info_t *ip_info)
{
    (void)netif;
    (void)ip_info;
    return ESP_FAIL;
}

static inline char *esp_ip4addr_ntoa(const esp_ip4_addr_t *addr, char *buf, int buflen)
{
    if (buf != NULL && buflen > 0) {
        snprintf(buf, (size_t)buflen, "%u.%u.%u.%u",
                 (unsigned)(addr->addr & 0xff), (unsigned)((addr->addr >> 8) & 0xff),
                 (unsigned)((addr->addr >> 16) & 0xff), (unsigned)((addr->addr >> 24) & 0xff));
    }
    return buf;
}
