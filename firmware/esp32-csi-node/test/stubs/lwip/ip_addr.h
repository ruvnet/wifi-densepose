/* Host-fuzzing stub for lwip/ip_addr.h (ADR-061). Minimal surface for the
 * #954 self-ping block; never functionally exercised in the fuzz env. */
#pragma once

#include <stdint.h>

typedef struct {
    uint32_t addr;
    uint8_t type;
} ip_addr_t;

static inline int ipaddr_aton(const char *cp, ip_addr_t *addr)
{
    (void)cp;
    if (addr != NULL) {
        addr->addr = 0;
        addr->type = 0;
    }
    return 1;
}
