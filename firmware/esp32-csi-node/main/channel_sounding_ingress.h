/** @file channel_sounding_ingress.h */

#ifndef CHANNEL_SOUNDING_INGRESS_H
#define CHANNEL_SOUNDING_INGRESS_H

#include "esp_err.h"

/** Start the optional external Channel Sounding UART ingress. */
esp_err_t channel_sounding_ingress_init(void);

#endif /* CHANNEL_SOUNDING_INGRESS_H */
