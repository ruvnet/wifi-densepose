#ifndef ESP32CAM_DUAL_STREAM_H
#define ESP32CAM_DUAL_STREAM_H

#include "esp_err.h"
#include "esp_http_server.h"

esp_err_t esp32cam_dual_stream_register(httpd_handle_t server);

#endif /* ESP32CAM_DUAL_STREAM_H */
