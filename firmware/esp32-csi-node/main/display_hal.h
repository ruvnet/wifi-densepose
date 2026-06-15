/**
 * @file display_hal.h
 * @brief Board-specific ST7789 LCD HAL for Cardputer-Adv and StickC Plus2.
 *
 * Hardware abstraction for the active board display panel.
 * Probes hardware at boot; returns ESP_ERR_NOT_FOUND if absent.
 */

#ifndef DISPLAY_HAL_H
#define DISPLAY_HAL_H

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#if defined(CONFIG_IDF_TARGET_ESP32S3)
#define BOARD_CARDPUTER_ADV 1
#define DISPLAY_PANEL_NAME    "Cardputer-Adv"
#define DISPLAY_PANEL_H_RES   240
#define DISPLAY_PANEL_V_RES   135
#define DISPLAY_PANEL_GAP_X    40
#define DISPLAY_PANEL_GAP_Y    53
#define DISPLAY_PANEL_BL_PIN   38
#define DISPLAY_PANEL_RST_PIN  33
#define DISPLAY_PANEL_DC_PIN   34
#define DISPLAY_PANEL_MOSI_PIN 35
#define DISPLAY_PANEL_SCLK_PIN 36
#define DISPLAY_PANEL_CS_PIN   37
#elif defined(CONFIG_IDF_TARGET_ESP32)
#define BOARD_M5STICKC_PLUS   1
#define DISPLAY_PANEL_NAME    "StickC Plus2"
#define DISPLAY_PANEL_H_RES   135
#define DISPLAY_PANEL_V_RES   240
#define DISPLAY_PANEL_GAP_X    52
#define DISPLAY_PANEL_GAP_Y    40
#define DISPLAY_PANEL_BL_PIN   27
#define DISPLAY_PANEL_RST_PIN  12
#define DISPLAY_PANEL_DC_PIN   14
#define DISPLAY_PANEL_MOSI_PIN 15
#define DISPLAY_PANEL_SCLK_PIN 13
#define DISPLAY_PANEL_CS_PIN    5
#else
#error "DISPLAY_ENABLE is only supported on ESP32-S3 Cardputer-Adv and ESP32 StickC Plus2 targets"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Probe and initialize the active target's ST7789 LCD panel.
 *
 * Configures SPI bus, sends panel init sequence, and fills
 * the screen with dark background to confirm it works.
 * Returns ESP_ERR_NOT_FOUND if the panel does not respond.
 *
 * @return ESP_OK on success, ESP_ERR_NOT_FOUND if no display detected.
 */
esp_err_t display_hal_init_panel(void);

/**
 * Copy a rectangle of pixels into the RAM framebuffer.
 *
 * @param x_start  Left column (inclusive).
 * @param y_start  Top row (inclusive).
 * @param x_end    Right column (exclusive).
 * @param y_end    Bottom row (exclusive).
 * @param color_data  RGB565 pixel data, (x_end-x_start)*(y_end-y_start) pixels.
 */
void display_hal_draw(int x_start, int y_start, int x_end, int y_end,
                      const void *color_data);

/**
 * Push the complete RAM framebuffer to the LCD.
 *
 * The active target screen is small enough for full-frame RGB565 updates.
 * Presenting whole frames avoids partial-window artifacts on the ST7789.
 */
void display_hal_present(void);

/**
 * Probe and initialize touch controller when present.
 *
 * @return ESP_OK on success, ESP_ERR_NOT_FOUND if no touch IC detected.
 */
esp_err_t display_hal_init_touch(void);

/**
 * Read touch point (non-blocking).
 *
 * @param[out] x  Touch X coordinate (0..535).
 * @param[out] y  Touch Y coordinate (0..239).
 * @return true if touch is active, false if released.
 */
bool display_hal_touch_read(uint16_t *x, uint16_t *y);

/**
 * Set LCD backlight state.
 *
 * @param percent  Brightness 0-100.
 */
void display_hal_set_brightness(uint8_t percent);

#ifdef __cplusplus
}
#endif

#endif /* DISPLAY_HAL_H */
