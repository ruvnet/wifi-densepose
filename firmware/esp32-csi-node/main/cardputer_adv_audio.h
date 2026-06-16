/**
 * @file cardputer_adv_audio.h
 * @brief Cardputer-Adv ES8311 speaker and SD WAV playback support.
 */

#ifndef CARDPUTER_ADV_AUDIO_H
#define CARDPUTER_ADV_AUDIO_H

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run the configured boot-time audio proof on Cardputer-Adv.
 *
 * On non-ADV targets or when disabled by Kconfig this returns ESP_OK without
 * touching GPIO, I2C, I2S, or SD hardware.
 */
esp_err_t cardputer_adv_audio_startup_probe(void);

/**
 * Play a PCM WAV file from the Cardputer-Adv microSD slot.
 *
 * The path must include the ESP-IDF mount point, for example
 * "/sdcard/ruview.wav". Supported input is unsigned 8-bit or signed 16-bit
 * PCM, mono or stereo.
 */
esp_err_t cardputer_adv_audio_play_wav_from_sd(const char *path);

/**
 * Feed a Cardputer-Adv keyboard event to the embedded Trautonium voice.
 *
 * The key code is the raw TCA8418 event code without the press bit. When the
 * synth is disabled this is a no-op, so display code can call it unconditionally
 * behind the audio feature flag.
 */
void cardputer_adv_audio_key_event(uint8_t key_code, bool pressed);

#ifdef __cplusplus
}
#endif

#endif /* CARDPUTER_ADV_AUDIO_H */
