# ESP32 build-only examples

**STATUS: `SYNTHETIC / L0` — build-only, never flashed.** These two minimal
ESP-IDF apps exist only to prove `veil_ris_controller` and
`veil_sensing_detector` actually compile and link against a real ESP-IDF
toolchain (v5.4, `esp32s3` target). Building successfully is not a runtime or
on-air claim — see `../README.md`.

```
idf.py set-target esp32s3
idf.py build
```

Both were built and verified locally against ESP-IDF v5.4 (`xtensa-esp32s3-elf`,
GCC 14.2.0); the resulting `.bin`/`.elf` are attached to the GitHub release.
Building surfaced two real compile errors in the underlying components, both
fixed here:

- `veil_sensing_detector/CMakeLists.txt` declared `PRIV_REQUIRES esp_mqtt`;
  the actual ESP-IDF v5.4 component is named `mqtt`.
- Two `ESP_LOGI(..., "%u", ...)` calls passed a bare `uint32_t` where the
  toolchain's `-Werror=format=` requires an explicit `(unsigned)` cast.
