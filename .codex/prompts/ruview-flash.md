# /ruview-flash — build + flash ESP32 firmware

Build and flash RuView ESP32 firmware. Variant + port: `$ARGUMENTS` (default `8mb`, port `COM8`).

1. **Variant.** `8mb` → ensure it builds from `firmware/esp32-csi-node/sdkconfig.defaults.template` (no mock — real WiFi CSI). `4mb` → `cp firmware/esp32-csi-node/sdkconfig.defaults.4mb firmware/esp32-csi-node/sdkconfig.defaults` first (display disabled, dual OTA via `partitions_4mb.csv`). `heltec` → `sdkconfig.defaults.heltec_n16r2`.
2. **Build (Windows).** ESP-IDF v5.4 does NOT work under Git Bash; `cmd.exe /C` hangs. Use the Espressif Python venv as a subprocess with `MSYSTEM*` env vars stripped — the exact command is in `CLAUDE.local.md` (`[python, idf_py, 'build']`, cwd = `firmware/esp32-csi-node`). Outputs in `firmware/esp32-csi-node/build/{bootloader/bootloader.bin, partition_table/partition-table.bin, esp32-csi-node.bin, ota_data_initial.bin}`.
3. **Flash.** Same subprocess with `[python, idf_py, '-p', 'COM8', 'flash']`, or:
   ```
   python -m esptool --chip esp32s3 --port COM8 --baud 460800 write_flash \
     0x0 firmware/esp32-csi-node/build/bootloader/bootloader.bin \
     0x8000 firmware/esp32-csi-node/build/partition_table/partition-table.bin \
     0xf000 firmware/esp32-csi-node/build/ota_data_initial.bin \
     0x20000 firmware/esp32-csi-node/build/esp32-csi-node.bin
   ```
4. **Confirm.** Serial monitor via pyserial on `COM8` @ 115200 (NOT `idf.py monitor` — it hangs in a subprocess). Then `cd v2 && cargo run -p wifi-densepose-sensing-server` — frames should arrive. If not: re-run `/ruview-provision`, match the AP channel, drop any `--filter-mac`.

Never test in mock mode — the Kconfig fall-threshold bug only showed up with real CSI.
