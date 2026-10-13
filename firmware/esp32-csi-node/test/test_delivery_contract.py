#!/usr/bin/env python3
"""Static delivery contracts that do not require an ESP-IDF toolchain."""

from pathlib import Path
import argparse
import hashlib
import sys
import tempfile
import unittest


FIRMWARE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = FIRMWARE_ROOT.parents[1]


def app_version(path: Path) -> str:
    """Read the first segment's ESP-IDF app descriptor, not arbitrary strings.

    IDF image header + segment header occupy 32 bytes; esp_app_desc_t has
    magic 0xABCD5432 and a 32-byte version at descriptor offset 16.
    """
    with path.open("rb") as stream:
        header = stream.read(80)
    if (len(header) != 80 or header[0] != 0xE9
            or header[32:36] != bytes.fromhex("3254cdab")):
        raise ValueError("missing ESP-IDF application descriptor")
    return header[48:80].split(b"\0", 1)[0].decode("ascii")


class FirmwareDeliveryContractTests(unittest.TestCase):
    def test_c6_uses_native_usb_serial_jtag_as_primary_console(self) -> None:
        config = (FIRMWARE_ROOT / "sdkconfig.defaults.esp32c6").read_text(encoding="utf-8")
        self.assertIn("CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y", config)
        self.assertNotIn("CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG=y", config)

        onboarding = (FIRMWARE_ROOT / "main/serial_onboarding.c").read_text(encoding="utf-8")
        self.assertNotIn('open("/dev/secondary"', onboarding)

    def test_historical_c6_snapshot_matches_its_own_version_and_checksums(self) -> None:
        snapshot = FIRMWARE_ROOT / "release_bins/c6-onboarding"
        expected = (snapshot / "version.txt").read_text(encoding="utf-8").strip()
        self.assertEqual(app_version(snapshot / "esp32-csi-node.bin"), expected)
        for line in (snapshot / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
            digest, filename = line.split()
            self.assertEqual(hashlib.sha256((snapshot / filename).read_bytes()).hexdigest(), digest)

    def test_app_version_rejects_truncated_or_wrong_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app.bin"
            for data in [b"", b"0.8.14\0" * 20, b"\xe9" + b"\0" * 79]:
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    app_version(path)

    def test_firmware_tags_publish_downloadable_release_assets(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/firmware-ci.yml").read_text(encoding="utf-8")
        self.assertIn("publish-release-assets:", workflow)
        self.assertIn("esp32-csi-node-firmware-*", workflow)
        self.assertIn("softprops/action-gh-release@", workflow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--built-app", type=Path)
    args, remaining = parser.parse_known_args()
    if args.built_app:
        expected = (FIRMWARE_ROOT / "version.txt").read_text(encoding="utf-8").strip()
        actual = app_version(args.built_app)
        if actual != expected:
            raise SystemExit(f"built application version {actual!r} does not match {expected!r}")
        print(f"PASS: built application descriptor version {actual}")
    unittest.main(argv=[sys.argv[0], *remaining])
