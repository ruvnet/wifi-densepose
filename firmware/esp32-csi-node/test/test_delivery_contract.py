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
RELEASE_BINS = FIRMWARE_ROOT / "release_bins"


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


def declared_version(path: Path) -> str:
    """The version a bundle claims in its `version.txt`.

    The first line is the version; the root bundle adds `git-sha`, `built`
    and `note` lines below it.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return line.strip()
    raise ValueError(f"{path} declares no version")


def checksum_entries(sums: Path, bundle: Path):
    """Yield (digest, path) for every entry of a `SHA256SUMS.txt`.

    Two formats are tracked: bundle-relative (`bootloader.bin`) and
    repository-relative (`*firmware/.../bundle/bootloader.bin`). Both are
    resolved so every bundle can be checked the same way.
    """
    for line in sums.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        digest, name = line.split()
        name = name.lstrip("*")
        candidate = bundle / name
        if not candidate.exists():
            candidate = REPOSITORY_ROOT / name
        yield digest, candidate


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

    def test_every_tracked_release_bundle_is_self_describing(self) -> None:
        """Each bundle must state the version it ships and list its own bytes.

        Only `c6-onboarding/` was checked before this, so a bundle could
        drift without anything noticing: the root bundle's `version.txt`
        said 0.6.7 while its application descriptor said 0.7.0, and three
        bundles were not covered at all.
        """
        bundles = sorted(p for p in RELEASE_BINS.iterdir() if p.is_dir())
        self.assertTrue(bundles, "no release bundle found under release_bins/")
        for bundle in bundles:
            with self.subTest(bundle=bundle.name):
                app = bundle / "esp32-csi-node.bin"
                self.assertTrue(app.exists(), f"{bundle.name} ships no application image")

                version_file = bundle / "version.txt"
                self.assertTrue(
                    version_file.exists(),
                    f"{bundle.name} declares no version.txt",
                )
                self.assertEqual(
                    app_version(app),
                    declared_version(version_file),
                    f"{bundle.name} version.txt disagrees with its application descriptor",
                )

                sums = bundle / "SHA256SUMS.txt"
                self.assertTrue(
                    sums.exists(),
                    f"{bundle.name} lists no SHA256SUMS.txt",
                )
                listed = set()
                for digest, path in checksum_entries(sums, bundle):
                    self.assertTrue(
                        path.exists(),
                        f"{bundle.name} lists a file that is not tracked: {path}",
                    )
                    self.assertEqual(
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        digest,
                        f"{bundle.name} checksum mismatch for {path.name}",
                    )
                    listed.add(path.name)
                for binary in sorted(bundle.glob("*.bin")):
                    self.assertIn(
                        binary.name,
                        listed,
                        f"{bundle.name} does not list {binary.name} in SHA256SUMS.txt",
                    )

    def test_release_bundle_root_note_is_backed_by_its_application_image(self) -> None:
        """A bundle must not claim a firmware fix its own image does not contain.

        The root bundle's note claimed the RuView#893 display-less
        DATA-capture fix, but the image next to it is v0.7.0 (2026-06-02)
        and predates the guard that implements that fix -- the FT3168
        readback bail-out in 65e29ef4 (2026-06-17), which first shipped in
        v0.8.1. A board flashed from this bundle on a bare DevKit keeps the
        MGMT-only filter and yields 0 pps.
        """
        note = " ".join(
            (RELEASE_BINS / "version.txt").read_text(encoding="utf-8").splitlines()[1:]
        )
        app = (RELEASE_BINS / "esp32-csi-node.bin").read_bytes()
        if b"FT3168 touch readback" in app:
            return  # the image carries the guard, so any claim is backed by it
        self.assertNotIn(
            "RuView#893",
            note,
            "the root bundle claims the #893 display-less fix, but its image "
            "predates the guard that implements it",
        )
        self.assertIn(
            "v0.8.1",
            note,
            "the root bundle note must name the release that carries the fix",
        )

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
