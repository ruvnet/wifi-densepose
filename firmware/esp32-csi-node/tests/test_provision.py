import csv
import contextlib
import importlib.util
import io
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROVISION_PATH = Path(__file__).resolve().parents[1] / "provision.py"
SPEC = importlib.util.spec_from_file_location("provision", PROVISION_PATH)
provision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provision)


def make_args(**overrides):
    values = {name: None for name, _ in provision.CONFIG_VALUE_CHECKS}
    values["hop_dwell"] = 200
    values.update(overrides)
    return types.SimpleNamespace(**values)


def csv_rows(content):
    return list(csv.DictReader(io.StringIO(content)))


class ProvisionConfigValueTests(unittest.TestCase):
    def test_swarm_and_hopping_flags_count_as_config_values(self):
        cases = [
            {"hop_channels": "1,6,11"},
            {"seed_token": "token-123"},
            {"swarm_hb": 15},
            {"swarm_ingest": 3},
            {"cs_source_id": 1},
            {"radio_envelope_key_id": 0},
        ]

        for values in cases:
            with self.subTest(values=values):
                self.assertTrue(provision.has_config_value(make_args(**values)))

    def test_operational_flags_alone_do_not_count_as_config_values(self):
        self.assertFalse(provision.has_config_value(make_args()))

    def test_swarm_and_hopping_values_are_written_to_csv(self):
        args = make_args(
            hop_channels="1,6,11",
            hop_dwell=250,
            seed_token="token-123",
            swarm_hb=15,
            swarm_ingest=3,
        )

        rows = csv_rows(provision.build_nvs_csv(args))
        values_by_key = {row["key"]: row["value"] for row in rows}

        self.assertEqual(values_by_key["hop_count"], "3")
        self.assertEqual(values_by_key["chan_list"], "01060b")
        self.assertEqual(values_by_key["dwell_ms"], "250")
        self.assertEqual(values_by_key["seed_token"], "token-123")
        self.assertEqual(values_by_key["swarm_hb"], "15")
        self.assertEqual(values_by_key["swarm_ingest"], "3")

    def test_radio_secrets_source_and_key_ids_are_written_as_typed_nvs_values(self):
        args = make_args(
            ble_identity_enable=1,
            ble_key_id=7,
            ble_secret_bytes=bytes(range(32)),
            radio_envelope_key_id=11,
            radio_envelope_secret_bytes=bytes([0xA5]) * 32,
        )
        rows = csv_rows(provision.build_nvs_csv(args))
        values_by_key = {row["key"]: row["value"] for row in rows}
        self.assertEqual(values_by_key["ble_enable"], "1")
        self.assertEqual(values_by_key["ble_key_id"], "7")
        self.assertEqual(values_by_key["ble_secret"], bytes(range(32)).hex())
        self.assertEqual(values_by_key["radio_key_id"], "11")
        self.assertEqual(values_by_key["radio_secret"], (bytes([0xA5]) * 32).hex())
        self.assertNotIn("ble_secret_bytes", provision.MERGEABLE_ATTRS)
        self.assertNotIn("radio_envelope_secret_bytes", provision.MERGEABLE_ATTRS)

        args.cs_ingress_enable = 1
        args.cs_key_id = 9
        args.cs_secret_bytes = bytes(reversed(range(32)))
        args.cs_source_id = 0x11223344
        rows = csv_rows(provision.build_nvs_csv(args))
        values_by_key = {row["key"]: row["value"] for row in rows}
        self.assertEqual(values_by_key["cs_enable"], "1")
        self.assertEqual(values_by_key["cs_key_id"], "9")
        self.assertEqual(values_by_key["cs_secret"], bytes(reversed(range(32))).hex())
        self.assertEqual(values_by_key["cs_source_id"], str(0x11223344))
        self.assertNotIn("cs_secret_bytes", provision.MERGEABLE_ATTRS)

    def test_ble_secret_file_accepts_raw_and_hex(self):
        encoded = bytes(range(32)).hex().encode("ascii")
        for payload in (bytes(range(32)), encoded, encoded + b"\n", encoded + b"\r\n"):
            with self.subTest(length=len(payload)):
                with tempfile.NamedTemporaryFile() as secret_file:
                    secret_file.write(payload)
                    secret_file.flush()
                    self.assertEqual(provision.load_ble_secret(secret_file.name), bytes(range(32)))

    def test_radio_secret_file_rejects_trailing_data(self):
        for payload in (
            bytes(range(32)) + b"x",
            b"00" * 32 + b"\nextra",
            b"00" * 32 + b"\n\nignored",
            b"00" * 32 + b"\r\nextra",
        ):
            with self.subTest(length=len(payload)):
                with tempfile.NamedTemporaryFile() as secret_file:
                    secret_file.write(payload)
                    secret_file.flush()
                    with self.assertRaisesRegex(
                        ValueError, "trailing data|exactly 32 raw bytes"
                    ):
                        provision.load_ble_secret(secret_file.name)

        with tempfile.NamedTemporaryFile() as secret_file:
            secret_file.write(b"gg" * 32)
            secret_file.flush()
            with self.assertRaisesRegex(ValueError, "not valid hexadecimal"):
                provision.load_ble_secret(secret_file.name)

    def test_radio_secrets_reject_zero_and_key_reuse(self):
        for payload in (bytes(32), b"00" * 32, b"00" * 32 + b"\n"):
            with self.subTest(length=len(payload)):
                with tempfile.NamedTemporaryFile() as secret_file:
                    secret_file.write(payload)
                    secret_file.flush()
                    with self.assertRaisesRegex(ValueError, "must not be all zero"):
                        provision.load_ble_secret(secret_file.name)

        key = bytes([0x5a]) * 32
        with self.assertRaisesRegex(ValueError, "independently generated"):
            provision.validate_distinct_radio_secrets(key, None, key)
        provision.validate_distinct_radio_secrets(
            bytes([0x11]) * 32,
            bytes([0x22]) * 32,
            bytes([0x33]) * 32,
        )

    def test_nvs_csv_rejects_non_32_byte_in_memory_secrets(self):
        for attribute in (
            "ble_secret_bytes",
            "cs_secret_bytes",
            "radio_envelope_secret_bytes",
        ):
            with self.subTest(attribute=attribute):
                args = make_args(**{attribute: b"short"})
                with self.assertRaisesRegex(ValueError, "exactly 32 bytes"):
                    provision.build_nvs_csv(args)

        args = make_args(ble_secret_bytes=bytes(32))
        with self.assertRaisesRegex(ValueError, "must not be all zero"):
            provision.build_nvs_csv(args)

    def test_radio_evidence_requires_gateway_envelope_key_and_secret(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ble_secret = Path(temp_dir) / "ble.key"
            ble_secret.write_bytes(bytes([0x11]) * 32)
            base = [
                "provision.py",
                "--port", "TEST",
                "--state-dir", str(Path(temp_dir) / "state"),
                "--force-partial",
                "--ble-identity-enable", "1",
                "--ble-key-id", "7",
                "--ble-secret-file", str(ble_secret),
            ]

            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", base), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    provision.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--radio-envelope-key-id is required", stderr.getvalue())

            stderr = io.StringIO()
            with mock.patch.object(
                sys, "argv", base + ["--radio-envelope-key-id", "8"]
            ), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    provision.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--radio-envelope-secret-file is required", stderr.getvalue())

    def test_channel_sounding_requires_enrolled_nonzero_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cs_secret = Path(temp_dir) / "cs.key"
            radio_secret = Path(temp_dir) / "radio.key"
            cs_secret.write_bytes(bytes([0x22]) * 32)
            radio_secret.write_bytes(bytes([0x33]) * 32)
            base = [
                "provision.py",
                "--port", "TEST",
                "--state-dir", str(Path(temp_dir) / "state"),
                "--force-partial",
                "--cs-ingress-enable", "1",
                "--cs-key-id", "9",
                "--cs-secret-file", str(cs_secret),
                "--radio-envelope-key-id", "10",
                "--radio-envelope-secret-file", str(radio_secret),
            ]

            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", base), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    provision.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--cs-source-id is required", stderr.getvalue())

            stderr = io.StringIO()
            with mock.patch.object(
                sys, "argv", base + ["--cs-source-id", "0"]
            ), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    provision.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("must be between 1 and 4294967295", stderr.getvalue())

    def test_secret_configuration_never_writes_fallback_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir, contextlib.chdir(temp_dir):
            radio_secret = Path(temp_dir) / "radio.key"
            radio_secret.write_bytes(bytes([0x44]) * 32)
            argv = [
                "provision.py",
                "--port", "TEST",
                "--state-dir", str(Path(temp_dir) / "state"),
                "--force-partial",
                "--radio-envelope-key-id", "12",
                "--radio-envelope-secret-file", str(radio_secret),
            ]
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                provision, "generate_nvs_binary", side_effect=RuntimeError("missing")
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    provision.main()

            self.assertEqual(raised.exception.code, 1)
            self.assertFalse(Path("nvs_config.csv").exists())
            self.assertIn("Refusing to persist fallback NVS CSV", stderr.getvalue())

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_secret_outputs_and_state_are_mode_0600(self):
        with tempfile.TemporaryDirectory() as temp_dir, contextlib.chdir(temp_dir):
            radio_secret = Path(temp_dir) / "radio.key"
            cs_secret = Path(temp_dir) / "cs.key"
            radio_secret.write_bytes(bytes([0x55]) * 32)
            cs_secret.write_bytes(bytes([0x66]) * 32)
            output = Path("nvs_provision.bin")
            output.write_bytes(b"old")
            output.chmod(0o644)
            state_dir = Path(temp_dir) / "state"
            state_dir.mkdir()
            state_path = Path(provision._state_path_for("TEST", str(state_dir)))
            state_path.write_text("{}\n")
            state_path.chmod(0o644)
            argv = [
                "provision.py",
                "--port", "TEST",
                "--state-dir", str(state_dir),
                "--ssid", "test-network",
                "--password", "test-password",
                "--target-ip", "192.0.2.10",
                "--dry-run",
                "--cs-ingress-enable", "1",
                "--cs-key-id", "9",
                "--cs-secret-file", str(cs_secret),
                "--cs-source-id", "0x11223344",
                "--radio-envelope-key-id", "13",
                "--radio-envelope-secret-file", str(radio_secret),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                provision, "generate_nvs_binary", return_value=b"private-nvs"
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                provision.main()

            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(state_path.stat().st_mode), 0o600)
            state_text = state_path.read_text()
            self.assertIn('"password": "test-password"', state_text)
            self.assertNotIn("radio_envelope_secret", state_text)


if __name__ == "__main__":
    unittest.main()
