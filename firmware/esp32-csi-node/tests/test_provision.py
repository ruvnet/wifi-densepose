import csv
import importlib.util
import io
import json
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
    @patch("provision.getpass.getpass", return_value="prompted-secret")
    def test_missing_password_uses_secure_prompt(self, getpass):
        self.assertEqual(provision.read_wifi_password(None), "prompted-secret")
        getpass.assert_called_once()

    def test_provisioning_source_has_no_plaintext_fallback_file(self):
        source = PROVISION_PATH.read_text(encoding="utf-8")
        self.assertNotIn('fallback_path = "nvs_config.csv"', source)

    def test_state_files_never_persist_or_reuse_passwords(self):
        with tempfile.TemporaryDirectory() as state_dir:
            path = provision.save_state(
                "COM7", state_dir, {"password": "secret", "ssid": "wifi"}
            )
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertNotIn("password", saved)

            Path(path).write_text(
                json.dumps({"password": "legacy-secret", "node_id": 2}),
                encoding="utf-8",
            )
            self.assertNotIn("password", provision.load_state("COM7", state_dir))

    def test_swarm_and_hopping_flags_count_as_config_values(self):
        cases = [
            {"hop_channels": "1,6,11"},
            {"seed_token": "token-123"},
            {"swarm_hb": 15},
            {"swarm_ingest": 3},
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


if __name__ == "__main__":
    unittest.main()
