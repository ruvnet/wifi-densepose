from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "main" / "wasm_upload.c"


class WasmAuthContractTests(unittest.TestCase):
    def test_every_wasm_handler_checks_ota_auth_before_work(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("ota_check_auth(req)", source[source.index("wasm_require_ota_auth"):])
        for handler in (
            "wasm_upload_handler",
            "wasm_list_handler",
            "wasm_start_handler",
            "wasm_stop_handler",
            "wasm_delete_handler",
        ):
            start = source.index(handler)
            body = source[start:source.find("\n}", start) + 2]
            self.assertIn("wasm_require_ota_auth(req)", body, handler)
