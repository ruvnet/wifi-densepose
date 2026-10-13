import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "evaluate-active-evidence.py"


def fixture():
    budget = dict(packets=10, airtime_ms=20, energy_mj=30,
                  available_sensors=["wifi", "radar"])
    sessions = []
    for index in range(4):
        sessions.append({
            "room": f"room{index}",
            "session": f"session{index}",
            "budget": {"baseline": budget.copy(),
                       "candidate": budget.copy()},
            "intervals": [
                {"start_ms": 0, "end_ms": 1000, "truth": "empty",
                 "baseline": "occupied", "candidate": "empty"},
                {"start_ms": 1000, "end_ms": 2000,
                 "truth": "occupied", "baseline": "unknown",
                 "candidate": "occupied"},
                {"start_ms": 2000, "end_ms": 3000,
                 "truth": "occupied", "baseline": "occupied",
                 "candidate": "occupied"},
                {"start_ms": 3000, "end_ms": 4000, "truth": "empty",
                 "baseline": "occupied", "candidate": "empty"},
                {"start_ms": 4000, "end_ms": 5000, "truth": "empty",
                 "baseline": "empty", "candidate": "empty"},
            ],
        })
    return {"schemaVersion": 1, "evidence": "SYNTHETIC",
            "sessions": sessions}


class ActiveEvaluationCliTests(unittest.TestCase):
    def run_cli(self, document, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            if isinstance(document, str):
                path.write_text(document)
            else:
                path.write_text(json.dumps(document))
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(SCRIPT.parent)
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(path), *arguments],
                text=True, capture_output=True, env=environment, timeout=10,
                check=False)

    def test_require_pass_returns_zero_only_for_overall_pass(self):
        process = self.run_cli(fixture(), "--require-pass")
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(process.stdout)
        self.assertEqual(report["overallGate"], "pass")
        self.assertFalse(report["promotionAuthorized"])

    def test_require_pass_rejects_fail_after_printing_report(self):
        document = fixture()
        for session in document["sessions"]:
            session["intervals"][1]["candidate"] = "unknown"
            session["intervals"][2]["candidate"] = "unknown"
        process = self.run_cli(document, "--require-pass")
        self.assertEqual(process.returncode, 3)
        self.assertEqual(json.loads(process.stdout)["overallGate"], "fail")
        self.assertIn("active evaluation gate: fail", process.stderr)

    def test_require_pass_rejects_inconclusive(self):
        document = fixture()
        document["sessions"] = document["sessions"][:1]
        process = self.run_cli(document, "--require-pass")
        self.assertEqual(process.returncode, 3)
        self.assertEqual(json.loads(process.stdout)["overallGate"],
                         "inconclusive")

    def test_default_remains_report_only_and_invalid_input_is_two(self):
        document = fixture()
        document["sessions"] = document["sessions"][:1]
        process = self.run_cli(document)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["overallGate"],
                         "inconclusive")
        invalid = self.run_cli("{not json", "--require-pass")
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(invalid.stdout, "")


if __name__ == "__main__":
    unittest.main()
