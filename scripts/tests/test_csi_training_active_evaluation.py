import copy
import json
from pathlib import Path
import tempfile
import unittest

from csi_training.active_evaluation import evaluate, load


def fixture():
    budget = dict(packets=10, airtime_ms=20, energy_mj=30, available_sensors=["wifi", "radar"])
    sessions = []
    for i in range(4):
        sessions.append(dict(room=f"room{i}", session=f"session{i}",
                             budget=dict(baseline=budget.copy(), candidate=budget.copy()), intervals=[
            dict(start_ms=0, end_ms=1000, truth="empty", baseline="occupied", candidate="empty"),
            dict(start_ms=1000, end_ms=2000, truth="occupied", baseline="occupied", candidate="occupied")]))
    return dict(schemaVersion=1, evidence="SYNTHETIC", sessions=sessions)


class ActiveEvaluationTests(unittest.TestCase):
    def test_duration_and_gate_never_authorize_promotion(self):
        result = evaluate(fixture())
        self.assertEqual(result["metrics"]["false_duration_relative_reduction"], 1)
        self.assertEqual(result["numericalGate"], "pass")
        self.assertEqual(result["fieldAcceptance"], "not_established")
        self.assertFalse(result["promotionAuthorized"])

    def test_abstention_cannot_hide_recall_loss(self):
        doc = fixture()
        for session in doc["sessions"]:
            session["intervals"][1]["candidate"] = "unknown"
        result = evaluate(doc)
        self.assertEqual(result["metrics"]["candidate_recall"], 0)
        self.assertEqual(result["numericalGate"], "fail")

    def test_zero_baseline_or_one_room_is_inconclusive(self):
        doc = fixture()
        for session in doc["sessions"]:
            session["intervals"][0]["baseline"] = "empty"
        self.assertEqual(evaluate(doc)["numericalGate"], "inconclusive")
        self.assertEqual(evaluate({**fixture(), "sessions": fixture()["sessions"][:1]})["numericalGate"], "inconclusive")

    def test_abstention_on_empty_cannot_game_false_occupied_duration(self):
        doc = fixture()
        for session in doc["sessions"]:
            session["intervals"][0]["candidate"] = "unknown"
        result = evaluate(doc)
        self.assertEqual(result["metrics"]["false_duration_relative_reduction"], 1)
        self.assertEqual(result["metrics"]["unknown_empty_delta"], 1)
        self.assertEqual(result["numericalGate"], "fail")

    def test_recall_interval_not_conditioned_on_false_events(self):
        doc = fixture()
        for session in doc["sessions"]:
            session["intervals"][0]["baseline"] = "empty"
        result = evaluate(doc)
        self.assertEqual(result["bootstrap"]["valid"], 0)
        self.assertEqual(result["bootstrap"]["recall_valid"], 1000)
        self.assertEqual(result["bootstrap"]["recall_delta_interval"], [0, 0])

    def test_room_clustering_keeps_sessions_together(self):
        doc = fixture()
        doc["sessions"][1]["room"] = "room0"
        self.assertEqual(evaluate(doc)["rooms"], 3)

    def test_invalid_budget_timestamp_schema_and_truth(self):
        mutations = [lambda d: d["sessions"][0]["budget"]["candidate"].update(packets=11),
                     lambda d: d["sessions"][0]["intervals"][1].update(start_ms=500),
                     lambda d: d["sessions"][0]["intervals"][1].update(end_ms=float("nan")),
                     lambda d: d["sessions"][0]["intervals"][1].update(truth="unknown"),
                     lambda d: d["sessions"][0]["budget"]["candidate"].update(available_sensors=["extra"]),
                     lambda d: d.update(schemaVersion=True),
                     lambda d: d["sessions"].append(copy.deepcopy(d["sessions"][0]))]
        for mutation in mutations:
            doc = fixture()
            mutation(doc)
            with self.assertRaises(ValueError):
                evaluate(doc)

    def test_strict_file_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(fixture()))
            self.assertEqual(evaluate(load(path))["numericalGate"], "pass")
            path.write_text('{"schemaVersion":1,"schemaVersion":2}')
            with self.assertRaises(ValueError):
                load(path)
            link = Path(directory) / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                load(link)


if __name__ == "__main__":
    unittest.main()
