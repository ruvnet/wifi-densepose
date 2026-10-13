#!/usr/bin/env python3
"""Verify the authored active replay and ranking authority invariants."""
import argparse
import json
from csi_training.active_evaluation import load


def verify(reference, candidate):
    expected = ["empty", "occupied", "unknown", "occupied", "unknown"]
    for report in (reference, candidate):
        if report.get("evidence") != "SYNTHETIC" or report.get("schemaVersion") != 1:
            raise ValueError("explicit synthetic replay required")
        if [step["state"] for step in report["steps"]] != expected:
            raise ValueError("incorrect evidence lifecycle")
        if report["authority"] != dict(actuation=False, promotion=False, rankingChangesEvidence=False):
            raise ValueError("authority expanded")
        if report["budgetStop"]["requested"] or not report["worldGraph"]["roundTripPreserved"]:
            raise ValueError("budget or graph invariant failed")
        for step in report["steps"]:
            if step["proposal"]["actuated"] or step["scoreIsProbability"]:
                raise ValueError("invalid evidence authority")
            if step["proposal"]["requested"] and step["proposal"]["sensorId"] not in ("wifi", "radar", "door"):
                raise ValueError("unregistered sensor")
    for left, right in zip(reference["steps"], candidate["steps"]):
        if {k: v for k, v in left.items() if k != "proposal"} != {k: v for k, v in right.items() if k != "proposal"}:
            raise ValueError("ranker changed evidence or belief")
    return dict(status="pass", evidence="SYNTHETIC", states=expected,
                rankingChangesEvidence=False, promotionAuthorized=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(load(args.reference), load(args.candidate))))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"active replay rejected: {type(error).__name__}\n")
