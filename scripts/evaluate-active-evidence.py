#!/usr/bin/env python3
"""Evaluate paired room intervals; prints aggregate metrics, never observations."""
import argparse
import json
from csi_training.active_evaluation import evaluate, load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument(
        "--require-pass", action="store_true",
        help="exit 3 unless the valid evaluation overallGate is pass")
    args = parser.parse_args(argv)
    try:
        result = evaluate(load(args.manifest))
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, f"invalid evaluation input: {type(error).__name__}\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if args.require_pass and result["overallGate"] != "pass":
        parser.exit(3, f"active evaluation gate: {result['overallGate']}\n")
    return result


if __name__ == "__main__":
    main()
