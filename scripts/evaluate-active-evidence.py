#!/usr/bin/env python3
"""Evaluate paired room intervals; prints aggregate metrics, never observations."""
import argparse
import json
from csi_training.active_evaluation import evaluate, load

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(load(args.manifest)), indent=2, allow_nan=False))
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, f"invalid evaluation input: {type(error).__name__}\n")
