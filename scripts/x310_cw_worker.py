#!/usr/bin/env python3
"""Retired X310 CW worker compatibility stub.

This entry point no longer sends data to RuView. The old experiment emitted an
unversioned feature shape with unvalidated human-sensing semantics and is kept
only at ``archive/experiments/x310_cw_unvalidated_experiment.py`` for audit
reproduction. Use ``scripts/openisac_to_ruview_bridge.py`` for the supported,
versioned observation-only transport.
"""

from __future__ import annotations

import argparse
import sys


RETIREMENT_MESSAGE = (
    "x310_cw_worker.py is retired and fails closed; use "
    "scripts/openisac_to_ruview_bridge.py for versioned rf-direct observations"
)


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main() -> int:
    build_parser().parse_known_args()
    print(RETIREMENT_MESSAGE, file=sys.stderr)
    return 78


if __name__ == "__main__":
    raise SystemExit(main())
