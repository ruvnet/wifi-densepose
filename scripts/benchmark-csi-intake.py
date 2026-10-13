#!/usr/bin/env python3
"""Synthetic intake benchmark. --baseline must be the exact pinned trusted source."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc
import types

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tests'))
from test_csi_training_intake_races import IntakeRaceTests
from csi_training import data

BASELINE_SHA = 'e50e5305b8678e7dc9039d99faa8cc71c20132af786bdd1b3538def73ef49055'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', required=True)
args = parser.parse_args()
with data._open_regular(args.baseline, 1024 * 1024) as (stream, _):
    source = stream.read(1024 * 1024 + 1)
if hashlib.sha256(source).hexdigest() != BASELINE_SHA:
    raise SystemExit('Expected pinned baseline source; arbitrary Python is not admitted')
baseline = types.ModuleType('intake_baseline');sys.modules[baseline.__name__] = baseline
exec(compile(source, args.baseline, 'exec'), baseline.__dict__)
case = IntakeRaceTests();case.setUp()
try:
    old = baseline.load_manifest(case.manifest, allow_synthetic=True)
    new = data.load_manifest(case.manifest, allow_synthetic=True)
    assert old.manifest_sha256 == new.manifest_sha256 and old.inventory == new.inventory
    for split in data.SPLITS:
        a, b = old.partitions[split], new.partitions[split]
        assert (a.features == b.features).all() and (a.timestamps == b.timestamps).all()
        assert a.node_ids == b.node_ids and a.groups == b.groups and a.session_ids == b.session_ids
    modules = {'baseline': baseline, 'candidate': data}
    timings = {key: [] for key in modules}
    for _ in range(20):
        for module in modules.values(): module.load_manifest(case.manifest, allow_synthetic=True)
    for i in range(1000):
        for key in (list(modules) if i % 2 else list(reversed(modules))):
            before = time.perf_counter_ns()
            modules[key].load_manifest(case.manifest, allow_synthetic=True)
            timings[key].append((time.perf_counter_ns() - before) / 1e6)
    peaks = {}
    for key, module in modules.items():
        tracemalloc.start();module.load_manifest(case.manifest, allow_synthetic=True)
        peaks[key] = tracemalloc.get_traced_memory()[1];tracemalloc.stop()
    print(json.dumps({'evidence': 'SYNTHETIC', 'baseline_commit': '721a094f09b97422e32555fc8444cdaa17c5fd3a',
        'baseline_source_sha256': BASELINE_SHA,
        'candidate_source_sha256': hashlib.sha256(Path(data.__file__).read_bytes()).hexdigest(),
        'valid_partition_parity': True, 'samples_per_arm': 1000, 'fixture_records': 9,
        'fixture_files': 3, 'warmups_per_arm': 20,
        'timing': {k: {'p50_ms': sorted(v)[500], 'p95_ms': sorted(v)[950]} for k,v in timings.items()},
        'python_tracemalloc_peak_bytes': peaks, 'python': platform.python_version(),
        'scope': 'Paired alternating full intake from warm filesystem cache. Shared CPU; tracemalloc excludes native allocations. No inference or accuracy claim.',
        'promotionAuthorized': False}, indent=2))
finally:
    case.doCleanups()
