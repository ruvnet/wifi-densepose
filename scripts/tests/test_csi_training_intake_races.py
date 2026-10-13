"""Synthetic mutation probes for the production training intake boundary."""
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from csi_training import data


class IntakeRaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.json'
        self.contents = []
        entries = []
        for i, split in enumerate(data.SPLITS):
            content = b''.join((json.dumps({'timestamp': t, 'node_id': i,
                          'features': [i + t + k / 10 for k in range(8)]})+'\n').encode()
                          for t in range(3))
            self.contents.append(content)
            (self.root / f'{i}.jsonl').write_bytes(content)
            entries.append({'path': f'{i}.jsonl', 'sha256': hashlib.sha256(content).hexdigest(),
                            'captured_at': '2026-10-13T00:00:00Z', 'room_id': str(i),
                            'session_id': str(i), 'split': split, 'evidence': 'synthetic',
                            'use_permitted': True})
        self.manifest.write_text(json.dumps({'schema_version': 1, 'files': entries}))

    def after_inventory(self, change):
        original = data._manifest
        def wrapped(path):
            result = original(path)
            change()
            return result
        return patch.object(data, '_manifest', wrapped)

    def load(self):
        return data.load_manifest(self.manifest, allow_synthetic=True)

    def test_symlink_swap_after_inventory(self):
        def swap():
            p = self.root / '0.jsonl'; target = self.root / 'moved.jsonl'
            p.rename(target); p.symlink_to(target)
        with self.after_inventory(swap):
            with self.assertRaises(data.DataError): self.load()

    def test_aggregate_bytes_rechecked_during_read(self):
        for i in range(3): (self.root / f'{i}.jsonl').write_bytes(b'')
        def grow():
            for i, content in enumerate(self.contents):
                (self.root / f'{i}.jsonl').write_bytes(content)
        with patch.object(data, 'MAX_TOTAL_BYTES', sum(map(len, self.contents))-1):
            with self.after_inventory(grow):
                with self.assertRaisesRegex(data.DataError, 'total dataset'):
                    self.load()

    def test_fifo_swap_rejected_without_blocking(self):
        # Subprocess timeout makes a regression to blocking open fail safely.
        code = """
from test_csi_training_intake_races import IntakeRaceTests
from csi_training import data
import os
case=IntakeRaceTests();case.setUp()
def swap():
    p=case.root/'0.jsonl';p.unlink();os.mkfifo(p)
try:
    with case.after_inventory(swap):
        try:case.load()
        except data.DataError:pass
        else:raise AssertionError('FIFO accepted')
finally:case.doCleanups()
"""
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([
            str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]))
        result = subprocess.run([sys.executable, '-c', code], env=env,
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_descriptor_stays_bound_after_path_replacement(self):
        original = data._open_regular
        replaced = False
        @contextmanager
        def wrapped(path, maximum):
            nonlocal replaced
            with original(path, maximum) as opened:
                # First encounter is the inventory check. Replace only after
                # inventory, on the descriptor that will actually be parsed.
                if Path(path).name == '0.jsonl' and replaced:
                    target = self.root/'0.jsonl'
                    target.rename(self.root/'old.jsonl')
                    target.write_bytes(b'not the opened data')
                yield opened
        def mark():
            nonlocal replaced
            replaced = True
        with self.after_inventory(mark), patch.object(data, '_open_regular', wrapped):
            self.assertEqual(len(self.load().partitions['train'].features), 3)

    def test_manifest_symlink_and_bounded_read(self):
        moved = self.root/'manifest-original.json'
        self.manifest.rename(moved);self.manifest.symlink_to(moved)
        with self.assertRaises(data.DataError):self.load()
        self.manifest.unlink();moved.rename(self.manifest)
        with patch.object(data, 'MAX_MANIFEST_BYTES', 10):
            with self.assertRaises(data.DataError):self.load()

    def test_manifest_growth_after_descriptor_check(self):
        original = data._open_regular
        @contextmanager
        def wrapped(path, maximum):
            with original(path, maximum) as opened:
                if Path(path) == self.manifest:
                    with self.manifest.open('ab') as writer:writer.write(b' ')
                yield opened
        with patch.object(data, '_open_regular', wrapped):
            with self.assertRaisesRegex(data.DataError, 'manifest changed'):
                self.load()

    def test_valid_reload_preserves_all_partitions(self):
        first, second = self.load(), self.load()
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(first.inventory, second.inventory)
        for split in data.SPLITS:
            self.assertTrue((first.partitions[split].features == second.partitions[split].features).all())

if __name__ == '__main__': unittest.main()
