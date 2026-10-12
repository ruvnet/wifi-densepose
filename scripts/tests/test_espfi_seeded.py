"""Lossless seeded-map integrity tests."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_training.espfi_kernel import fit, predict_features
from csi_training.espfi_generalize import features
from csi_training.espfi_seeded import SPEC, export_seeded, SeededKernelBundle

class SeededTests(unittest.TestCase):
    def test_exact_export_and_tamper_rejection(self):
        amplitudes = np.exp(np.random.default_rng(2).normal(size=(14,52,190)))
        x = features(amplitudes, SPEC['family'])
        state = fit(x, np.arange(14)%7, dict(SPEC, kind='rff'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model'
            report = export_seeded(state, path)
            loaded = SeededKernelBundle(path)
            np.testing.assert_array_equal(predict_features(state,x), loaded.predict(amplitudes))
            self.assertLess(report['model_bytes'], 17000)
            self.assertEqual(report['stored_scalars'], 1967)
            self.assertEqual(report['runtime_scalars'], 61999)
            with (path/'model.safetensors').open('ab') as f:
                f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                SeededKernelBundle(path)

    def test_wrong_projection_rejected(self):
        x = np.random.default_rng(1).normal(size=(14,468))
        state = fit(x,np.arange(14)%7,dict(SPEC,kind='rff'))
        state['projection'][0,0] += .1
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'seed recipe'):
                export_seeded(state,Path(tmp)/'model')
