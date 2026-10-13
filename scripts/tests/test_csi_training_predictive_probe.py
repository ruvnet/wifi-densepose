"""Synthetic probe mathematics and train-only information boundaries."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi_training.predictive_probe import NAMES, _fit, metrics, run_probes


def fixture():
    train = np.tile(np.eye(7), (2, 1))
    dev = np.eye(7)
    return {name: {'train': train.copy(), 'dev': dev.copy()} for name in NAMES}, np.tile(np.arange(7), 2), np.arange(7)


class ProbeTests(unittest.TestCase):
    def test_independent_hand_computed_metrics_and_ridge(self):
        result = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 2]))
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macro_f1'], 1/6)
        np.testing.assert_allclose(result['per_class_f1'], [2/3, .5, 0, 0, 0, 0, 0])
        expected = np.zeros((7, 7), dtype=int)
        expected[0, 0] = expected[0, 1] = expected[1, 1] = expected[1, 2] = 1
        np.testing.assert_array_equal(result['confusion'], expected)
        state = _fit(np.array([[-1.], [1.]]), np.array([0, 1]))
        np.testing.assert_allclose(state['weights'], [[-1/3, 1/3, 0, 0, 0, 0, 0]])
        np.testing.assert_allclose(state['bias'], [.5, .5, 0, 0, 0, 0, 0])

    def test_dev_mutation_does_not_fit_and_train_labels_do(self):
        parts, yt, yd = fixture()
        first = run_probes(parts, yt, yd)
        for part in parts.values():
            part['dev'] = part['dev'] * 23 + 11
        second = run_probes(parts, yt, (yd+1) % 7)
        for name in NAMES:
            self.assertEqual(first['probes'][name]['artifact_state_sha256'], second['probes'][name]['artifact_state_sha256'])
        third = run_probes(parts, (yt+1) % 7, yd)
        self.assertNotEqual(first['probes']['baseline']['artifact_state_sha256'], third['probes']['baseline']['artifact_state_sha256'])
        self.assertFalse(first['gate']['passed'])  # Equal perfect probes do not demonstrate lift.
        self.assertEqual(first['majority']['predicted_activity'], 0)

    def test_zero_variance_and_dual_solution(self):
        parts, yt, yd = fixture()
        for part in parts.values():
            part['train'][:] = 5.
            part['dev'][:] = 8.
        result = run_probes(parts, yt, yd)
        self.assertEqual(result['probes']['candidate']['accuracy'], 1/7)
        state = _fit(np.ones((2, 4)), np.array([1, 1]))
        np.testing.assert_array_equal(state['weights'], np.zeros((4, 7)))
        np.testing.assert_array_equal(state['scale'], np.ones(4))
        self.assertEqual(state['bias'][1], 1.)

    def test_new_zero_class_blocks_otherwise_improved_candidate(self):
        parts, yt, yd = fixture()
        for name in ('baseline', 'untrained', 'statistics'):
            parts[name]['dev'][:] = np.eye(7)[0]
            parts[name]['dev'][1] = np.eye(7)[1]
        parts['candidate']['dev'][1] = np.eye(7)[0]
        result = run_probes(parts, yt, yd)
        checks = result['gate']['checks']
        self.assertTrue(checks['macro_f1_at_least_best_plus_002'])
        self.assertTrue(checks['accuracy_within_001_of_best'])
        self.assertFalse(checks['no_new_zero_f1_class'])
        self.assertFalse(result['gate']['passed'])

    def test_input_rejections(self):
        invalid = [np.ones((14, 7), dtype=np.int64), np.ones((14, 7), dtype=bool),
                   np.ones((14, 7), dtype=np.complex128), np.ones((14, 7), dtype=object),
                   np.ones((4097, 7)), np.ones((14, 1025)), np.ones((0, 7)),
                   np.full((14, 7), np.nan), np.full((14, 7), np.inf), np.full((14, 7), 1e101)]
        for value in invalid:
            parts, yt, yd = fixture()
            parts['candidate']['train'] = value
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                run_probes(parts, yt, yd)
        for labels in [np.arange(14, dtype=float), np.ones(14, dtype=bool), np.full(14, 7),
                       np.full(14, -1), np.zeros(13, dtype=int)]:
            parts, _, yd = fixture()
            with self.assertRaises(ValueError): run_probes(parts, labels, yd)
        parts, yt, yd = fixture()
        parts['statistics']['dev'] = np.ones((7, 8))
        with self.assertRaises(ValueError): run_probes(parts, yt, yd)
        with self.assertRaises(ValueError): run_probes({}, yt, yd)

    def test_candidate_lift_gate_and_no_input_mutation(self):
        parts, yt, yd = fixture()
        for name in ('baseline', 'untrained', 'statistics'):
            parts[name]['train'][:] = 1.
            parts[name]['dev'][:] = 1.
        before = {name: {key: value.copy() for key, value in part.items()} for name, part in parts.items()}
        result = run_probes(parts, yt, yd)
        self.assertTrue(result['gate']['passed'])
        self.assertEqual(result['probes']['candidate']['accuracy'], 1.)
        for name in NAMES:
            for split in ('train', 'dev'):
                np.testing.assert_array_equal(parts[name][split], before[name][split])


if __name__ == '__main__':
    unittest.main()
