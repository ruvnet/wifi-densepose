"""Fixed label-dependent ridge probes for frozen activity representations.

Only training labels/features fit the probes. Development metrics never select
hyperparameters. Numeric activity IDs 0..6 have no occupancy semantics.
"""
import hashlib
import json

import numpy as np

NAMES = ('baseline', 'candidate', 'untrained', 'statistics')
CONFIG = {'alpha': 1.0, 'intercept': True, 'scaling': 'train_mean_population_std',
          'zero_variance_scale': 1.0, 'classes': list(range(7)),
          'macro_f1_classes': 'all_seven_including_absent', 'selection': 'none'}


def _features(value):
    if (not isinstance(value, np.ndarray) or value.ndim != 2
            or value.dtype.kind != 'f' or value.dtype.itemsize not in (4, 8)
            or not 1 <= value.shape[0] <= 4096 or not 1 <= value.shape[1] <= 1024):
        raise ValueError('Expected bounded float32/float64 N-by-D features')
    if not np.isfinite(value).all() or np.any(np.abs(value) > 1e100):
        raise ValueError('Features must be finite and magnitude <= 1e100')
    return value.astype(np.float64, copy=True)


def _labels(value, length=None):
    if (not isinstance(value, np.ndarray) or value.ndim != 1
            or value.dtype.kind not in 'iu' or not 1 <= len(value) <= 4096
            or (length is not None and len(value) != length)
            or np.any(value < 0) or np.any(value > 6)):
        raise ValueError('Expected aligned integer activity labels 0..6 (no bool)')
    return value.astype(np.int64, copy=True)


def metrics(y, prediction):
    y = _labels(y)
    prediction = _labels(prediction, len(y))
    confusion = np.bincount(7*y + prediction, minlength=49).reshape(7, 7)
    true_positive = np.diag(confusion)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2.*true_positive, denominator, out=np.zeros(7), where=denominator != 0)
    return {'accuracy': float(true_positive.sum()/len(y)), 'macro_f1': float(f1.mean()),
            'per_class_f1': f1.tolist(), 'confusion': confusion.tolist(),
            'confusion_axes': 'rows=true numeric activity, columns=predicted numeric activity'}


def _state_hash(state):
    result = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode())
    for name in ('mean', 'scale', 'weights', 'bias'):
        value = np.ascontiguousarray(state[name], dtype='<f8')
        result.update(name.encode())
        result.update(json.dumps(value.shape).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def _fit(x, y):
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale == 0] = 1.
    z = (x-mean)/scale
    # Center after scaling to account for floating-point residual mean; this
    # fits an unpenalized intercept even for nearly constant input columns.
    z_mean = z.mean(0)
    centered = z-z_mean
    target = np.eye(7)[y]
    target_mean = target.mean(0)
    centered_target = target-target_mean
    if x.shape[1] <= x.shape[0]:
        weights = np.linalg.solve(centered.T@centered + np.eye(x.shape[1]),
                                  centered.T@centered_target)
    else:
        weights = centered.T@np.linalg.solve(centered@centered.T + np.eye(x.shape[0]),
                                             centered_target)
    state = {'mean': mean, 'scale': scale, 'weights': weights,
             'bias': target_mean-z_mean@weights}
    if not all(np.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite fitted probe state')
    return state


def _predict(state, x):
    # Reject overflow in transformation or prediction instead of reporting a
    # spurious class zero from nonfinite scores.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            scores = ((x-state['mean'])/state['scale'])@state['weights']+state['bias']
        except FloatingPointError as error:
            raise ValueError('Probe transformation overflow') from error
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite probe scores')
    return scores.argmax(1)


def run_probes(embeddings, y_train, y_dev):
    """Return JSON-compatible metrics and fitted-state hashes, never raw data.

    ``embeddings`` must have exactly baseline/candidate/untrained/statistics,
    each mapping to train and dev floating arrays with matching feature width.
    Hashes attest fitted numeric state, not externally verified source identity;
    callers must bind this report to their split/encoder/data provenance.
    """
    if not isinstance(embeddings, dict) or set(embeddings) != set(NAMES):
        raise ValueError('Expected four frozen representation comparators')
    y_train, y_dev = _labels(y_train), _labels(y_dev)
    parts = {}
    for name in NAMES:
        part = embeddings[name]
        if not isinstance(part, dict) or set(part) != {'train', 'dev'}:
            raise ValueError('Expected train/dev features only')
        train, dev = _features(part['train']), _features(part['dev'])
        if (train.shape[0] != len(y_train) or dev.shape[0] != len(y_dev)
                or train.shape[1] != dev.shape[1]):
            raise ValueError('Feature/label alignment or width mismatch')
        parts[name] = train, dev
    probes = {}
    for name, (train, dev) in parts.items():
        state = _fit(train, y_train)
        probes[name] = dict(metrics(y_dev, _predict(state, dev)),
                            feature_count=train.shape[1], training_samples=len(y_train),
                            training_parameters=json.loads(json.dumps(CONFIG)), artifact_state_sha256=_state_hash(state),
                            fitted_scalar_count=sum(v.size for v in state.values()))
    majority_class = int(np.bincount(y_train, minlength=7).argmax())
    majority = dict(metrics(y_dev, np.full(len(y_dev), majority_class, dtype=np.int64)),
                    predicted_activity=majority_class, tie_break='lowest numeric activity')
    candidate = probes['candidate']
    comparisons = [probes[name] for name in ('baseline', 'untrained', 'statistics')]
    best_f1, best_accuracy = max(p['macro_f1'] for p in comparisons), max(p['accuracy'] for p in comparisons)
    checks = {'macro_f1_at_least_best_plus_002': candidate['macro_f1'] >= best_f1+.02,
              'accuracy_within_001_of_best': candidate['accuracy'] >= best_accuracy-.01,
              'no_new_zero_f1_class': all(c > 0 or b == 0 for c, b in
                                         zip(candidate['per_class_f1'], probes['baseline']['per_class_f1']))}
    return {'schema_version': 1, 'task': 'frozen_numeric_activity_linear_probe',
            'promotionAuthorized': False, 'classes': list(range(7)), 'probes': probes,
            'majority': majority, 'gate': {'checks': checks, 'passed': all(checks.values()),
                'best_comparator_macro_f1': best_f1, 'best_comparator_accuracy': best_accuracy,
                'macro_f1_margin': .02, 'accuracy_tolerance': .01},
            'scope': 'Fixed alpha=1; train-only scaling and label fit. Development labels score only. '
                     'No hyperparameter search or representation selection. Caller supplies provenance '
                     'and separately evaluates representation collapse and latency.'}
