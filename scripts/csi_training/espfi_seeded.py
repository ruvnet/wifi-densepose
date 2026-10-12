"""Lossless seed encoding for the fixed 128-dimensional CSI RFF candidate."""
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file, save_file
from .espfi_generalize import features
from .espfi_kernel import predict_features
from .espfi_train import digest, write_json

SPEC = {'family': 'relative_combined', 'dimensions': 128, 'gamma_multiplier': .1,
        'alpha': .01, 'seed': 42, 'input_features': 468}
SHAPES = {'feature_mean': (468,), 'feature_scale': (468,), 'output_mean': (128,),
          'weights': (128, 7), 'bias': (7,)}
def random_map():
    rng = np.random.default_rng(42)
    return {'projection': rng.normal(0, np.sqrt(2*.1/468), (468, 128)),
            'phase': rng.uniform(0, 2*np.pi, 128)}
def tensor_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def validate(state):
    if set(state) != set(SHAPES) or any(
        state[k].shape != shape or state[k].dtype != np.float64 or
        not np.isfinite(state[k]).all() for k, shape in SHAPES.items()
    ) or not (state['feature_scale'] > 0).all():
        raise ValueError('Invalid compact state')

def export_seeded(state, output):
    generated = random_map()
    if set(state) != set(SHAPES) | set(generated) or any(
        not np.array_equal(state[k], generated[k]) for k in generated
    ):
        raise ValueError('Projection does not match frozen seed recipe')
    compact = {k: np.ascontiguousarray(state[k]) for k in SHAPES}
    validate(compact)
    root = Path(output)
    root.mkdir(exist_ok=False, parents=True)
    save_file(compact, str(root/'model.safetensors'))
    report = {'status': 'complete', 'promotionAuthorized': False, 'spec': SPEC,
              'model_sha256': digest(root/'model.safetensors'),
              'model_bytes': (root/'model.safetensors').stat().st_size,
              'random_map_sha256': {k: tensor_hash(v) for k,v in generated.items()},
              'numpy_version': np.__version__, 'implementation_sha256': digest(__file__),
              'feature_implementation_sha256': digest(Path(__file__).with_name('espfi_generalize.py')),
              'stored_scalars': sum(v.size for v in compact.values()),
              'runtime_scalars': sum(v.size for v in state.values()),
              'scope': 'Lossless on-disk encoding only; projection regenerated at load, runtime memory unchanged'}
    write_json(root/'metadata.json', report)
    return report

class SeededKernelBundle:
    def __init__(self, directory):
        root = Path(directory)
        for name in ('metadata.json', 'model.safetensors'):
            p = root/name
            if p.is_symlink() or not p.is_file() or p.stat().st_size > 65536:
                raise ValueError('Invalid bounded compact artifact')
        report = json.loads((root/'metadata.json').read_text())
        if report.get('status') != 'complete' or report.get('promotionAuthorized') is not False or report.get('spec') != SPEC:
            raise ValueError('Invalid compact contract')
        if digest(root/'model.safetensors') != report.get('model_sha256'):
            raise ValueError('Compact model hash mismatch')
        state = load_file(str(root/'model.safetensors'))
        validate(state)
        generated = random_map()
        if {k:tensor_hash(v) for k,v in generated.items()} != report.get('random_map_sha256'):
            raise ValueError('Runtime random generator differs from export')
        state.update(generated)
        self.state = state

    def predict(self, amplitudes):
        return predict_features(self.state, features(amplitudes, SPEC['family']))
