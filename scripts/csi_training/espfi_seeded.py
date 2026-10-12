"""Seed encoding with optional measured float32 conversion for fixed CSI RFF."""
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
def validate(state, dtype=np.float64):
    if set(state) != set(SHAPES) or any(
        state[k].shape != shape or state[k].dtype != dtype or
        not np.isfinite(state[k]).all() for k, shape in SHAPES.items()
    ) or not (state['feature_scale'] > 0).all():
        raise ValueError('Invalid compact state')

def export_seeded(state, output, precision='float64'):
    if precision not in ('float64', 'float32'):
        raise ValueError('Unsupported precision')
    dtype = np.dtype(precision)
    generated = random_map()
    if set(state) != set(SHAPES) | set(generated) or any(
        not np.array_equal(state[k], generated[k]) for k in generated
    ):
        raise ValueError('Projection does not match frozen seed recipe')
    compact = {k: np.ascontiguousarray(state[k], dtype=dtype) for k in SHAPES}
    validate(compact, dtype)
    root = Path(output)
    root.mkdir(exist_ok=False, parents=True)
    save_file(compact, str(root/'model.safetensors'))
    report = {'status': 'complete', 'promotionAuthorized': False, 'spec': SPEC,
              'precision': precision, 'model_sha256': digest(root/'model.safetensors'),
              'model_bytes': (root/'model.safetensors').stat().st_size,
              'random_map_sha256': {k: tensor_hash(v) for k,v in generated.items()},
              'numpy_version': np.__version__, 'implementation_sha256': digest(__file__),
              'feature_implementation_sha256': digest(Path(__file__).with_name('espfi_generalize.py')),
              'stored_scalars': sum(v.size for v in compact.values()),
              'runtime_scalars': sum(v.size for v in state.values()),
              'runtime_state_bytes': sum(v.size for v in state.values()) * dtype.itemsize,
              'kernel_implementation_sha256': digest(Path(__file__).with_name('espfi_kernel.py')),
              'scope': 'Seed encoded map; float32 requires measured parity, float64 preserves original state'}
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
        precision = report.get('precision', 'float64')
        if precision not in ('float64', 'float32'):
            raise ValueError('Unsupported precision')
        dtype = np.dtype(precision)
        state = load_file(str(root/'model.safetensors'))
        validate(state, dtype)
        generated = random_map()
        if {k:tensor_hash(v) for k,v in generated.items()} != report.get('random_map_sha256'):
            raise ValueError('Runtime random generator differs from export')
        state.update({k: np.ascontiguousarray(v, dtype=dtype) for k, v in generated.items()})
        self.state = state

    def predict(self, amplitudes):
        return predict_features(self.state, features(amplitudes, SPEC['family']))
