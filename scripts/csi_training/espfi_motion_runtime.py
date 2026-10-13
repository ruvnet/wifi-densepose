"""Bounded float64 ACF inference using only NumPy and safetensors.

One locked workspace per instance; outputs own their storage. No training,
Torch, SciPy, recording persistence or artifact mutation occurs here.
Scratch retains transformed last-window values until reuse; it is not memory erasure.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
from threading import Lock
import numpy as np
from safetensors.numpy import load

BASELINE = dict(family='motion_acf', kind='rff', dimensions=128, gamma_multiplier=.1, alpha=.1)
SHAPES = {'feature_mean': (468,), 'feature_scale': (468,), 'output_mean': (128,),
          'weights': (128, 7), 'bias': (7,)}
SPEC = {'candidate': BASELINE, 'seed': 42, 'rng': 'numpy.default_rng.PCG64',
        'recipe': 'normal(0,sqrt(2*.1/468),(468,128)), then uniform(0,2*pi,128)',
        'precision': 'float64', 'features': 'unchanged motion_features motion_acf',
        'max_abs_logit_error': 1e-10, 'promotionAuthorized': False}


def random_map():
    rng = np.random.default_rng(42)
    return {'projection': rng.normal(0, np.sqrt(2 * (.1 / 468)), (468, 128)),
            'phase': rng.uniform(0, 2*np.pi, 128)}


def tensor_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate(state):
    if (set(state) != set(SHAPES) or any(state[k].shape != shape or state[k].dtype != np.float64
            or not np.isfinite(state[k]).all() for k, shape in SHAPES.items())
            or not (state['feature_scale'] > 0).all()):
        raise ValueError('Invalid compact float64 ACF state')


def _file_identity(details):
    return (details.st_dev, details.st_ino, stat.S_IFMT(details.st_mode),
            details.st_size, details.st_mtime_ns, details.st_ctime_ns)


def _open_directory(path):
    """Open and pin the artifact directory used by one bundle load."""
    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_DIRECTORY'):
        raise ValueError('Secure artifact intake unavailable')
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    if hasattr(os, 'O_CLOEXEC'):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as exc:
        raise ValueError('Expected artifact directory') from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError('Expected artifact directory')
        return descriptor
    except (OSError, ValueError):
        os.close(descriptor)
        raise


def _read(path, directory_fd=None):
    """Read a small bounded artifact through the descriptor that was checked."""
    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_NONBLOCK'):
        raise ValueError('Secure artifact intake unavailable')
    target = Path(path)
    if directory_fd is not None and target != Path(target.name):
        raise ValueError('Artifact name must be relative to its directory')
    if (directory_fd is not None and
            (not hasattr(os, 'supports_dir_fd') or os.open not in os.supports_dir_fd)):
        raise ValueError('Secure artifact intake unavailable')
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, 'O_CLOEXEC'):
        flags |= os.O_CLOEXEC
    if hasattr(os, 'O_BINARY'):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(target.name if directory_fd is not None else target,
                             flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError('Expected regular bounded artifact') from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size <= 0:
            raise ValueError('Expected regular bounded artifact')
        if details.st_size > 65536:
            raise ValueError('Oversized artifact')
        chunks = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise ValueError('Artifact changed while reading')
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError('Artifact changed while reading')
        after = os.fstat(descriptor)
        if _file_identity(after) != _file_identity(details):
            raise ValueError('Artifact changed while reading')
        return b''.join(chunks)
    except OSError as exc:
        raise ValueError('Unable to read bounded artifact') from exc
    finally:
        os.close(descriptor)


def load_state(directory):
    root = Path(directory)
    directory_fd = _open_directory(root)
    try:
        status = json.loads(_read('status.json', directory_fd))
        if not isinstance(status,dict) or status.get('status') != 'complete' or status.get('promotionAuthorized') is not False:
            raise ValueError('Incomplete compact ACF artifact')
        metadata = json.loads(_read('metadata.json', directory_fd))
        if (not isinstance(metadata,dict) or metadata.get('spec') != SPEC or metadata.get('status') != 'complete'
                or metadata.get('promotionAuthorized') is not False):
            raise ValueError('Invalid compact ACF metadata')
        data = _read('model.safetensors', directory_fd)
        if hashlib.sha256(data).hexdigest() != metadata.get('model_sha256'):
            raise ValueError('Compact model hash mismatch')
        compact = load(data)
        validate(compact)
        generated = random_map()
        if {k:tensor_hash(v) for k,v in generated.items()} != metadata.get('random_map_sha256'):
            raise ValueError('Regenerated map differs from export')
    finally:
        os.close(directory_fd)
    projection = np.ascontiguousarray(generated['projection']/compact['feature_scale'][:,None])
    state = {'projection':projection,
             'phase':generated['phase']-compact['feature_mean']@projection,
             'bias':compact['bias']-compact['output_mean']@compact['weights'],
             'weights':compact['weights']}
    if any(not np.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite folded state')
    return state


class BufferedMotionBundle:
    """Serializes calls per instance; use separate instances for parallel streams."""
    def __init__(self, directory):
        self.state = load_state(directory)
        self._lock = Lock()
        self._centered = np.empty((52,190),dtype=np.float64)
        self._scratch = np.empty_like(self._centered)
        self._mean = np.empty((52,1),dtype=np.float64)
        self._energy = np.empty(52,dtype=np.float64)
        self._features = np.empty((1,468),dtype=np.float64)
        self._per_channel = self._features.reshape(52,9)
        self._hidden = np.empty((1,128),dtype=np.float64)
        self._scores = np.empty((1,7),dtype=np.float64)
        self._normalizer = np.float64(np.sqrt(2./128))

    @property
    def workspace_bytes(self):
        """Retained scratch, excluding caller input/owned output and model state."""
        return sum(getattr(self,k).nbytes for k in
                   ('_centered','_scratch','_mean','_energy','_features','_hidden','_scores'))

    @property
    def model_state_bytes(self):
        return sum(v.nbytes for v in self.state.values())

    def _extract(self, sample):
        # Explicit float64 loop prevents float32 floor rounding near constant data.
        np.maximum(sample,1e-6,out=self._centered,dtype=np.float64)
        np.log(self._centered,out=self._centered)
        self._centered.mean(axis=-1,keepdims=True,out=self._mean)
        self._centered -= self._mean
        np.square(self._centered,out=self._scratch)
        scale = max(float(np.sqrt(self._scratch.mean())),1e-6)
        self._scratch.sum(axis=-1,out=self._energy)
        np.maximum(self._energy,190e-12,out=self._energy)
        for i,lag in enumerate((1,2,4,8,16,32)):
            # Direct dot reduction removes a product temporary. Reduction order
            # differs from the reference; parity is measured at <=1e-10 logits.
            np.einsum('ij,ij->i',self._centered[:,lag:],self._centered[:,:-lag],
                      out=self._per_channel[:,i],optimize=False)
            self._per_channel[:,i] /= self._energy
        self._centered /= scale
        for i,lag in enumerate((1,4,16),6):
            scratch = self._scratch[:,:190-lag]
            np.subtract(self._centered[:,lag:],self._centered[:,:-lag],out=scratch)
            np.abs(scratch,out=scratch)
            scratch.mean(axis=-1,out=self._per_channel[:,i])
        return self._features

    def predict(self, values):
        x = np.asarray(values)
        if (x.dtype.kind not in 'fiu' or x.ndim != 3 or x.shape[1:] != (52,190)
                or not 0 < len(x) <= 4096):
            raise ValueError('Expected 1..4096 real amplitude windows N52T190')
        # Validate within the fixed-window loop: no batch-sized temporary mask.
        result = np.empty((len(x),7),dtype=np.float64)
        with self._lock:
            for i,sample in enumerate(x):
                if not np.isfinite(sample).all() or (sample < 0).any():
                    raise ValueError('Expected finite nonnegative amplitudes')
                self._extract(sample)
                np.matmul(self._features,self.state['projection'],out=self._hidden)
                self._hidden += self.state['phase']
                np.cos(self._hidden,out=self._hidden)
                self._hidden *= self._normalizer
                np.matmul(self._hidden,self.state['weights'],out=self._scores)
                self._scores += self.state['bias']
                if not np.isfinite(self._scores).all():
                    raise ValueError('Nonfinite runtime prediction')
                result[i] = self._scores[0]
        return result
