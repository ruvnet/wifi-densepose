"""Bounded v7.3 CSI diagnostic. Native axes retained; not a training adapter."""
import argparse
import hashlib
import io
import json
import math

import h5py
import numpy as np

from .data import _open_regular
from .zenodo_sample import MAX_MEMBER


def inspect(path, expected_sha256):
    """Read at most 32 frames after checking file identity and HDF metadata.

    Run in a resource-limited process for hostile files: these checks do not
    sandbox the native HDF5 parser. Parent directories must be trusted.
    """
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('Expected SHA256 required')
    with _open_regular(path, MAX_MEMBER) as (stream, size):
        raw = stream.read(MAX_MEMBER + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Sample size or hash mismatch')
    with h5py.File(io.BytesIO(raw), 'r') as source:
        if not isinstance(source.get('csi', getlink=True), h5py.HardLink):
            raise ValueError('CSI must be a local hard link')
        data = source['csi']
        if not isinstance(data, h5py.Dataset):
            raise ValueError('CSI dataset required')
        shape = data.shape
        if (shape is None or len(shape) != 3 or not 1 <= shape[0] <= 8
                or not 1 <= shape[1] <= 1024 or not 1 <= shape[2] <= 100000
                or data.dtype != np.dtype([('real', '<f8'), ('imag', '<f8')])):
            raise ValueError('Unsupported CSI dimensions or compound dtype')
        if data.is_virtual or data.external:
            raise ValueError('External or virtual CSI storage forbidden')
        properties = data.id.get_create_plist()
        if any(properties.get_filter(i)[0] not in (1, 2, 3)
               for i in range(properties.get_nfilters())):
            raise ValueError('Only builtin deflate, shuffle and Fletcher32 filters allowed')
        if data.chunks and math.prod(data.chunks) * data.dtype.itemsize > MAX_MEMBER:
            raise ValueError('Oversized decompression chunk')
        # MATLAB v7.3 reverses dimension order: antenna, subcarrier, frame.
        values = data[:, :, :min(32, shape[2])]
        real, imag = values['real'], values['imag']
        finite = np.isfinite(real) & np.isfinite(imag)
        if not finite.all():
            raise ValueError('Nonfinite CSI in inspected window')
        amplitude = np.hypot(real, imag)
        if not np.isfinite(amplitude).all():
            raise ValueError('Amplitude overflow')
        return {'sha256': expected_sha256, 'source_shape_antenna_subcarrier_frame': list(shape),
                'inspected_frames': values.shape[2], 'finite_fraction': float(finite.mean()),
                'zero_amplitude_fraction': float((amplitude == 0).mean()),
                'amplitude_min': float(amplitude.min()), 'amplitude_max': float(amplitude.max()),
                'full_recording_validated': False, 'promotionAuthorized': False,
                'occupancy_inferred': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.path, args.sha256), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
