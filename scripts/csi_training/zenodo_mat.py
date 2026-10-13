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

FRAME_CHUNK = 32


def _diagnostics(data, frames):
    """Aggregate native CSI quality without materializing the full recording."""
    zero_count = total_count = chunks_read = 0
    amplitude_min, amplitude_max = math.inf, 0.0
    antenna_zeros = np.zeros(data.shape[0], dtype=np.uint64)
    frame_nonzero = np.empty(frames, dtype=np.float64)
    worst_fraction = math.inf
    worst_index = 0
    worst_antenna_nonzero = None
    for start in range(0, frames, FRAME_CHUNK):
        stop = min(start + FRAME_CHUNK, frames)
        values = data[:, :, start:stop]
        real, imag = values['real'], values['imag']
        finite = np.isfinite(real) & np.isfinite(imag)
        if not finite.all():
            raise ValueError('Nonfinite CSI in inspected frames')
        amplitude = np.hypot(real, imag)
        if not np.isfinite(amplitude).all():
            raise ValueError('Amplitude overflow')
        zeros = amplitude == 0
        count = amplitude.size
        zero_count += int(zeros.sum(dtype=np.uint64))
        total_count += count
        antenna_zeros += zeros.sum(axis=(1, 2), dtype=np.uint64)
        frame_nonzero[start:stop] = (~zeros).mean(axis=(0, 1))
        local_index = int(np.argmin(frame_nonzero[start:stop]))
        local_min = float(frame_nonzero[start + local_index])
        # Strict comparison keeps the earliest frame when minima tie, including
        # across chunk boundaries. Copy only one bounded antenna vector.
        if local_min < worst_fraction:
            worst_fraction = local_min
            worst_index = start + local_index
            worst_antenna_nonzero = (~zeros[:, :, local_index]).mean(axis=1).tolist()
        amplitude_min = min(amplitude_min, float(amplitude.min()))
        amplitude_max = max(amplitude_max, float(amplitude.max()))
        chunks_read += 1
    per_antenna_denominator = data.shape[1] * frames
    return {
        'inspected_frames': frames,
        'decoded_complex_values': total_count,
        'chunks_read': chunks_read,
        'frame_chunk_limit': FRAME_CHUNK,
        'finite_fraction': 1.0,
        'zero_amplitude_fraction': zero_count / total_count,
        'zero_amplitude_fraction_by_antenna': [
            int(value) / per_antenna_denominator for value in antenna_zeros],
        'frame_nonzero_fraction_min': float(frame_nonzero.min()),
        'worst_frame_index_zero_based': worst_index,
        'worst_frame_tie_count': int((frame_nonzero == worst_fraction).sum()),
        'worst_frame_nonzero_fraction_by_antenna': worst_antenna_nonzero,
        'frame_nonzero_fraction_p05': float(np.quantile(frame_nonzero, .05)),
        'frame_nonzero_fraction_median': float(np.median(frame_nonzero)),
        'frame_nonzero_fraction_mean': float(frame_nonzero.mean()),
        'all_zero_frames': int((frame_nonzero == 0).sum()),
        'amplitude_min': amplitude_min,
        'amplitude_max': amplitude_max,
    }


def inspect(path, expected_sha256, all_frames=False):
    """Inspect the first chunk or whole recording after identity/metadata checks.

    Run in a resource-limited process for hostile files: these checks do not
    sandbox the native HDF5 parser. Parent directories must be trusted.
    """
    if type(all_frames) is not bool:
        raise ValueError('all_frames must be boolean')
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
        frames = shape[2] if all_frames else min(FRAME_CHUNK, shape[2])
        report = _diagnostics(data, frames)
        report.update({'sha256': expected_sha256,
                       'source_shape_antenna_subcarrier_frame': list(shape),
                       'all_csi_frames_decoded': frames == shape[2],
                       'validation_scope': 'csi_dataset_values_only',
                       'training_eligible': False, 'promotionAuthorized': False,
                       'occupancy_inferred': False})
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--all-frames', action='store_true')
    args = parser.parse_args()
    print(json.dumps(inspect(args.path, args.sha256, args.all_frames), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
