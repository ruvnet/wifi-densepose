"""Offline authored ZIP/HTTP/HDF fixtures; never public sensing accuracy."""
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import h5py
import numpy as np

from csi_training import zenodo_sample as z
from csi_training.zenodo_mat import inspect


class Response(io.BytesIO):
    status = 206
    def __init__(self, value=b'ab', **headers):
        super().__init__(value)
        self.headers = {'Content-Range': f'bytes 0-1/{z.ARCHIVE_BYTES}', **headers}
    def geturl(self): return z.URL


class ZenodoTests(unittest.TestCase):
    def test_http_requires_exact_range_and_stable_etag(self):
        ranges = z.HttpRanges()
        with patch.object(ranges.opener, 'open', return_value=Response()):
            self.assertEqual(ranges(0, 2), b'ab')
        for response in [Response(b'a'), Response(b'abc'), Response(**{'ETag': '"new"'}),
                         Response(**{'Content-Encoding': 'gzip'}),
                         Response(**{'Content-Range': 'bytes 0-1/3'})]:
            with self.subTest(headers=response.headers), patch.object(ranges.opener, 'open', return_value=response):
                with self.assertRaises(ValueError): ranges(0, 2)
        response = Response(); response.status = 200
        with patch.object(ranges.opener, 'open', return_value=response):
            with self.assertRaises(ValueError): ranges(0, 2)
        with self.assertRaises(ValueError):
            z.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere/')

    def test_strong_etag_is_sent_on_followup(self):
        ranges = z.HttpRanges()
        for _ in range(2):
            with patch.object(ranges.opener, 'open', return_value=Response(ETag='"fixed"')) as opened:
                ranges(0, 2)
        self.assertEqual(opened.call_args.args[0].get_header('If-match'), '"fixed"')
        weak = z.HttpRanges()
        with patch.object(weak.opener, 'open', return_value=Response(ETag='W/"weak"')):
            with self.assertRaises(ValueError): weak(0, 2)

    def test_range_budgets_and_failed_response(self):
        reader = z.RangeReader(lambda start, count: b'x' * count)
        for args in [(-1,), (True,), (0, 3), (z.ARCHIVE_BYTES + 1,)]:
            with self.assertRaises(ValueError): reader.seek(*args)
        with self.assertRaises(ValueError): reader.read(z.MAX_REQUEST + 1)
        for _ in range(3): self.assertEqual(len(reader.read(z.MAX_REQUEST)), z.MAX_REQUEST)
        with self.assertRaises(ValueError): reader.read(1)
        reader = z.RangeReader(lambda start, count: b'')
        with self.assertRaises(ValueError): reader.read(2)
        self.assertEqual(reader.tell(), 0)
        reader.started -= 121
        with self.assertRaises(ValueError): reader.read(1)
        reader = z.RangeReader(lambda start, count: b'x' * count)
        reader.requests = 128
        with self.assertRaises(ValueError): reader.read(1)

    def test_zip_duplicates_symlinks_crc_and_size(self):
        for kind in ('duplicate', 'symlink', 'oversized', 'crc'):
            buf = io.BytesIO()
            with warnings.catch_warnings(), zipfile.ZipFile(buf, 'w') as archive:
                warnings.simplefilter('ignore', UserWarning)
                info = zipfile.ZipInfo('selected')
                if kind == 'symlink': info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, b'fixture')
                if kind == 'duplicate': archive.writestr('selected', b'duplicate')
            value = buf.getvalue()
            if kind == 'crc': value = value.replace(b'fixture', b'changed')
            with zipfile.ZipFile(io.BytesIO(value)) as archive:
                if kind == 'oversized': archive.getinfo('selected').file_size = z.MAX_MEMBER + 1
                with self.subTest(kind=kind), self.assertRaises((ValueError, zipfile.BadZipFile)):
                    z.member_bytes(archive, 'selected')

    def test_complete_intake_is_synthetic_and_never_overwrites(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as archive:
            for session in z.SESSIONS.values():
                for name in ('config.json', 'capture1_extracted.mat'):
                    archive.writestr('Zenodo/NoShield/beacon/' + session + '/' + name, b'fixture')
        value = buf.getvalue()
        original = z.RangeReader
        with tempfile.TemporaryDirectory() as tmp, patch.object(z, 'RangeReader', side_effect=lambda f: original(f, len(value))):
            path = Path(tmp) / 'new'
            result = z.intake(path, lambda start, count: value[start:start+count])
            self.assertEqual(result['evidence'], 'SYNTHETIC')
            self.assertEqual(len(result['artifacts']), 6)
            self.assertFalse(result['snapshot_verified'])
            self.assertFalse(result['promotionAuthorized'])
            with self.assertRaises(FileExistsError): z.intake(path)
            with self.assertRaises(ValueError): z.intake(Path(tmp)/'bad', lambda s, c: b'')
            self.assertEqual(json.loads((Path(tmp)/'bad/status.json').read_text())['status'], 'failed')

    def fixture(self, path, mode='normal'):
        with h5py.File(path, 'w') as f:
            if mode == 'link':
                f['csi'] = h5py.ExternalLink('never-open.mat', '/csi'); return
            if mode == 'soft':
                f['csi'] = h5py.SoftLink('/never-open'); return
            dtype = np.dtype([('real', '<f8'), ('imag', '<f8')])
            if mode == 'dtype': dtype = np.dtype('float64')
            if mode == 'external':
                f.create_dataset('csi', (2, 3, 40), dtype=dtype, external=[('never-read', 0, 4000)]); return
            if mode == 'huge':
                f.create_dataset('csi', (2, 3, 100001), dtype=dtype); return
            if mode == 'chunk':
                f.create_dataset('csi', (8, 1024, 40), dtype=dtype, chunks=(8, 1024, 40)); return
            if mode == 'virtual':
                layout = h5py.VirtualLayout((2, 3, 40), dtype=dtype)
                layout[:] = h5py.VirtualSource('never-read', 'source', shape=(2, 3, 40))
                f.create_virtual_dataset('csi', layout); return
            values = np.zeros((2, 3, 40), dtype=dtype)
            if mode != 'dtype':
                values['real'] = 3; values['imag'] = 4
                if mode == 'nan': values['real'][0, 0, 0] = np.nan
                if mode == 'tail_nan': values['real'][0, 0, 39] = np.nan
                if mode == 'tail_zero':
                    values['real'][:, :, 39] = 0
                    values['imag'][:, :, 39] = 0
            f.create_dataset('csi', data=values, chunks=(2, 3, 4), compression='lzf' if mode == 'filter' else 'gzip')

    def test_mat_native_axes_bounded_window_and_no_occupancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sample.mat'; self.fixture(path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            report = inspect(path, sha)
            self.assertEqual(report['source_shape_antenna_subcarrier_frame'], [2, 3, 40])
            self.assertEqual(report['inspected_frames'], 32)
            self.assertEqual(report['amplitude_min'], 5)
            self.assertFalse(report['occupancy_inferred'])
            self.assertFalse(report['all_csi_frames_decoded'])
            self.assertEqual(report['validation_scope'], 'csi_dataset_values_only')
            self.assertFalse(report['training_eligible'])
            self.assertEqual(report['chunks_read'], 1)
            complete = inspect(path, sha, True)
            self.assertTrue(complete['all_csi_frames_decoded'])
            self.assertEqual(complete['inspected_frames'], 40)
            self.assertEqual(complete['chunks_read'], 2)
            self.assertEqual(complete['decoded_complex_values'], 240)
            self.assertEqual(complete['frame_nonzero_fraction_min'], 1)
            self.assertEqual(complete['worst_frame_index_zero_based'], 0)
            self.assertEqual(complete['worst_frame_tie_count'], 40)
            self.assertEqual(complete['worst_frame_nonzero_fraction_by_antenna'], [1, 1])
            self.assertEqual(complete['zero_amplitude_fraction_by_antenna'], [0, 0])
            with self.assertRaises(ValueError): inspect(path, '0' * 64)
            with self.assertRaises(ValueError): inspect(path, sha, 1)
            link = Path(tmp)/'link'; link.symlink_to(path)
            with self.assertRaises(ValueError): inspect(link, sha)

    def test_full_scan_detects_unread_tail_and_reports_dropouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)/'bad'; self.fixture(bad, 'tail_nan')
            sha = hashlib.sha256(bad.read_bytes()).hexdigest()
            self.assertEqual(inspect(bad, sha)['inspected_frames'], 32)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                inspect(bad, sha, True)
            zeros = Path(tmp)/'zeros'; self.fixture(zeros, 'tail_zero')
            report = inspect(zeros, hashlib.sha256(zeros.read_bytes()).hexdigest(), True)
            self.assertEqual(report['all_zero_frames'], 1)
            self.assertEqual(report['frame_nonzero_fraction_min'], 0)
            self.assertEqual(report['zero_amplitude_fraction'], 1/40)
            self.assertEqual(report['zero_amplitude_fraction_by_antenna'], [1/40, 1/40])
            self.assertEqual(report['worst_frame_index_zero_based'], 39)
            self.assertEqual(report['worst_frame_tie_count'], 1)
            self.assertEqual(report['worst_frame_nonzero_fraction_by_antenna'], [0, 0])

    def test_worst_frame_ties_scope_and_native_antenna_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sample.mat'; self.fixture(path)
            with h5py.File(path, 'r+') as source:
                values = source['csi'][:]
                # Equal aggregate minima on either side of the chunk boundary,
                # but different antenna coverage. Keep the earlier vector.
                values[0, :, 31] = (0, 0)
                values[1, :, 32] = (0, 0)
                source['csi'][:] = values
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            window = inspect(path, sha)
            complete = inspect(path, sha, True)
            for report in (window, complete):
                self.assertEqual(report['worst_frame_index_zero_based'], 31)
                self.assertEqual(report['worst_frame_nonzero_fraction_by_antenna'], [0, 1])
                self.assertEqual(report['frame_nonzero_fraction_min'], .5)
                self.assertFalse(report['training_eligible'])
                self.assertFalse(report['promotionAuthorized'])
            self.assertEqual(window['worst_frame_tie_count'], 1)
            self.assertEqual(complete['worst_frame_tie_count'], 2)

    def test_worst_frame_does_not_read_beyond_bounded_chunks(self):
        from csi_training.zenodo_mat import _diagnostics, FRAME_CHUNK
        class BoundedDataset:
            shape = (2, 3, 65)
            calls = []
            def __getitem__(self, index):
                first, second, frames = index
                self.calls.append((frames.start, frames.stop))
                self.assert_bounded(frames)
                values = np.zeros((2, 3, frames.stop - frames.start),
                                  dtype=[('real', '<f8'), ('imag', '<f8')])
                values['real'] = 1
                if frames.stop == 65:
                    values[:, :, -1] = (0, 0)
                return values
            def assert_bounded(self, frames):
                assert 0 < frames.stop - frames.start <= FRAME_CHUNK
        data = BoundedDataset()
        report = _diagnostics(data, 65)
        self.assertEqual(data.calls, [(0, 32), (32, 64), (64, 65)])
        self.assertEqual(report['worst_frame_index_zero_based'], 64)
        self.assertEqual(report['worst_frame_tie_count'], 1)
        self.assertEqual(report['worst_frame_nonzero_fraction_by_antenna'], [0, 0])

    def test_mat_rejects_links_external_storage_dimensions_dtype_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('link', 'soft', 'external', 'huge', 'dtype', 'nan', 'chunk', 'virtual', 'filter'):
                path = Path(tmp)/mode; self.fixture(path, mode)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    inspect(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__': unittest.main()
