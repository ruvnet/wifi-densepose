"""Buffered inference parity, ownership and standalone dependency boundaries."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from safetensors.numpy import save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from csi_training.espfi_motion_runtime import BufferedMotionBundle, SPEC, SHAPES, _open_directory, _read, random_map, tensor_hash
from csi_training.espfi_motion_compact import CompactMotionBundle
from csi_training.espfi_motion import MotionBundle, motion_features
from csi_training.espfi_train import write_json
from test_espfi_context import make_baseline
from test_espfi_generalize import fixture


def make_models(root):
    parts,provenance=fixture()
    make_baseline(parts,provenance,root/'baseline')
    baseline=MotionBundle(root/'baseline')
    compact=root/'compact';compact.mkdir()
    save_file({k:baseline.state[k] for k in SHAPES},str(compact/'model.safetensors'))
    write_json(compact/'status.json',{'status':'complete','promotionAuthorized':False})
    write_json(compact/'metadata.json',{'status':'complete','promotionAuthorized':False,'spec':SPEC,
        'model_sha256':hashlib.sha256((compact/'model.safetensors').read_bytes()).hexdigest(),
        'random_map_sha256':{k:tensor_hash(v) for k,v in random_map().items()}})
    return parts,baseline,compact


class BufferedMotionTests(unittest.TestCase):
    def test_features_logits_floors_strides_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts,old,path=make_models(Path(tmp))
            bundle=BufferedMotionBundle(path);compact=CompactMotionBundle(path)
            rng=np.random.default_rng(44)
            near=np.full((2,52,190),np.float32(1e-6));near[...,::2]=np.nextafter(near[...,::2],np.float32(0))
            mixed=np.zeros((2,52,190),dtype=np.float32)
            mixed[...,::2]=np.nextafter(np.float32(1e-6),np.float32(np.inf))
            inputs=[parts['validation']['x'],near,mixed,np.zeros((2,52,190)),np.ones((2,52,190)),
                    np.full((2,52,190),1e100),1e50*np.exp(rng.normal(0,1e-9,(2,52,190))),
                    np.exp(rng.normal(size=(2,52,380)))[...,::2],parts['validation']['x'][...,::-1],
                    np.ones((1,52,190),dtype=np.int64)]
            for x in inputs:
                reference=motion_features(x,'motion_acf')
                for i,sample in enumerate(x):
                    actual_features=bundle._extract(sample).reshape(52,9)
                    expected_features=reference[i].reshape(52,9)
                    # Only the six ACF sums change reduction order with einsum.
                    np.testing.assert_allclose(actual_features[:,:6],expected_features[:,:6],atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(actual_features[:,6:],expected_features[:,6:])
                actual=bundle.predict(x)
                np.testing.assert_allclose(actual,compact.predict(x),atol=1e-10,rtol=0)
                np.testing.assert_allclose(actual,old.predict(x),atol=1e-10,rtol=0)
                np.testing.assert_array_equal(actual.argmax(1),old.predict(x).argmax(1))
                np.testing.assert_array_equal(actual,np.concatenate([bundle.predict(v[None]) for v in x]))
            self.assertGreater(bundle.workspace_bytes,0)
            self.assertEqual(bundle.model_state_bytes,sum(v.nbytes for v in bundle.state.values()))
            size=bundle.workspace_bytes
            bundle.predict(parts['train']['x'])
            self.assertEqual(bundle.workspace_bytes,size)

    def test_concurrent_owned_outputs_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts,_,path=make_models(Path(tmp));bundle=BufferedMotionBundle(path)
            x=parts['validation']['x'];expected=bundle.predict(x)
            with ThreadPoolExecutor(max_workers=4) as pool:
                outputs=list(pool.map(lambda _:bundle.predict(x),range(8)))
            for result in outputs:
                np.testing.assert_array_equal(result,expected)
                self.assertTrue(result.flags.owndata)
            outputs[0][:]=999
            np.testing.assert_array_equal(outputs[1],expected)
            for invalid in (np.ones((1,52,950)),np.ones((0,52,190)),np.ones((4097,52,190),dtype=np.uint8),
                            np.ones((1,52,190),dtype=complex),np.ones((1,52,190),dtype=bool),
                            np.full((1,52,190),np.inf),-np.ones((1,52,190))):
                with self.assertRaises(ValueError):bundle.predict(invalid)
            np.testing.assert_array_equal(bundle.predict(x),expected)

    def test_artifact_bounds_corruption_and_standalone_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts,_,path=make_models(Path(tmp))
            script="""import sys,numpy as np
from csi_training.espfi_motion_runtime import BufferedMotionBundle
b=BufferedMotionBundle(sys.argv[1]);assert b.predict(np.ones((1,52,190))).shape==(1,7)
assert not any(n=='torch' or n.startswith('torch.') or n=='scipy' or n.startswith('scipy.') for n in sys.modules)
assert 'csi_training.espfi_motion' not in sys.modules
assert 'csi_training.espfi_motion_compact' not in sys.modules
"""
            import os
            env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]))
            subprocess.run([sys.executable,'-c',script,str(path)],check=True,env=env,capture_output=True)
            model=path/'model.safetensors';original=model.read_bytes();model.write_bytes(original+b'bad')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):BufferedMotionBundle(path)
            model.write_bytes(original)
            metadata=path/'metadata.json';original_meta=metadata.read_bytes();metadata.write_bytes(b'x'*65537)
            with self.assertRaisesRegex(ValueError,'Oversized'):BufferedMotionBundle(path)
            metadata.write_bytes(original_meta)
            state={k:np.ones(shape,dtype=np.float64) for k,shape in SHAPES.items()};state['feature_scale'][0]=0
            save_file(state,str(model));meta=json.loads(original_meta);meta['model_sha256']=hashlib.sha256(model.read_bytes()).hexdigest();write_json(metadata,meta)
            with self.assertRaisesRegex(ValueError,'Invalid compact'):BufferedMotionBundle(path)
            write_json(path/'status.json',{'status':'failed','promotionAuthorized':False})
            with self.assertRaisesRegex(ValueError,'Incomplete'):BufferedMotionBundle(path)

    @unittest.skipUnless(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_NONBLOCK')
                         and hasattr(os, 'mkfifo'), 'POSIX secure open required')
    def test_artifact_reader_rejects_special_files_and_path_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regular = root/'artifact';regular.write_bytes(b'original')
            link = root/'link';link.symlink_to(regular)
            fifo = root/'fifo';os.mkfifo(fifo)
            for invalid in (link, fifo, root):
                with self.subTest(path=invalid):
                    with self.assertRaisesRegex(ValueError, 'regular bounded artifact'):
                        _read(invalid)
            empty = root/'empty';empty.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'regular bounded artifact'):_read(empty)

            replacement = root/'replacement';replacement.write_bytes(b'replaced')
            displaced = root/'displaced'
            real_open = os.open
            def open_then_replace(path, flags, **kwargs):
                descriptor = real_open(path, flags, **kwargs)
                Path(path).rename(displaced)
                replacement.rename(path)
                return descriptor
            with patch('csi_training.espfi_motion_runtime.os.open', side_effect=open_then_replace):
                self.assertEqual(_read(regular), b'original')
            self.assertEqual(regular.read_bytes(), b'replaced')

    @unittest.skipUnless(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_NONBLOCK'),
                         'POSIX secure open required')
    def test_artifact_reader_detects_size_change_and_closes_descriptor(self):
        def changed_stat(details, **changes):
            fields = ('st_dev','st_ino','st_mode','st_size',
                      'st_mtime_ns','st_ctime_ns')
            values = {field:changes.get(field,getattr(details,field)) for field in fields}
            return type('Stat', (), values)()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'artifact';path.write_bytes(b'12345678')
            real_fstat = os.fstat
            for delta in (-1, 1):
                with self.subTest(delta=delta):
                    def changed_size(descriptor, delta=delta):
                        details = real_fstat(descriptor)
                        return changed_stat(details,st_size=details.st_size + delta)
                    with patch('csi_training.espfi_motion_runtime.os.fstat', side_effect=changed_size):
                        with self.assertRaisesRegex(ValueError, 'changed while reading'):
                            _read(path)

            before = os.stat(path)
            changed = changed_stat(before,st_ctime_ns=before.st_ctime_ns + 1)
            with patch('csi_training.espfi_motion_runtime.os.fstat',
                       side_effect=(before,changed)):
                with self.assertRaisesRegex(ValueError, 'changed while reading'):
                    _read(path)

            captured = []
            real_open = os.open
            def capture_open(path, flags, **kwargs):
                descriptor = real_open(path, flags, **kwargs);captured.append(descriptor);return descriptor
            with patch('csi_training.espfi_motion_runtime.os.open', side_effect=capture_open):
                with self.assertRaisesRegex(ValueError, 'regular bounded artifact'):_read(Path(tmp))
            with self.assertRaises(OSError):real_fstat(captured[0])

    @unittest.skipUnless(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_NONBLOCK')
                         and hasattr(os, 'pwrite'), 'POSIX mutation test required')
    def test_artifact_reader_rejects_metadata_visible_same_inode_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'artifact';path.write_bytes(b'12345678')
            before = path.stat();real_read = os.read;mutated = False
            def read_then_mutate(descriptor, size):
                nonlocal mutated
                result = real_read(descriptor, size)
                if not mutated:
                    mutated = True
                    writer = os.open(path, os.O_WRONLY)
                    try:
                        os.pwrite(writer,b'87654321',0);os.fsync(writer)
                    finally:
                        os.close(writer)
                    current = path.stat()
                    os.utime(path,ns=(current.st_atime_ns,before.st_mtime_ns + 1_000_000_000))
                return result
            with patch('csi_training.espfi_motion_runtime.os.read', side_effect=read_then_mutate):
                with self.assertRaisesRegex(ValueError, 'changed while reading'):_read(path)
            after = path.stat()
            self.assertEqual(after.st_ino,before.st_ino)
            self.assertEqual(after.st_size,before.st_size)

    @unittest.skipUnless(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_DIRECTORY'),
                         'POSIX secure directory open required')
    def test_bundle_load_pins_one_directory_and_rejects_symlink_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base/'original').mkdir();(base/'replacement').mkdir()
            _,_,original = make_models(base/'original')
            _,_,replacement = make_models(base/'replacement')
            write_json(replacement/'status.json',{'status':'failed','promotionAuthorized':False})
            displaced = base/'displaced'
            real_open_directory = _open_directory
            def open_then_replace(path):
                descriptor = real_open_directory(path)
                Path(path).rename(displaced)
                replacement.rename(path)
                return descriptor
            with patch('csi_training.espfi_motion_runtime._open_directory',
                       side_effect=open_then_replace):
                bundle = BufferedMotionBundle(original)
            self.assertEqual(set(bundle.state), {'projection','phase','bias','weights'})
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                BufferedMotionBundle(original)

            link = base/'linked';link.symlink_to(displaced, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'artifact directory'):
                BufferedMotionBundle(link)

    @unittest.skipUnless(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_DIRECTORY'),
                         'POSIX secure directory open required')
    def test_bundle_load_closes_directory_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,_,path = make_models(Path(tmp))
            captured = []
            real_open_directory = _open_directory
            def capture_directory(path):
                descriptor = real_open_directory(path);captured.append(descriptor);return descriptor
            with patch('csi_training.espfi_motion_runtime._open_directory',
                       side_effect=capture_directory):
                BufferedMotionBundle(path)
            self.assertEqual(len(captured), 1)
            with self.assertRaises(OSError):os.fstat(captured[0])

            with patch.object(os, 'supports_dir_fd', set()):
                with self.assertRaisesRegex(ValueError, 'Secure artifact intake unavailable'):
                    BufferedMotionBundle(path)

if __name__=='__main__':unittest.main()
