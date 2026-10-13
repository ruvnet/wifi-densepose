"""Synthetic storage and numerical-parity contracts, not sensing evidence."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from safetensors.numpy import load, save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from csi_training.espfi_motion_compact import CompactMotionBundle, SHAPES, random_map, export
from csi_training.espfi_motion import MotionBundle
from csi_training.espfi_train import digest, write_json
from test_espfi_context import make_baseline
from test_espfi_generalize import fixture
import csi_training.espfi_motion_compact as compact_module


class CompactMotionTests(unittest.TestCase):
    def test_export_regeneration_folded_edges_storage_and_corruption(self):
        parts,provenance=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            make_baseline(parts,provenance,root/'baseline')
            original=(root/'baseline'/'model.safetensors').read_bytes()
            report=export(parts,provenance,root/'baseline',root/'compact')
            self.assertEqual(report['stored_scalars'],1967)
            self.assertLess(report['model_bytes'],.1*report['baseline_model_bytes'])
            self.assertFalse(report['promotionAuthorized'])
            self.assertEqual((root/'baseline'/'model.safetensors').read_bytes(),original)
            saved=load((root/'compact'/'model.safetensors').read_bytes())
            self.assertEqual(set(saved),set(SHAPES))
            old=MotionBundle(root/'baseline')
            unfused=CompactMotionBundle(root/'compact',folded=False)
            folded=CompactMotionBundle(root/'compact')
            self.assertFalse(hasattr(folded,'state'))
            rng=np.random.default_rng(41)
            for x in (parts['validation']['x'],np.zeros((2,52,190)),np.ones((2,52,190)),
                      np.full((2,52,190),1e100),
                      1e50*np.exp(rng.normal(0,1e-9,(2,52,190))),
                      parts['validation']['x'].astype(np.float64)*7):
                expected=old.predict(x)
                np.testing.assert_array_equal(unfused.predict(x),expected)
                np.testing.assert_allclose(folded.predict(x),expected,rtol=0,atol=1e-10)
                np.testing.assert_array_equal(folded.predict(x).argmax(1),expected.argmax(1))
            self.assertTrue(report['parity']['train']['runtimes']['folded']['identical_predictions'])
            metadata=root/'compact'/'metadata.json'
            status=json.loads((root/'compact'/'status.json').read_text())
            original_metadata=metadata.read_bytes()
            self.assertEqual(status['metadata_sha256'],digest(metadata))
            metadata.write_bytes(original_metadata+b'\n')
            with self.assertRaisesRegex(ValueError,'metadata hash mismatch'):
                CompactMotionBundle(root/'compact')
            metadata.write_bytes(original_metadata)
            path=root/'compact'/'model.safetensors'
            path.write_bytes(path.read_bytes()+b'corruption')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                CompactMotionBundle(root/'compact')

    def test_recipe_mismatch_fails_lifecycle_and_provenance(self):
        parts,provenance=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            make_baseline(parts,provenance,root/'baseline')
            with self.assertRaisesRegex(ValueError,'provenance'):
                export(parts,dict(provenance,manifest_sha256='wrong'),root/'baseline',root/'wrong')
            path=root/'baseline'/'model.safetensors'
            state=load(path.read_bytes())
            for name,value in random_map().items():
                np.testing.assert_array_equal(state[name],value)
            state['projection'][0,0]+=1e-9
            save_file(state,str(path))
            report=json.loads((root/'baseline'/'metrics.json').read_text())
            report['model_sha256']=digest(path)
            write_json(root/'baseline'/'metrics.json',report)
            with self.assertRaisesRegex(ValueError,'seed recipe'):
                export(parts,provenance,root/'baseline',root/'failed')
            self.assertEqual(json.loads((root/'failed'/'status.json').read_text())['status'],'failed')
            with self.assertRaises(ValueError):
                CompactMotionBundle(root/'failed')
            with self.assertRaises(FileExistsError):
                export(parts,provenance,root/'baseline',root/'failed')

    def test_rehashed_invalid_state_and_map_metadata_rejected(self):
        parts,provenance=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            make_baseline(parts,provenance,root/'baseline')
            export(parts,provenance,root/'baseline',root/'compact')
            meta_path=root/'compact'/'metadata.json'
            original=json.loads(meta_path.read_text())
            changed=dict(original,random_map_sha256={})
            write_json(meta_path,changed)
            status_path=root/'compact'/'status.json'
            status=json.loads(status_path.read_text())
            status['metadata_sha256']=digest(meta_path)
            write_json(status_path,status)
            with self.assertRaisesRegex(ValueError,'Regenerated map'):
                CompactMotionBundle(root/'compact')
            state_path=root/'compact'/'model.safetensors'
            state=load(state_path.read_bytes());state['feature_scale'][0]=0.
            save_file(state,str(state_path))
            write_json(meta_path,dict(original,model_sha256=digest(state_path)))
            status['metadata_sha256']=digest(meta_path)
            write_json(status_path,status)
            with self.assertRaisesRegex(ValueError,'Invalid compact'):
                CompactMotionBundle(root/'compact')

    def test_export_parent_lineage_uses_verified_baseline_generation(self):
        parts,provenance=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            baseline=root/'baseline'
            make_baseline(parts,provenance,baseline)
            expected_model=digest(baseline/'model.safetensors')
            expected_receipt=digest(baseline/'metrics.json')
            expected_model_bytes=(baseline/'model.safetensors').stat().st_size
            forged=json.loads((baseline/'metrics.json').read_text())
            forged['model_sha256']='f'*64
            forged_bytes=json.dumps(forged,separators=(',',':'),sort_keys=True).encode()
            forged_receipt=hashlib.sha256(forged_bytes).hexdigest()
            real_read=compact_module._read
            injected=[]
            swapped=[]

            def replace_second_receipt(path,limit):
                if Path(path)==baseline/'metrics.json':
                    injected.append(Path(path))
                    return forged_bytes
                result=real_read(path,limit)
                if Path(path)==root/'compact'/'model.safetensors' and not swapped:
                    (baseline/'model.safetensors').write_bytes(b'x')
                    swapped.append(Path(path))
                return result

            with patch.object(compact_module,'_read',side_effect=replace_second_receipt):
                report=export(parts,provenance,baseline,root/'compact')
            metadata=json.loads((root/'compact'/'metadata.json').read_text())
            self.assertEqual(injected,[])
            self.assertEqual(len(swapped),1)
            self.assertEqual(metadata['baseline_model_sha256'],expected_model)
            self.assertEqual(metadata['baseline_receipt_sha256'],expected_receipt)
            self.assertNotEqual(metadata['baseline_receipt_sha256'],forged_receipt)
            self.assertEqual(report['baseline_model_bytes'],expected_model_bytes)
            self.assertTrue(report['retention_checks']['model_below_10percent'])

if __name__=='__main__':
    unittest.main()
