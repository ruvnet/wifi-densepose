"""Synthetic learning/lineage checks; no occupancy or public-task accuracy."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from csi_training.predictive import (Predictor, PatchEncoder, make_mask, objective,
    ema_update, synthetic, train, FrozenEncoder, split_training, validate_parts, normalize, _read, _json)


class PredictiveTests(unittest.TestCase):
    def test_mask_exclusion_gradient_and_stopped_ema(self):
        torch.manual_seed(42);torch.set_num_threads(1)
        student=Predictor();teacher=copy.deepcopy(student.encoder).eval().requires_grad_(False)
        x=torch.randn(3,52,190);mask=make_mask(3,torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(mask.sum(1),torch.full((3,),52)))
        self.assertFalse(mask.all())
        # Perturb only masked target patch values; context predictions cannot see them.
        patch_mask=mask.reshape(3,13,10)[:,:,None,:,None].expand(3,13,4,10,19).reshape(3,52,190)
        altered=x.clone();altered[patch_mask]+=100
        np.testing.assert_array_equal(student(x,mask).detach().numpy(),student(altered,mask).detach().numpy())
        loss,diagnostics,target=objective(student,teacher,x,mask)
        self.assertFalse(target.requires_grad);self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        before=next(teacher.parameters()).clone()
        with torch.no_grad():next(student.encoder.parameters()).add_(.1)
        ema_update(teacher,student)
        torch.testing.assert_close(next(teacher.parameters()),before+.001)
        self.assertTrue(torch.isfinite(diagnostics['sample_std']))

    def test_group_leakage_duplicates_and_original_train_only(self):
        parts,_=synthetic();validate_parts(parts)
        leaking=copy.deepcopy(parts);leaking['heldout']['groups'][:,0]=1
        with self.assertRaisesRegex(ValueError,'leakage'):validate_parts(leaking)
        duplicate=copy.deepcopy(parts);duplicate['heldout']['x'][0]=duplicate['train']['x'][0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate_parts(duplicate)
        with self.assertRaises(ValueError):validate_parts(dict(parts,test=parts['heldout']))
        x=np.concatenate([parts['train']['x'],parts['heldout']['x']])
        ids=np.column_stack([np.concatenate([parts['train']['groups'],parts['heldout']['groups']]),np.ones((len(x),2),dtype=int)])
        derived=split_training(x,ids)
        self.assertEqual(len(derived['train']['x']),24);self.assertEqual(len(derived['heldout']['x']),8)
        ids[0,0]=3
        with self.assertRaisesRegex(ValueError,'original training'):split_training(x,ids)

    def test_train_save_load_baselines_and_no_heldout_influence(self):
        parts,lineage=synthetic()
        # Keep CPU verification bounded; no outcome requirement on synthetic loss.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=train(parts,lineage,root/'a',epochs=1,batch_size=8,max_seconds=120)
            changed=copy.deepcopy(parts);changed['heldout']['x']*=3
            b=train(changed,lineage,root/'b',epochs=1,batch_size=8,max_seconds=120)
            self.assertEqual(a['model_sha256'],b['model_sha256'])
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(set(a['heldout_latent_mse']),{'trained','untrained','constant','position_constant'})
            self.assertTrue(all(np.isfinite(v) for v in a['heldout_latent_mse'].values()))
            self.assertEqual(len(a['inference_latency']['rounds']),3)
            self.assertEqual(a['inference_latency']['calls_per_round'],100)
            self.assertIn('collapse_warning',a['anti_collapse']);self.assertFalse(a['promotionAuthorized'])
            bundle=FrozenEncoder(root/'a');second=FrozenEncoder(root/'a')
            values=parts['heldout']['x'];np.testing.assert_array_equal(bundle.encode(values),second.encode(values))
            self.assertEqual(bundle.encode(values[:1]).shape,(1,32))
            logged=np.log(np.maximum(parts['train']['x'].astype(np.float64),1e-6))
            np.testing.assert_array_equal(bundle.mean,logged.mean((0,2)))
            self.assertTrue(all(not p.requires_grad for p in bundle.encoder.parameters()))
            adapter=bundle.new_adapter(3);adapter(values).sum().backward()
            self.assertTrue(all(p.grad is None for p in bundle.encoder.parameters()))
            self.assertIsNotNone(adapter.head.weight.grad)
            path=root/'a'/'model.safetensors';path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):FrozenEncoder(root/'a')
            with self.assertRaises(FileExistsError):train(parts,lineage,root/'b',epochs=1)

    def test_failure_lifecycle_and_bounds(self):
        parts,lineage=synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'failed'
            with patch('csi_training.predictive.objective',side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError):train(parts,lineage,root,epochs=1)
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'failed')
            with self.assertRaises(ValueError):FrozenEncoder(root)
        for bad in [True,False,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):train(parts,lineage,'unused',max_seconds=bad)
        for x in [np.ones((1,52,189)),np.full((2,52,190),np.nan),-np.ones((2,52,190))]:
            with self.assertRaises(ValueError):normalize(x,np.zeros(52),np.ones(52))

    def test_strict_json_and_opened_file_security(self):
        for text in ['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}','{"a":1e999}']:
            with self.assertRaises(ValueError):_json(text)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);regular=root/'regular';regular.write_bytes(b'123')
            self.assertEqual(_read(regular,3),b'123')
            with self.assertRaises(ValueError):_read(regular,2)
            link=root/'link';link.symlink_to(regular)
            with self.assertRaises(ValueError):_read(link,100)
            if hasattr(os,'mkfifo'):
                fifo=root/'fifo';os.mkfifo(fifo)
                with self.assertRaises(ValueError):_read(fifo,100)
            with self.assertRaises(ValueError):_read(root,100)

if __name__=='__main__':unittest.main()
