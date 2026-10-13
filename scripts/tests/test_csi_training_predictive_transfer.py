"""Synthetic fixed-arm, gradient, baseline reproduction and artifact contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from csi_training.predictive import synthetic, train, Predictor, make_mask
from csi_training.predictive_transfer import (SPEC, record_penalties, transfer_objective,
    TransferEncoder, run)


def fixture():
    parts,lineage=synthetic()
    for part in parts.values():part['y']=np.arange(len(part['x']),dtype=np.int64)%7
    return parts,lineage


class TransferTests(unittest.TestCase):
    def test_record_regularization_gradient_and_exact_baseline_objective(self):
        from csi_training.predictive import objective
        torch.manual_seed(42);torch.set_num_threads(1)
        student=Predictor();teacher=copy.deepcopy(student.encoder).eval().requires_grad_(False)
        x=torch.randn(4,52,190);mask=make_mask(4,torch.Generator().manual_seed(42))
        baseline,_,_=transfer_objective(student,teacher,x,mask,'baseline')
        expected,_,_=objective(student,teacher,x,mask)
        torch.testing.assert_close(baseline,expected,rtol=0,atol=0)
        candidate,diagnostic,target=transfer_objective(student,teacher,x,mask,'candidate')
        variance,covariance=record_penalties(student.encoder(x).mean(1))
        torch.testing.assert_close(candidate,expected+variance+.1*covariance)
        candidate.backward()
        self.assertFalse(target.requires_grad)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in student.encoder.parameters()))
        collapsed=torch.zeros(4,32);v,c=record_penalties(collapsed)
        self.assertAlmostEqual(float(v),.99,places=6);self.assertEqual(float(c),0.)
        self.assertIn('record_covariance',diagnostic)

    def test_fixed_campaign_reproduces_original_and_ignores_dev_updates(self):
        parts,lineage=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=train(parts,lineage,root/'original',epochs=5,batch_size=8,max_seconds=120)
            reference=root/'original'/'model.safetensors'
            first=run(parts,lineage,root/'first',max_seconds=240,baseline_artifact=reference)
            changed=copy.deepcopy(parts);changed['heldout']['x']*=1.2
            changed['heldout']['y']=(changed['heldout']['y']+1)%7
            second=run(changed,lineage,root/'second',max_seconds=240,baseline_artifact=reference)
            self.assertTrue(first['original_baseline_exact'])
            self.assertEqual(first['models']['baseline']['model_sha256'],original['model_sha256'])
            self.assertEqual(len(set(first['mask_order_sha256'].values())),1)
            for name in ('baseline','candidate','untrained'):
                self.assertEqual(first['models'][name]['model_sha256'],second['models'][name]['model_sha256'])
                loaded=TransferEncoder(root/'first'/name)
                self.assertEqual(loaded.encode(parts['heldout']['x']).shape,(8,32))
                self.assertTrue(all(not p.requires_grad for p in loaded.encoder.parameters()))
                self.assertEqual(len(first['inference_latency'][name]),3)
            self.assertEqual(first['histories'],second['histories'])
            self.assertEqual(first['spec'],SPEC);self.assertFalse(first['promotionAuthorized'])
            self.assertEqual(set(first['probes']['probes']),{'baseline','candidate','untrained','statistics'})
            frozen=json.loads((root/'first'/'representation_frozen.json').read_text())
            self.assertNotIn('probes',frozen)
            model=root/'first'/'candidate'/'model.safetensors';model.write_bytes(model.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):TransferEncoder(model.parent)

    def test_failure_provenance_budget_and_no_overwrite(self):
        parts,lineage=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for budget in [True,float('nan'),float('inf'),0]:
                with self.assertRaises(ValueError):run(parts,lineage,root/'invalid',max_seconds=budget)
            with self.assertRaisesRegex(ValueError,'required'):
                run(parts,dict(lineage,evidence='PUBLIC_TRAIN_ONLY'),root/'missing')
            with patch('csi_training.predictive_transfer.transfer_objective',side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError):run(parts,lineage,root/'failed')
            self.assertEqual(json.loads((root/'failed'/'status.json').read_text())['status'],'failed')
            with self.assertRaises(FileExistsError):run(parts,lineage,root/'failed')
            with self.assertRaises(ValueError):run(dict(parts,test=parts['heldout']),lineage,root/'test')

if __name__=='__main__':unittest.main()
