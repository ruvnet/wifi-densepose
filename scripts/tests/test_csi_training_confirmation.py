import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from csi_training.confirmation import evaluate, read_json, regular_sha256, training_gate, KNOWN_EXPOSED


def fixture():
    records = []
    for split in ('train', 'development', 'confirmation'):
        for activity in range(7):
            for trial in range(10 if split == 'confirmation' else 1):
                name = f'{split}:{activity}:{trial}'
                records.append(dict(id=name, signal_sha256=hashlib.sha256(name.encode()).hexdigest(),
                    source={'dataset':'synthetic', 'revision':'1'}, split=split,
                    room=f'{split}:room:{trial%2}', participant=f'{split}:person:{trial%2}',
                    session=f'{split}:session:{trial%2}', device=f'{split}:device', activity=activity))
    return {'schema_version':1, 'protocol_sha256':'a'*64,
            'claim':'new_room_person_device', 'records':records}, {'schema_version':1,'exposed_sources':[]}


class ConfirmationTests(unittest.TestCase):
    def test_success_is_only_metadata(self):
        m,l=fixture(); r=evaluate(m,l)
        self.assertTrue(r['preflight_passed']);self.assertFalse(r['promotionAuthorized'])
        self.assertEqual(r['evidence'],'METADATA_ONLY')
        self.assertEqual(r,evaluate(copy.deepcopy(m),copy.deepcopy(l)))
        self.assertNotIn('records',r)

    def test_each_leakage_axis(self):
        for axis in ('room','participant','session','device'):
            with self.subTest(axis=axis):
                m,l=fixture();m['records'][-1][axis]=m['records'][0][axis]
                self.assertIn('overlapping_'+axis,evaluate(m,l)['reasons'])

    def test_duplicates_even_with_renamed_record(self):
        for key,reason in [('id','duplicate_record_id'),('signal_sha256','duplicate_signal_digest')]:
            m,l=fixture();m['records'][-1][key]=m['records'][0][key]
            self.assertIn(reason,evaluate(m,l)['reasons'])

    def test_exposure_cannot_be_cleared_for_known_source(self):
        m,l=fixture();dataset,revision=next(iter(KNOWN_EXPOSED))
        m['records'][-1]['source']={'dataset':dataset,'revision':revision}
        self.assertIn('confirmation_source_previously_exposed',evaluate(m,l)['reasons'])
        m,l=fixture();l['exposed_sources']=[m['records'][0]['source']]
        self.assertIn('confirmation_source_previously_exposed',evaluate(m,l)['reasons'])

    def test_coverage_and_claim_scope(self):
        m,l=fixture();m['records'].pop()
        self.assertIn('confirmation_class_coverage',evaluate(m,l)['reasons'])
        m,l=fixture()
        for row in m['records']:
            if row['split']=='confirmation': row['room']='onlyroom'
        self.assertIn('insufficient_confirmation_room_coverage',evaluate(m,l)['reasons'])
        m,l=fixture();m['claim']='new_room_person'
        for row in m['records']:row['device']='same_device'
        self.assertTrue(evaluate(m,l)['preflight_passed'])

    def test_strict_contract(self):
        for field,value in [('schema_version',True),('protocol_sha256','bad'),('claim','occupancy'),('records',[])]:
            m,l=fixture();m[field]=value
            with self.assertRaises(ValueError):evaluate(m,l)
        m,l=fixture();m['records'][0]['activity']=True
        with self.assertRaises(ValueError):evaluate(m,l)
        m,l=fixture();m['records']*=50
        with self.assertRaises(ValueError):evaluate(m,l)

    def test_unsafe_json_files(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            for content in ['{"x":1,"x":2}', '{"x":NaN}', ' '* (2*1024*1024+1)]:
                p.write_text(content)
                with self.assertRaises(ValueError):read_json(p)
            q=Path(d)/'link';q.symlink_to(p)
            with self.assertRaises(OSError):read_json(q)
            q.unlink();os.mkfifo(q)
            with self.assertRaises(ValueError):read_json(q)

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            m,l=fixture();p=Path(d)/'m.json';q=Path(d)/'l.json';q.write_text(json.dumps(l))
            for expected in (0,2,1):
                if expected==2:m['records'][-1]['room']=m['records'][0]['room']
                if expected==1:m['protocol_sha256']='bad'
                p.write_text(json.dumps(m))
                r=subprocess.run([sys.executable,'-m','csi_training.confirmation','--manifest',str(p),'--ledger',str(q)],capture_output=True,text=True)
                self.assertEqual(r.returncode,expected,r.stderr)
                self.assertFalse(json.loads(r.stdout)['promotionAuthorized'])

    def test_training_gate_binds_frozen_protocol_and_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            m,l=fixture();root=Path(d);dataset=root/'dataset.json'
            p=root/'m.json';q=root/'l.json'
            dataset.write_text('{"schema_version":1}')
            p.write_text(json.dumps(m));q.write_text(json.dumps(l))
            frozen=regular_sha256(dataset)
            report=training_gate(p,q,dataset,'a'*64,frozen)
            self.assertTrue(report['preflight_passed'])
            self.assertFalse(report['promotionAuthorized'])
            self.assertNotIn('records',report)
            self.assertIn('protocol_digest_mismatch',training_gate(p,q,dataset,'b'*64,frozen)['reasons'])
            self.assertIn('dataset_manifest_digest_mismatch',training_gate(p,q,dataset,'a'*64,'0'*64)['reasons'])

    def test_regular_sha256_rejects_unsafe_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'target';target.write_bytes(b'ok')
            link=root/'link';link.symlink_to(target)
            with self.assertRaises(OSError):regular_sha256(link)
            fifo=root/'fifo';os.mkfifo(fifo)
            with self.assertRaises(ValueError):regular_sha256(fifo)
            oversized=root/'oversized';oversized.write_bytes(b'x'*33)
            with self.assertRaises(ValueError):regular_sha256(oversized,32)

if __name__=='__main__':unittest.main()
