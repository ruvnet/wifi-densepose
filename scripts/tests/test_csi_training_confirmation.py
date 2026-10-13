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
                group = {'train': 1, 'development': 2}.get(split, 3 + trial % 2)
                records.append(dict(id=name, signal_sha256=hashlib.sha256(name.encode()).hexdigest(),
                    source={'dataset':'synthetic', 'revision':'1'}, split=split,
                    room=f'room:{group}', participant=f'participant:{group}',
                    session=f'{split}:session:{trial%2}', device=f'{split}:device', activity=activity))
    return {'schema_version':1, 'protocol_sha256':'a'*64,
            'claim':'new_room_person_device', 'records':records}, {'schema_version':1,'exposed_sources':[]}


def dataset_fixture(manifest):
    split_map = {'train':'train', 'development':'validation', 'confirmation':'test'}
    return {'schema_version':1, 'dataset':'synthetic', 'source_revision':'1',
            'records':[{'signal_sha256':row['signal_sha256'],
                        'split':split_map[row['split']],
                        'identity':[int(row['room'].split(':')[1]),
                                    int(row['participant'].split(':')[1]),
                                    row['activity'] + 1, index + 1],
                        'session_id':row['session'], 'device_id':row['device']}
                       for index,row in enumerate(manifest['records'])]}


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
            dataset.write_text(json.dumps(dataset_fixture(m)))
            p.write_text(json.dumps(m));q.write_text(json.dumps(l))
            frozen=regular_sha256(dataset)
            report=training_gate(p,q,dataset,'a'*64,frozen)
            self.assertTrue(report['preflight_passed'])
            self.assertFalse(report['promotionAuthorized'])
            self.assertNotIn('records',report)
            self.assertIn('protocol_digest_mismatch',training_gate(p,q,dataset,'b'*64,frozen)['reasons'])
            self.assertIn('dataset_manifest_digest_mismatch',training_gate(p,q,dataset,'a'*64,'0'*64)['reasons'])

    def test_training_gate_binds_each_dataset_record_field(self):
        changes = [('signal_sha256','dataset_record_set_mismatch'),
                   ('source','dataset_source_mismatch'),
                   ('split','dataset_split_mismatch'),
                   ('activity','dataset_activity_mismatch'),
                   ('room','dataset_room_mismatch'),
                   ('participant','dataset_participant_mismatch'),
                   ('session','dataset_session_mismatch'),
                   ('device','dataset_device_mismatch')]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for field,reason in changes:
                with self.subTest(field=field):
                    m,l=fixture();dataset=dataset_fixture(m)
                    if field == 'signal_sha256':
                        dataset['records'][0]['signal_sha256']='f'*64
                    elif field == 'source':
                        m['records'][0]['source']={'dataset':'other','revision':'1'}
                    elif field == 'split':
                        dataset['records'][0]['split']='validation'
                    elif field == 'activity':
                        dataset['records'][0]['identity'][2]=2 if m['records'][0]['activity'] != 1 else 3
                    elif field == 'room':
                        dataset['records'][0]['identity'][0]=8
                    elif field == 'participant':
                        dataset['records'][0]['identity'][1]=8
                    elif field == 'session':
                        dataset['records'][0]['session_id']='other:session'
                    else:
                        dataset['records'][0]['device_id']='other:device'
                    p=root/'m.json';q=root/'l.json';data=root/'dataset.json'
                    p.write_text(json.dumps(m));q.write_text(json.dumps(l));data.write_text(json.dumps(dataset))
                    report=training_gate(p,q,data,'a'*64,regular_sha256(data))
                    self.assertIn(reason,report['reasons'])
                    self.assertFalse(report['preflight_passed'])

    def test_training_gate_rejects_duplicate_retained_signal(self):
        with tempfile.TemporaryDirectory() as d:
            m,l=fixture();dataset=dataset_fixture(m)
            dataset['records'][1]['signal_sha256']=dataset['records'][0]['signal_sha256']
            root=Path(d);p=root/'m';q=root/'l';data=root/'data'
            p.write_text(json.dumps(m));q.write_text(json.dumps(l));data.write_text(json.dumps(dataset))
            reasons=training_gate(p,q,data,'a'*64,regular_sha256(data))['reasons']
            self.assertIn('dataset_duplicate_signal_digest',reasons)
            self.assertIn('dataset_record_set_mismatch',reasons)

    def test_training_gate_ignores_excluded_dataset_record(self):
        with tempfile.TemporaryDirectory() as d:
            m,l=fixture();dataset=dataset_fixture(m)
            dataset['records'].append({'signal_sha256':'f'*64, 'split':None,
                                       'identity':[9,9,1,1]})
            root=Path(d);p=root/'m';q=root/'l';data=root/'data'
            p.write_text(json.dumps(m));q.write_text(json.dumps(l));data.write_text(json.dumps(dataset))
            report=training_gate(p,q,data,'a'*64,regular_sha256(data))
            self.assertTrue(report['preflight_passed'],report['reasons'])
            self.assertTrue(report['dataset_binding']['exact_record_set'])

    def test_training_gate_rejects_malformed_dataset_binding(self):
        with tempfile.TemporaryDirectory() as d:
            m,l=fixture();dataset=dataset_fixture(m);dataset['records'][0]['identity'][2]=True
            root=Path(d);p=root/'m';q=root/'l';data=root/'data'
            p.write_text(json.dumps(m));q.write_text(json.dumps(l));data.write_text(json.dumps(dataset))
            with self.assertRaises(ValueError):
                training_gate(p,q,data,'a'*64,regular_sha256(data))

    def test_training_gate_fails_closed_without_session_or_device_identity(self):
        with tempfile.TemporaryDirectory() as d:
            m,l=fixture();dataset=dataset_fixture(m)
            for row in dataset['records']:
                row.pop('session_id');row.pop('device_id')
            root=Path(d);p=root/'m';q=root/'l';data=root/'data'
            p.write_text(json.dumps(m));q.write_text(json.dumps(l));data.write_text(json.dumps(dataset))
            reasons=training_gate(p,q,data,'a'*64,regular_sha256(data))['reasons']
            self.assertIn('dataset_session_identity_unbound',reasons)
            self.assertIn('dataset_device_identity_unbound',reasons)

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
