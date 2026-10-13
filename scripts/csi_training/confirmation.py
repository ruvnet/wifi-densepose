"""Metadata-only preflight. A pass never authenticates captures or authorizes promotion."""
import argparse
import hashlib
import json
import os
import re
import stat

MAX_BYTES = 2 * 1024 * 1024
MAX_DATASET_MANIFEST_BYTES = 3_000_000
MAX_RECORDS = 4096
SPLITS = ('train', 'development', 'confirmation')
AXES = ('room', 'participant', 'session')
TOKEN = re.compile(r'[A-Za-z0-9_.:]{1,128}\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
# This source has already informed development and final-test inspection.
KNOWN_EXPOSED = {('ESP-Fi HAR', 'c8bf6aa0d680fa02695c85a82b05e79b3ebb10c4')}


def exact(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('Unexpected schema keys')


def token(value):
    if type(value) is not str or not TOKEN.fullmatch(value):
        raise ValueError('Expected bounded opaque identifier')
    return value


def source(value):
    exact(value, ('dataset', 'revision'))
    if type(value['dataset']) is not str or not 1 <= len(value['dataset']) <= 128:
        raise ValueError('Expected dataset identifier')
    return value['dataset'], token(value['revision'])


def digest(value):
    if type(value) is not str or not SHA.fullmatch(value):
        raise ValueError('Expected lowercase SHA256')
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def evaluate(manifest, ledger):
    exact(manifest, ('schema_version', 'protocol_sha256', 'claim', 'records'))
    exact(ledger, ('schema_version', 'exposed_sources'))
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        raise ValueError('Unsupported manifest version')
    if type(ledger['schema_version']) is not int or ledger['schema_version'] != 1:
        raise ValueError('Unsupported ledger version')
    digest(manifest['protocol_sha256'])
    if manifest['claim'] not in ('new_room_person', 'new_room_person_device'):
        raise ValueError('Unsupported claim')
    exposed = ledger['exposed_sources']
    if type(exposed) is not list or len(exposed) > MAX_RECORDS:
        raise ValueError('Invalid exposure list')
    exposed = KNOWN_EXPOSED | {source(row) for row in exposed}
    records = manifest['records']
    if type(records) is not list or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError('Invalid record count')
    axes = AXES + (('device',) if manifest['claim'].endswith('_device') else ())
    groups = {s: {a: set() for a in axes} for s in SPLITS}
    classes = {s: [0] * 7 for s in SPLITS}
    seen_ids, seen_signals, reasons = set(), set(), set()
    for row in records:
        exact(row, ('id', 'signal_sha256', 'source', 'split', 'room', 'participant',
                    'session', 'device', 'activity'))
        for key in ('id', 'room', 'participant', 'session', 'device'):
            token(row[key])
        signal = digest(row['signal_sha256']); origin = source(row['source'])
        split = row['split']; activity = row['activity']
        if type(split) is not str or split not in SPLITS:
            raise ValueError('Invalid split')
        if type(activity) is not int or not 0 <= activity < 7:
            raise ValueError('Invalid numeric activity')
        if row['id'] in seen_ids: reasons.add('duplicate_record_id')
        if signal in seen_signals: reasons.add('duplicate_signal_digest')
        seen_ids.add(row['id']); seen_signals.add(signal)
        classes[split][activity] += 1
        for axis in axes: groups[split][axis].add(row[axis])
        if split == 'confirmation' and origin in exposed:
            reasons.add('confirmation_source_previously_exposed')
    for axis in axes:
        for i, left in enumerate(SPLITS):
            for right in SPLITS[i+1:]:
                if groups[left][axis] & groups[right][axis]:
                    reasons.add('overlapping_' + axis)
    for split in SPLITS:
        minimum = 10 if split == 'confirmation' else 1
        if min(classes[split]) < minimum: reasons.add(split + '_class_coverage')
    for axis in ('room', 'participant'):
        if len(groups['confirmation'][axis]) < 2:
            reasons.add('insufficient_confirmation_' + axis + '_coverage')
    return {'schema_version': 1, 'evidence': 'METADATA_ONLY',
            'preflight_passed': not reasons, 'promotionAuthorized': False,
            'protocol_sha256': manifest['protocol_sha256'],
            'manifest_sha256': canonical_hash(manifest), 'ledger_sha256': canonical_hash(ledger),
            'reasons': sorted(reasons), 'class_counts': classes,
            'group_counts': {s: {a: len(v) for a, v in g.items()} for s, g in groups.items()},
            'scope': 'Declared metadata only. No capture authentication, measured accuracy, or independent confirmation result.'}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _read_regular(path, maximum):
    if (type(maximum) is not int or not 1 <= maximum <= 16 * 1024 * 1024
            or not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_NONBLOCK')):
        raise ValueError('Safe file intake unavailable on this platform')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError('Expected bounded regular file')
        raw = handle.read(info.st_size + 1)
    if len(raw) != info.st_size:
        raise ValueError('File changed during read')
    return raw


def read_json_and_sha256(path, maximum=MAX_BYTES):
    raw = _read_regular(path, maximum)
    def invalid(value): raise ValueError('Nonfinite JSON constant')
    return (json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid),
            hashlib.sha256(raw).hexdigest())


def read_json(path):
    return read_json_and_sha256(path)[0]


def regular_sha256(path, maximum=MAX_DATASET_MANIFEST_BYTES):
    """Hash one bounded regular file through the descriptor that was checked."""
    return hashlib.sha256(_read_regular(path, maximum)).hexdigest()


def _dataset_binding(manifest, dataset, required_source=None):
    """Bind declared confirmation records to a supervised ESP-Fi manifest."""
    if type(dataset) is not dict:
        raise ValueError('Dataset manifest must be an object')
    if type(dataset.get('schema_version')) is not int or dataset['schema_version'] != 1:
        raise ValueError('Unsupported dataset manifest version')
    dataset_name = dataset.get('dataset')
    revision = dataset.get('source_revision')
    if type(dataset_name) is not str or not 1 <= len(dataset_name) <= 128:
        raise ValueError('Invalid dataset identity')
    token(revision)
    dataset_source = (dataset_name, revision)
    records = dataset.get('records')
    if type(records) is not list or not 1 <= len(records) <= MAX_RECORDS:
        raise ValueError('Invalid dataset record count')
    split_map = {'train': 'train', 'validation': 'development', 'test': 'confirmation'}
    retained, duplicates = {}, False
    for row in records:
        if type(row) is not dict or row.get('split') not in (*split_map, None):
            raise ValueError('Invalid dataset record')
        signal = digest(row.get('signal_sha256'))
        identity = row.get('identity')
        if (type(identity) is not list or len(identity) != 4
                or any(type(value) is not int for value in identity)
                or not 1 <= identity[2] <= 7):
            raise ValueError('Invalid dataset identity tuple')
        if row['split'] is not None:
            if signal in retained:
                duplicates = True
            retained[signal] = row
    planned = {row['signal_sha256']: row for row in manifest['records']}
    common = set(retained) & set(planned)
    reasons = set()
    if required_source is not None:
        if (type(required_source) is not tuple or len(required_source) != 2
                or type(required_source[0]) is not str or type(required_source[1]) is not str):
            raise ValueError('Invalid required source')
        if dataset_source != required_source:
            reasons.add('canonical_dataset_source_mismatch')
    if duplicates:
        reasons.add('dataset_duplicate_signal_digest')
    if set(retained) != set(planned):
        reasons.add('dataset_record_set_mismatch')
    expected_source = dataset_source
    for signal in common:
        plan, row = planned[signal], retained[signal]
        if source(plan['source']) != expected_source:
            reasons.add('dataset_source_mismatch')
        if plan['split'] != split_map[row['split']]:
            reasons.add('dataset_split_mismatch')
        if plan['activity'] != row['identity'][2] - 1:
            reasons.add('dataset_activity_mismatch')
        if plan['room'] != 'room:' + str(row['identity'][0]):
            reasons.add('dataset_room_mismatch')
        if plan['participant'] != 'participant:' + str(row['identity'][1]):
            reasons.add('dataset_participant_mismatch')
        session = row.get('session_id')
        if session is None:
            reasons.add('dataset_session_identity_unbound')
        elif plan['session'] != token(session):
            reasons.add('dataset_session_mismatch')
        if manifest['claim'].endswith('_device'):
            device = row.get('device_id')
            if device is None:
                reasons.add('dataset_device_identity_unbound')
            elif plan['device'] != token(device):
                reasons.add('dataset_device_mismatch')
    return reasons, {'retained_records': len(retained), 'planned_records': len(planned),
                     'matched_records': len(common), 'exact_record_set': set(retained) == set(planned)}


def training_gate(manifest_path, ledger_path, dataset_manifest_path,
                  expected_protocol_sha256, expected_dataset_manifest_sha256,
                  required_source=None):
    """Evaluate declared metadata and bind it to frozen protocol/dataset bytes.

    This remains a metadata gate. It does not authenticate captures, consent,
    licensing, source aliases, or the completeness of the exposure ledger.
    """
    digest(expected_protocol_sha256)
    digest(expected_dataset_manifest_sha256)
    manifest = read_json(manifest_path)
    report = evaluate(manifest, read_json(ledger_path))
    reasons = set(report['reasons'])
    if report['protocol_sha256'] != expected_protocol_sha256:
        reasons.add('protocol_digest_mismatch')
    dataset, actual = read_json_and_sha256(dataset_manifest_path, MAX_DATASET_MANIFEST_BYTES)
    if actual != expected_dataset_manifest_sha256:
        reasons.add('dataset_manifest_digest_mismatch')
    binding_reasons, binding = _dataset_binding(manifest, dataset, required_source)
    reasons.update(binding_reasons)
    result = dict(report)
    result.update({'preflight_passed': not reasons,
                   'reasons': sorted(reasons),
                   'dataset_manifest_sha256': actual,
                   'expected_dataset_manifest_sha256': expected_dataset_manifest_sha256,
                   'dataset_binding': binding,
                   'scope': report['scope'] + ' Dataset records and bytes are bound to declarations; provenance claims remain unauthenticated.'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--ledger', required=True)
    args = parser.parse_args()
    try:
        report = evaluate(read_json(args.manifest), read_json(args.ledger))
    except (ValueError, OSError, TypeError, RecursionError, OverflowError):
        print(json.dumps({'status': 'invalid_input', 'promotionAuthorized': False}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report['preflight_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
