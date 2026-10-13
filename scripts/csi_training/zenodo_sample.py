"""Bounded research-only sample intake from Zenodo 18924941; no archive extraction."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import stat
import time
import urllib.request
import zipfile

URL = 'https://zenodo.org/api/records/18924941/files/Zenodo.zip/content'
ARCHIVE_BYTES = 16125799937
MAX_REQUEST = 4 * 1024 * 1024
MAX_TRANSFER = 12 * 1024 * 1024
MAX_MEMBER = 4 * 1024 * 1024
SESSIONS = {
    'empty': 'E - Empty/2025-07-23_19-54-37',
    'sitting': 'S - Sitting/2025-07-23_18-06-23',
    'walking': 'W - Walking/2025-07-23_19-28-30',
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Unexpected redirect from pinned source')


class HttpRanges:
    def __init__(self):
        self.opener = urllib.request.build_opener(NoRedirect)
        self.etag = None
        self.seen_response = False

    def __call__(self, start, count):
        headers = {'Range': f'bytes={start}-{start+count-1}', 'Accept-Encoding': 'identity'}
        if self.etag is not None: headers['If-Match'] = self.etag
        request = urllib.request.Request(URL, headers=headers)
        with self.opener.open(request, timeout=20) as response:
            if (response.status != 206 or response.geturl() != URL
                    or response.headers.get('Content-Range') != f'bytes {start}-{start+count-1}/{ARCHIVE_BYTES}'
                    or response.headers.get('Content-Encoding', 'identity') != 'identity'):
                raise ValueError('Server did not honor exact bounded range')
            etag = response.headers.get('ETag')
            if etag and etag.startswith('W/'):
                raise ValueError('Weak ETag is not supported')
            if self.seen_response and etag != self.etag:
                raise ValueError('Source changed between range reads')
            self.etag = etag
            self.seen_response = True
            result = response.read(count + 1)
        if len(result) != count: raise ValueError('Range length mismatch')
        return result


class RangeReader(io.RawIOBase):
    def __init__(self, fetch, size=ARCHIVE_BYTES):
        super().__init__()
        if type(size) is not int or not 1 <= size <= ARCHIVE_BYTES:
            raise ValueError('Invalid archive size')
        self.fetch, self.size, self.position = fetch, size, 0
        self.transferred = self.requests = 0
        self.started = time.monotonic()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        if type(offset) is not int or type(whence) is not int or whence not in (0, 1, 2):
            raise ValueError('Invalid seek')
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if not 0 <= position <= self.size: raise ValueError('Seek outside archive')
        self.position = position
        return position

    def read(self, count=-1):
        if type(count) is not int or count < -1: raise ValueError('Invalid read count')
        count = self.size-self.position if count == -1 else min(count, self.size-self.position)
        if not count: return b''
        if (count > MAX_REQUEST or self.transferred+count > MAX_TRANSFER
                or self.requests >= 128 or time.monotonic()-self.started > 120):
            raise ValueError('Archive read budget exceeded')
        self.requests += 1;self.transferred += count
        result = self.fetch(self.position, count)
        if type(result) is not bytes or len(result) != count:
            raise ValueError('Invalid range response')
        self.position += count
        return result


def member_bytes(archive, name):
    matches = [info for info in archive.infolist() if info.filename == name]
    if len(matches) != 1: raise ValueError('Missing or duplicate selected member')
    info = matches[0]
    mode = info.external_attr >> 16
    if (info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 1
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            or not 0 < info.file_size <= MAX_MEMBER or info.compress_size > MAX_MEMBER):
        raise ValueError('Unsupported or oversized selected member')
    with archive.open(info) as handle:
        value = handle.read(info.file_size + 1)
    if len(value) != info.file_size: raise ValueError('Expanded member length mismatch')
    return value


def intake(output, fetch=None):
    root = Path(output);root.mkdir(parents=True, exist_ok=False)
    status = root/'status.json'
    status.write_text('{"status":"running","promotionAuthorized":false}\n')
    reader = RangeReader(fetch if fetch is not None else HttpRanges())
    try:
        artifacts = []
        with zipfile.ZipFile(reader) as archive:
            if len(archive.infolist()) > 4096: raise ValueError('Too many archive members')
            for label, session in SESSIONS.items():
                prefix = 'Zenodo/NoShield/beacon/' + session + '/'
                for source_name, suffix in [('config.json', 'json'), ('capture1_extracted.mat', 'mat')]:
                    member = prefix + source_name
                    value = member_bytes(archive, member)
                    destination = label + '.' + suffix
                    with (root/destination).open('xb') as handle: handle.write(value)
                    artifacts.append({'label': label, 'member': member, 'file': destination,
                                      'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()})
        report = {'status': 'complete', 'evidence': 'PUBLIC_SAMPLE_INTAKE' if fetch is None else 'SYNTHETIC',
                  'source': URL, 'record': '10.5281/zenodo.18924941', 'archive_bytes': ARCHIVE_BYTES,
                  'archive_md5_publisher': '0eacb8856c2f37d43bda98a988d07a56', 'full_archive_hash_verified': False,
                  'snapshot_verified': False,
                  'integrity': 'HTTPS exact ranges, ZIP CRC and local member SHA256; not full archive or snapshot verification',
                  'license_catalog': 'CC-BY-4.0', 'license_readme': 'CC-BY-SA-4.0', 'license_conflict': True,
                  'usage': 'Private intake research only; retain both notices; no redistribution or model promotion',
                  'transferred_bytes': reader.transferred, 'requests': reader.requests,
                  'artifacts': artifacts, 'elapsed_seconds': time.monotonic()-reader.started,
                  'promotionAuthorized': False, 'training_performed': False}
        (root/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
        status.write_text('{"status":"complete","promotionAuthorized":false}\n')
        return report
    except Exception:
        status.write_text('{"status":"failed","promotionAuthorized":false}\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(intake(args.output), indent=2))

if __name__ == '__main__': main()
