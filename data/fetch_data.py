#!/usr/bin/env python3
"""Fetch optional source inputs; the core simulation uses bundled derived CSVs.

Run from any directory. --verify performs no downloads and exits nonzero on
missing automatic inputs or any checksum mismatch. Manual inputs are optional.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
DEFAULT_DEST = HERE.parent / 'code' / 'emd1_simulation' / 'data'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_copy(stream, destination, expected):
    """Never install a partial or mismatched download as an analysis input."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
        staging = Path(f.name)
        try:
            shutil.copyfileobj(stream, f)
            f.flush()
            if sha256(staging) != expected:
                raise ValueError(f'Checksum mismatch: {destination.name}')
            f.close()
            staging.replace(destination)
        finally:
            staging.unlink(missing_ok=True)


def load_sources():
    with (HERE / 'SOURCES.tsv').open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def verify(rows, destination):
    failures = 0
    for row in rows:
        path = destination / row['path']
        if not path.is_file():
            if row['automatic'] == '1':
                print('MISSING', row['path']); failures += 1
            else:
                print('OPTIONAL, not supplied:', row['path'])
        elif sha256(path) != row['sha256']:
            print('MISMATCH', row['path']); failures += 1
        else:
            print('OK', row['path'])
    return failures


def fetch(rows, destination):
    pending = {}
    for row in rows:
        path = destination / row['path']
        if row['automatic'] != '1':
            continue
        if path.is_file():
            if sha256(path) != row['sha256']:
                raise ValueError(f'Existing file differs from reference: {path}; preserve or move it before retrying')
            continue
        pending.setdefault(row['url'], []).append(row)
    for url, group in pending.items():
        print('Fetching', url, flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            if group[0]['archive_member']:
                # Stream the GEO archive once; write only explicitly named members.
                wanted = {r['archive_member']: r for r in group}
                with tarfile.open(fileobj=response, mode='r|*') as archive:
                    for member in archive:
                        row = wanted.get(member.name)
                        if row is None:
                            continue
                        if not member.isfile():
                            raise ValueError(f'Not a regular source file: {member.name}')
                        with archive.extractfile(member) as stream:
                            checked_copy(stream, destination / row['path'], row['sha256'])
                        del wanted[member.name]
                if wanted:
                    raise ValueError('Missing archive members: ' + ', '.join(wanted))
            else:
                if len(group) != 1:
                    raise ValueError('Ambiguous direct download in manifest')
                row = group[0]
                checked_copy(response, destination / row['path'], row['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    rows = load_sources()
    try:
        if not args.verify:
            fetch(rows, args.dest)
        return 1 if verify(rows, args.dest) else 0
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(f'ERROR: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
