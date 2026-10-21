"""Verify public evidence copies and raw ZIP filename headers after a v4 audit."""
from pathlib import Path, PurePosixPath
import argparse
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import sha_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--public', type=Path, required=True)
    a = p.parse_args()
    audit = json.loads((a.review/'audit.json').read_text())
    provenance = json.loads((a.public/'PROVENANCE.json').read_text())
    if provenance['archive_sha256'] != audit['archive_sha256'] or sha_file(a.archive) != audit['archive_sha256']:
        raise ValueError('Original archive identity changed')
    with zipfile.ZipFile(a.archive) as z:
        for entry in z.infolist():
            path = PurePosixPath(entry.filename)
            if entry.orig_filename != entry.filename or path.is_absolute() or '..' in path.parts or '\\' in entry.filename or ':' in entry.filename:
                raise ValueError('Raw ZIP header path is unsafe or normalized')
    for name, entry in provenance['files'].items():
        if sha_file(a.public/name) != entry['sha256']:
            raise ValueError(f'Public evidence copy differs: {name}')
    if sha_file(a.review/'artifacts/runs/development/verification.json') != audit['imported_gpu_verification_sha256']:
        raise ValueError('Imported GPU verification changed')
    if audit['cpu_checkpoints_verified'] != 78 or audit['cpu_initializations_verified'] != 6 or audit['test_split_present']:
        raise ValueError('Unexpected independent audit scope')
    print(f"PASS: original archive hash, {audit['members_crc_checked']} raw path headers, {len(provenance['files'])} public evidence hashes, preserved GPU verification, 78-checkpoint/6-initialization scope.")


if __name__ == '__main__':
    main()
