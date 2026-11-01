"""Fetch or verify a small immutable CLRS-Text reference; never overwrite it."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'd1c2ad7af8437c7536cf329d9cef8fdf93184d9d'
FILES = ['LICENSE', 'README.md', 'clrs/_src/clrs_text/README.md',
         'clrs/_src/clrs_text/clrs_utils.py', 'clrs/_src/clrs_text/generate_clrs_text.py',
         'clrs/_src/algorithms/graphs.py', 'clrs/_src/specs.py']


def main():
    out = ROOT/'sources/clrs-text-reference'
    manifest = out/'reference.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        assert record['commit'] == COMMIT and set(record['files']) == set(FILES)
        for name, row in record['files'].items():
            assert hashlib.sha256((out/COMMIT/name).read_bytes()).hexdigest() == row['sha256']
        print('PASS: seven pinned CLRS-Text reference files and Apache license unchanged.')
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Preserve partial/reference evidence; use a fresh destination')
    values = {}
    for name in FILES:
        url = f'https://raw.githubusercontent.com/google-deepmind/clrs/{COMMIT}/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            values[name] = (url, response.read())
    record = {'repository': 'https://github.com/google-deepmind/clrs', 'commit': COMMIT,
              'tag': 'v2.0.1', 'paper': 'https://arxiv.org/html/2406.04229v1',
              'scope': 'Reference only; local graph diagnostic is an explicit task/output adaptation, not official CLRS-Text scores', 'files': {}}
    for name, (url, content) in values.items():
        p = out/COMMIT/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        record['files'][name] = {'url': url, 'sha256': hashlib.sha256(content).hexdigest()}
    manifest.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print('Pinned seven CLRS-Text files:', COMMIT)


if __name__ == '__main__':
    main()
