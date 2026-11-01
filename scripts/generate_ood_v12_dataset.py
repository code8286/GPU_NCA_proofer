"""Generate the declared development suite without historical checkpoints."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ood_v12.data import generate, write_jsonl
from ood_v12.plan import read, sources
from transfer_v4.common import dump, sha_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/ood_v12.json')
    a = p.parse_args()
    c = read(a.config)
    if c['language_training_allowed'] or c['reserved_confirmation_generated']:
        raise ValueError('Development data only; no fitting or reserved confirmation generation')
    data, info = generate(c)
    a.out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rows in data.items():
        write_jsonl(a.out/(name+'.jsonl'), rows)
        hashes[name+'.jsonl'] = sha_file(a.out/(name+'.jsonl'))
    dump(a.out/'manifest.json', {'config': c, 'source_sha256': sources(), 'files_sha256': hashes,
                               'inventory': info, 'scope': 'CLRS-inspired development data, not official CLRS or untouched confirmation'})
    print(info)


if __name__ == '__main__':
    main()
