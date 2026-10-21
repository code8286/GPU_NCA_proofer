"""Repeat the frozen CPU bridge and first-transition checks in one fresh folder."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True, help='New output folder; never overwrite evidence')
    a = p.parse_args()
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError('A fresh output directory is required')

    def run(*args):
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)

    run('-m', 'interaction_v7.run', 'freeze', '--out', out)
    run('scripts/diagnose_interaction_v7_first_transition.py', 'freeze', '--parent', out, '--out', out/'first_transition')
    for command in ('train', 'assess', 'verify'):
        run('-m', 'interaction_v7.run', command, '--out', out)
    run('scripts/diagnose_interaction_v7_first_transition.py', 'score', '--parent', out, '--out', out/'first_transition')
    run('scripts/check_interaction_v7_review.py', '--run', out, '--replay-first-transition')
    print('WORKFLOW FINISHED: see primary and first_transition results separately. Language remains disabled.')


if __name__ == '__main__':
    main()
