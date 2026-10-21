"""Retained four-arm tiny execution fixture, separate from spatial learning."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interaction_v7.plan import read, freeze
from interaction_v7.run import train, assess, verify


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    c = read(ROOT/'configs/interaction_v7.json')
    c.update(purpose='execution_check', seeds=[1], splits={'train': 32, 'development': 4, 'assessment': 4}, batch=1, accumulate=1, eval_every=1)
    c['model'].update(layers=1, width=16, heads=2)
    c['ppt']['steps'] = 2
    out = ROOT/'runs'/('interaction_v7_execution_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    freeze(out, c)
    train(out)
    assess(out)
    verify(out)
    print('PASS: retained four-arm/two-update execution fixture; no learning inference.')


if __name__ == '__main__':
    main()
