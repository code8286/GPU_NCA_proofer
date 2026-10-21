"""Retained tiny CPU execution fixture; not a model trainability result."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainability_v6.plan import freeze
from trainability_v6.run import train, assess, verify


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    c = json.loads((ROOT/'configs/trainability_v6.json').read_text())
    c.update(purpose='execution_check', seeds=[1], splits={'train': 32, 'development': 4, 'assessment': 4}, batch=1, accumulate=1, eval_every=1)
    c['model'].update(layers=1, width=16, heads=2)
    c['ppt']['steps'] = 2
    out = ROOT/'runs'/('trainability_v6_execution_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    freeze(out, c)
    train(out)
    assess(out)
    verify(out)
    print('PASS: retained two-arm CPU execution fixture; no trainability/transfer inference.')


if __name__ == '__main__':
    main()
