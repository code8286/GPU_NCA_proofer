"""Two-update local execution fixture using retained stage 7 inputs/controls."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loss_weight_v8.plan import read, freeze
from loss_weight_v8.run import train, assess, verify


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    s = read(ROOT/'configs/loss_weight_v8.json')
    s['purpose'] = 'execution_check'
    c = copy.deepcopy(read(ROOT/'configs/interaction_v7.json'))
    c.update(seeds=[131], batch=1, accumulate=1, eval_every=1)
    c['ppt']['steps'] = 2
    out = ROOT/'runs'/('loss_weight_v8_execution_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    freeze(out, s, fixture_config=c)
    train(out)
    assess(out)
    verify(out)
    print('PASS: local two-update fixture. Imported controls have their historical full budget; no scientific comparison is inferred.')


if __name__ == '__main__':
    main()
