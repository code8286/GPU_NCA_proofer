"""Small CPU check of all six new default cohorts; no model training."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from transfer_v4.common import load_config
from transfer_v4.controls import prefix_shuffled, validate_prefix_control
from transfer_v4.graph import generate_paired, rewire


def main():
    torch.set_num_threads(2)
    c = load_config(ROOT/'configs/transfer_v4.json')
    results = []
    for seed in c['seeds']:
        edges, topology = rewire(c['nca']['grid'], c['graph']['rewire_fraction'], c['graph']['topology_seed']+seed)
        paired, _, stats = generate_paired(c, 16, 'train', seed, 'cpu', edges)
        for tokens in paired.values():
            validate_prefix_control(tokens, prefix_shuffled(tokens, c, 100003*seed+30), c)
        results.append({'seed': seed, 'accepted': stats['accepted'], 'candidates': stats['candidates'],
                        'shape': list(paired['graph'].shape), 'topology_sha256': topology['sha256'],
                        'changed_fraction': topology['actual_changed_fraction']})
    if len({r['topology_sha256'] for r in results}) != 6:
        raise ValueError('Default cohort topologies unexpectedly coincide')
    print(json.dumps({'scope': 'CPU feasibility for 16 rules/cohort, not full generation, learning or GPU timing',
                      'replicates': results}, indent=2))


if __name__ == '__main__':
    main()
