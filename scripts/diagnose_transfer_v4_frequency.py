"""Post-hoc causal frequency diagnostic; never changes the declared admission gate.

An exchangeable Dirichlet-multinomial predictor updates counts using only past
patch tokens. Its joint conditional likelihood has a closed form in final counts;
using that identity does not give the predictor future tokens. Fit the prior on
synthetic training rules, then evaluate only the declared held-out rule prefix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from transfer_v4.common import dump, frame_length, sha_file


def count_parts(tokens, c):
    cutoff, first, end = c['nca']['states']**4, frame_length(c), c['model']['context']+1
    initial = np.zeros((len(tokens), cutoff), dtype=np.float64)
    target = np.zeros_like(initial)
    for i, row in enumerate(tokens):
        a, b = row[:first], row[first:end]
        initial[i] = np.bincount(a[a < cutoff], minlength=cutoff)
        target[i] = np.bincount(b[b < cutoff], minlength=cutoff)
    return initial, target


def conditional_patch_nll(initial, target, probabilities, concentration):
    initial = torch.as_tensor(initial, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    prior = torch.as_tensor(probabilities, dtype=torch.float64)*concentration
    total0 = initial.sum(1)+concentration
    logp = (torch.lgamma(total0)-torch.lgamma(total0+target.sum(1))
            +(torch.lgamma(prior+initial+target)-torch.lgamma(prior+initial)).sum(1))
    return float(-logp.sum()/target.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Use a fresh diagnostic output')
    torch.set_num_threads(1)
    data, run = a.review/'artifacts/data/development', a.review/'artifacts/runs/development'
    manifest = json.loads((data/'manifest.json').read_text())
    c = manifest['config']
    results = []
    for seed in c['seeds']:
        for kind, arm, control in (('nca', 'grid_nca', 'prefix_grid'), ('graph', 'graph_nca', 'prefix_graph')):
            folder = data/f'replicate_{seed}'
            files = {split: folder/f'{kind}_{split}.npy' for split in ('train', 'validation')}
            for path in files.values():
                if sha_file(path) != manifest['files'][path.relative_to(data).as_posix()]:
                    raise ValueError('Diagnostic dataset checksum mismatch')
            train, val = (np.load(files[s], allow_pickle=False) for s in ('train', 'validation'))
            v = c['nca']['states']**4
            targets = train[:, frame_length(c):c['model']['context']+1]
            counts = np.bincount(targets[targets < v], minlength=v).astype(float)+1
            probabilities = counts/counts.sum()
            fit_n = min(256, len(train))
            train_initial, train_target = count_parts(train[:fit_n], c)
            candidates = [.1, 1., 10., 100., 1000., 10000.]
            fit = [{'concentration': alpha, 'train_patch_nll': conditional_patch_nll(train_initial, train_target, probabilities, alpha)} for alpha in candidates]
            selected = min(fit, key=lambda r:r['train_patch_nll'])['concentration']
            val_initial, val_target = count_parts(val[:c['ppt_eval_rules']], c)
            loss = conditional_patch_nll(val_initial, val_target, probabilities, selected)
            prefix = np.load(folder/f'{control}_validation.npy', allow_pickle=False)[:c['ppt_eval_rules']]
            same_initial, same_target = count_parts(prefix, c)
            if not np.array_equal(val_initial, same_initial) or not np.array_equal(val_target, same_target):
                raise ValueError('Matched prefix control does not share the exchangeable likelihood')
            parent_loss = json.loads((run/f'{arm}_{seed}/ppt_summary.json').read_text())['selected']['patch_nll']
            control_loss = json.loads((run/f'{control}_{seed}/ppt_summary.json').read_text())['selected']['patch_nll']
            results.append({'seed': seed, 'arm': arm, 'control_arm': control, 'train_rules_for_concentration_fit': fit_n,
                            'concentration_fit': fit, 'selected_concentration': selected, 'held_out_rules': len(val_initial),
                            'patch_targets': int(val_target.sum()), 'causal_exchangeable_patch_nll': loss,
                            'imported_ppt_patch_nll': parent_loss, 'imported_control_patch_nll': control_loss,
                            'ppt_minus_exchangeable_nll': parent_loss-loss,
                            'matched_control_exchangeable_nll_identical': True})
    result = {'scope': 'Post-hoc CPU diagnostic, not a revised admission gate or predeclared language-transfer outcome',
              'prior': 'Add-one-smoothed global frequencies from all training rule prefixes; concentration selected only on first 256 training rules',
              'conditioning': 'Actual first frame; marker tokens excluded; posterior updates after each observed patch',
              'data_manifest_sha256': sha_file(data/'manifest.json'), 'results': results}
    dump(a.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
