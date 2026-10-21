"""Publish compact CPU readiness evidence; preserve the frozen private run."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transfer_v4.common import dump, sha_file
from readiness_v5.gate import decision
from readiness_v5.plan import verify_plan
from readiness_v5.run import require_readiness


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, default=Path('Kaggle Outputs/nca_transfer_v4_run1_review'))
    p.add_argument('--run', type=Path, default=Path('runs/readiness_v5_run1'))
    p.add_argument('--out', type=Path, default=Path('docs/results/readiness_v5_run1'))
    a = p.parse_args()
    torch.set_num_threads(2)
    plan = verify_plan(a.review, a.run)
    results = json.loads((a.run/'results.json').read_text())
    if sha_file(a.run/'results.json') != json.loads((a.run/'results_receipt.json').read_text())['results_sha256']:
        raise ValueError('Result receipt differs')
    if len(results['cells']) != 48 or len(results['cohorts']) != 12:
        raise ValueError('Incomplete cross-evaluation')
    expected = {(seed, arm, dataset) for seed in plan['settings']['seeds'] for structured, control, _ in plan['settings']['families']
                for arm in (structured, control) for dataset in (structured, control)}
    if {(r['seed'], r['checkpoint_arm'], r['data_arm']) for r in results['cells']} != expected:
        raise ValueError('Cross-evaluation cell identities differ')
    for row in results['cohorts']:
        if decision(row['structured_coherent_nll'], row['structured_shuffled_nll'], row['frequency_nll'], plan['settings']) != row['gate']:
            raise ValueError('Decision differs from measured scores')
    try:
        approved = require_readiness(a.review, a.run)
        blocked = False
    except ValueError as exc:
        if 'Language training blocked:' not in str(exc):
            raise
        blocked, approved = True, []
    if blocked == results['language_allowed']:
        raise ValueError('Authorization guard disagrees with result')
    a.out.mkdir(parents=True, exist_ok=False)
    provenance = {'scope': 'New CPU diagnostic on inspected v4 rules; no neural training or language evaluation', 'files': {}}
    for name in ('plan.json', 'plan_receipt.json', 'FROZEN_PROTOCOL.md', 'results.json', 'results_receipt.json', 'positive_control.json', 'status.json'):
        shutil.copyfile(a.run/name, a.out/name)
        provenance['files'][name] = {'sha256': sha_file(a.out/name), 'origin': 'Frozen readiness CPU run: '+name}
    flat = [{k:v for k,v in row.items() if k != 'per_rule'} for row in results['cells']]
    with (a.out/'cross_evaluation_cells.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    families = []
    for arm, _, _ in plan['settings']['families']:
        rows = [r for r in results['cohorts'] if r['arm'] == arm]
        families.append({'arm': arm, 'cohorts': len(rows),
                         'frequency_passes': sum(r['gate']['frequency_passed'] for r in rows),
                         'ordering_passes': sum(r['gate']['ordering_passed'] for r in rows),
                         'both_passes': sum(r['gate']['passed'] for r in rows),
                         'mean_frequency_advantage_nats': statistics.mean(r['gate']['frequency_advantage_nats'] for r in rows),
                         'mean_ordering_advantage_nats': statistics.mean(r['gate']['ordering_advantage_nats'] for r in rows),
                         'mean_prefix_model_ordering_gap': statistics.mean(r['control_ordering_gap'] for r in rows),
                         'mean_difference_in_ordering_gaps': statistics.mean(r['difference_in_ordering_gaps'] for r in rows)})
    summary = {'families': families, 'language_blocked_by_guard': blocked, 'approved_initializations': approved,
               'maximum_native_cell_replay_error': max(r['native_replay_absolute_error'] for r in results['cells'] if r['native_replay_absolute_error'] is not None),
               'positive_control_passed': results['positive_control']['passed'], 'plan_sha256': results['plan_sha256']}
    dump(a.out/'summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(plan['settings']['seeds']))
    for i, (arm, _, _) in enumerate(plan['settings']['families']):
        rows = [next(r for r in results['cohorts'] if r['seed'] == seed and r['arm'] == arm) for seed in plan['settings']['seeds']]
        for ax, key in zip(axes, ('frequency_advantage_nats', 'ordering_advantage_nats')):
            ax.bar(x+(i-.5)*.36, [r['gate'][key] for r in rows], width=.34, label=arm, color=('#2563eb', '#059669')[i])
    for ax, title in zip(axes, ('Advantage over causal frequency prediction', 'Advantage from coherent ordering')):
        ax.axhline(.01, color='#dc2626', linestyle='--', label='Frozen 0.01-nat requirement')
        ax.axhline(0, color='#64748b', linewidth=.7)
        ax.set_xticks(x, [str(s) for s in plan['settings']['seeds']])
        ax.set(title=title, xlabel='Paired model/data/topology cohort', ylabel='Nats per patch; larger is better')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Stage 5: checkpoint-only readiness on inspected v4 rules')
    fig.savefig(a.out/'readiness_advantages.png', dpi=180)
    fig.savefig(a.out/'readiness_advantages.svg')
    plt.close(fig)
    dump(a.out/'PROVENANCE.json', provenance)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
