"""Check retained copies, paired budgets/orders, nested scores and freeze chronology."""
import hashlib
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import sha_file, effective_tokens
from readiness_v5.gate import decision
from loss_weight_v8.plan import read, verify_plan
from loss_weight_v8.run import verify_training


def main():
    out, public = ROOT/'runs/loss_weight_v8_run1', ROOT/'docs/results/loss_weight_v8_run1'
    plan = verify_plan(out)
    c, models = plan['training_config'], verify_training(out, plan)
    assert len(models) == len(plan['standard_controls']) == 6
    copies = read(public/'PROVENANCE.json')['files']
    assert len(copies) == 21
    for name, entry in copies.items():
        assert sha_file(public/name) == entry['sha256'] == sha_file(out/name)
    assert sha_file(out/'plan.json') == read(out/'plan_receipt.json')['plan_sha256']
    for seed in c['seeds']:
        order, rng = hashlib.sha256(), np.random.default_rng(seed+101)
        for _ in range(c['ppt']['steps']*c['accumulate']):
            order.update(rng.integers(0, c['splits']['train'], size=c['batch']).astype('<i8').tobytes())
        initial = torch.load(out/f'initialization_{seed}.pt', map_location='cpu', weights_only=True)
        for arm in ('coherent', 'prefix'):
            row = next(r for r in models if (r['seed'], r['arm']) == (seed, arm))
            old = next(r for r in plan['standard_controls'] if (r['seed'], r['treatment']) == (seed, arm))
            for key in ('steps', 'input_tokens', 'prediction_targets', 'initial_core_sha256', 'initial_head_sha256', 'train_order_sha256'):
                assert row[key] == old[key]
            assert row['input_tokens'] == c['ppt']['steps']*effective_tokens(c) == 188000
            assert row['prediction_targets'] == 168000
            assert row['train_order_sha256'] == order.hexdigest()
            assert row['initial_core_sha256'] == initial['core_sha256'] and row['initial_head_sha256'] == initial['head_sha256']
    result = read(public/'results.json')
    assert sha_file(public/'results.json') == read(public/'results_receipt.json')['results_sha256'] == read(public/'verification.json')['results_sha256']
    full = {(r['seed'], r['condition'], r['arm'], r['data_arm']): r for r in result['full_cells']}
    assert len(full) == len(result['full_cells']) == 30
    for row in full.values():
        values = row['per_initial_condition']
        assert len(values) == 128 and all(v['patch_targets'] == 28 for v in values)
        assert abs(sum(v['patch_nll'] for v in values)/128-row['patch_nll']) < 1e-12
    first = {(r['seed'], r['condition'], r['arm']): r for r in result['first_transition_cells']}
    assert len(first) == len(result['first_transition_cells']) == 15
    mask = np.array(read(out/'queries/eligibility.json')['eligible'], dtype=bool)
    assert len(mask) == 128 and mask.sum() == result['eligible_pairs'] == 122
    for row in first.values():
        for kind, n in (('native', 128), ('changed', 122)):
            values, correct = row[kind+'_per_query_nll'], row[kind+'_per_query_correct']
            assert len(values) == len(correct) == n
            assert float(np.array(values).mean()) == row[kind+'_nll']
            assert float(np.array(correct).mean()) == row[kind+'_accuracy']
        assert float((np.array(row['native_per_query_correct'])[mask] & np.array(row['changed_per_query_correct'])).mean()) == row['paired_both_correct']
    for row in result['paired_comparisons']:
        seed = row['seed']
        standard, weighted = first[seed, 'standard', 'coherent'], first[seed, 'weighted', 'coherent']
        assert row['first_patch_nll_improvement'] == standard['native_nll']-weighted['native_nll']
        assert row['intervention_nll_improvement'] == standard['changed_nll']-weighted['changed_nll']
        assert row['first_patch_improved'] == (row['first_patch_nll_improvement'] >= .01)
        assert row['intervention_improved'] == (row['intervention_nll_improvement'] >= .01)
        assert row['weighted_readiness'] == decision(full[seed, 'weighted', 'coherent', 'coherent']['patch_nll'],
                                                  full[seed, 'weighted', 'coherent', 'prefix']['patch_nll'], result['frequency_nll'], c)
        control = first[seed, 'standard', 'prefix']['native_nll']-first[seed, 'weighted', 'prefix']['native_nll']
        assert row['shuffled_first_patch_improvement'] == control
        assert row['difference_in_objective_gains'] == row['first_patch_nll_improvement']-control
    assert result['all_seed_primary_improvement'] == all(r['first_patch_improved'] and r['intervention_improved'] and r['weighted_readiness']['passed'] for r in result['paired_comparisons'])
    assert not result['language_allowed']
    events = [json.loads(line) for line in (ROOT/'logs/checks.jsonl').read_text().splitlines()]
    def event(label):
        return next(e for e in events if e['label'] == label and e['exit_code'] == 0)
    freeze, training, assessment = [event(label) for label in ('loss-weight-v8-freeze-before-weighted-training',
                                                              'loss-weight-v8-six-fixed-budget-weighted-models',
                                                              'loss-weight-v8-final-only-assessment')]
    finish = lambda e: datetime.fromisoformat(e['started_utc'])+timedelta(seconds=e['duration_seconds'])
    assert finish(freeze) < datetime.fromisoformat(training['started_utc'])
    assert finish(training) < datetime.fromisoformat(assessment['started_utc'])
    print('PASS: 21 exact evidence copies; six new/imported paired starts, budgets and recreated batch orders; all 30 full and 15 first-transition score aggregates/decisions; freeze-before-training-before-assessment chronology.')


if __name__ == '__main__':
    main()
