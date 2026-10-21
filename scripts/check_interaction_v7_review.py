"""Verify frozen bridge evidence and separately replay first-transition queries."""
import argparse
import hashlib
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import dump, sha_file, effective_tokens
from readiness_v5.gate import decision
from interaction_v7.plan import read, verify_plan
from interaction_v7.run import verify_training, load_model
from interaction_v7.data import canonical
from diagnose_interaction_v7_first_transition import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-first-transition', action='store_true')
    p.add_argument('--run', type=Path, default=ROOT/'runs/interaction_v7_run1')
    a = p.parse_args()
    torch.set_num_threads(2)
    out, public = a.run, ROOT/'docs/results/interaction_v7_run1'
    plan = verify_plan(out)
    models, c = verify_training(out, plan), plan['config']
    old = read(ROOT/'docs/results/trainability_v6_run1/plan.json')
    assert plan['v4_source_sha256'] == old['v4_source_sha256']
    for name, checksum in old['source_sha256'].items():
        assert plan['source_sha256'][name] == checksum
    for key in old['config']:
        if key not in ('schema_version', 'purpose', 'rule'):
            assert c[key] == old['config'][key]
    ids = read(out/'data/split_identities.json')
    all_classes = sum(ids['joint_orbit_ids'].values(), [])
    assert len(set(all_classes)) == len(all_classes) == 704
    for split, values in ids['initial_ids'].items():
        np.testing.assert_array_equal(canonical(np.array(values)), ids['joint_orbit_ids'][split])
        x = np.load(out/f'data/complement_{split}.npy', allow_pickle=False)
        y = np.load(out/f'data/spatial_{split}.npy', allow_pickle=False)
        np.testing.assert_array_equal(x[:, :6], y[:, :6])
    for seed in c['seeds']:
        order, rng = hashlib.sha256(), np.random.default_rng(seed+101)
        for _ in range(c['ppt']['steps']*c['accumulate']):
            order.update(rng.integers(0, 512, size=c['batch']).astype('<i8').tobytes())
        initial = torch.load(out/f'initialization_{seed}.pt', map_location='cpu', weights_only=True)
        old_initial = read(ROOT/f'docs/results/trainability_v6_run1/coherent_{seed}/summary.json')
        assert initial['core_sha256'] == old_initial['initial_core_sha256'] and initial['head_sha256'] == old_initial['initial_head_sha256']
        for row in [r for r in models if r['seed'] == seed]:
            assert row['train_order_sha256'] == order.hexdigest()
            assert row['initial_core_sha256'] == initial['core_sha256'] and row['initial_head_sha256'] == initial['head_sha256']
            assert row['input_tokens'] == c['ppt']['steps']*effective_tokens(c)
            assert row['prediction_targets'] == 168000
    result = read(out/'results.json')
    assert sha_file(out/'results.json') == read(out/'results_receipt.json')['results_sha256'] == read(out/'verification.json')['results_sha256']
    assert not result['language_allowed']
    cells = {(r['rule'], r['seed'], r['checkpoint_arm'], r['data_arm']):r for r in result['cells']}
    assert len(cells) == len(result['cells']) == 36
    for row in cells.values():
        scores = row['per_initial_condition']
        assert len(scores) == 128 and all(v['patch_targets'] == 28 for v in scores)
        assert abs(sum(v['patch_nll'] for v in scores)/128-row['patch_nll']) < 1e-12
    for row in result['decisions']:
        family = next(f for f in result['families'] if f['rule'] == row['rule'])
        assert row['coherent_nll'] == cells[row['rule'], row['seed'], 'coherent', 'coherent']['patch_nll']
        assert row['shuffled_nll'] == cells[row['rule'], row['seed'], 'coherent', 'prefix']['patch_nll']
        assert row['gate'] == decision(row['coherent_nll'], row['shuffled_nll'], family['frequency_nll'], c)
        assert row['local_table_advantage_nats'] == family['local_table_nll']-row['coherent_nll']
        assert row['local_table_advantage_passed'] == (row['local_table_advantage_nats'] >= .01)
    sub = out/'first_transition'
    frozen, first = read(sub/'plan.json'), read(sub/'results.json')
    assert sha_file(sub/'plan.json') == read(sub/'plan_receipt.json')['plan_sha256'] == first['plan_sha256']
    assert frozen['source_sha256'] == sha_file(ROOT/'scripts/diagnose_interaction_v7_first_transition.py')
    assert frozen['protocol_sha256'] == sha_file(sub/'FROZEN_PROTOCOL.md') == sha_file(ROOT/'docs/INTERACTION_V7_FIRST_TRANSITION.md')
    assert first['parent_results_sha256'] == sha_file(out/'results.json')
    assert sha_file(sub/'results.json') == read(sub/'results_receipt.json')['results_sha256']
    for name, checksum in frozen['query_files_sha256'].items():
        assert sha_file(sub/name) == checksum
    mask = np.array(read(sub/'eligibility.json')['eligible'], dtype=bool)
    assert len(mask) == 128 and int(mask.sum()) == first['eligible_pairs']
    forbidden = set(ids['joint_orbit_ids']['train']+ids['joint_orbit_ids']['development'])
    expected_mask = [int(k) not in forbidden for k in canonical(np.array(ids['initial_ids']['assessment']) ^ 8)]
    np.testing.assert_array_equal(mask, expected_mask)
    assert len(first['cells']) == 18
    for row in first['cells']:
        baseline = next(b for b in first['baselines'] if b['rule'] == row['rule'])
        assert row['stricter_first_transition_passed'] == (min(baseline['first_patch_table_nll'], baseline['uniform_nll'])-row['native_nll'] >= .01)
        expected_hash = plan['input_files_sha256'][f"initialization_{row['seed']}.pt"] if row['arm'] == 'initial' else next(r['checkpoint_sha256'] for r in models if r['seed'] == row['seed'] and r['arm'] == f"{row['rule']}_{row['arm']}")
        assert row['checkpoint_sha256'] == expected_hash
        correct = np.array(row['native_per_query_correct'])[mask]
        other = np.array(row['changed_per_query_correct'])
        assert float((correct & other).mean()) == row['paired_both_correct']
    assert first['stricter_first_transition_passed'] == all(r['stricter_first_transition_passed'] for r in first['cells'] if r['rule'] == 'spatial' and r['arm'] == 'coherent')
    if a.replay_first_transition:
        chronology = {}
        if out.resolve() == (ROOT/'runs/interaction_v7_run1').resolve():
            events = [json.loads(line) for line in (ROOT/'logs/checks.jsonl').read_text().splitlines()]
            def event(label):
                return next(e for e in events if e['label'] == label and e['exit_code'] == 0)
            fitting = event('interaction-v7-twelve-full-budget-cpu-models')
            freezing = event('interaction-v7-supplementary-first-transition-freeze-before-assessment')
            assessment = event('interaction-v7-final-only-primary-assessment')
            start = datetime.fromisoformat(fitting['started_utc'])
            frozen_at = datetime.fromisoformat(freezing['started_utc'])+timedelta(seconds=freezing['duration_seconds'])
            assessed_at = datetime.fromisoformat(assessment['started_utc'])
            assert start < frozen_at < start+timedelta(seconds=fitting['duration_seconds']) < assessed_at
            chronology = {'supplement_completed_utc': frozen_at.isoformat(), 'primary_assessment_started_utc': assessed_at.isoformat(),
                          'supplement_frozen_during_training_before_assessment': True}
        for row in first['cells']:
            rule, seed, arm = row['rule'], row['seed'], row['arm']
            model, _ = load_model(out, c, seed, None if arm == 'initial' else f'{rule}_{arm}')
            predictions = {}
            for kind, query, loss_key, correct_key in (
                ('native', np.load(sub/f'{rule}_native.npy', allow_pickle=False), 'native_per_query_nll', 'native_per_query_correct'),
                ('changed', np.load(sub/f'{rule}_changed.npy', allow_pickle=False)[mask], 'changed_per_query_nll', 'changed_per_query_correct')):
                loss, correct, predictions[kind] = score(model, query)
                np.testing.assert_array_equal(loss, row[loss_key])
                np.testing.assert_array_equal(correct, row[correct_key])
                assert float(loss.mean()) == row[kind+'_nll'] and float(correct.mean()) == row[kind+'_accuracy']
            expected_prediction = predictions['native'][mask] ^ (1 if rule == 'spatial' else 0)
            assert float((expected_prediction == predictions['changed']).mean()) == row['argmax_changes_as_required']
            del model
        dump(sub/'verification.json', {'passed': True, 'replayed_cells': len(first['cells']), 'results_sha256': sha_file(sub/'results.json'), 'language_allowed': False})
        if chronology:
            verification = read(sub/'verification.json') | {'logged_chronology': chronology}
            dump(sub/'verification.json', verification)
        print('PASS: all 18 supplementary original/intervention cell scores replay exactly.')
    else:
        for name, entry in read(public/'PROVENANCE.json')['files'].items():
            assert sha_file(public/name) == entry['sha256'] == sha_file(out/name)
        print('PASS: public copies/receipts, all 36 primary cells, twelve recreated training orders/budgets, common starting weights and jointly disjoint orbits.')


if __name__ == '__main__':
    main()
