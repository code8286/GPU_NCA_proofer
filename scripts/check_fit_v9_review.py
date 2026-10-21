"""Check split-score aggregates, original replay identities and freeze chronology."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import sha_file
from fit_v9.diagnostic import read, verify_plan, comparisons
from report_fit_v9 import summarize


def main():
    out, public = ROOT/'runs/fit_v9_run1', ROOT/'docs/results/fit_v9_run1'
    plan = verify_plan(out)
    copies = read(public/'PROVENANCE.json')['files']
    assert len(copies) == 7
    for name, row in copies.items():
        assert sha_file(public/name) == row['sha256'] == sha_file(out/name)
    result = read(public/'results.json')
    assert sha_file(public/'results.json') == read(public/'results_receipt.json')['results_sha256'] == read(public/'verification.json')['results_sha256']
    keys = {(r['seed'], r['condition'], r['arm'], r['data_arm'], r['split']) for r in result['cells']}
    assert len(keys) == len(result['cells']) == 90
    assert result['models_scored'] == 15 and result['new_training_updates'] == 0 and not result['language_allowed']
    for row in result['cells']:
        n = plan['training_config']['splits'][row['split']]
        values = row['per_query']
        assert row['queries'] == n and all(len(v) == n for v in values.values())
        assert float(np.mean(values['first_patch_nll'])) == row['first_patch_nll']
        assert float(np.mean(values['first_patch_correct'])) == row['first_patch_accuracy']
        assert float(np.mean(values['transition_patch_nll'])) == row['transition_patch_nll']
        assert float(np.mean(values['transition_patch_correct_count'])/4) == row['transition_patch_accuracy']
        assert float(np.mean(values['transition_marker_nll'])) == row['transition_marker_nll']
        assert (4*row['transition_patch_nll']+2*row['transition_marker_nll'])/6 == row['transition_all_token_nll']
        assert 0 <= row['short_long_max_nll_difference'] <= 1e-5
    pairs, gaps = comparisons(result['cells'], plan['training_config'])
    assert pairs == result['paired_comparisons'] and gaps == result['generalization_gaps']
    assert read(public/'summary.json') == summarize(result) | {'plan_sha256': sha_file(out/'plan.json')}
    principal = [r for r in pairs if r['arm'] == r['data_arm'] == 'coherent']
    assert [(r['seed'], r['category']) for r in principal] == [(131, 'fit_regression_on_train_and_assessment'),
                                                            (149, 'training_gain_with_assessment_reversal'),
                                                            (167, 'training_gain_with_assessment_reversal')]
    index = {(r['seed'], r['condition'], r['arm'], r['data_arm'], r['split']): r for r in result['cells']}
    for seed in plan['training_config']['seeds']:
        for split in ('train', 'assessment'):
            a, b = [index[seed, k, 'coherent', 'coherent', split] for k in ('standard', 'weighted')]
            assert b['transition_patch_nll'] < a['transition_patch_nll'] and b['transition_patch_accuracy'] > a['transition_patch_accuracy']
    events = [json.loads(line) for line in (ROOT/'logs/checks.jsonl').read_text().splitlines()]
    def event(label):
        return next(e for e in events if e['label'] == label and e['exit_code'] == 0)
    freeze, assess = [event(label) for label in ('fit-v9-freeze-before-new-split-scoring', 'fit-v9-frozen-existing-checkpoint-split-assessment')]
    frozen_at = datetime.fromisoformat(freeze['started_utc'])+timedelta(seconds=freeze['duration_seconds'])
    assert frozen_at < datetime.fromisoformat(assess['started_utc'])
    print('PASS: seven copied evidence identities; 90 nested score aggregates; all paired gaps/categories; whole-transition improvement and mixed first-patch finding; freeze-before-new-scoring chronology; zero new training updates.')


if __name__ == '__main__':
    main()
