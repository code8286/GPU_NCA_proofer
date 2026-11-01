"""Verify nested score evidence, exposure masks, all-seed decisions and chronology."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import sha_file
from diversity_v10.plan import read, verify_plan
from diversity_v10.run import verify_training, subset_metrics, decisions, compact
from diversity_v10.data import exposure_mask
from report_diversity_v10 import raw_evidence, summarize


def main():
    out, public = ROOT/'runs/diversity_v10_run1', ROOT/'docs/results/diversity_v10_run1'
    plan = verify_plan(out)
    models = verify_training(out, plan)
    provenance = read(public/'PROVENANCE.json')['files']
    assert len(provenance) == 42
    for name, row in provenance.items():
        assert sha_file(public/name) == row['sha256'] == sha_file(out/name)
    result = read(public/'results.json')
    assert sha_file(public/'results.json') == read(public/'results_receipt.json')['results_sha256'] == read(public/'verification.json')['results_sha256']
    raw = raw_evidence(public, result)
    assert len(raw['full_cells']) == 54 and len(raw['fit_cells']) == 108 and len(raw['intervention_cells']) == 27
    full_index = {(r['seed'], r['level'], r['objective'], r['arm'], r['data_arm']): r for r in raw['full_cells']}
    fit_index = {(r['seed'], r['level'], r['objective'], r['arm'], r['probe']): r for r in raw['fit_cells']}
    intervention_index = {(r['seed'], r['level'], r['objective'], r['arm']): r for r in raw['intervention_cells']}
    assert (len(full_index), len(fit_index), len(intervention_index)) == (54, 108, 27)
    for row in full_index.values():
        values = row['per_initial_condition']
        assert len(values) == 128 and all(v['patch_targets'] == 28 for v in values)
        assert abs(sum(v['patch_nll'] for v in values)/128-row['patch_nll']) < 1e-12
    for row in fit_index.values():
        n = {'anchor': 512, 'extension': 512, 'development': 64, 'assessment': 128}[row['probe']]
        values = row['per_query']
        assert row['queries'] == n and all(len(v) == n for v in values.values())
        assert float(np.mean(values['first_patch_nll'])) == row['first_patch_nll']
        assert float(np.mean(values['first_patch_correct'])) == row['first_patch_accuracy']
        assert float(np.mean(values['transition_patch_nll'])) == row['transition_patch_nll']
        assert float(np.mean(values['transition_patch_correct_count'])/4) == row['transition_patch_accuracy']
        assert float(np.mean(values['transition_marker_nll'])) == row['transition_marker_nll']
        assert (4*row['transition_patch_nll']+2*row['transition_marker_nll'])/6 == row['transition_all_token_nll']
        assert row['short_long_max_nll_difference'] <= 1e-5
        mask = exposure_mask(row['probe'], row['level'], row['objective'], row['seed'], plan['expected_orders'], n)
        assert mask.tolist() == row['query_exposed']
        for key, subset in (('drawn_subset', mask), ('undrawn_subset', ~mask)):
            assert subset_metrics(values['first_patch_nll'], values['first_patch_correct'], subset) == row[key]
    eligible = np.array(read(out/'queries/eligibility.json')['eligible'], dtype=bool)
    assert int(eligible.sum()) == 122
    for key, row in intervention_index.items():
        fit = fit_index[*key, 'assessment']
        assert row['native_nll'] == fit['first_patch_nll'] and row['native_accuracy'] == fit['first_patch_accuracy']
        assert len(row['changed_per_query_nll']) == len(row['changed_per_query_correct']) == 122
        assert float(np.mean(row['changed_per_query_nll'])) == row['changed_nll']
        assert float(np.mean(row['changed_per_query_correct'])) == row['changed_accuracy']
        assert float((np.array(fit['per_query']['first_patch_correct'])[eligible] & np.array(row['changed_per_query_correct'])).mean()) == row['paired_both_correct']
    pairs, strata = decisions(raw['full_cells'], raw['intervention_cells'], result['baselines'], plan['training_config'])
    assert pairs == result['paired_comparisons'] and strata == result['objective_strata_passed']
    assert result['study_primary_passed'] == all(strata.values()) and not result['language_allowed']
    combined = {k: v for k, v in result.items() if k not in ('evidence_sha256', 'full_cells', 'fit_cells', 'intervention_cells')} | raw
    assert compact(combined, result['evidence_sha256']) == result
    summary, _ = summarize(plan, result, models, raw)
    assert summary == read(public/'summary.json')
    for row in models:
        old = next(r['source_summary'] for r in plan['standard_controls'] if (r['seed'], r['objective'], r['arm']) == (row['seed'], row['objective'], row['arm']))
        for key in ('steps', 'input_tokens', 'prediction_targets', 'initial_core_sha256', 'initial_head_sha256'):
            assert row[key] == old[key]
        assert row['actual_order'] == plan['expected_orders'][f"4096_{row['seed']}"]
    events = [json.loads(line) for line in (ROOT/'logs/checks.jsonl').read_text().splitlines()]
    def event(label):
        return next(e for e in events if e['label'] == label and e['exit_code'] == 0)
    freeze, train, assess = [event(label) for label in ('diversity-v10-freeze-before-larger-cache-fitting',
                                                     'diversity-v10-twelve-fixed-budget-larger-cache-models',
                                                     'diversity-v10-final-only-frozen-assessment')]
    finish = lambda e: datetime.fromisoformat(e['started_utc'])+timedelta(seconds=e['duration_seconds'])
    assert finish(freeze) < datetime.fromisoformat(train['started_utc']) and finish(train) < datetime.fromisoformat(assess['started_utc'])
    print('PASS: 42 exact copies; 54 full, 108 fit and 27 intervention nested cells; cache/exposure distinction; all paired objective decisions; twelve matched budgets/starts and frozen sampling; freeze-before-training-before-assessment chronology.')


if __name__ == '__main__':
    main()
