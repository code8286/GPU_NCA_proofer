"""Verify nested score evidence, exposure masks, all-seed decisions and chronology."""
from datetime import datetime, timedelta
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import sha_file
from confirmation_v11.plan import read, verify_plan
from confirmation_v11.run import verify_training, subset_metrics, decisions, compact
from confirmation_v11.data import exposure_mask
from report_confirmation_v11 import raw_evidence, summarize


def main():
    out, public = ROOT/'runs/confirmation_v11_run1', ROOT/'docs/results/confirmation_v11_run1'
    plan = verify_plan(out)
    models = verify_training(out, plan)
    provenance = read(public/'PROVENANCE.json')['files']
    assert len(provenance) == 71
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
    assert int(eligible.sum()) == 128
    for key, row in intervention_index.items():
        fit = fit_index[*key, 'assessment']
        assert row['native_nll'] == fit['first_patch_nll'] and row['native_accuracy'] == fit['first_patch_accuracy']
        assert len(row['changed_per_query_nll']) == len(row['changed_per_query_correct']) == 128
        assert float(np.mean(row['changed_per_query_nll'])) == row['changed_nll']
        assert float(np.mean(row['changed_per_query_correct'])) == row['changed_accuracy']
        assert float((np.array(fit['per_query']['first_patch_correct'])[eligible] & np.array(row['changed_per_query_correct'])).mean()) == row['paired_both_correct']
    pairs, strata = decisions(raw['full_cells'], raw['intervention_cells'], result['baselines'], plan['training_config'])
    assert pairs == result['paired_comparisons'] and strata == result['objective_strata_passed']
    # Independent arithmetic/accounting of the frozen decision, rather than
    # merely replaying the decision helper that generated the result.
    frequency = next(r['frequency_nll'] for r in result['baselines'] if r['level'] == 4096)
    independent = {}
    for objective in ('standard', 'weighted'):
        passed = []
        for seed in (179, 193, 211):
            small = intervention_index[seed, 512, objective, 'coherent']
            large = intervention_index[seed, 4096, objective, 'coherent']
            coherent = full_index[seed, 4096, objective, 'coherent', 'coherent']['patch_nll']
            shuffled = full_index[seed, 4096, objective, 'coherent', 'prefix']['patch_nll']
            passed.append(small['native_nll']-large['native_nll'] >= .01 and
                          small['changed_nll']-large['changed_nll'] >= .01 and
                          frequency-coherent >= .01 and shuffled-coherent >= .01)
        independent[objective] = all(passed)
    assert independent == strata
    with (public/'paired_comparisons.csv').open(newline='', encoding='utf-8') as f:
        csv_pairs = list(csv.DictReader(f))
    assert len(csv_pairs) == len(pairs)
    for row, pair in zip(csv_pairs, pairs):
        assert row['objective'] == pair['objective'] and int(row['seed']) == pair['seed']
        assert row['passed'] == str(pair['passed'])
        assert row['readiness_passed'] == str(pair['readiness']['passed'])
    assert result['study_primary_passed'] == strata['weighted'] and result['both_objectives_passed'] == all(strata.values()) and not result['language_allowed']
    combined = {k: v for k, v in result.items() if k not in ('evidence_sha256', 'full_cells', 'fit_cells', 'intervention_cells')} | raw
    assert compact(combined, result['evidence_sha256']) == result
    recovery = read(public/'execution_recovery.json')
    assert sha_file(public/'execution_recovery.json') == read(public/'execution_recovery_receipt.json')['sha256']
    assert read(public/'execution_recovery_verification_6950.json')['passed']
    assert sha_file(ROOT/'scripts/recover_confirmation_v11_6950.py') == recovery['driver_sha256']
    for name, files in recovery['retained_models'].items():
        assert all(sha_file(out/name/p) == h for p,h in files.items())
    for name, entry in recovery['interrupted_models'].items():
        old = [json.loads(line) for line in (public/entry['archive']/'metrics.jsonl').read_text().splitlines()]
        current = [json.loads(line) for line in (public/name/'metrics.jsonl').read_text().splitlines()][:len(old)]
        clean = lambda rows: [{k:v for k,v in r.items() if k != 'stage_seconds'} for r in rows]
        assert clean(old) == clean(current) and old[-1]['step'] == 800
    summary, _ = summarize(plan, result, models, raw, recovery)
    assert summary == read(public/'summary.json')
    assert summary['actual_updates_including_aborted'] == 24800
    assert summary['actual_input_tokens_including_aborted'] == 4662400
    assert summary['actual_prediction_targets_including_aborted'] == 4166400
    assert len(models) == 24
    for row in models:
        assert row['steps'] == 1000 and row['input_tokens'] == 188000 and row['prediction_targets'] == 168000
        assert row['actual_order'] == plan['expected_orders'][f"{row['level']}_{row['seed']}"]
    for seed in plan['training_config']['seeds']:
        paired = [r for r in models if r['seed'] == seed]
        assert len(paired) == 8
        assert len({r['initial_core_sha256'] for r in paired}) == 1
        assert len({r['initial_head_sha256'] for r in paired}) == 1
        for level in (512, 4096):
            assert len({r['actual_order']['sha256'] for r in paired if r['level'] == level}) == 1
    events = [json.loads(line) for line in (ROOT/'logs/checks.jsonl').read_text().splitlines()]
    def event(command):
        return next(e for e in events if e['command'][:4] == ['python', '-m', 'confirmation_v11.run', command] and e['exit_code'] == 0)
    freeze, assess = [event(command) for command in ('freeze', 'assess')]
    failed = next(e for e in events if e['label'] == 'confirmation-v11-twenty-four-fresh-fixed-budget-models')
    recovery_freeze = next(e for e in events if e['label'] == 'confirmation-v11-freeze-status-io-recovery' and e['exit_code'] == 0)
    train = next(e for e in events if e['label'] == 'confirmation-v11-recover-three-unchanged-endpoints' and e['exit_code'] == 0)
    finish = lambda e: datetime.fromisoformat(e['started_utc'])+timedelta(seconds=e['duration_seconds'])
    assert finish(freeze) < datetime.fromisoformat(failed['started_utc'])
    assert failed['exit_code'] == 1 and finish(failed) < datetime.fromisoformat(recovery_freeze['started_utc'])
    assert finish(recovery_freeze) < datetime.fromisoformat(train['started_utc']) and finish(train) < datetime.fromisoformat(assess['started_utc'])
    for e in (freeze, failed, recovery_freeze, train, assess):
        assert e['confirmation_v11_sources_sha256'] == {k:v for k,v in plan['source_sha256'].items() if k.startswith('confirmation_v11/')}
    print('PASS: 71 exact copies; 54 full, 108 fit and 27 intervention nested cells; all paired objective decisions; 24 matched budgets/starts and frozen sampling; retained recovery identities/prefix replay, discarded costs and freeze-training-recovery-assessment chronology.')


if __name__ == '__main__':
    main()
