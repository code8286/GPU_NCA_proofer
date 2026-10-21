"""Check compact readiness evidence against the retained frozen CPU run."""
from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import sha_file
from readiness_v5.gate import decision


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    public = ROOT/'docs/results/readiness_v5_run1'
    private = ROOT/'runs/readiness_v5_run1'
    provenance = read(public/'PROVENANCE.json')
    for name, entry in provenance['files'].items():
        assert sha_file(public/name) == entry['sha256'] == sha_file(private/name), name
    plan, result, summary = (read(public/name) for name in ('plan.json', 'results.json', 'summary.json'))
    assert sha_file(public/'plan.json') == read(public/'plan_receipt.json')['plan_sha256'] == result['plan_sha256']
    assert sha_file(public/'results.json') == read(public/'results_receipt.json')['results_sha256']
    assert sha_file(public/'FROZEN_PROTOCOL.md') == plan['protocol_sha256']
    assert not result['language_allowed'] and summary['language_blocked_by_guard']
    assert result['positive_control'] == read(public/'positive_control.json')
    assert result['positive_control']['passed'] and not result['positive_control']['frequency_negative_gate']['passed']
    cells = {(r['seed'], r['checkpoint_arm'], r['data_arm']): r for r in result['cells']}
    assert len(cells) == len(result['cells']) == 48 and len(result['cohorts']) == 12
    for row in cells.values():
        assert len(row['per_rule']) == 32
        assert all(r['patch_targets'] == 936 for r in row['per_rule'])
        average = sum(r['patch_nll']*r['patch_targets'] for r in row['per_rule'])/29952
        assert math.isclose(average, row['patch_nll'], abs_tol=1e-12)
        name = f"runs/development/{row['checkpoint_arm']}_{row['seed']}/ppt_selected.pt"
        assert row['checkpoint_sha256'] == plan['input_files_sha256'][name]
        if row['checkpoint_arm'] == row['data_arm']:
            assert row['native_replay_absolute_error'] < 1e-4
    old = {(r['seed'], r['arm']): r for r in read(ROOT/'docs/results/transfer_v4_run1/frequency_diagnostic.json')['results']}
    for row in result['cohorts']:
        seed, arm, control = row['seed'], row['arm'], row['control_arm']
        for field, checkpoint, dataset in (
            ('structured_coherent_nll', arm, arm), ('structured_shuffled_nll', arm, control),
            ('prefix_trained_coherent_nll', control, arm), ('prefix_trained_shuffled_nll', control, control)):
            assert row[field] == cells[seed, checkpoint, dataset]['patch_nll']
        assert math.isclose(sum(row['frequency_per_rule_nll'])/32, row['frequency_nll'], abs_tol=1e-12)
        assert math.isclose(row['frequency_nll'], old[seed, arm]['causal_exchangeable_patch_nll'], abs_tol=1e-10)
        assert row['gate'] == decision(row['structured_coherent_nll'], row['structured_shuffled_nll'], row['frequency_nll'], plan['settings'])
        assert not row['gate']['frequency_passed'] and not row['gate']['ordering_passed']
    print('PASS: 7 frozen copies/receipts, 48 per-rule cell aggregates, 24 checkpoint identities/native replays, 12 prior frequency matches and blocked decisions.')


if __name__ == '__main__':
    main()
