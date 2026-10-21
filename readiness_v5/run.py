"""CLI for frozen checkpoint-only cross-evaluation. No training command exists."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transfer_v4.common import dump, sha_file
from transfer_v4.controls import validate_prefix_control
from transfer_v4.model import LM
from .evaluate import score, mean_nll
from .frequency import counts, fit, per_rule_nll
from .gate import decision
from .plan import read, freeze, verify_plan
from .positive_control import positive_control


def require_readiness(review, out):
    """Guard for a future runner; no old v4 language gate is modified or invoked."""
    out = Path(out)
    plan = verify_plan(review, out)
    results = read(out/'results.json')
    if sha_file(out/'results.json') != read(out/'results_receipt.json')['results_sha256']:
        raise ValueError('Readiness results changed')
    control = positive_control(plan['settings'])
    if results['plan_sha256'] != sha_file(out/'plan.json') or results['positive_control'] != control or not control['passed']:
        raise ValueError('Readiness measurement control or plan mismatch')
    expected = {(arm, seed) for arm, _, _ in plan['settings']['families'] for seed in plan['settings']['seeds']}
    if {(r['arm'], r['seed']) for r in results['cohorts']} != expected or len(results['cohorts']) != len(expected):
        raise ValueError('Incomplete structured cohort decisions')
    failed = []
    for row in results['cohorts']:
        measured = decision(row['structured_coherent_nll'], row['structured_shuffled_nll'], row['frequency_nll'], plan['settings'])
        if measured != row['gate']:
            raise ValueError('Stored gate disagrees with scores')
        if not measured['passed']:
            failed.append((row['arm'], row['seed']))
    if failed:
        raise ValueError(f'Language training blocked: {len(failed)} structured cohort(s) fail readiness')
    return [{'arm': arm, 'seed': seed, 'checkpoint_sha256': plan['input_files_sha256'][f'runs/development/{arm}_{seed}/ppt_selected.pt']}
            for arm, seed in sorted(expected)]


def evaluate(review, out):
    out = Path(out)
    plan = verify_plan(review, out)
    if (out/'results.json').exists() or (out/'progress.json').exists():
        raise FileExistsError('Use a fresh frozen output for another evaluation')
    s, c = plan['settings'], plan['v4_config']
    measurement = positive_control(s)
    dump(out/'positive_control.json', measurement)
    if not measurement['passed']:
        dump(out/'status.json', {'state': 'measurement_control_failed', 'language_allowed': False})
        raise ValueError('Known-rule measurement control failed; main evaluation stopped')
    base = Path(review)/'artifacts'
    cells, cohorts = [], []
    for seed in s['seeds']:
        for structured, control, kind in s['families']:
            folder = base/f'data/development/replicate_{seed}'
            train = np.load(folder/f'{kind}_train.npy', allow_pickle=False)
            native = np.load(folder/f'{kind}_validation.npy', allow_pickle=False)[:s['held_out_rules']]
            shuffled = np.load(folder/f'{control}_validation.npy', allow_pickle=False)[:s['held_out_rules']]
            validate_prefix_control(native, shuffled, c)
            prior, alpha, trials = fit(train, c, s)
            ai, at = counts(native, c)
            bi, bt = counts(shuffled, c)
            if not np.array_equal(ai, bi) or not np.array_equal(at, bt):
                raise ValueError('Frequency conditioning/targets differ across the control')
            freq_rows = per_rule_nll(ai, at, prior, alpha)
            freq = float(np.average(freq_rows, weights=at.sum(1)))
            values = {}
            for arm in (structured, control):
                name = f'runs/development/{arm}_{seed}/ppt_selected.pt'
                saved = torch.load(base/name, map_location='cpu', weights_only=True)
                if saved['config'] != c or saved['arm'] != arm or saved['seed'] != seed or saved['stage'] != 'ppt' or saved['data_manifest_sha256'] != s['reference_data_manifest_sha256']:
                    raise ValueError('Checkpoint identity differs')
                model = LM(c['model'], saved['vocab'], seed, seed)
                model.load_state_dict(saved['model'], strict=True)
                for data_arm, tokens in ((structured, native), (control, shuffled)):
                    rows = score(model, tokens, c)
                    mean = mean_nll(rows)
                    native_error = abs(mean-saved['validation']['patch_nll']) if arm == data_arm else None
                    if native_error is not None and native_error > 1e-4:
                        raise ValueError('Native cell does not reproduce the audited checkpoint')
                    if arm == data_arm:
                        accuracy = sum(r['patch_accuracy']*r['patch_targets'] for r in rows)/sum(r['patch_targets'] for r in rows)
                        if abs(accuracy-saved['validation']['patch_accuracy']) > 1e-4:
                            raise ValueError('Native accuracy does not reproduce the audited checkpoint')
                    values[arm, data_arm] = mean
                    cells.append({'seed': seed, 'checkpoint_arm': arm, 'data_arm': data_arm,
                                  'checkpoint_sha256': plan['input_files_sha256'][name],
                                  'patch_nll': mean, 'native_replay_absolute_error': native_error, 'per_rule': rows})
                    dump(out/'progress.json', {'completed_cells': len(cells), 'cells': cells})
                    print(f'CPU cell {len(cells)}/48: {arm} seed={seed} on {data_arm}: patch NLL={mean:.6f}', flush=True)
                del model, saved
            lss, lsp, lps, lpp = (values[structured, structured], values[structured, control], values[control, structured], values[control, control])
            cohorts.append({'seed': seed, 'arm': structured, 'control_arm': control,
                            'structured_coherent_nll': lss, 'structured_shuffled_nll': lsp,
                            'prefix_trained_coherent_nll': lps, 'prefix_trained_shuffled_nll': lpp,
                            'frequency_nll': freq, 'frequency_per_rule_nll': freq_rows,
                            'concentration': alpha, 'train_concentration_fit': trials,
                            'control_ordering_gap': lpp-lps, 'difference_in_ordering_gaps': (lsp-lss)-(lpp-lps),
                            'gate': decision(lss, lsp, freq, s)})
    allowed = measurement['passed'] and all(row['gate']['passed'] for row in cohorts)
    results = {'schema_version': 1, 'scope': 'New CPU cross-evaluation of inspected v4 development checkpoints; no neural training',
               'plan_sha256': sha_file(out/'plan.json'), 'positive_control': measurement,
               'cells': cells, 'cohorts': cohorts, 'language_allowed': allowed,
               'note': 'Admission is not proof of downstream transfer; do not revise old v4 results'}
    dump(out/'results.json', results)
    dump(out/'results_receipt.json', {'results_sha256': sha_file(out/'results.json')})
    dump(out/'status.json', {'state': 'readiness_passed' if allowed else 'blocked_at_readiness', 'language_allowed': allowed,
                            'completed_cells': len(cells), 'structured_cohorts': len(cohorts)})
    print(json.dumps({'status': read(out/'status.json'), 'cohort_gates': [{'arm': r['arm'], 'seed': r['seed'], **r['gate']} for r in cohorts]}, indent=2))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'positive-control', 'evaluate', 'authorize', 'all'))
    p.add_argument('--config', type=Path, default=Path('configs/readiness_v5.json'))
    p.add_argument('--review', type=Path, default=Path('Kaggle Outputs/nca_transfer_v4_run1_review'))
    p.add_argument('--out', type=Path, default=Path('runs/readiness_v5_run1'))
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if a.command in ('freeze', 'all'):
        freeze(a.review, a.out, read(a.config))
    if a.command == 'positive-control':
        print(json.dumps(positive_control(read(a.config)), indent=2))
    elif a.command in ('evaluate', 'all'):
        evaluate(a.review, a.out)
    elif a.command == 'authorize':
        print(json.dumps(require_readiness(a.review, a.out), indent=2))


if __name__ == '__main__':
    main()
