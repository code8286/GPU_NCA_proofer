"""Frozen CPU transformer trainability workflow; no language-training command."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from transfer_v4.common import dump, sha_file, state_sha, seed_all, effective_tokens
from transfer_v4.model import LM
from transfer_v4.training import update, optimizer_for, evaluate, append
from transfer_v4.controls import validate_prefix_control
from readiness_v5.evaluate import score, mean_nll
from readiness_v5.frequency import counts, fit, per_rule_nll
from readiness_v5.gate import decision
from readiness_v5.positive_control import previous_and_truth, table_scores
from .plan import freeze, verify_plan, read

ARMS = ('coherent', 'prefix')


def load_model(out, c, seed, arm=None):
    out = Path(out)
    path = out/f'initialization_{seed}.pt' if arm is None else out/f'{arm}_{seed}/final.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = LM(c['model'], 18, seed, seed+50_000)
    model.load_state_dict(saved['model'], strict=True)
    return model, saved


def train(out):
    out = Path(out)
    plan = verify_plan(out)
    c = plan['config']
    if (out/'training_complete.json').exists():
        raise FileExistsError('Training already complete; preserve it and use a fresh run')
    # No assessment file is loaded in this function.
    train_data = {arm: np.load(out/f"data/{'prefix_' if arm == 'prefix' else ''}train.npy", allow_pickle=False) for arm in ARMS}
    dev_data = {arm: np.load(out/f"data/{'prefix_' if arm == 'prefix' else ''}development.npy", allow_pickle=False) for arm in ARMS}
    summaries = []
    for seed in c['seeds']:
        for arm in ARMS:
            folder = out/f'{arm}_{seed}'
            folder.mkdir(exist_ok=False)
            started = time.perf_counter()
            seed_all(seed)
            model, initial = load_model(out, c, seed)
            model.train()
            optimizer = optimizer_for(model, c['ppt'], synthetic=True)
            if optimizer.state:
                raise ValueError('Expected fresh optimizer')
            rng, order = np.random.default_rng(seed+101), hashlib.sha256()
            before = evaluate(model, dev_data[arm], c, 'cpu', True)
            append(folder/'metrics.jsonl', {'step': 0, 'development': before})
            predicted, loss_sum, window = 0, 0., 0
            for k in range(c['ppt']['steps']):
                loss, count, lr = update(model, optimizer, train_data[arm], c, c['ppt'], k, rng, 'cpu', True, order)
                predicted += count
                loss_sum += loss
                window += 1
                if (k+1) % c['eval_every'] == 0 or k+1 == c['ppt']['steps']:
                    measured = evaluate(model, dev_data[arm], c, 'cpu', True)
                    elapsed = time.perf_counter()-started
                    append(folder/'metrics.jsonl', {'step': k+1, 'train_loss': loss_sum/window, 'lr': lr,
                                                    'development': measured, 'elapsed_seconds': elapsed})
                    dump(out/'status.json', {'state': 'training', 'arm': arm, 'seed': seed, 'step': k+1,
                                              'finished_models': len(summaries), 'language_allowed': False})
                    print(f"{arm} seed={seed} {k+1}/{c['ppt']['steps']}: development patch NLL={measured['patch_nll']:.6f}", flush=True)
                    loss_sum, window = 0., 0
            saved = {'schema_version': 6, 'plan_sha256': sha_file(out/'plan.json'), 'config': c,
                     'arm': arm, 'seed': seed, 'steps': c['ppt']['steps'], 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'core_initial_sha256': initial['core_sha256'],
                     'head_initial_sha256': initial['head_sha256']}
            torch.save(saved, folder/'final.pt')
            row = {'arm': arm, 'seed': seed, 'steps': c['ppt']['steps'],
                   'input_tokens': c['ppt']['steps']*effective_tokens(c), 'prediction_targets': predicted,
                   'checkpoint_sha256': sha_file(folder/'final.pt'), 'initial_core_sha256': initial['core_sha256'],
                   'initial_head_sha256': initial['head_sha256'], 'final_core_sha256': model.core_hash(),
                   'final_head_sha256': model.head_hash(), 'train_order_sha256': order.hexdigest(),
                   'final_development': measured, 'stage_seconds': time.perf_counter()-started,
                   'parameter_count': sum(p.numel() for p in model.parameters()),
                   'backbone_parameters': sum(p.numel() for p in model.core.parameters())}
            dump(folder/'summary.json', row)
            row['summary_sha256'] = sha_file(folder/'summary.json')
            summaries.append(row)
            del model, saved, optimizer
        pair = summaries[-2:]
        if any(pair[0][key] != pair[1][key] for key in ('train_order_sha256', 'initial_core_sha256', 'initial_head_sha256', 'input_tokens', 'prediction_targets')):
            raise ValueError('Paired initialization or training budget/order differs')
    dump(out/'training_complete.json', {'models': summaries, 'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'training_receipt.json', {'training_sha256': sha_file(out/'training_complete.json')})
    dump(out/'status.json', {'state': 'training_complete_assessment_pending', 'language_allowed': False})


def verify_training(out, plan):
    out = Path(out)
    training = read(out/'training_complete.json')
    if sha_file(out/'training_complete.json') != read(out/'training_receipt.json')['training_sha256'] or training['plan_sha256'] != sha_file(out/'plan.json'):
        raise ValueError('Training receipt or plan differs')
    c = plan['config']
    rows = training['models']
    if len(rows) != len(c['seeds'])*2 or {(r['seed'], r['arm']) for r in rows} != {(s, a) for s in c['seeds'] for a in ARMS}:
        raise ValueError('All fixed-budget models must finish before assessment')
    for row in rows:
        folder = out/f"{row['arm']}_{row['seed']}"
        if sha_file(folder/'summary.json') != row['summary_sha256'] or sha_file(folder/'final.pt') != row['checkpoint_sha256']:
            raise ValueError('Final checkpoint or summary changed')
        if row['steps'] != c['ppt']['steps']:
            raise ValueError('Incomplete training budget')
    return rows


def compute_assessment(out):
    out = Path(out)
    plan = verify_plan(out)
    c = plan['config']
    summaries = verify_training(out, plan)
    data = {arm: np.load(out/f"data/{'prefix_' if arm == 'prefix' else ''}assessment.npy", allow_pickle=False) for arm in ARMS}
    validate_prefix_control(data['coherent'], data['prefix'], c)
    train_data = np.load(out/'data/train.npy', allow_pickle=False)
    prior, alpha, trials = fit(train_data, c, c)
    ai, at = counts(data['coherent'], c)
    bi, bt = counts(data['prefix'], c)
    if not np.array_equal(ai, bi) or not np.array_equal(at, bt):
        raise ValueError('Frequency conditioning/targets differ')
    frequency_rows = per_rule_nll(ai, at, prior, alpha)
    frequency = float(np.average(frequency_rows, weights=at.sum(1)))
    table = np.ones((16, 16), dtype=np.float64)
    a, b = previous_and_truth(train_data, c)
    mask = b < 16
    np.add.at(table, (a[mask], b[mask]), 1)
    table /= table.sum(1, keepdims=True)
    if not np.array_equal(table.argmax(1), np.arange(16)^15):
        raise ValueError('Training-fitted table failed known-rule recovery')
    table_nll = float(np.mean(table_scores(table, data['coherent'], c)))
    table_shuffled = float(np.mean(table_scores(table, data['prefix'], c)))
    table_gate = decision(table_nll, table_shuffled, frequency, c)
    negative_gate = decision(frequency, frequency, frequency, c)
    cells, decisions = [], []
    for seed in c['seeds']:
        values = {}
        for checkpoint_arm in ('initial', *ARMS):
            model, saved = load_model(out, c, seed, None if checkpoint_arm == 'initial' else checkpoint_arm)
            if checkpoint_arm != 'initial':
                if saved['plan_sha256'] != sha_file(out/'plan.json') or saved['config'] != c or saved['arm'] != checkpoint_arm or saved['seed'] != seed or saved['steps'] != c['ppt']['steps']:
                    raise ValueError('Saved model identity differs')
                row = next(r for r in summaries if r['seed'] == seed and r['arm'] == checkpoint_arm)
                initial = torch.load(out/f'initialization_{seed}.pt', map_location='cpu', weights_only=True)
                if saved['core_initial_sha256'] != initial['core_sha256'] or saved['head_initial_sha256'] != initial['head_sha256']:
                    raise ValueError('Starting snapshot differs')
                if model.core_hash() != row['final_core_sha256'] or model.head_hash() != row['final_head_sha256']:
                    raise ValueError('Model state differs from training summary')
                if any(int(s['step']) != c['ppt']['steps'] for s in saved['optimizer']['state'].values()):
                    raise ValueError('Optimizer update count differs')
            for data_arm in ARMS:
                rows = score(model, data[data_arm], c)
                nll = mean_nll(rows)
                values[checkpoint_arm, data_arm] = nll
                cells.append({'seed': seed, 'checkpoint_arm': checkpoint_arm, 'data_arm': data_arm,
                              'patch_nll': nll, 'per_initial_condition': rows})
            del model, saved
        coherent = values['coherent', 'coherent']
        shuffled = values['coherent', 'prefix']
        control_gap = values['prefix', 'prefix']-values['prefix', 'coherent']
        decisions.append({'seed': seed, 'coherent_nll': coherent, 'shuffled_nll': shuffled,
                          'initial_coherent_nll': values['initial', 'coherent'],
                          'relative_nll_improvement': (values['initial', 'coherent']-coherent)/values['initial', 'coherent'],
                          'prefix_model_ordering_gap': control_gap,
                          'difference_in_ordering_gaps': (shuffled-coherent)-control_gap,
                          'gate': decision(coherent, shuffled, frequency, c)})
    passed = table_gate['passed'] and not negative_gate['passed'] and all(r['gate']['passed'] for r in decisions)
    return {'schema_version': 6, 'plan_sha256': sha_file(out/'plan.json'), 'training_sha256': sha_file(out/'training_complete.json'),
            'scope': 'CPU known-rule transformer trainability; held-out initial conditions, not held-out rules or LM transfer',
            'frequency_nll': frequency, 'frequency_per_initial_condition_nll': frequency_rows,
            'frequency_concentration': alpha, 'frequency_train_fit': trials,
            'table_control': {'coherent_nll': table_nll, 'shuffled_nll': table_shuffled, 'gate': table_gate},
            'frequency_negative_gate': negative_gate, 'cells': cells, 'decisions': decisions,
            'trainability_passed': passed, 'language_allowed': False}


def assess(out):
    out = Path(out)
    if (out/'results.json').exists():
        raise FileExistsError('Assessment already saved; use verify, never overwrite it')
    result = compute_assessment(out)
    dump(out/'results.json', result)
    dump(out/'results_receipt.json', {'results_sha256': sha_file(out/'results.json')})
    dump(out/'status.json', {'state': 'trainability_passed' if result['trainability_passed'] else 'trainability_failed', 'language_allowed': False})
    print({k:v for k,v in result.items() if k in ('trainability_passed', 'language_allowed', 'decisions', 'frequency_nll')}, flush=True)


def verify(out):
    out = Path(out)
    if sha_file(out/'results.json') != read(out/'results_receipt.json')['results_sha256']:
        raise ValueError('Result receipt differs')
    expected = read(out/'results.json')
    if compute_assessment(out) != expected:
        raise ValueError('Independent final-checkpoint score replay differs')
    dump(out/'verification.json', {'passed': True, 'results_sha256': sha_file(out/'results.json'),
                                  'replayed_cells': len(expected['cells']), 'language_allowed': False})
    print('PASS: all initial/final checkpoint scores and decisions replay exactly; language remains disabled.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'train', 'assess', 'verify', 'all'))
    p.add_argument('--config', type=Path, default=Path('configs/trainability_v6.json'))
    p.add_argument('--out', type=Path, default=Path('runs/trainability_v6_run1'))
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if a.command in ('freeze', 'all'):
        freeze(a.out, read(a.config))
    if a.command in ('train', 'all'):
        train(a.out)
    if a.command in ('assess', 'all'):
        assess(a.out)
    if a.command in ('verify', 'all'):
        verify(a.out)


if __name__ == '__main__':
    main()
