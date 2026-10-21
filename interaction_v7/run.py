"""Paired CPU spatial bridge; numerical updates and masks reuse frozen v4 code."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from transfer_v4.common import dump, sha_file, seed_all, effective_tokens
from transfer_v4.model import LM
from transfer_v4.training import update, optimizer_for, evaluate, append
from transfer_v4.controls import validate_prefix_control
from readiness_v5.evaluate import score, mean_nll
from readiness_v5.frequency import counts, fit, per_rule_nll
from readiness_v5.gate import decision
from readiness_v5.positive_control import previous_and_truth, table_scores
from .data import RULES, oracle_nll
from .plan import freeze, verify_plan, read

ARMS = ('coherent', 'prefix')


def load_model(out, c, seed, arm=None):
    path = Path(out)/(f'initialization_{seed}.pt' if arm is None else f'{arm}_{seed}/final.pt')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = LM(c['model'], 18, seed, seed+50_000)
    model.load_state_dict(saved['model'], strict=True)
    return model, saved


def train(out):
    out = Path(out)
    plan = verify_plan(out)
    c, summaries = plan['config'], []
    if (out/'training_complete.json').exists():
        raise FileExistsError('Use a fresh run; preserve completed training')
    # Assessment likelihoods are not accessed during any training arm.
    for seed in c['seeds']:
        for rule in RULES:
            for treatment in ARMS:
                arm = f'{rule}_{treatment}'
                folder = out/f'{arm}_{seed}'
                folder.mkdir(exist_ok=False)
                started = time.perf_counter()
                seed_all(seed)
                model, initial = load_model(out, c, seed)
                suffix = 'prefix_' if treatment == 'prefix' else ''
                training = np.load(out/f'data/{rule}_{suffix}train.npy', allow_pickle=False)
                development = np.load(out/f'data/{rule}_{suffix}development.npy', allow_pickle=False)
                optimizer = optimizer_for(model, c['ppt'], synthetic=True)
                if optimizer.state:
                    raise ValueError('Optimizer must start empty')
                rng, order = np.random.default_rng(seed+101), hashlib.sha256()
                append(folder/'metrics.jsonl', {'step': 0, 'development': evaluate(model, development, c, 'cpu', True)})
                predicted, loss_sum, window = 0, 0., 0
                for k in range(c['ppt']['steps']):
                    loss, n, lr = update(model, optimizer, training, c, c['ppt'], k, rng, 'cpu', True, order)
                    predicted += n
                    loss_sum += loss
                    window += 1
                    if (k+1) % c['eval_every'] == 0 or k+1 == c['ppt']['steps']:
                        measured = evaluate(model, development, c, 'cpu', True)
                        append(folder/'metrics.jsonl', {'step': k+1, 'train_loss': loss_sum/window, 'lr': lr,
                                                        'development': measured, 'elapsed_seconds': time.perf_counter()-started})
                        dump(out/'status.json', {'state': 'training', 'arm': arm, 'seed': seed, 'step': k+1,
                                                 'finished_models': len(summaries), 'language_allowed': False})
                        print(f"{arm} seed={seed} {k+1}/{c['ppt']['steps']}: development patch NLL={measured['patch_nll']:.6f}", flush=True)
                        loss_sum, window = 0., 0
                saved = {'schema_version': 7, 'config': c, 'plan_sha256': sha_file(out/'plan.json'), 'arm': arm,
                         'seed': seed, 'steps': c['ppt']['steps'], 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'core_initial_sha256': initial['core_sha256'], 'head_initial_sha256': initial['head_sha256']}
                torch.save(saved, folder/'final.pt')
                row = {'rule': rule, 'treatment': treatment, 'arm': arm, 'seed': seed, 'steps': c['ppt']['steps'],
                       'input_tokens': c['ppt']['steps']*effective_tokens(c), 'prediction_targets': predicted,
                       'checkpoint_sha256': sha_file(folder/'final.pt'), 'initial_core_sha256': initial['core_sha256'],
                       'initial_head_sha256': initial['head_sha256'], 'final_core_sha256': model.core_hash(),
                       'final_head_sha256': model.head_hash(), 'train_order_sha256': order.hexdigest(),
                       'final_development': measured, 'stage_seconds': time.perf_counter()-started}
                dump(folder/'summary.json', row)
                row['summary_sha256'] = sha_file(folder/'summary.json')
                summaries.append(row)
                del model, saved, optimizer
        paired = summaries[-4:]
        for key in ('train_order_sha256', 'initial_core_sha256', 'initial_head_sha256', 'input_tokens', 'prediction_targets'):
            if len({r[key] for r in paired}) != 1:
                raise ValueError('Paired setting differs: '+key)
    dump(out/'training_complete.json', {'models': summaries, 'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'training_receipt.json', {'training_sha256': sha_file(out/'training_complete.json')})
    dump(out/'status.json', {'state': 'training_complete_assessment_pending', 'language_allowed': False})


def verify_training(out, plan):
    out = Path(out)
    saved = read(out/'training_complete.json')
    if sha_file(out/'training_complete.json') != read(out/'training_receipt.json')['training_sha256'] or saved['plan_sha256'] != sha_file(out/'plan.json'):
        raise ValueError('Training receipt differs')
    rows, c = saved['models'], plan['config']
    expected = {(s, f'{r}_{a}') for s in c['seeds'] for r in RULES for a in ARMS}
    if len(rows) != len(expected) or {(r['seed'], r['arm']) for r in rows} != expected:
        raise ValueError('All fixed-budget arms must finish before assessment')
    for row in rows:
        folder = out/f"{row['arm']}_{row['seed']}"
        if row['steps'] != c['ppt']['steps'] or sha_file(folder/'final.pt') != row['checkpoint_sha256'] or sha_file(folder/'summary.json') != row['summary_sha256']:
            raise ValueError('Final model/summary/budget differs')
    return rows


def compute_assessment(out):
    out = Path(out)
    plan = verify_plan(out)
    c = plan['config']
    summaries = verify_training(out, plan)
    families, cells, decisions = [], [], []
    for rule in RULES:
        training = np.load(out/f'data/{rule}_train.npy', allow_pickle=False)
        data = {a: np.load(out/f"data/{rule}_{'prefix_' if a == 'prefix' else ''}assessment.npy", allow_pickle=False) for a in ARMS}
        validate_prefix_control(data['coherent'], data['prefix'], c)
        prior, alpha, trials = fit(training, c, c)
        ai, at = counts(data['coherent'], c)
        bi, bt = counts(data['prefix'], c)
        if not np.array_equal(ai, bi) or not np.array_equal(at, bt):
            raise ValueError('Frequency control conditioning differs')
        frequency_rows = per_rule_nll(ai, at, prior, alpha)
        frequency = float(np.mean(frequency_rows))
        table = np.ones((16, 16), dtype=np.float64)
        a, b = previous_and_truth(training, c)
        mask = b < 16
        np.add.at(table, (a[mask], b[mask]), 1)
        table /= table.sum(1, keepdims=True)
        local_table = float(np.mean(table_scores(table, data['coherent'], c)))
        oracle = {a: float(np.mean(oracle_nll(data[a], rule, c['oracle_probability']))) for a in ARMS}
        oracle_gate = decision(oracle['coherent'], oracle['prefix'], frequency, c)
        negative = decision(frequency, frequency, frequency, c)
        families.append({'rule': rule, 'frequency_nll': frequency, 'frequency_per_initial_condition_nll': frequency_rows,
                         'frequency_concentration': alpha, 'frequency_train_fit': trials,
                         'local_table_nll': local_table, 'oracle_nll': oracle, 'oracle_gate': oracle_gate,
                         'frequency_negative_gate': negative})
        for seed in c['seeds']:
            values = {}
            for treatment in ('initial', *ARMS):
                arm = None if treatment == 'initial' else f'{rule}_{treatment}'
                model, saved = load_model(out, c, seed, arm)
                if arm is not None:
                    row = next(r for r in summaries if r['arm'] == arm and r['seed'] == seed)
                    if saved['config'] != c or saved['plan_sha256'] != sha_file(out/'plan.json') or saved['arm'] != arm or saved['seed'] != seed or saved['steps'] != c['ppt']['steps']:
                        raise ValueError('Final checkpoint identity differs')
                    initial = torch.load(out/f'initialization_{seed}.pt', map_location='cpu', weights_only=True)
                    if saved['core_initial_sha256'] != initial['core_sha256'] or saved['head_initial_sha256'] != initial['head_sha256']:
                        raise ValueError('Actual starting state differs')
                    if model.core_hash() != row['final_core_sha256'] or model.head_hash() != row['final_head_sha256']:
                        raise ValueError('Final model state differs')
                    if not saved['optimizer']['state'] or any(int(v['step']) != c['ppt']['steps'] for v in saved['optimizer']['state'].values()):
                        raise ValueError('Optimizer counters differ')
                for data_arm in ARMS:
                    scores = score(model, data[data_arm], c)
                    nll = mean_nll(scores)
                    values[treatment, data_arm] = nll
                    cells.append({'rule': rule, 'seed': seed, 'checkpoint_arm': treatment, 'data_arm': data_arm,
                                  'patch_nll': nll, 'per_initial_condition': scores})
                del model, saved
            coherent, shuffled = values['coherent', 'coherent'], values['coherent', 'prefix']
            control_gap = values['prefix', 'prefix']-values['prefix', 'coherent']
            decisions.append({'rule': rule, 'seed': seed, 'coherent_nll': coherent, 'shuffled_nll': shuffled,
                              'initial_coherent_nll': values['initial', 'coherent'], 'gate': decision(coherent, shuffled, frequency, c),
                              'local_table_advantage_nats': local_table-coherent,
                              'local_table_advantage_passed': bool(local_table-coherent >= c['minimum_local_table_advantage_nats']),
                              'prefix_model_ordering_gap': control_gap, 'difference_in_ordering_gaps': shuffled-coherent-control_gap})
    readiness = all(f['oracle_gate']['passed'] and not f['frequency_negative_gate']['passed'] for f in families) and all(r['gate']['passed'] for r in decisions)
    coupling = all(r['local_table_advantage_passed'] for r in decisions if r['rule'] == 'spatial')
    return {'schema_version': 7, 'scope': 'Paired known-rule spatial bridge on CPU; no language-transfer experiment',
            'plan_sha256': sha_file(out/'plan.json'), 'training_sha256': sha_file(out/'training_complete.json'),
            'families': families, 'cells': cells, 'decisions': decisions,
            'readiness_passed': readiness, 'spatial_coupling_qualified': coupling,
            'bridge_passed': readiness and coupling, 'language_allowed': False}


def assess(out):
    out = Path(out)
    if (out/'results.json').exists():
        raise FileExistsError('Preserve the first assessment; use verify')
    result = compute_assessment(out)
    dump(out/'results.json', result)
    dump(out/'results_receipt.json', {'results_sha256': sha_file(out/'results.json')})
    dump(out/'status.json', {'state': 'bridge_passed' if result['bridge_passed'] else 'bridge_incomplete', 'language_allowed': False})
    print({k:result[k] for k in ('readiness_passed', 'spatial_coupling_qualified', 'bridge_passed', 'decisions')}, flush=True)


def verify(out):
    out = Path(out)
    if sha_file(out/'results.json') != read(out/'results_receipt.json')['results_sha256'] or compute_assessment(out) != read(out/'results.json'):
        raise ValueError('Assessment receipt or checkpoint replay differs')
    dump(out/'verification.json', {'passed': True, 'results_sha256': sha_file(out/'results.json'),
                                  'replayed_cells': len(read(out/'results.json')['cells']), 'language_allowed': False})
    print('PASS: all initial/final rule-bridge cells and decisions replay exactly; language disabled.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'train', 'assess', 'verify', 'all'))
    p.add_argument('--config', type=Path, default=Path('configs/interaction_v7.json'))
    p.add_argument('--out', type=Path, default=Path('runs/interaction_v7_run1'))
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
