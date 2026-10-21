"""Six new weighted models versus six verified imported standard controls."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from transfer_v4.common import dump, sha_file, seed_all, effective_tokens
from transfer_v4.model import LM
from transfer_v4.training import optimizer_for, evaluate, append
from readiness_v5.evaluate import score, mean_nll
from readiness_v5.gate import decision
from scripts.diagnose_interaction_v7_first_transition import score as first_score
from .plan import PARENT, freeze, verify_plan, read
from .objective import update

ARMS = ('coherent', 'prefix')


def load_model(out, c, seed, condition='initial', arm=None):
    if condition == 'initial':
        path = Path(out)/f'initialization_{seed}.pt'
    elif condition == 'standard':
        path = PARENT/f'spatial_{arm}_{seed}/final.pt'
    elif condition == 'weighted':
        path = Path(out)/f'weighted_{arm}_{seed}/final.pt'
    else:
        raise ValueError('Unknown checkpoint condition')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = LM(c['model'], 18, seed, seed+50_000)
    model.load_state_dict(saved['model'], strict=True)
    return model, saved


def train(out):
    out = Path(out)
    plan = verify_plan(out)
    c, s = plan['training_config'], plan['settings']
    if (out/'training_complete.json').exists():
        raise FileExistsError('Preserve completed training; use a fresh run')
    summaries = []
    for seed in c['seeds']:
        for arm in ARMS:
            folder = out/f'weighted_{arm}_{seed}'
            folder.mkdir(exist_ok=False)
            started = time.perf_counter()
            seed_all(seed)
            model, initial = load_model(out, c, seed)
            suffix = 'prefix_' if arm == 'prefix' else ''
            training = np.load(out/f'data/{suffix}train.npy', allow_pickle=False)
            development = np.load(out/f'data/{suffix}development.npy', allow_pickle=False)
            optimizer = optimizer_for(model, c['ppt'], True)
            if optimizer.state:
                raise ValueError('Expected empty Adam state')
            rng, order = np.random.default_rng(seed+101), hashlib.sha256()
            append(folder/'metrics.jsonl', {'step': 0, 'unweighted_development': evaluate(model, development, c, 'cpu', True)})
            predicted, window, losses = 0, 0, 0.
            for k in range(c['ppt']['steps']):
                loss, n, lr = update(model, optimizer, training, c, c['ppt'], k, rng, order, s['first_transition_weight'])
                predicted += n
                losses += loss
                window += 1
                if (k+1) % c['eval_every'] == 0 or k+1 == c['ppt']['steps']:
                    measured = evaluate(model, development, c, 'cpu', True)
                    append(folder/'metrics.jsonl', {'step': k+1, 'weighted_train_loss': losses/window, 'lr': lr,
                                                    'unweighted_development': measured, 'stage_seconds': time.perf_counter()-started})
                    dump(out/'status.json', {'state': 'training', 'arm': arm, 'seed': seed, 'step': k+1,
                                             'finished_models': len(summaries), 'language_allowed': False})
                    print(f"weighted {arm} seed={seed} {k+1}/{c['ppt']['steps']}: unweighted dev patch NLL={measured['patch_nll']:.6f}", flush=True)
                    losses, window = 0., 0
            saved = {'schema_version': 8, 'config': c, 'settings': s, 'plan_sha256': sha_file(out/'plan.json'),
                     'arm': arm, 'seed': seed, 'steps': c['ppt']['steps'], 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'core_initial_sha256': initial['core_sha256'], 'head_initial_sha256': initial['head_sha256']}
            torch.save(saved, folder/'final.pt')
            row = {'arm': arm, 'seed': seed, 'steps': c['ppt']['steps'], 'input_tokens': c['ppt']['steps']*effective_tokens(c),
                   'prediction_targets': predicted, 'checkpoint_sha256': sha_file(folder/'final.pt'),
                   'initial_core_sha256': initial['core_sha256'], 'initial_head_sha256': initial['head_sha256'],
                   'final_core_sha256': model.core_hash(), 'final_head_sha256': model.head_hash(),
                   'train_order_sha256': order.hexdigest(), 'final_unweighted_development': measured,
                   'stage_seconds': time.perf_counter()-started}
            dump(folder/'summary.json', row)
            row['summary_sha256'] = sha_file(folder/'summary.json')
            summaries.append(row)
            if s['purpose'] == 'development_loss_weighting':
                ref = next(r for r in plan['standard_controls'] if r['seed'] == seed and r['treatment'] == arm)
                for key in ('steps', 'input_tokens', 'prediction_targets', 'initial_core_sha256', 'initial_head_sha256', 'train_order_sha256'):
                    if row[key] != ref[key]:
                        raise ValueError('Matched budget/start/order differs: '+key)
            del model, saved, optimizer
    dump(out/'training_complete.json', {'models': summaries, 'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'training_receipt.json', {'training_sha256': sha_file(out/'training_complete.json')})
    dump(out/'status.json', {'state': 'weighted_training_complete_assessment_pending', 'language_allowed': False})


def verify_training(out, plan):
    out = Path(out)
    rows = read(out/'training_complete.json')
    if sha_file(out/'training_complete.json') != read(out/'training_receipt.json')['training_sha256'] or rows['plan_sha256'] != sha_file(out/'plan.json'):
        raise ValueError('Weighted training receipt differs')
    expected = {(s, a) for s in plan['training_config']['seeds'] for a in ARMS}
    models = rows['models']
    if len(models) != len(expected) or {(r['seed'], r['arm']) for r in models} != expected:
        raise ValueError('Every weighted model must finish before assessment')
    for row in models:
        folder = out/f"weighted_{row['arm']}_{row['seed']}"
        if row['steps'] != plan['training_config']['ppt']['steps'] or sha_file(folder/'final.pt') != row['checkpoint_sha256'] or sha_file(folder/'summary.json') != row['summary_sha256']:
            raise ValueError('Weighted final state changed')
    return models


def compute_assessment(out):
    out = Path(out)
    plan = verify_plan(out)
    c, settings = plan['training_config'], plan['settings']
    trained = verify_training(out, plan)
    data = {a: np.load(out/f"data/{'prefix_' if a == 'prefix' else ''}assessment.npy", allow_pickle=False) for a in ARMS}
    native = np.load(out/'queries/spatial_native.npy', allow_pickle=False)
    mask = np.array(read(out/'queries/eligibility.json')['eligible'], dtype=bool)
    changed = np.load(out/'queries/spatial_changed.npy', allow_pickle=False)[mask]
    reference_full = read(PARENT/'results.json')
    reference_first = read(PARENT/'first_transition/results.json')
    baseline = next(r for r in reference_full['families'] if r['rule'] == 'spatial')
    cells, first_cells, comparisons = [], [], []
    for seed in c['seeds']:
        first_by_arm = {}
        full_by_arm = {}
        for condition, arm in [('initial', None)]+[(k, a) for k in ('standard', 'weighted') for a in ARMS]:
            model, saved = load_model(out, c, seed, condition, arm)
            if condition != 'initial':
                if condition == 'weighted':
                    row = next(r for r in trained if r['seed'] == seed and r['arm'] == arm)
                    if saved['plan_sha256'] != sha_file(out/'plan.json') or saved['config'] != c or saved['settings'] != settings or saved['steps'] != c['ppt']['steps']:
                        raise ValueError('Weighted checkpoint identity differs')
                    if model.core_hash() != row['final_core_sha256'] or model.head_hash() != row['final_head_sha256']:
                        raise ValueError('Weighted state differs')
                    if not saved['optimizer']['state'] or any(int(v['step']) != c['ppt']['steps'] for v in saved['optimizer']['state'].values()):
                        raise ValueError('Weighted Adam update count differs')
                else:
                    row = next(r for r in plan['standard_controls'] if r['seed'] == seed and r['treatment'] == arm)
                initial = torch.load(out/f'initialization_{seed}.pt', map_location='cpu', weights_only=True)
                if saved['core_initial_sha256'] != initial['core_sha256'] or saved['head_initial_sha256'] != initial['head_sha256']:
                    raise ValueError('Actual paired starting state differs')
            for data_arm in ARMS:
                values = score(model, data[data_arm], c)
                mean = mean_nll(values)
                cells.append({'seed': seed, 'condition': condition, 'arm': arm, 'data_arm': data_arm, 'patch_nll': mean, 'per_initial_condition': values})
                full_by_arm[condition, arm, data_arm] = mean
                if condition == 'standard':
                    old = next(r for r in reference_full['cells'] if r['rule'] == 'spatial' and r['seed'] == seed and r['checkpoint_arm'] == arm and r['data_arm'] == data_arm)
                    if mean != old['patch_nll']:
                        raise ValueError('Imported standard full score does not replay exactly')
            loss, correct, pred = first_score(model, native)
            other_loss, other_correct, other_pred = first_score(model, changed)
            first = {'seed': seed, 'condition': condition, 'arm': arm, 'native_nll': float(loss.mean()),
                     'native_accuracy': float(correct.mean()), 'changed_nll': float(other_loss.mean()),
                     'changed_accuracy': float(other_correct.mean()), 'paired_both_correct': float((correct[mask] & other_correct).mean()),
                     'argmax_changes_as_required': float(((pred[mask] ^ 1) == other_pred).mean()),
                     'native_per_query_nll': loss.tolist(), 'native_per_query_correct': correct.tolist(),
                     'changed_per_query_nll': other_loss.tolist(), 'changed_per_query_correct': other_correct.tolist()}
            first_cells.append(first)
            first_by_arm[condition, arm] = first
            if condition == 'standard':
                old = next(r for r in reference_first['cells'] if r['rule'] == 'spatial' and r['seed'] == seed and r['arm'] == arm)
                if any(first[k] != old[k] for k in ('native_nll', 'native_accuracy', 'changed_nll', 'changed_accuracy', 'paired_both_correct')):
                    raise ValueError('Imported standard first-transition score differs')
            del model, saved
        standard, weighted = first_by_arm['standard', 'coherent'], first_by_arm['weighted', 'coherent']
        native_gain = standard['native_nll']-weighted['native_nll']
        changed_gain = standard['changed_nll']-weighted['changed_nll']
        readiness = decision(full_by_arm['weighted', 'coherent', 'coherent'], full_by_arm['weighted', 'coherent', 'prefix'], baseline['frequency_nll'], c)
        control_gain = first_by_arm['standard', 'prefix']['native_nll']-first_by_arm['weighted', 'prefix']['native_nll']
        comparisons.append({'seed': seed, 'first_patch_nll_improvement': native_gain, 'intervention_nll_improvement': changed_gain,
                            'first_patch_improved': bool(native_gain >= settings['minimum_first_patch_nll_improvement']),
                            'intervention_improved': bool(changed_gain >= settings['minimum_intervention_nll_improvement']),
                            'weighted_readiness': readiness, 'shuffled_first_patch_improvement': control_gain,
                            'difference_in_objective_gains': native_gain-control_gain})
    success = all(r['first_patch_improved'] and r['intervention_improved'] and r['weighted_readiness']['passed'] for r in comparisons)
    return {'schema_version': 8, 'scope': plan['scope'], 'plan_sha256': sha_file(out/'plan.json'),
            'training_sha256': sha_file(out/'training_complete.json'), 'frequency_nll': baseline['frequency_nll'],
            'eligible_pairs': int(mask.sum()), 'original_queries': len(native), 'full_cells': cells,
            'first_transition_cells': first_cells, 'paired_comparisons': comparisons,
            'all_seed_primary_improvement': success, 'language_allowed': False}


def assess(out):
    out = Path(out)
    if (out/'results.json').exists():
        raise FileExistsError('Preserve first assessment; use verify')
    result = compute_assessment(out)
    dump(out/'results.json', result)
    dump(out/'results_receipt.json', {'results_sha256': sha_file(out/'results.json')})
    dump(out/'status.json', {'state': 'primary_improvement_passed' if result['all_seed_primary_improvement'] else 'primary_improvement_not_established', 'language_allowed': False})
    print({'all_seed_primary_improvement': result['all_seed_primary_improvement'], 'comparisons': result['paired_comparisons']}, flush=True)


def verify(out):
    out = Path(out)
    if sha_file(out/'results.json') != read(out/'results_receipt.json')['results_sha256'] or compute_assessment(out) != read(out/'results.json'):
        raise ValueError('Receipt or full/first-transition replay differs')
    result = read(out/'results.json')
    dump(out/'verification.json', {'passed': True, 'results_sha256': sha_file(out/'results.json'),
                                  'full_cases': len(result['full_cells']), 'first_transition_cases': len(result['first_transition_cells']), 'language_allowed': False})
    print('PASS: all imported/new full-trajectory and first-transition scores replay exactly.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'train', 'assess', 'verify', 'all'))
    p.add_argument('--config', type=Path, default=Path('configs/loss_weight_v8.json'))
    p.add_argument('--out', type=Path, default=Path('runs/loss_weight_v8_run1'))
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
