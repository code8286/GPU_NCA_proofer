"""Twenty-four fresh fixed-budget fits; final-only confirmation and full replay."""
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
from readiness_v5.frequency import counts, fit, per_rule_nll
from readiness_v5.gate import decision
from fit_v9.diagnostic import score_cell, first_score
from loss_weight_v8.objective import update
from .plan import ROOT, OBJECTIVES, ARMS, read, freeze, verify_plan
from .data import exposure_mask


class Trace:
    def __init__(self):
        self.digest, self.observed, self.draws = hashlib.sha256(), set(), 0

    def update(self, values):
        self.digest.update(values)
        indices = np.frombuffer(values, dtype='<i8')
        self.observed.update(indices.tolist())
        self.draws += len(indices)

    def summary(self):
        return {'sha256': self.digest.hexdigest(), 'draws': self.draws, 'unique_rows': len(self.observed),
                'observed_indices': sorted(self.observed)}


def conditions(c):
    return [(seed, level, objective, arm) for seed in c['seeds'] for level, objective, arm in
            [(0, None, None)]+[(n, o, a) for n in (512, 4096) for o in OBJECTIVES for a in ARMS]]


def model_path(out, plan, seed, level, objective, arm):
    if level == 0:
        return Path(out)/f'initialization_{seed}.pt'
    return Path(out)/f'{level}_{objective}_{arm}_{seed}/final.pt'


def load_model(out, plan, seed, level, objective=None, arm=None):
    saved = torch.load(model_path(out, plan, seed, level, objective, arm), map_location='cpu', weights_only=True)
    model = LM(plan['training_config']['model'], 18, seed, seed+50000)
    model.load_state_dict(saved['model'], strict=True)
    return model, saved


def train(out):
    out = Path(out)
    plan = verify_plan(out)
    c, rows = plan['training_config'], []
    if (out/'training_complete.json').exists():
        raise FileExistsError('Preserve completed training; use a fresh run')
    for seed in c['seeds']:
        for level in (512, 4096):
            for objective in OBJECTIVES:
                factor = 1. if objective == 'standard' else 6.
                for arm in ARMS:
                    folder = out/f'{level}_{objective}_{arm}_{seed}'
                    folder.mkdir(exist_ok=False)
                    started = time.perf_counter()
                    seed_all(seed)
                    model, initial = load_model(out, plan, seed, 0)
                    suffix = 'prefix_' if arm == 'prefix' else ''
                    data = np.load(out/f'data/{suffix}train.npy', allow_pickle=False)[:level]
                    development = np.load(out/f'data/{suffix}development.npy', allow_pickle=False)
                    optimizer = optimizer_for(model, c['ppt'], True)
                    if optimizer.state:
                        raise ValueError('Fresh Adam required')
                    rng, trace = np.random.default_rng(seed+101), Trace()
                    append(folder/'metrics.jsonl', {'step': 0, 'unweighted_development': evaluate(model, development, c, 'cpu', True)})
                    predicted, loss_sum, window = 0, 0., 0
                    for k in range(c['ppt']['steps']):
                        loss, n, lr = update(model, optimizer, data, c, c['ppt'], k, rng, trace, factor)
                        predicted += n
                        loss_sum += loss
                        window += 1
                        if (k+1) % c['eval_every'] == 0 or k+1 == c['ppt']['steps']:
                            measured = evaluate(model, development, c, 'cpu', True)
                            append(folder/'metrics.jsonl', {'step': k+1, 'objective_train_loss': loss_sum/window, 'factor': factor,
                                                            'lr': lr, 'unweighted_development': measured,
                                                            'stage_seconds': time.perf_counter()-started})
                            dump(out/'status.json', {'state': 'training', 'seed': seed, 'objective': objective,
                                                     'arm': arm, 'level': level, 'step': k+1, 'finished_models': len(rows), 'language_allowed': False})
                            print(f'cache={level} {objective}/{arm} seed={seed}: {k+1}/{c["ppt"]["steps"]}, unweighted dev patch NLL={measured["patch_nll"]:.6f}', flush=True)
                            loss_sum, window = 0., 0
                    actual = trace.summary()
                    if actual != plan['expected_orders'][f'{level}_{seed}']:
                        raise ValueError('Actual larger-cache row order/exposure differs')
                    saved = {'schema_version': 11, 'config': c, 'settings': plan['settings'],
                             'plan_sha256': sha_file(out/'plan.json'), 'objective': objective, 'arm': arm,
                             'seed': seed, 'level': level, 'steps': c['ppt']['steps'], 'factor': factor, 'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(), 'core_initial_sha256': initial['core_sha256'],
                             'head_initial_sha256': initial['head_sha256']}
                    torch.save(saved, folder/'final.pt')
                    row = {'seed': seed, 'objective': objective, 'arm': arm, 'level': level, 'factor': factor,
                           'steps': c['ppt']['steps'], 'input_tokens': c['ppt']['steps']*effective_tokens(c),
                           'prediction_targets': predicted, 'actual_order': actual,
                           'initial_core_sha256': initial['core_sha256'], 'initial_head_sha256': initial['head_sha256'],
                           'final_core_sha256': model.core_hash(), 'final_head_sha256': model.head_hash(),
                           'checkpoint_sha256': sha_file(folder/'final.pt'), 'stage_seconds': time.perf_counter()-started,
                           'final_unweighted_development': measured}
                    dump(folder/'summary.json', row)
                    row['summary_sha256'] = sha_file(folder/'summary.json')
                    rows.append(row)
                    del model, saved, optimizer
    dump(out/'training_complete.json', {'models': rows, 'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'training_receipt.json', {'training_sha256': sha_file(out/'training_complete.json')})
    dump(out/'status.json', {'state': 'all_declared_models_complete_assessment_pending', 'finished_models': len(rows), 'language_allowed': False})


def verify_training(out, plan):
    out = Path(out)
    record = read(out/'training_complete.json')
    if sha_file(out/'training_complete.json') != read(out/'training_receipt.json')['training_sha256'] or record['plan_sha256'] != sha_file(out/'plan.json'):
        raise ValueError('Training receipt differs')
    rows = record['models']
    expected = {(s, n, o, a) for s in plan['training_config']['seeds'] for n in (512, 4096) for o in OBJECTIVES for a in ARMS}
    if len(rows) != len(expected) or {(r['seed'], r['level'], r['objective'], r['arm']) for r in rows} != expected:
        raise ValueError('Every declared model must finish before assessment')
    for row in rows:
        folder = out/f"{row['level']}_{row['objective']}_{row['arm']}_{row['seed']}"
        if row['steps'] != plan['training_config']['ppt']['steps'] or sha_file(folder/'final.pt') != row['checkpoint_sha256'] or sha_file(folder/'summary.json') != row['summary_sha256']:
            raise ValueError('Final checkpoint/summary changed')
        if row['actual_order'] != plan['expected_orders'][f"{row['level']}_{row['seed']}"]:
            raise ValueError('Actual sampling differs')
        expected_row = read(folder/'summary.json')
        if expected_row != {k: v for k, v in row.items() if k != 'summary_sha256'}:
            raise ValueError('Training record/summary disagreement')
        steps = plan['training_config']['ppt']['steps']
        if row['input_tokens'] != steps*effective_tokens(plan['training_config']) or row['prediction_targets'] != steps*168 or row['factor'] != (1. if row['objective'] == 'standard' else 6.):
            raise ValueError('Declared objective or budget differs')
    return rows


def subset_metrics(loss, correct, mask):
    mask = np.asarray(mask, dtype=bool)
    return {'count': int(mask.sum()), 'nll': float(np.mean(np.asarray(loss)[mask])) if mask.any() else None,
            'accuracy': float(np.mean(np.asarray(correct)[mask])) if mask.any() else None}


def decisions(full, interventions, baselines, c):
    fi = {(r['seed'], r['level'], r['objective'], r['arm'], r['data_arm']): r['patch_nll'] for r in full}
    ii = {(r['seed'], r['level'], r['objective'], r['arm']): r for r in interventions}
    frequency = next(r['frequency_nll'] for r in baselines if r['level'] == 4096)
    pairs = []
    for objective in OBJECTIVES:
        for seed in c['seeds']:
            a, b = [ii[seed, level, objective, 'coherent'] for level in (512, 4096)]
            first, changed = a['native_nll']-b['native_nll'], a['changed_nll']-b['changed_nll']
            gate = decision(fi[seed, 4096, objective, 'coherent', 'coherent'], fi[seed, 4096, objective, 'coherent', 'prefix'], frequency, c)
            control = ii[seed, 512, objective, 'prefix']['native_nll']-ii[seed, 4096, objective, 'prefix']['native_nll']
            pairs.append({'objective': objective, 'seed': seed, 'first_patch_improvement_nats': first,
                          'intervention_improvement_nats': changed, 'readiness': gate,
                          'passed': bool(first >= .01 and changed >= .01 and gate['passed']),
                          'shuffled_first_patch_improvement_nats': control,
                          'coherent_minus_shuffled_diversity_gain': first-control})
    strata = {o: all(r['passed'] for r in pairs if r['objective'] == o) for o in OBJECTIVES}
    return pairs, strata


def compute(out):
    out = Path(out)
    plan = verify_plan(out)
    rows = verify_training(out, plan)  # Block all new likelihoods until completion.
    c = plan['training_config']
    arrays = {a: np.load(out/f"data/{'prefix_' if a == 'prefix' else ''}assessment.npy", allow_pickle=False) for a in ARMS}
    training = np.load(out/'data/train.npy', allow_pickle=False)
    probes = {'anchor': training[:512], 'extension': training[512:1024],
              'development': np.load(out/'data/development.npy', allow_pickle=False), 'assessment': arrays['coherent']}
    mask = np.array(read(out/'queries/eligibility.json')['eligible'], dtype=bool)
    changed = np.load(out/'queries/spatial_changed.npy', allow_pickle=False)[mask]
    baselines = []
    for level, train_data in ((512, probes['anchor']), (4096, training)):
        prior, alpha, trials = fit(train_data, c, c)
        ai, at = counts(arrays['coherent'], c)
        bi, bt = counts(arrays['prefix'], c)
        if not np.array_equal(ai, bi) or not np.array_equal(at, bt):
            raise ValueError('Frequency conditioning/counts differ across controls')
        nll = float(np.mean(per_rule_nll(ai, at, prior, alpha)))
        baselines.append({'level': level, 'prior': prior.tolist(), 'concentration': alpha, 'trials': trials, 'frequency_nll': nll})
    full, fits, interventions = [], [], []
    for seed, level, objective, arm in conditions(c):
        model, saved = load_model(out, plan, seed, level, objective, arm)
        if level:
            row = next(r for r in rows if (r['seed'], r['level'], r['objective'], r['arm']) == (seed, level, objective, arm))
            if saved['config'] != c or saved['settings'] != plan['settings'] or saved['plan_sha256'] != sha_file(out/'plan.json') or saved['steps'] != 1000 or saved['level'] != level or saved['objective'] != objective or saved['arm'] != arm or saved['seed'] != seed or saved['factor'] != row['factor']:
                raise ValueError('New checkpoint identity differs')
            if model.core_hash() != row['final_core_sha256'] or model.head_hash() != row['final_head_sha256']:
                raise ValueError('Final parameters differ')
            if not saved['optimizer']['state'] or any(int(v['step']) != 1000 for v in saved['optimizer']['state'].values()):
                raise ValueError('Final Adam update counts differ')
        if level:
            initial = torch.load(out/f'initialization_{seed}.pt', map_location='cpu', weights_only=True)
            if saved['core_initial_sha256'] != initial['core_sha256'] or saved['head_initial_sha256'] != initial['head_sha256']:
                raise ValueError('Common initial weights differ')
        key = {'seed': seed, 'level': level, 'objective': objective, 'arm': arm}
        for data_arm in ARMS:
            values = score(model, arrays[data_arm], c)
            row = key | {'data_arm': data_arm, 'patch_nll': mean_nll(values), 'per_initial_condition': values}
            full.append(row)
        by_probe = {}
        for probe, data in probes.items():
            values = score_cell(model, data)
            observed = exposure_mask(probe, level, objective, seed, plan['expected_orders'], len(data))
            values['query_exposed'] = observed.tolist()
            values['drawn_subset'] = subset_metrics(values['per_query']['first_patch_nll'], values['per_query']['first_patch_correct'], observed)
            values['undrawn_subset'] = subset_metrics(values['per_query']['first_patch_nll'], values['per_query']['first_patch_correct'], ~observed)
            fits.append(key | {'probe': probe, **values})
            by_probe[probe] = values
        native = by_probe['assessment']
        loss, correct, pred = first_score(model, changed)
        native_correct = np.array(native['per_query']['first_patch_correct'])[mask]
        row = key | {'native_nll': native['first_patch_nll'], 'native_accuracy': native['first_patch_accuracy'],
                     'changed_nll': float(loss.mean()), 'changed_accuracy': float(correct.mean()),
                     'paired_both_correct': float((native_correct & correct).mean()),
                     'changed_per_query_nll': loss.tolist(), 'changed_per_query_correct': correct.tolist()}
        interventions.append(row)
        del model, saved
        print(f'Assessed seed={seed} cache={level} objective={objective} arm={arm}', flush=True)
    pairs, strata = decisions(full, interventions, baselines, c)
    return {'schema_version': 11, 'plan_sha256': sha_file(out/'plan.json'), 'scope': plan['scope'],
            'baselines': baselines, 'full_cells': full, 'fit_cells': fits, 'intervention_cells': interventions,
            'paired_comparisons': pairs, 'objective_strata_passed': strata, 'study_primary_passed': strata['weighted'], 'both_objectives_passed': all(strata.values()),
            'language_allowed': False}


def compact(result, hashes):
    value = {k: v for k, v in result.items() if k not in ('full_cells', 'fit_cells', 'intervention_cells')}
    value['full_cells'] = [{k: v for k, v in r.items() if k != 'per_initial_condition'} for r in result['full_cells']]
    value['fit_cells'] = [{k: v for k, v in r.items() if k not in ('per_query', 'query_exposed')} for r in result['fit_cells']]
    value['intervention_cells'] = [{k: v for k, v in r.items() if not k.startswith('changed_per_query_')} for r in result['intervention_cells']]
    value['evidence_sha256'] = hashes
    return value


def assess(out):
    out = Path(out)
    if (out/'results.json').exists():
        raise FileExistsError('Preserve first assessment; use verify')
    result = compute(out)
    hashes = {}
    for seed in read(out/'plan.json')['training_config']['seeds']:
        for key in ('full_cells', 'fit_cells', 'intervention_cells'):
            name = f'evidence/{key}_{seed}.json'
            dump(out/name, [r for r in result[key] if r['seed'] == seed])
            hashes[name] = sha_file(out/name)
    dump(out/'results.json', compact(result, hashes))
    dump(out/'results_receipt.json', {'results_sha256': sha_file(out/'results.json')})
    dump(out/'status.json', {'state': 'weighted_confirmation_passed' if result['study_primary_passed'] else 'weighted_confirmation_failed', 'language_allowed': False})
    print('Objective strata:', result['objective_strata_passed'], 'study primary:', result['study_primary_passed'])
    print('Comparisons:', result['paired_comparisons'])


def verify(out):
    out = Path(out)
    result = read(out/'results.json')
    if sha_file(out/'results.json') != read(out/'results_receipt.json')['results_sha256']:
        raise ValueError('Results receipt changed')
    replay = compute(out)
    hashes = result['evidence_sha256']
    expected_names = {f'evidence/{k}_{s}.json' for s in read(out/'plan.json')['training_config']['seeds'] for k in ('full_cells', 'fit_cells', 'intervention_cells')}
    if set(hashes) != expected_names:
        raise ValueError('Raw evidence inventory differs')
    for name, checksum in hashes.items():
        key, seed = name.split('/')[-1].rsplit('_', 1)
        seed = int(seed[:-5])
        if sha_file(out/name) != checksum or read(out/name) != [r for r in replay[key] if r['seed'] == seed]:
            raise ValueError('Per-query checkpoint replay differs: '+name)
    if compact(replay, hashes) != result:
        raise ValueError('All score/decision replay differs')
    dump(out/'verification.json', {'passed': True, 'full_cells': 54, 'fit_cells': 108, 'intervention_cells': 27,
                                  'models': 27, 'results_sha256': sha_file(out/'results.json'), 'language_allowed': False})
    print('PASS: 54 full, 108 fit and 27 intervention cases replay exactly; all 24 new final checkpoints checked.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'train', 'assess', 'verify', 'all'))
    p.add_argument('--out', type=Path, default=Path('runs/confirmation_v11_run1'))
    p.add_argument('--config', type=Path, default=Path('configs/confirmation_v11.json'))
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
