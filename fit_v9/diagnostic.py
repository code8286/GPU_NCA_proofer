"""Freeze, assess and replay first-transition fit; no training entry point."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from transfer_v4.common import dump, sha_file
from loss_weight_v8.plan import ROOT, PARENT, read, verify_plan as verify_parent
from loss_weight_v8.run import load_model, verify_training
from scripts.diagnose_interaction_v7_first_transition import score as first_score

PARENT_V8 = ROOT/'runs/loss_weight_v8_run1'
PROTOCOL = ROOT/'docs/FIT_V9_PROTOCOL.md'
SPLITS = ('train', 'development', 'assessment')
ARMS = ('coherent', 'prefix')
MARGIN = .01


def classify(train_delta, assessment_delta):
    if train_delta >= MARGIN and assessment_delta >= MARGIN:
        return 'fit_regression_on_train_and_assessment'
    if train_delta <= -MARGIN and assessment_delta >= MARGIN:
        return 'training_gain_with_assessment_reversal'
    return 'mixed_or_below_margin'


def conditions(c):
    return [(s, condition, arm) for s in c['seeds'] for condition, arm in
            [('initial', None)]+[(k, a) for k in ('standard', 'weighted') for a in ARMS]]


def checkpoint(seed, condition, arm):
    if condition == 'initial':
        return PARENT_V8/f'initialization_{seed}.pt'
    if condition == 'standard':
        return PARENT/f'spatial_{arm}_{seed}/final.pt'
    if condition == 'weighted':
        return PARENT_V8/f'weighted_{arm}_{seed}/final.pt'
    raise ValueError('Unknown checkpoint condition')


def references():
    parent = verify_parent(PARENT_V8)
    verify_training(PARENT_V8, parent)
    hashes = {}
    for name in ('plan.json', 'plan_receipt.json', 'results.json', 'results_receipt.json', 'verification.json', 'training_complete.json', 'training_receipt.json'):
        path = PARENT_V8/name
        hashes[path.relative_to(ROOT).as_posix()] = sha_file(path)
    result, verification = read(PARENT_V8/'results.json'), read(PARENT_V8/'verification.json')
    if not verification['passed'] or verification['results_sha256'] != sha_file(PARENT_V8/'results.json') or read(PARENT_V8/'results_receipt.json')['results_sha256'] != verification['results_sha256']:
        raise ValueError('Verified parent results required')
    if result['all_seed_primary_improvement'] or result['language_allowed']:
        raise ValueError('Keep the inspected negative parent result and language restriction')
    return parent, hashes


def source_hashes(parent):
    return parent['source_sha256'] | {p.relative_to(ROOT).as_posix(): sha_file(p) for p in sorted((ROOT/'fit_v9').glob('*.py'))}


def freeze(out):
    out = Path(out)
    parent, hashes = references()
    out.mkdir(parents=True, exist_ok=False)
    files, origins = {}, {}
    for split in SPLITS:
        for arm in ARMS:
            name = f"data/{'prefix_' if arm == 'prefix' else ''}{split}.npy"
            path = out/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT_V8/name, path)
            files[name], origins[name] = sha_file(path), (PARENT_V8/name).relative_to(ROOT).as_posix()
    checkpoints = []
    for seed, condition, arm in conditions(parent['training_config']):
        path = checkpoint(seed, condition, arm)
        checkpoints.append({'seed': seed, 'condition': condition, 'arm': arm, 'path': path.relative_to(ROOT).as_posix(), 'sha256': sha_file(path)})
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    plan = {'schema_version': 9, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Post-hoc inspected-development inference only; no new training or confirmation',
            'training_config': parent['training_config'], 'source_sha256': source_hashes(parent),
            'protocol_sha256': sha_file(PROTOCOL), 'reference_files_sha256': hashes,
            'input_files_sha256': files, 'input_origins': origins, 'checkpoints': checkpoints,
            'descriptive_margin_nats': MARGIN, 'language_allowed': False,
            'environment': parent['environment'] | {'torch': str(torch.__version__), 'numpy': np.__version__}}
    dump(out/'plan.json', plan)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'status.json', {'state': 'frozen_before_new_split_scoring', 'language_allowed': False})
    print('Frozen fit diagnostic:', sha_file(out/'plan.json'))


def verify_plan(out):
    out = Path(out)
    plan = read(out/'plan.json')
    parent, hashes = references()
    if sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256'] or source_hashes(parent) != plan['source_sha256']:
        raise ValueError('Frozen plan/source changed')
    if sha_file(PROTOCOL) != plan['protocol_sha256'] or sha_file(out/'FROZEN_PROTOCOL.md') != plan['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    if hashes != plan['reference_files_sha256'] or plan['training_config'] != parent['training_config'] or plan['descriptive_margin_nats'] != MARGIN or plan['language_allowed']:
        raise ValueError('Reference/configuration/decision changed')
    expected = []
    for seed, condition, arm in conditions(parent['training_config']):
        path = checkpoint(seed, condition, arm)
        expected.append({'seed': seed, 'condition': condition, 'arm': arm, 'path': path.relative_to(ROOT).as_posix(), 'sha256': sha_file(path)})
    if plan['checkpoints'] != expected:
        raise ValueError('Checkpoint identity changed')
    for name, checksum in plan['input_files_sha256'].items():
        if sha_file(out/name) != checksum or sha_file(ROOT/plan['input_origins'][name]) != checksum:
            raise ValueError('Copied input/origin changed')
    return plan


@torch.no_grad()
def transition_score(model, array):
    if array.ndim != 2 or array.shape[1] < 12 or len(array) == 0:
        raise ValueError('Nonempty complete first-transition sequences required')
    model.eval()
    patch_losses, patch_counts, marker_losses, first_losses, first_correct = [], [], [], [], []
    for start in range(0, len(array), 16):
        z = torch.tensor(array[start:start+16, :12].astype(np.int64))
        if not torch.all((z[:, 7:11] >= 0) & (z[:, 7:11] < 16)) or not torch.all(z[:, [6, 11]] >= 16):
            raise ValueError('Fixed binary patch/marker layout differs')
        logits = model(z[:, :11])[:, 5:11]
        labels = z[:, 6:12]
        losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
        correct = logits.argmax(-1) == labels
        patch_losses.extend(losses[:, 1:5].double().mean(1).tolist())
        patch_counts.extend(correct[:, 1:5].sum(1).tolist())
        marker_losses.extend(losses[:, [0, 5]].double().mean(1).tolist())
        first_losses.extend(losses[:, 1].double().tolist())
        first_correct.extend(correct[:, 1].tolist())
    return {'transition_patch_nll': patch_losses, 'transition_patch_correct_count': patch_counts,
            'transition_marker_nll': marker_losses, 'long_prefix_first_nll': first_losses,
            'long_prefix_first_correct': first_correct}


def score_cell(model, array):
    first_loss, first_correct, _ = first_score(model, array[:, :8])
    transition = transition_score(model, array)
    error = float(np.max(np.abs(first_loss-np.array(transition.pop('long_prefix_first_nll')))))
    if error > 1e-5 or not np.array_equal(first_correct, transition.pop('long_prefix_first_correct')):
        raise ValueError('Short/long causal first-patch scores disagree')
    patch_nll = float(np.mean(transition['transition_patch_nll']))
    marker_nll = float(np.mean(transition['transition_marker_nll']))
    return {'queries': len(array), 'first_patch_nll': float(first_loss.mean()), 'first_patch_accuracy': float(first_correct.mean()),
            'transition_patch_nll': patch_nll, 'transition_patch_accuracy': float(np.mean(transition['transition_patch_correct_count'])/4),
            'transition_marker_nll': marker_nll, 'transition_all_token_nll': (4*patch_nll+2*marker_nll)/6,
            'short_long_max_nll_difference': error,
            'per_query': {'first_patch_nll': first_loss.tolist(), 'first_patch_correct': first_correct.tolist(), **transition}}


def comparisons(cells, c):
    index = {(r['seed'], r['condition'], r['arm'], r['data_arm'], r['split']): r for r in cells}
    pairs, gaps = [], []
    for seed, condition, arm in conditions(c):
        for data_arm in ARMS:
            rows = {s: index[seed, condition, arm, data_arm, s] for s in SPLITS}
            gaps.append({'seed': seed, 'condition': condition, 'arm': arm, 'data_arm': data_arm,
                         'development_minus_train_nll': rows['development']['first_patch_nll']-rows['train']['first_patch_nll'],
                         'assessment_minus_train_nll': rows['assessment']['first_patch_nll']-rows['train']['first_patch_nll']})
    for seed in c['seeds']:
        for arm in ARMS:
            for data_arm in ARMS:
                delta = {s: index[seed, 'weighted', arm, data_arm, s]['first_patch_nll']-index[seed, 'standard', arm, data_arm, s]['first_patch_nll'] for s in SPLITS}
                pairs.append({'seed': seed, 'arm': arm, 'data_arm': data_arm,
                              'weighted_minus_standard_nll': delta, 'category': classify(delta['train'], delta['assessment'])})
    return pairs, gaps


def compute(out):
    out = Path(out)
    plan = verify_plan(out)
    c = plan['training_config']
    arrays = {(a, s): np.load(out/f"data/{'prefix_' if a == 'prefix' else ''}{s}.npy", allow_pickle=False) for a in ARMS for s in SPLITS}
    old = read(PARENT_V8/'results.json')
    cells = []
    for seed, condition, arm in conditions(c):
        model, _ = load_model(PARENT_V8, c, seed, condition, arm)
        for data_arm in ARMS:
            for split in SPLITS:
                values = score_cell(model, arrays[data_arm, split])
                if values['queries'] != c['splits'][split]:
                    raise ValueError('Split count differs')
                row = {'seed': seed, 'condition': condition, 'arm': arm, 'data_arm': data_arm, 'split': split, **values}
                if data_arm == 'coherent' and split == 'assessment':
                    original = next(r for r in old['first_transition_cells'] if (r['seed'], r['condition'], r['arm']) == (seed, condition, arm))
                    if row['first_patch_nll'] != original['native_nll'] or row['first_patch_accuracy'] != original['native_accuracy'] or row['per_query']['first_patch_nll'] != original['native_per_query_nll'] or row['per_query']['first_patch_correct'] != original['native_per_query_correct']:
                        raise ValueError('Original assessment score/vector does not replay exactly')
                cells.append(row)
        del model
        print(f"Scored existing {condition}/{arm} seed={seed}: all splits and both data versions", flush=True)
    pairs, gaps = comparisons(cells, c)
    return {'schema_version': 9, 'scope': plan['scope'], 'plan_sha256': sha_file(out/'plan.json'),
            'cells': cells, 'paired_comparisons': pairs, 'generalization_gaps': gaps,
            'models_scored': len(conditions(c)), 'new_training_updates': 0, 'language_allowed': False}


def assess(out):
    out = Path(out)
    if (out/'results.json').exists():
        raise FileExistsError('Preserve first assessment; use verify')
    result = compute(out)
    dump(out/'results.json', result)
    dump(out/'results_receipt.json', {'results_sha256': sha_file(out/'results.json')})
    dump(out/'status.json', {'state': 'split_fit_diagnostic_complete', 'language_allowed': False})
    print('Coherent diagnostic:', [r for r in result['paired_comparisons'] if r['arm'] == r['data_arm'] == 'coherent'])


def verify(out):
    out = Path(out)
    if sha_file(out/'results.json') != read(out/'results_receipt.json')['results_sha256'] or compute(out) != read(out/'results.json'):
        raise ValueError('Results identity or exact checkpoint replay differs')
    dump(out/'verification.json', {'passed': True, 'cells': 90, 'models': 15, 'new_training_updates': 0,
                                  'results_sha256': sha_file(out/'results.json'), 'language_allowed': False})
    print('PASS: all 90 split/model/data cells replay exactly; 15 original assessment vectors agree with stage 8.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'assess', 'verify', 'all'))
    p.add_argument('--out', type=Path, default=Path('runs/fit_v9_run1'))
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if a.command in ('freeze', 'all'):
        freeze(a.out)
    if a.command in ('assess', 'all'):
        assess(a.out)
    if a.command in ('verify', 'all'):
        verify(a.out)


if __name__ == '__main__':
    main()
