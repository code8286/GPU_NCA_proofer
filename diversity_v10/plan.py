"""Freeze nested data, imported endpoints and replacement-sampling coverage."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import platform
import numpy as np
import torch
from transfer_v4.common import dump, sha_file
from fit_v9.diagnostic import verify_plan as verify_fit, checkpoint, PARENT_V8
from loss_weight_v8.plan import ROOT, PARENT, read
from .data import generate, draw_order

PARENT_FIT = ROOT/'runs/fit_v9_run1'
PROTOCOL = ROOT/'docs/DIVERSITY_V10_PROTOCOL.md'
OBJECTIVES = ('standard', 'weighted')
ARMS = ('coherent', 'prefix')


def validate(settings):
    expected = {'schema_version': 10, 'purpose': 'development_initial_condition_diversity',
                'training_conditions': 4096, 'expansion_seed': 20261021,
                'minimum_first_patch_improvement_nats': .01, 'minimum_intervention_improvement_nats': .01,
                'selection': 'final_declared_update', 'language_allowed': False}
    if settings != expected:
        raise ValueError('Keep the declared single diversity treatment, thresholds, final endpoint and language restriction')


def sources(parent):
    return parent['source_sha256'] | {p.relative_to(ROOT).as_posix(): sha_file(p) for p in sorted((ROOT/'diversity_v10').glob('*.py'))}


def reference():
    parent = verify_fit(PARENT_FIT)
    hashes = {}
    for root in (PARENT_FIT, PARENT_V8):
        public = ROOT/'docs/results'/root.name
        for name in ('plan.json', 'results.json', 'verification.json'):
            if sha_file(root/name) != sha_file(public/name):
                raise ValueError('Historical reference differs from preserved public copy')
            hashes[(root/name).relative_to(ROOT).as_posix()] = sha_file(root/name)
        if not read(root/'verification.json')['passed']:
            raise ValueError('Verified parent required')
    return parent, hashes


def control_rows():
    old = read(PARENT/'training_complete.json')['models']
    weighted = read(PARENT_V8/'training_complete.json')['models']
    rows = []
    for s in (131, 149, 167):
        for objective in OBJECTIVES:
            for arm in ARMS:
                base = next(r for r in (old if objective == 'standard' else weighted)
                            if r['seed'] == s and (r['arm'] == f'spatial_{arm}' if objective == 'standard' else r['arm'] == arm))
                path = checkpoint(s, objective, arm)
                rows.append({'objective': objective, 'arm': arm, 'seed': s, 'level': 512,
                             'path': path.relative_to(ROOT).as_posix(), 'source_summary': base})
    return rows


def freeze(out, settings):
    validate(settings)
    parent, refs = reference()
    original = parent['training_config']
    c = copy.deepcopy(original)
    c['splits']['train'] = settings['training_conditions']
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    hashes, origins = {}, {}
    def copy_file(source, name):
        dest = out/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        hashes[name], origins[name] = sha_file(dest), source.relative_to(ROOT).as_posix()
    def save_array(name, array):
        path = out/name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
        hashes[name] = sha_file(path)
    for split in ('train', 'development', 'assessment'):
        for arm in ARMS:
            suffix = 'prefix_' if arm == 'prefix' else ''
            copy_file(PARENT_V8/f'data/{suffix}{split}.npy', f'data/original_{suffix}{split}.npy')
    copy_file(PARENT/'data/split_identities.json', 'data/original_identities.json')
    ids = read(out/'data/original_identities.json')
    tokens, shuffled, expanded = generate(ids, np.load(out/'data/original_train.npy', allow_pickle=False),
                                          np.load(out/'data/original_prefix_train.npy', allow_pickle=False), c, settings)
    save_array('data/train.npy', tokens)
    save_array('data/prefix_train.npy', shuffled)
    dump(out/'data/expanded_identities.json', expanded)
    hashes['data/expanded_identities.json'] = sha_file(out/'data/expanded_identities.json')
    for name in ('spatial_native.npy', 'spatial_changed.npy', 'eligibility.json'):
        copy_file(PARENT_V8/('queries/'+name), 'queries/'+name)
    for seed in c['seeds']:
        copy_file(PARENT_V8/f'initialization_{seed}.pt', f'initialization_{seed}.pt')
    orders = {f'{level}_{s}': draw_order(s, c, level) for s in c['seeds'] for level in (512, 4096)}
    controls = control_rows()
    for row in controls:
        expected = orders[f"512_{row['seed']}"]
        if expected['sha256'] != row['source_summary']['train_order_sha256']:
            raise ValueError('Historical index-order reconstruction differs')
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    plan = {'schema_version': 10, 'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
            'training_config': c, 'original_training_config': original, 'source_sha256': sources(parent),
            'protocol_sha256': sha_file(PROTOCOL), 'reference_files_sha256': refs,
            'input_files_sha256': hashes, 'input_origins': origins, 'standard_controls': controls,
            'expected_orders': orders, 'language_allowed': False,
            'scope': 'Inspected-development nested cache comparison; twelve new CPU models versus twelve imported controls',
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__}}
    dump(out/'plan.json', plan)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'status.json', {'state': 'frozen_before_diversity_training', 'language_allowed': False})
    print('Frozen diversity study:', sha_file(out/'plan.json'))
    print('Distinct row exposure:', {k: v['unique_rows'] for k, v in orders.items()})


def verify_plan(out):
    out = Path(out)
    plan = read(out/'plan.json')
    validate(plan['settings'])
    parent, refs = reference()
    if sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256'] or sources(parent) != plan['source_sha256']:
        raise ValueError('Frozen plan or source changed')
    if refs != plan['reference_files_sha256'] or control_rows() != plan['standard_controls']:
        raise ValueError('Historical control identity changed')
    if sha_file(PROTOCOL) != plan['protocol_sha256'] or sha_file(out/'FROZEN_PROTOCOL.md') != plan['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    original = parent['training_config']
    expected = copy.deepcopy(original)
    expected['splits']['train'] = 4096
    if original != plan['original_training_config'] or expected != plan['training_config'] or plan['language_allowed']:
        raise ValueError('Only training availability may change')
    for name, checksum in plan['input_files_sha256'].items():
        if sha_file(out/name) != checksum:
            raise ValueError('Frozen input changed: '+name)
        if name in plan['input_origins'] and sha_file(ROOT/plan['input_origins'][name]) != checksum:
            raise ValueError('Copied origin changed: '+name)
    for seed in expected['seeds']:
        for level in (512, 4096):
            if draw_order(seed, expected, level) != plan['expected_orders'][f'{level}_{seed}']:
                raise ValueError('Expected sampling changed')
    return plan
