"""Freeze fresh inputs, new paired starts, budgets and the confirmation target."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import platform
import numpy as np
import torch
from transfer_v4.common import dump, sha_file
from transfer_v4.model import LM
from trainability_v6.plan import verify_plan as verify_v6
from diversity_v10.plan import ROOT, OBJECTIVES, ARMS, read, verify_plan as verify_v10
from diversity_v10.run import verify_training as verify_v10_training
from interaction_v7.data import canonical
from .data import generate, draw_order

PROTOCOL = ROOT/'docs/CONFIRMATION_V11_PROTOCOL.md'
PARENT = ROOT/'runs/diversity_v10_run1'


def validate(s):
    expected = {'schema_version': 11, 'purpose': 'fresh_phase_weighted_diversity_confirmation',
                'training_conditions': [512, 4096], 'data_seed': 20261023, 'seeds': [179, 193, 211],
                'minimum_first_patch_improvement_nats': .01, 'minimum_intervention_improvement_nats': .01,
                'primary_objective': 'weighted', 'selection': 'final_declared_update', 'language_allowed': False}
    if s != expected:
        raise ValueError('Preserve the fresh-phase confirmation, both objectives/controls and frozen margins/budgets')


def reference():
    parent = verify_v10(PARENT)
    verify_v10_training(PARENT, parent)
    verify_v6(ROOT/'runs/trainability_v6_run1')
    refs = {}
    for folder in ('trainability_v6_run1', 'diversity_v10_run1'):
        for name in ('plan.json', 'results.json', 'verification.json'):
            source = ROOT/'runs'/folder/name
            if sha_file(source) != sha_file(ROOT/'docs/results'/folder/name):
                raise ValueError('Historical public/private reference changed')
            refs[source.relative_to(ROOT).as_posix()] = sha_file(source)
        if not read(ROOT/'runs'/folder/'verification.json')['passed']:
            raise ValueError('Verified history required')
    files = ('runs/trainability_v6_run1/data/split_identities.json',
             'runs/interaction_v7_run1/data/split_identities.json',
             'runs/diversity_v10_run1/data/expanded_identities.json')
    refs.update({name: sha_file(ROOT/name) for name in files})
    old6, old7, old10 = [read(ROOT/name) for name in files]
    # V6 records complement canonical representatives, which cover both phases;
    # joint canonicalization also covers their spatial partners conservatively.
    values = sum(old6.values(), []) + sum(old7['initial_ids'].values(), [])
    values += (np.array(old7['initial_ids']['assessment']) ^ 8).tolist()
    values += old10['initial_ids']
    forbidden = sorted(set(canonical(np.array(values)).tolist()))
    return parent, refs, forbidden


def sources(parent):
    return parent['source_sha256'] | {p.relative_to(ROOT).as_posix(): sha_file(p)
                                    for p in sorted((ROOT/'confirmation_v11').glob('*.py'))}


def configuration(parent, settings):
    c = copy.deepcopy(parent['training_config'])
    c['seeds'], c['data_seed'] = settings['seeds'], settings['data_seed']
    # All architecture, rule, serializer, weighting, optimizer and update/token
    # settings remain the same; both cache sizes are sliced from this one cache.
    return c


def freeze(out, settings):
    validate(settings)
    parent, refs, forbidden = reference()
    c = configuration(parent, settings)
    arrays, ids, native, changed = generate(forbidden, c)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    def save(name, value):
        path = out/name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value, allow_pickle=False)
        hashes[name] = sha_file(path)
    for name, array in arrays.items():
        save('data/'+name+'.npy', array)
    dump(out/'data/split_identities.json', ids)
    hashes['data/split_identities.json'] = sha_file(out/'data/split_identities.json')
    save('queries/spatial_native.npy', native)
    save('queries/spatial_changed.npy', changed)
    dump(out/'queries/eligibility.json', {'eligible': [True]*len(native), 'pairs': len(native), 'original_queries': len(native)})
    hashes['queries/eligibility.json'] = sha_file(out/'queries/eligibility.json')
    for seed in c['seeds']:
        model = LM(c['model'], 18, seed, seed+50000)
        path = out/f'initialization_{seed}.pt'
        torch.save({'model': model.state_dict(), 'core_sha256': model.core_hash(), 'head_sha256': model.head_hash()}, path)
        hashes[path.name] = sha_file(path)
    orders = {f'{n}_{s}': draw_order(s, c, n) for s in c['seeds'] for n in (512, 4096)}
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    plan = {'schema_version': 11, 'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
            'training_config': c, 'parent_training_config': parent['training_config'], 'source_sha256': sources(parent),
            'protocol_sha256': sha_file(PROTOCOL), 'reference_files_sha256': refs, 'input_files_sha256': hashes,
            'expected_orders': orders, 'historical_forbidden_phase_count': len(forbidden), 'language_allowed': False,
            'scope': 'Fresh-phase confirmation of weighted diversity gain; both objectives/controls, 24 new CPU models, one data split and three fresh initialization seeds',
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__}}
    dump(out/'plan.json', plan)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'status.json', {'state': 'frozen_before_confirmation_training', 'language_allowed': False})
    print('Frozen confirmation:', sha_file(out/'plan.json'), 'excluded historical phases:', len(forbidden), flush=True)
    print('Distinct row exposures:', {k: v['unique_rows'] for k, v in orders.items()}, flush=True)


def verify_plan(out):
    out = Path(out)
    plan = read(out/'plan.json')
    validate(plan['settings'])
    parent, refs, forbidden = reference()
    if sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256'] or sources(parent) != plan['source_sha256']:
        raise ValueError('Frozen plan/source changed')
    if refs != plan['reference_files_sha256'] or len(forbidden) != plan['historical_forbidden_phase_count']:
        raise ValueError('Historical reference/phase exclusion changed')
    if sha_file(PROTOCOL) != plan['protocol_sha256'] or sha_file(out/'FROZEN_PROTOCOL.md') != plan['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    if parent['training_config'] != plan['parent_training_config'] or configuration(parent, plan['settings']) != plan['training_config'] or plan['language_allowed']:
        raise ValueError('Only declared fresh data/seeds may differ from development')
    for name, checksum in plan['input_files_sha256'].items():
        if sha_file(out/name) != checksum:
            raise ValueError('Frozen input changed: '+name)
    for s in plan['training_config']['seeds']:
        for n in (512, 4096):
            if draw_order(s, plan['training_config'], n) != plan['expected_orders'][f'{n}_{s}']:
                raise ValueError('Frozen sampling changed')
    return plan
