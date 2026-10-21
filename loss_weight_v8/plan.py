"""Pin imported standard controls and identical cached data before weighted fitting."""
from pathlib import Path
from datetime import datetime, timezone
import platform
import shutil
import numpy as np
import torch
from transfer_v4.common import dump, sha_file
from interaction_v7.plan import read, verify_plan as verify_parent, sources as parent_sources
from interaction_v7.run import verify_training

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'runs/interaction_v7_run1'


def sources():
    return parent_sources() | {p.relative_to(ROOT).as_posix(): sha_file(p) for p in sorted((ROOT/'loss_weight_v8').glob('*.py'))} | {
        'scripts/diagnose_interaction_v7_first_transition.py': sha_file(ROOT/'scripts/diagnose_interaction_v7_first_transition.py')}


def reference(settings):
    parent = verify_parent(PARENT)
    public = ROOT/'docs/results/interaction_v7_run1'
    if sha_file(PARENT/'plan.json') != settings['reference_plan_sha256'] or sha_file(PARENT/'plan.json') != sha_file(public/'plan.json'):
        raise ValueError('Different verified standard-control study')
    rows = verify_training(PARENT, parent)
    for name in ('results.json', 'verification.json', 'first_transition/results.json', 'first_transition/verification.json'):
        if sha_file(PARENT/name) != sha_file(public/name):
            raise ValueError('Imported reference evidence changed: '+name)
    if not read(PARENT/'verification.json')['passed'] or not read(PARENT/'first_transition/verification.json')['passed']:
        raise ValueError('Verified standard controls required')
    return parent, [r for r in rows if r['rule'] == 'spatial']


def validate(s):
    if s['schema_version'] != 8 or s['purpose'] not in ('development_loss_weighting', 'execution_check'):
        raise ValueError('Expected the declared loss-weighting study')
    if s['first_transition_weight'] != 6. or not s['require_all_seeds'] or s['selection'] != 'final_declared_update' or s['language_allowed']:
        raise ValueError('Keep the frozen treatment/endpoint/all-seed rule; no language')
    if s['minimum_first_patch_nll_improvement'] != .01 or s['minimum_intervention_nll_improvement'] != .01:
        raise ValueError('Keep both primary improvement margins')


def freeze(out, settings, fixture_config=None):
    validate(settings)
    parent, controls = reference(settings)
    c = parent['config']
    if fixture_config is not None:
        if settings['purpose'] != 'execution_check':
            raise ValueError('Tiny configurations are execution fixtures only')
        c = fixture_config
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    files, origins = {}, {}

    def copy(source, destination):
        dest = out/destination
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PARENT/source, dest)
        files[destination] = sha_file(dest)
        origins[destination] = source

    for split in ('train', 'development', 'assessment'):
        for arm in ('coherent', 'prefix'):
            suffix = 'prefix_' if arm == 'prefix' else ''
            copy(f'data/spatial_{suffix}{split}.npy', f'data/{suffix}{split}.npy')
    for name in ('spatial_native.npy', 'spatial_changed.npy', 'eligibility.json'):
        copy('first_transition/'+name, 'queries/'+name)
    for seed in c['seeds']:
        copy(f'initialization_{seed}.pt', f'initialization_{seed}.pt')
    protocol = ROOT/'docs/LOSS_WEIGHT_V8_PROTOCOL.md'
    (out/'FROZEN_PROTOCOL.md').write_bytes(protocol.read_bytes())
    selected = [r for r in controls if r['seed'] in c['seeds']]
    plan = {'schema_version': 8, 'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
            'training_config': c, 'source_sha256': sources(), 'protocol_sha256': sha_file(protocol),
            'reference_plan_sha256': sha_file(PARENT/'plan.json'), 'standard_controls': selected,
            'reference_results_sha256': sha_file(PARENT/'results.json'),
            'reference_first_transition_sha256': sha_file(PARENT/'first_transition/results.json'),
            'input_files_sha256': files, 'input_origins': origins,
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
            'scope': 'Imported verified standard controls versus new weighted CPU models; inspected development data only'}
    dump(out/'plan.json', plan)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'status.json', {'state': 'frozen_before_weighted_training', 'language_allowed': False})
    print('Frozen weighting plan:', sha_file(out/'plan.json'), flush=True)


def verify_plan(out):
    out = Path(out)
    p = read(out/'plan.json')
    if sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256']:
        raise ValueError('Plan checksum differs')
    if sources() != p['source_sha256'] or sha_file(ROOT/'docs/LOSS_WEIGHT_V8_PROTOCOL.md') != p['protocol_sha256'] or sha_file(out/'FROZEN_PROTOCOL.md') != p['protocol_sha256']:
        raise ValueError('Frozen source/protocol changed')
    parent, controls = reference(p['settings'])
    expected = [r for r in controls if r['seed'] in p['training_config']['seeds']]
    if expected != p['standard_controls'] or sha_file(PARENT/'results.json') != p['reference_results_sha256'] or sha_file(PARENT/'first_transition/results.json') != p['reference_first_transition_sha256']:
        raise ValueError('Bound standard evidence changed')
    for name, checksum in p['input_files_sha256'].items():
        if sha_file(out/name) != checksum or checksum != sha_file(PARENT/p['input_origins'][name]):
            raise ValueError('Frozen common input changed: '+name)
    if p['settings']['purpose'] == 'development_loss_weighting' and p['training_config'] != parent['config']:
        raise ValueError('Training setting changed')
    return p
