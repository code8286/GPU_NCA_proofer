"""Pre-training freeze and immutable-input verification for this new task."""
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import numpy as np
import torch
from transfer_v4.common import dump, sha_file, source_hashes
from transfer_v4.model import LM
from .data import generate

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources():
    return {p.relative_to(ROOT).as_posix(): sha_file(p)
            for folder in ('trainability_v6', 'readiness_v5') for p in sorted((ROOT/folder).glob('*.py'))}


def validate(c):
    if c['schema_version'] != 6 or c['purpose'] not in ('development_trainability', 'execution_check'):
        raise ValueError('Expected a declared trainability task')
    if c['language_allowed'] or c['selection'] != 'final_declared_update':
        raise ValueError('No language stage or held-out checkpoint selection')
    if not c['seeds'] or len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('Unique model seeds required')
    if min(c[k] for k in ('batch', 'accumulate', 'eval_batch', 'eval_every', 'frequency_fit_rules')) < 1 or c['ppt']['steps'] < 1:
        raise ValueError('Positive budgets required')
    if min(c['minimum_frequency_advantage_nats'], c['minimum_ordering_advantage_nats']) != .01 or max(c['minimum_frequency_advantage_nats'], c['minimum_ordering_advantage_nats']) != .01:
        raise ValueError('Preserve both 0.01-nat readiness margins')
    if c['purpose'] == 'development_trainability':
        if {k:c['model'][k] for k in ('layers', 'heads', 'width')} != {'layers': 6, 'heads': 4, 'width': 192}:
            raise ValueError('Preserve the original full backbone')
        if len(c['seeds']) != 3 or c['ppt'] != read(ROOT/'configs/transfer_v4.json')['ppt']:
            raise ValueError('Use the declared three paired seeds and original PPT optimizer/budget')
    generate(c)  # Split and framing validation; no model fitting or score inspection.


def freeze(out, c):
    validate(c)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    arrays, ids = generate(c)
    hashes = {}
    for name, value in arrays.items():
        path = out/'data'/f'{name}.npy'
        path.parent.mkdir(exist_ok=True)
        np.save(path, value, allow_pickle=False)
        hashes[f'data/{name}.npy'] = sha_file(path)
    dump(out/'data/split_identities.json', ids)
    hashes['data/split_identities.json'] = sha_file(out/'data/split_identities.json')
    for seed in c['seeds']:
        model = LM(c['model'], 18, seed, seed+50_000)
        path = out/f'initialization_{seed}.pt'
        torch.save({'model': model.state_dict(), 'core_sha256': model.core_hash(), 'head_sha256': model.head_hash()}, path)
        hashes[path.name] = sha_file(path)
    protocol = ROOT/'docs/TRAINABILITY_V6_PROTOCOL.md'
    (out/'FROZEN_PROTOCOL.md').write_bytes(protocol.read_bytes())
    plan = {'schema_version': 6, 'created_utc': datetime.now(timezone.utc).isoformat(), 'config': c,
            'source_sha256': sources(), 'v4_source_sha256': source_hashes(),
            'protocol_sha256': sha_file(protocol), 'input_files_sha256': hashes,
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
            'scope': 'Known self-local rule, held-out initial conditions, final checkpoints; no language stage'}
    dump(out/'plan.json', plan)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'status.json', {'state': 'frozen_before_training', 'language_allowed': False})
    print('Frozen training plan:', sha_file(out/'plan.json'), flush=True)
    return plan


def verify_plan(out):
    out = Path(out)
    plan = read(out/'plan.json')
    if sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256']:
        raise ValueError('Plan checksum differs')
    if plan['source_sha256'] != sources() or plan['v4_source_sha256'] != source_hashes():
        raise ValueError('Frozen source changed')
    protocol = ROOT/'docs/TRAINABILITY_V6_PROTOCOL.md'
    if sha_file(protocol) != plan['protocol_sha256'] or sha_file(out/'FROZEN_PROTOCOL.md') != plan['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    for name, expected in plan['input_files_sha256'].items():
        if sha_file(out/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    return plan
