"""Freeze a matched rule-only bridge while preserving the historical packages."""
from pathlib import Path
from datetime import datetime, timezone
import platform
import numpy as np
import torch
from transfer_v4.common import dump, sha_file, source_hashes
from transfer_v4.model import LM
from trainability_v6.plan import read, sources as old_sources
from .data import generate

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return old_sources() | {p.relative_to(ROOT).as_posix(): sha_file(p) for p in sorted((ROOT/'interaction_v7').glob('*.py'))}


def validate(c):
    if c['schema_version'] != 7 or c['purpose'] not in ('development_interaction', 'execution_check') or c['rule'] != 'paired_complement_and_neighbor_parity':
        raise ValueError('Expected the declared paired spatial bridge')
    if c['language_allowed'] or c['selection'] != 'final_declared_update':
        raise ValueError('No language or checkpoint selection')
    if any(c[k] != .01 for k in ('minimum_frequency_advantage_nats', 'minimum_ordering_advantage_nats', 'minimum_local_table_advantage_nats')) or c['oracle_probability'] != .99:
        raise ValueError('Keep frozen readiness/diagnostic margins and oracle smoothing')
    if c['purpose'] == 'development_interaction':
        old = read(ROOT/'docs/results/trainability_v6_run1/plan.json')['config']
        for key in old:
            if key not in ('schema_version', 'purpose', 'rule') and c[key] != old[key]:
                raise ValueError('Setting changed from v6: '+key)
    generate(c)


def freeze(out, c):
    validate(c)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    arrays, ids = generate(c)
    hashes = {}
    for name, array in arrays.items():
        path = out/'data'/f'{name}.npy'
        path.parent.mkdir(exist_ok=True)
        np.save(path, array, allow_pickle=False)
        hashes[f'data/{name}.npy'] = sha_file(path)
    dump(out/'data/split_identities.json', ids)
    hashes['data/split_identities.json'] = sha_file(out/'data/split_identities.json')
    for seed in c['seeds']:
        model = LM(c['model'], 18, seed, seed+50_000)
        path = out/f'initialization_{seed}.pt'
        torch.save({'model': model.state_dict(), 'core_sha256': model.core_hash(), 'head_sha256': model.head_hash()}, path)
        hashes[path.name] = sha_file(path)
    protocol = ROOT/'docs/INTERACTION_V7_PROTOCOL.md'
    (out/'FROZEN_PROTOCOL.md').write_bytes(protocol.read_bytes())
    plan = {'schema_version': 7, 'created_utc': datetime.now(timezone.utc).isoformat(), 'config': c,
            'source_sha256': sources(), 'v4_source_sha256': source_hashes(),
            'protocol_sha256': sha_file(protocol), 'input_files_sha256': hashes,
            'reference_v6_plan_sha256': sha_file(ROOT/'docs/results/trainability_v6_run1/plan.json'),
            'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
            'scope': 'Matched spatial/complement fixed-rule bridge, shared joint-orbit-disjoint split; no language stage'}
    dump(out/'plan.json', plan)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'status.json', {'state': 'frozen_before_training', 'language_allowed': False})
    print('Frozen spatial bridge:', sha_file(out/'plan.json'), flush=True)


def verify_plan(out):
    out = Path(out)
    plan = read(out/'plan.json')
    if sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256']:
        raise ValueError('Plan checksum differs')
    if sources() != plan['source_sha256'] or source_hashes() != plan['v4_source_sha256']:
        raise ValueError('Frozen source changed')
    if sha_file(ROOT/'docs/INTERACTION_V7_PROTOCOL.md') != plan['protocol_sha256'] or sha_file(out/'FROZEN_PROTOCOL.md') != plan['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    if sha_file(ROOT/'docs/results/trainability_v6_run1/plan.json') != plan['reference_v6_plan_sha256']:
        raise ValueError('Historical reference plan changed')
    for name, expected in plan['input_files_sha256'].items():
        if sha_file(out/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    return plan
