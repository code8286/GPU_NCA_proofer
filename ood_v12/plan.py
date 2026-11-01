"""Freeze OOD measurement inputs before any new neural prediction."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from transfer_v4.common import dump, sha_file
from .data import generate, write_jsonl, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/OOD_V12_PROTOCOL.md'
REFERENCE = ROOT/'sources/clrs-text-reference/reference.json'
LEGACY = ROOT/'Kaggle Outputs/nca_transfer_v4_run1_review/artifacts/runs/development'


def tokenizer():
    import tiktoken
    # Sandboxed and network-enabled processes can have different temp roots.
    # Keep one explicit private workspace cache, excluded from publication.
    os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT/'.uv-cache/tiktoken_ood_v12')
    return tiktoken.get_encoding('gpt2')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def validate(c):
    if c != read(ROOT/'configs/ood_v12.json') or c['schema_version'] != 12 or c['language_training_allowed'] or c['reserved_confirmation_generated']:
        raise ValueError('Use the declared OOD measurement config; no new fitting or confirmation data')
    if c['tasks'] != ['bfs_distances', 'dijkstra_distances'] or c['legacy_seed'] != 59:
        raise ValueError('Do not select tasks/cohorts after inspecting outcomes')


def sources():
    return {p.relative_to(ROOT).as_posix(): sha_file(p) for p in sorted((ROOT/'ood_v12').glob('*.py'))} | {
        'transfer_v4/model.py': sha_file(ROOT/'transfer_v4/model.py')}


def reference():
    r = read(REFERENCE)
    assert r['commit'] == 'd1c2ad7af8437c7536cf329d9cef8fdf93184d9d'
    for name, value in r['files'].items():
        assert sha_file(REFERENCE.parent/r['commit']/name) == value['sha256']
    return sha_file(REFERENCE)


def legacy_records(c):
    rows = []
    for arm in c['legacy_arms']:
        folder = LEGACY/f"{arm}_{c['legacy_seed']}"
        summary = read(folder/'language_summary.json')
        if sha_file(folder/'language_final.pt') != summary['checkpoint_sha256'] or summary['arm'] != arm or summary['seed'] != c['legacy_seed']:
            raise ValueError('Historical checkpoint/summary identity mismatch')
        rows.append({'arm': arm, 'seed': c['legacy_seed'], 'checkpoint': (folder/'language_final.pt').relative_to(ROOT).as_posix(),
                     'checkpoint_sha256': summary['checkpoint_sha256'], 'summary_sha256': sha_file(folder/'language_summary.json'),
                     'language_updates': summary['language_updates'], 'language_tokens': summary['language_tokens'],
                     'language_head_initial_sha256': summary['language_head_initial_sha256'],
                     'language_order_sha256': summary['language_train_order_sha256'],
                     'language_schedule_sha256': summary['language_schedule_sha256']})
    for key in ('seed', 'language_updates', 'language_tokens', 'language_head_initial_sha256', 'language_order_sha256', 'language_schedule_sha256'):
        if len({r[key] for r in rows}) != 1:
            raise ValueError('Historical paired language stages differ: '+key)
    return rows


def freeze(out, c):
    validate(c)
    ref, legacy = reference(), legacy_records(c)
    # Resolve tokenizer before fitting/scoring; only its vocabulary/codec are used.
    import tiktoken
    tok = tokenizer()
    arrays, info = generate(c)
    pilot = []
    for cell in c['legacy_cases']:
        candidates = [r for r in arrays['assessment'] if r['auxiliary']['cell'] == cell]
        base = candidates[0]['base_id']
        pilot += [r for r in candidates if r['base_id'] == base]
    assert len(pilot) == 32
    lengths = [(len(tok.encode(r['prompt'])), len(tok.encode(r['references'][0]))) for r in arrays['assessment']]
    if max(a for a, _ in lengths)+c['generation_tokens'] > c['context_limit'] or max(b for _, b in lengths) >= c['generation_tokens']:
        raise ValueError('Whole prompt and answer budget must fit; do not truncate OOD graphs')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split, rows in arrays.items():
        name = f'data/{split}.jsonl'
        write_jsonl(out/name, rows)
        hashes[name] = sha_file(out/name)
    write_jsonl(out/'data/legacy_pilot.jsonl', pilot)
    hashes['data/legacy_pilot.jsonl'] = sha_file(out/'data/legacy_pilot.jsonl')
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    plan = {'schema_version': 12, 'created_utc': datetime.now(timezone.utc).isoformat(), 'config': c, 'source_sha256': sources(),
            'protocol_sha256': sha_file(PROTOCOL), 'reference_sha256': ref, 'input_files_sha256': hashes,
            'data_inventory': info, 'legacy_checkpoints': legacy, 'tokenizer': {'name': 'gpt2', 'vocab': tok.n_vocab,
                'max_assessment_prompt_tokens': max(a for a, _ in lengths), 'max_reference_tokens': max(b for _, b in lengths)},
            'environment': {'numpy': np.__version__, 'torch': str(torch.__version__), 'tiktoken': tiktoken.__version__},
            'scope': 'New OOD measurement controls and single-cohort historical zero-shot development diagnostic; zero optimizer updates',
            'language_training_allowed': False, 'reserved_confirmation_generated': False}
    dump(out/'plan.json', plan)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    print('Frozen OOD study:', sha_file(out/'plan.json'), info, plan['tokenizer'])


def verify_plan(out):
    out = Path(out)
    p = read(out/'plan.json')
    validate(p['config'])
    if sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256'] or p['source_sha256'] != sources():
        raise ValueError('Frozen source or plan changed')
    if p['protocol_sha256'] != sha_file(PROTOCOL) or p['protocol_sha256'] != sha_file(out/'FROZEN_PROTOCOL.md') or p['reference_sha256'] != reference():
        raise ValueError('Frozen protocol/reference changed')
    if p['legacy_checkpoints'] != legacy_records(p['config']):
        raise ValueError('Historical evidence changed')
    for name, value in p['input_files_sha256'].items():
        if sha_file(out/name) != value:
            raise ValueError('Frozen input changed: '+name)
    return p
