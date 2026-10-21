"""Freeze trusted sources and exact private input identities before cross-scoring."""
from datetime import datetime, timezone
import json
from pathlib import Path
from transfer_v4.common import dump, sha_file, source_hashes as v4_sources

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return {p.name: sha_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def references(review, settings):
    review = Path(review)
    artifacts = review/'artifacts'
    run, data = artifacts/'runs/development', artifacts/'data/development'
    audit, env, manifest = read(review/'audit.json'), read(run/'environment.json'), read(data/'manifest.json')
    published = ROOT/'docs/results/transfer_v4_run1'
    if sha_file(review/'audit.json') != sha_file(published/'audit.json') or sha_file(run/'environment.json') != sha_file(published/'gpu/environment.json'):
        raise ValueError('Private reference records differ from the preserved reviewed evidence')
    if audit['cpu_checkpoints_verified'] != 78 or audit['cpu_initializations_verified'] != 6:
        raise ValueError('Completed independent v4 audit required')
    if audit['archive_sha256'] != settings['reference_archive_sha256'] or sha_file(data/'manifest.json') != settings['reference_data_manifest_sha256']:
        raise ValueError('Different reference archive or dataset')
    if env['source_sha256'] != v4_sources() or env['data_manifest_sha256'] != settings['reference_data_manifest_sha256']:
        raise ValueError('Trusted v4 sources or recorded input identity differ')
    if settings['seeds'] != env['config']['seeds'] or settings['held_out_rules'] != env['config']['ppt_eval_rules']:
        raise ValueError('Diagnostic cohort/rule selection changed')
    return artifacts, run, data, env, manifest


def freeze(review, out, settings):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if settings['schema_version'] != 1 or not settings['require_all_structured_cohorts']:
        raise ValueError('Use the declared all-cohort readiness study')
    if min(settings['minimum_frequency_advantage_nats'], settings['minimum_ordering_advantage_nats']) <= 0:
        raise ValueError('Both margins must be strictly positive')
    if settings['families'] != [['grid_nca', 'prefix_grid', 'nca'], ['graph_nca', 'prefix_graph', 'graph']]:
        raise ValueError('Keep both declared structured families and their matched controls')
    artifacts, run, data, env, manifest = references(review, settings)
    files = {'data/development/manifest.json': sha_file(data/'manifest.json'),
             'runs/development/environment.json': sha_file(run/'environment.json')}
    for seed in settings['seeds']:
        for structured, control, kind in settings['families']:
            for dataset in (f'{kind}_train', f'{kind}_validation', f'{control}_validation'):
                name = f'data/development/replicate_{seed}/{dataset}.npy'
                expected = manifest['files'][f'replicate_{seed}/{dataset}.npy']
                if sha_file(artifacts/name) != expected:
                    raise ValueError('Cached diagnostic data changed')
                files[name] = expected
            for arm in (structured, control):
                folder = run/f'{arm}_{seed}'
                summary = read(folder/'ppt_summary.json')
                if sha_file(folder/'ppt_summary.json') != sha_file(ROOT/f'docs/results/transfer_v4_run1/gpu/{arm}_{seed}/ppt_summary.json'):
                    raise ValueError('PPT selection metadata differs from preserved reviewed evidence')
                name = f'runs/development/{arm}_{seed}/ppt_selected.pt'
                if summary['checkpoint'] != 'ppt_selected.pt' or sha_file(artifacts/name) != summary['checkpoint_sha256']:
                    raise ValueError('Selected PPT checkpoint differs')
                files[name] = summary['checkpoint_sha256']
                files[f'runs/development/{arm}_{seed}/ppt_summary.json'] = sha_file(folder/'ppt_summary.json')
    protocol = ROOT/'docs/READINESS_V5_PROTOCOL.md'
    (out/'FROZEN_PROTOCOL.md').write_bytes(protocol.read_bytes())
    plan = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
            'v4_config': env['config'], 'v4_source_sha256': v4_sources(), 'diagnostic_source_sha256': sources(),
            'protocol_sha256': sha_file(protocol), 'reference_audit_sha256': sha_file(Path(review)/'audit.json'),
            'input_files_sha256': files, 'scope': 'Development diagnostic on inspected v4 rules; no new neural training or confirmation'}
    dump(out/'plan.json', plan)
    dump(out/'plan_receipt.json', {'plan_sha256': sha_file(out/'plan.json')})
    dump(out/'status.json', {'state': 'frozen_before_cross_evaluation', 'language_allowed': False})
    return plan


def verify_plan(review, out):
    out = Path(out)
    plan, receipt = read(out/'plan.json'), read(out/'plan_receipt.json')
    if sha_file(out/'plan.json') != receipt['plan_sha256']:
        raise ValueError('Frozen plan checksum differs')
    if sources() != plan['diagnostic_source_sha256'] or v4_sources() != plan['v4_source_sha256']:
        raise ValueError('Source changed after freezing the diagnostic')
    if sha_file(out/'FROZEN_PROTOCOL.md') != plan['protocol_sha256'] or sha_file(ROOT/'docs/READINESS_V5_PROTOCOL.md') != plan['protocol_sha256']:
        raise ValueError('Frozen diagnostic protocol changed')
    if sha_file(Path(review)/'audit.json') != plan['reference_audit_sha256']:
        raise ValueError('Reference audit changed')
    artifacts, *_ = references(review, plan['settings'])
    for name, expected in plan['input_files_sha256'].items():
        if sha_file(artifacts/name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    return plan
