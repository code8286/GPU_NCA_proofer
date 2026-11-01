"""Audit copied OOD evidence, score accounting, frozen timing and privacy boundary."""
from datetime import datetime, timedelta
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transfer_v4.common import sha_file
from ood_v12.plan import verify_plan, read
from ood_v12.data import read_jsonl
from ood_v12.scoring import score, parse_answer
from ood_v12.run import compact
from scripts.check_release import selected_files


def main():
    run, public = ROOT/'runs/ood_v12_run1', ROOT/'docs/results/ood_v12_run1'
    p = verify_plan(run)
    copies = read(public/'PROVENANCE.json')['files']
    assert len(copies) == 13
    assert all(sha_file(run/name) == sha_file(public/name) == checksum for name,checksum in copies.items())
    r, c, v = [read(public/name) for name in ('legacy_results.json', 'controls.json', 'verification.json')]
    assert v['passed'] and v['legacy_predictions_replayed'] == 160
    assert v['controls_sha256'] == sha_file(public/'controls.json') and v['results_sha256'] == sha_file(public/'legacy_results.json')
    assert c['independent_oracle_examples'] == {'train':512, 'development':128, 'assessment':5120}
    assert c['controls']['oracle']['overall_exact_match'] == c['controls']['oracle']['native_and_cut_correct'] == 1
    assert c['controls']['cached_native_answer']['overall_exact_match'] == .75
    assert c['controls']['cached_native_answer']['native_and_cut_correct'] == c['controls']['source_only']['native_and_cut_correct'] == 0
    assert all(len(value['cells']) == 160 for value in c['controls'].values())
    rows = read_jsonl(run/'data/legacy_pilot.jsonl')
    assert len(rows) == 32 and len({row['base_id'] for row in rows}) == 4
    result_index = {m['arm']:m for m in r['models']}
    assert set(result_index) == set(p['config']['legacy_arms'])
    for record in p['legacy_checkpoints']:
        name = f"predictions/{record['arm']}_{record['seed']}.jsonl"
        preds = read_jsonl(public/name)
        assert len(preds) == 32 and {x['id'] for x in preds} == {x['id'] for x in rows}
        value = score(rows, preds)
        assert result_index[record['arm']]['score'] == compact(value)
        # Independently verify this observed format floor, without using the
        # same parse/aggregation helper to manufacture the conclusion.
        invalid = 0
        for item in preds:
            try:
                parsed = json.loads(item['text'])
            except (json.JSONDecodeError, ValueError):
                parsed = None
            invalid += not isinstance(parsed, list)
            assert item['status'] == 'ok' and item['stop_reason'] == 'irrecoverable_format_prefix'
        assert invalid == 32 and value['overall_valid_format'] == value['overall_exact_match'] == 0
        assert result_index[record['arm']]['checkpoint_sha256'] == record['checkpoint_sha256']
    with (public/'legacy_cells.csv').open(encoding='utf-8', newline='') as f:
        table = list(csv.DictReader(f))
    assert len(table) == 160
    assert all(float(row['exact_match']) == float(row['valid_format']) == 0 and float(row['coverage']) == 1 and int(row['count']) == 1 for row in table)
    summary = read(public/'summary.json')
    assert summary['scientific_optimizer_updates'] == 0 and summary['new_gpu_measurements'] == 0
    assert summary['graph_acceleration_claim'] is None and not summary['reserved_confirmation_generated']
    assert not r['language_training_allowed'] and r['original_random_rule_readiness'] == 'still_failed'
    events = [json.loads(x) for x in (ROOT/'logs/checks.jsonl').read_text(encoding='utf-8').splitlines()]
    def event(label):
        return next(e for e in events if e['label'] == label and e['exit_code'] == 0)
    labels = ['ood-v12-freeze-offline-with-explicit-cache', 'ood-v12-independent-oracles-and-matched-measurement-controls',
              'ood-v12-five-retained-final-language-checkpoints-zero-shot', 'ood-v12-independent-label-audit-and-checkpoint-generation-replay',
              'ood-v12-export-verified-development-evidence']
    chain = [event(label) for label in labels]
    for a,b in zip(chain[:-1], chain[1:]):
        assert datetime.fromisoformat(a['started_utc'])+timedelta(seconds=a['duration_seconds']) < datetime.fromisoformat(b['started_utc'])
    expected = {k:v for k,v in p['source_sha256'].items() if k.startswith('ood_v12/')}
    assert all(e['ood_v12_sources_sha256'] == expected for e in chain)
    names = selected_files(ROOT)
    assert not any(n.startswith(('runs/', 'data/', 'Kaggle Outputs/')) or 'petar_reply_private' in n for n in names)
    assert any(n.startswith('sources/clrs-text-reference/') and n.endswith('/LICENSE') for n in names)
    assert 'proofs/lean' in read(ROOT/'publication.json')['directories'] and 'docs/theory' in read(ROOT/'publication.json')['directories']
    print('PASS: 13 exact evidence copies; 5760 independent-oracle labels; all controls; 160 historical predictions/cells; independently counted format floor; fixed paired language stages, freeze-before-prediction chronology and private-correspondence exclusion.')


if __name__ == '__main__':
    main()
