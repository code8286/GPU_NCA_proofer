"""Audit exact copies, independent correctness accounting and prospective timing."""
from datetime import datetime, timedelta
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adaptation_v13.run import verify_plan, load_inputs
from adaptation_v13.training import decision
from ood_v12.plan import read
from ood_v12.data import read_jsonl
from ood_v12.scoring import score
from transfer_v4.common import sha_file
from scripts.check_release import selected_files


def main():
    run, out = ROOT/'runs/adaptation_v13_run1', ROOT/'docs/results/adaptation_v13_run1'
    p = verify_plan(run)
    copies = read(out/'PROVENANCE.json')['files']
    assert len(copies) == 32
    assert all(sha_file(run/n) == sha_file(out/n) == v for n,v in copies.items())
    result, summary, verification = [read(out/n) for n in ('results.json','summary.json','verification.json')]
    assert summary['learning_curves_sha256'] == sha_file(out/'learning_curves.png')
    assert verification['passed'] and verification['checkpoint_replays'] == 20 and verification['generated_answers_replayed'] == 2560
    assert verification['results_sha256'] == sha_file(out/'results.json')
    rows = read_jsonl(run/'data/development.jsonl')
    index = {r['id']:r for r in rows}
    _, _, encoded = load_inputs(run,p)
    train_index = {r['id']:r for r in encoded['train']}
    expected_cells = []
    for m, public_model in zip(result['models'],summary['models']):
        assert m['updates'] == 600 and m['counts'] == p['expected_exposure_per_arm']
        assert m['sampled_order_sha256'] == p['sampled_order_sha256'] and m['schedule_sha256'] == p['schedule_sha256']
        assert decision(m['curve'],p['config']) == m['decision']
        assert public_model == {'arm':m['arm'],'passed':m['decision']['passed'],
                                'crossing':m['decision']['first_pooled_crossing'],'final':m['curve'][-1]}
        counts = dict.fromkeys(p['expected_exposure_per_arm'],0)
        for trace in read_jsonl(run/f"models/{m['arm']}/updates.jsonl"):
            batch = [train_index[i] for i in trace['sampled_ids']]
            counts['input_tokens'] += sum(len(r['input']) for r in batch)
            counts['supervised_targets'] += sum(sum(v != -100 for v in r['target']) for r in batch)
            counts['padded_positions'] += len(batch)*max(len(r['input']) for r in batch)
            assert all(trace[k] == v for k,v in counts.items())
        assert counts == p['expected_exposure_per_arm']
        for r in m['curve']:
            predictions = read_jsonl(out/f"models/{m['arm']}/predictions_{r['step']}.jsonl")
            value = score(rows,predictions)
            assert value['overall_exact_match'] == r['exact_match'] and value['overall_valid_format'] == r['valid_format']
            assert value['cells'] == r['cells'] and value['coverage'] == r['coverage'] == 1
            # Independent JSON parsing for observed full-vector correctness and
            # vector validity; no shared regex/loss is used for these counts.
            exact = valid = 0
            for pred in predictions:
                try:
                    parsed = json.loads(pred['text'])
                except (ValueError,TypeError):
                    parsed = None
                row = index[pred['id']]
                isvalid = (type(parsed) is list and len(parsed) == row['graph']['n']
                           and all(type(x) is int and x >= -1 for x in parsed))
                valid += isvalid
                exact += isvalid and parsed == row['truth']
            assert exact/128 == r['exact_match'] and valid/128 == r['valid_format']
            expected_cells += [dict(arm=m['arm'],step=r['step'],input_tokens=r['tokens'],supervised_targets=r['targets'],
                                   training_seconds=r['seconds'],evaluation_seconds=r['evaluation_seconds'],
                                   train_answer_eos_nll=r['train_nll']['answer_eos_nll'],dev_answer_eos_nll=r['development_nll']['answer_eos_nll'],
                                   **cell) for cell in r['cells']]
    assert len(summary['models']) == len(result['models']) == 5
    assert result['id_calibration_passed'] == all(m['decision']['passed'] for m in result['models']) == summary['passed']
    assert result['new_scientific_optimizer_updates'] == summary['scientific_optimizer_updates'] == 3000
    assert result['new_ood_predictions'] == summary['new_ood_predictions'] == result['new_language_updates'] == 0
    assert summary['training_seconds'] == sum(m['train_seconds'] for m in result['models'])
    assert summary['evaluation_seconds'] == sum(m['evaluation_seconds'] for m in result['models'])
    assert result['ood_decision'] == summary['ood_decision'] == ('eligible_for_separate_fixed_endpoint_development_evaluation' if summary['passed'] else 'skipped_failed_ID_calibration')
    assert summary['graph_acceleration_claim'] is None and summary['original_random_rule_readiness'] == 'still_failed'
    with (out/'id_cells.csv').open(encoding='utf-8',newline='') as f:
        table = list(csv.DictReader(f))
    assert len(table) == 80
    assert table == [{k:str(v) for k,v in row.items()} for row in expected_cells]
    events = [json.loads(line) for line in (ROOT/'logs/checks.jsonl').read_text(encoding='utf-8').splitlines()]
    labels = ['adaptation-v13-freeze-before-first-update','adaptation-v13-five-matched-ID-adaptation-fits',
              'adaptation-v13-separate-twenty-checkpoint-generation-replay','adaptation-v13-export-verified-evidence']
    chain = [next(e for e in events if e['label'] == label and e['exit_code'] == 0) for label in labels]
    for a,b in zip(chain[:-1],chain[1:]):
        assert datetime.fromisoformat(a['started_utc'])+timedelta(seconds=a['duration_seconds']) < datetime.fromisoformat(b['started_utc'])
    expected = {k:v for k,v in p['source_sha256'].items() if k.startswith('adaptation_v13/')}
    assert all(e['adaptation_v13_sources_sha256'] == expected for e in chain)
    start = read(out/'fit_started.json')
    assert datetime.fromisoformat(p['created_utc']) < datetime.fromisoformat(start['started_utc'])
    names = selected_files(ROOT)
    assert not any(n.startswith(('runs/','data/','Kaggle Outputs/','.uv-cache/')) for n in names)
    assert all(n in names for n in ('adaptation_v13/run.py','configs/adaptation_v13.json','docs/ADAPTATION_V13_PROTOCOL.md'))
    print('PASS: 32 exact copies, 20 checkpoint/2560 answer replay receipts, independently parsed ID correctness, 80 cells, common exposure/order/schedule, frozen decisions/chronology and private artifact exclusions.')


if __name__ == '__main__':
    main()
