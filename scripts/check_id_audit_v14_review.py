"""Audit independent node/token counts, frozen chronology and public copies."""
from collections import Counter,defaultdict
import csv
from datetime import datetime,timedelta
import math
from pathlib import Path
from statistics import mean
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from id_audit_v14.run import verify_plan
from id_audit_v14.metrics import interpret
from ood_v12.plan import read
from ood_v12.data import read_jsonl
from ood_v12.graphs import floyd
from transfer_v4.common import sha_file
from scripts.report_id_audit_v14 import compact
from scripts.check_release import selected_files


def decimal_vector(text,n):
    t = text.strip()
    if not t.startswith('[') or not t.endswith(']'): return None
    values = []
    for chunk in t[1:-1].split(','):
        token = chunk.strip(); body = token[1:] if token.startswith('-') else token
        if not body or not all('0' <= c <= '9' for c in body): return None
        values.append(int(token))
    return values if len(values) == n and min(values) >= -1 else None


def main():
    run,public = ROOT/'runs/id_audit_v14_run1',ROOT/'docs/results/id_audit_v14_run1'
    p = verify_plan(run)
    r,v,s = [read(public/name) for name in ('results.json','verification.json','summary.json')]
    copies = read(public/'PROVENANCE.json')['files']
    assert len(copies) == 22 and all(sha_file(run/n)==sha_file(public/n)==h for n,h in copies.items())
    assert v['passed'] and v['predictions_replayed']==1280 and v['models']==5 and v['results_sha256']==sha_file(public/'results.json')
    assert s['plot_sha256']==sha_file(public/'matched_errors.png')
    assert r['new_optimizer_updates']==s['new_optimizer_updates']==v['new_optimizer_updates']==0
    assert s['graph_acceleration_claim'] is None and not s['ood_evaluation_allowed'] and s['original_random_rule_readiness']=='still_failed'
    assert s['models']==[{'arm':m['arm'],**compact(m['metrics'])} for m in r['models']]
    expected_csv = []
    for model in r['models']:
        m = model['metrics']
        assert m['interpretation']==interpret(m['train']['generated'],m['development']['generated'],p['config'])
        for split in ('train','development'):
            rows = read_jsonl(run/f'data/{split}.jsonl'); index={row['id']:row for row in rows}
            preds = read_jsonl(public/f"models/{model['arm']}/{split}_predictions.jsonl")
            generated = m[split]['generated']
            aggregate = defaultdict(Counter)
            nodes = defaultdict(list)
            saved_cases = {c['id']:c for c in generated['cases']}
            saved_errors = {c['id']:c for c in generated['errors']}
            for pred in preds:
                row=index[pred['id']]; g=row['graph']; parsed=decimal_vector(pred['text'],g['n']) if pred['status']=='ok' else None
                key=row['task'],g['n']; counts=aggregate[key]
                counts['examples']+=1;counts['valid']+=parsed is not None;counts['exact']+=parsed==row['truth']
                hops=floyd(g['n'],g['edges'],g['source'],False)
                assert min(hops)>=0  # This protocol uses connected native ID graphs.
                correct_nodes = 0
                signed = Counter()
                wrong = []
                for i,hop in enumerate(hops):
                    role='source' if hop==0 else 'one_hop' if hop==1 else 'multi_hop'
                    correct=int(parsed is not None and parsed[i]==row['truth'][i])
                    counts[role+'_nodes']+=1
                    counts[role+'_correct']+=correct; correct_nodes+=correct
                    if not correct: wrong.append(i)
                    if parsed is not None:
                        signed['valid_nodes']+=1
                        signed['under']+=parsed[i]<row['truth'][i]
                        signed['equal']+=correct
                        signed['over']+=parsed[i]>row['truth'][i]
                        signed['negative_reachable']+=parsed[i]<0
                node_accuracy=correct_nodes/g['n'];nodes[key].append(node_accuracy)
                case=saved_cases[row['id']]; error=saved_errors[row['id']]
                assert case['parsed']==parsed and case['exact_match']==(parsed==row['truth']) and case['node_accuracy']==node_accuracy
                assert error['invalid_examples']==int(parsed is None)
                first=wrong[0] if wrong and parsed is not None else None
                assert error['first_wrong_node']==first
                assert error['normalized_first_wrong']==(first/(g['n']-1) if first is not None else None)
                for name in ('valid_nodes','under','equal','over','negative_reachable'):
                    assert error.get(name,0)==signed[name]
            tf=m[split]['teacher_forced']; tfindex={(c['task'],c['n']):c for c in tf['cells']}
            tfcounts=defaultdict(Counter)
            for case in tf['cases']:
                row=index[case['id']]; totals={k:v for k,v in case.items() if k!='id'}
                assert totals['targets']==totals['digit_targets']+totals['other_targets']
                assert totals['correct']==totals.get('digit_correct',0)+totals.get('other_correct',0)
                tfcounts[row['task'],row['graph']['n']].update(totals)
            for cell in generated['cells']:
                key=cell['task'],cell['n']; counts=aggregate[key]; teacher=tfindex[key]
                assert counts['examples']==cell['count']==32
                assert counts['exact']/32==cell['exact_match'] and counts['valid']/32==cell['valid_format']
                assert mean(nodes[key])==cell['node_accuracy']
                for role in ('source','one_hop','multi_hop'):
                    assert counts[role+'_nodes']==cell.get(role+'_nodes',0) and counts[role+'_correct']==cell.get(role+'_correct',0)
                for k,value in tfcounts[key].items():
                    assert math.isclose(value,teacher[k],rel_tol=1e-12,abs_tol=1e-12)
                for prefix in ('','digit_','other_'):
                    assert teacher[prefix+'accuracy']==teacher[prefix+'correct']/teacher[prefix+'targets']
                    assert teacher[prefix+'nll']==teacher[prefix+'nll_sum']/teacher[prefix+'targets']
                expected_csv.append({'arm':model['arm'],'split':split,**cell,
                                     **{'teacher_'+k:teacher[k] for k in ('nll','accuracy','digit_accuracy','other_accuracy','targets','digit_targets','other_targets')}})
            assert generated['exact_match']==mean(c['exact_match'] for c in generated['cells'])
            assert generated['valid_format']==mean(c['valid_format'] for c in generated['cells'])
    with (public/'matched_cells.csv').open(encoding='utf-8',newline='') as f: table=list(csv.DictReader(f))
    fields={k for row in expected_csv for k in row}
    assert len(table)==40 and table==[{k:str(row[k]) if row.get(k) is not None else '' for k in fields} for row in expected_csv]
    import json
    events=[json.loads(line) for line in (ROOT/'logs/checks.jsonl').read_text(encoding='utf-8').splitlines()]
    labels=['id-audit-v14-freeze-before-new-metrics','id-audit-v14-five-matched-probes','id-audit-v14-separate-checkpoint-replay','id-audit-v14-export-verified-evidence']
    chain=[next(e for e in events if e['label']==label and e['exit_code']==0) for label in labels]
    for a,b in zip(chain[:-1],chain[1:]):
        assert datetime.fromisoformat(a['started_utc'])+timedelta(seconds=a['duration_seconds'])<datetime.fromisoformat(b['started_utc'])
    expected={k:h for k,h in p['source_sha256'].items() if k.startswith('id_audit_v14/')}
    assert all(e['id_audit_v14_sources_sha256']==expected for e in chain)
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(read(public/'evaluation_started.json')['started_utc'])
    names=selected_files(ROOT)
    assert not any(n.startswith(('runs/','data/','Kaggle Outputs/','.uv-cache/')) for n in names)
    print('PASS: 22 exact copies; matched IDs/exposures; 1280 replay receipts; independent decimal-vector/hop correctness; token partition counts/NLL; 40 CSV cells/summary; frozen source/chronology and privacy exclusions.')


if __name__=='__main__': main()
