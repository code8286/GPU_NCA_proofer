"""Audit numerical treatment identity, exposures and independent generated scores."""
from collections import Counter,defaultdict
import csv
from datetime import datetime,timedelta
import json
import math
from pathlib import Path
from statistics import mean
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from overfit_v15.run import verify_plan
from overfit_v15.core import competent,first_crossing
from adaptation_v13.training import encode,learning_rate
from ood_v12.plan import read,tokenizer
from ood_v12.data import read_jsonl
from transfer_v4.common import sha_file
from scripts.check_id_audit_v14_review import decimal_vector
from scripts.report_overfit_v15 import point
from scripts.check_release import selected_files


def check_metrics(rows,predictions,metrics):
    index={r['id']:r for r in rows};cells=defaultdict(Counter);nodes=defaultdict(list)
    assert len(predictions)==len(rows) and {v['id'] for v in predictions}==set(index)
    for pred in predictions:
        row=index[pred['id']];key=row['task'],row['graph']['n']
        parsed=decimal_vector(pred['text'],row['graph']['n']) if pred['status']=='ok' else None
        c=cells[key];c['examples']+=1;c['valid']+=parsed is not None;c['exact']+=parsed==row['truth'];c['coverage']+=pred['status']=='ok'
        nodes[key].append(sum(a==b for a,b in zip(parsed,row['truth']))/len(row['truth']) if parsed is not None else 0.)
    g=metrics['generated']
    for c in g['cells']:
        key=c['task'],c['n'];v=cells[key]
        assert v['examples']==c['count'] and v['exact']/v['examples']==c['exact_match']
        assert v['valid']/v['examples']==c['valid_format'] and v['coverage']/v['examples']==c['coverage']==1
        assert mean(nodes[key])==c['node_accuracy']
    assert g['exact_match']==mean(c['exact_match'] for c in g['cells'])
    assert g['valid_format']==mean(c['valid_format'] for c in g['cells'])
    assert g['node_accuracy']==mean(x for group in nodes.values() for x in group)
    totals=defaultdict(Counter)
    for case in metrics['teacher_forced']['cases']:
        row=index[case['id']];v={k:x for k,x in case.items() if k!='id'}
        assert v['targets']==v['digit_targets']+v['other_targets']
        assert v['correct']==v.get('digit_correct',0)+v.get('other_correct',0)
        totals[row['task'],row['graph']['n']].update(v)
    for c in metrics['teacher_forced']['cells']:
        for k,value in totals[c['task'],c['n']].items(): assert math.isclose(value,c[k],rel_tol=1e-12,abs_tol=1e-12)
        for prefix in ('','digit_','other_'):
            assert c[prefix+'nll']==c[prefix+'nll_sum']/c[prefix+'targets']
            assert c[prefix+'accuracy']==c[prefix+'correct']/c[prefix+'targets']


def main():
    run,public=ROOT/'runs/overfit_v15_run1',ROOT/'docs/results/overfit_v15_run1'
    p=verify_plan(run);c=p['adaptation_config'];r,v,s=[read(public/n) for n in ('results.json','verification.json','summary.json')]
    copies=read(public/'PROVENANCE.json')['files']
    assert len(copies)==52 and all(sha_file(run/n)==sha_file(public/n)==h for n,h in copies.items())
    assert v['passed'] and v['checkpoints']==20 and v['new_predictions_replayed']==960 and v['results_sha256']==sha_file(public/'results.json')
    assert s['plot_sha256']==sha_file(public/'cache_fit.png')
    assert r['new_scientific_optimizer_updates']==s['new_scientific_optimizer_updates']==3000
    assert r['imported_reference_optimizer_updates']==s['imported_reference_optimizer_updates']==3000
    assert r['new_language_or_ppt_updates']==r['new_ood_predictions']==0 and r['graph_acceleration_claim'] is None and r['original_random_rule_readiness']=='still_failed'
    assert s['new_training_seconds']==sum(m['training_seconds'] for m in r['models'])
    assert s['new_evaluation_seconds']==sum(m['evaluation_seconds'] for m in r['models'])
    assert r['imported_reference_optimizer_updates']==sum(v['historical_reference_updates'] for v in p['records'])
    rows={split:read_jsonl(run/f'data/{split}.jsonl') for split in ('train','development')}
    tok=tokenizer();encoded={v['id']:encode(v,tok,1024) for v in rows['train']}
    expected_csv=[]
    for record,m,sm in zip(p['records'],r['models'],s['models']):
        arm=m['arm'];assert arm==record['arm']==sm['arm']
        assert read(public/f'models/{arm}/summary.json')==m
        trace=read_jsonl(run/f'models/{arm}/updates.jsonl');seen=Counter();counts=dict.fromkeys(p['expected_counts_per_arm'],0)
        assert len(trace)==600 and m['sampled_order_sha256']==p['sampled_order_sha256'] and m['schedule_sha256']==p['schedule_sha256']
        for i,update in enumerate(trace,1):
            batch=[encoded[x] for x in update['sampled_ids']];seen.update(update['sampled_ids'])
            actual={'input_tokens':sum(len(x['input']) for x in batch),'supervised_targets':sum(sum(v!=-100 for v in x['target']) for x in batch),
                    'padded_positions':len(batch)*max(len(x['input']) for x in batch)}
            assert update['step']==i and update['lr']==learning_rate(i,c) and math.isfinite(update['loss']) and math.isfinite(update['gradient_norm'])
            for key in counts:
                counts[key]+=actual[key];assert update[key]==actual[key] and update['cumulative_counts'][key]==counts[key]
        assert seen==Counter(p['exposures_per_example']) and set(seen.values())=={300}
        assert counts==p['expected_counts_per_arm']==m['counts']
        ref=read(public/f'reference/{arm}/metrics.json')
        groups=[('small16','cache',e['step'],e['metrics'],f'models/{arm}/cache_predictions_{e["step"]}.jsonl',rows['train']) for e in m['curve']]
        groups += [('small16','development',600,m['development'],f'models/{arm}/development_predictions.jsonl',rows['development'])]
        groups += [('reference512',probe,600,ref[split],f'reference/{arm}/{split}_predictions.jsonl',rows[split]) for probe,split in (('cache','train'),('development','development'))]
        for variant,probe,step,metrics,name,values in groups:
            check_metrics(values,read_jsonl(public/name),metrics)
            tf={(x['task'],x['n']):x for x in metrics['teacher_forced']['cells']}
            for cell in metrics['generated']['cells']:
                expected_csv.append({'arm':arm,'variant':variant,'probe':probe,'step':step,**cell,
                                     **{'teacher_'+k:tf[cell['task'],cell['n']][k] for k in ('nll','digit_accuracy','other_accuracy','targets')}})
        assert m['cache_competent']==competent(m['curve'][-1]['metrics']['generated'],c)
        assert m['development_competent']==competent(m['development']['generated'],c) and m['first_cache_crossing']==first_crossing(m['curve'],c)
        assert sm=={'arm':arm,'small_cache':point(m['curve'][-1]['metrics']),'development':point(m['development']),
                    'reference_same_cache':point(ref['train']),'reference_development':point(ref['development']),
                    'cache_competent':m['cache_competent'],'development_competent':m['development_competent'],'first_cache_crossing':m['first_cache_crossing']}
    with (public/'cells.csv').open(encoding='utf-8',newline='') as f:table=list(csv.DictReader(f))
    keys={k for row in expected_csv for k in row}
    assert len(table)==140 and table==[{k:str(row[k]) if row.get(k) is not None else '' for k in keys} for row in expected_csv]
    assert r['small_cache_control_passed']==s['small_cache_control_passed']==all(m['cache_competent'] for m in r['models'])
    events=[json.loads(x) for x in (ROOT/'logs/checks.jsonl').read_text(encoding='utf-8').splitlines()]
    labels=['overfit-v15-freeze-before-first-update','overfit-v15-five-fixed-budget-fits','overfit-v15-separate-checkpoint-replay','overfit-v15-export-verified-evidence']
    chain=[next(e for e in events if e['label']==label and e['exit_code']==0) for label in labels]
    for a,b in zip(chain[:-1],chain[1:]): assert datetime.fromisoformat(a['started_utc'])+timedelta(seconds=a['duration_seconds'])<datetime.fromisoformat(b['started_utc'])
    expected={k:h for k,h in p['source_sha256'].items() if k.startswith('overfit_v15/')}
    assert all(e['overfit_v15_sources_sha256']==expected for e in chain)
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(read(public/'fit_started.json')['started_utc'])
    names=selected_files(ROOT);assert not any(n.startswith(('runs/','data/','Kaggle Outputs/','.uv-cache/')) for n in names)
    print('PASS: 52 exact copies, 20/960 replay receipts, original numerical configuration/starts, imported same-cache reference, actual 300 exposures and token budgets, independent vector/cell accounting, 140 CSV rows/summary and frozen decisions/source/chronology/privacy.')


if __name__=='__main__':main()
