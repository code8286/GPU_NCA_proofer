"""Audit independent endpoint counts, actual budgets, references and chronology."""
from collections import Counter
import csv
from datetime import datetime,timedelta
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cache_v17.run import verify_plan
from cache_v17.core import competent,first_crossing
from adaptation_v13.training import encode,learning_rate
from ood_v12.plan import read,tokenizer
from ood_v12.data import read_jsonl
from scripts.check_overfit_v15_review import check_metrics
from scripts.report_cache_v17 import RUN,PUBLIC,groups,csv_rows,summary,render
from scripts.check_release import selected_files
from transfer_v4.common import sha_file
from error_audit_v16.core import annotate,aggregate,ROLES


def main():
    p=verify_plan(RUN);c=p['adaptation_config'];r,v,s=[read(PUBLIC/n) for n in ('results.json','verification.json','summary.json')]
    copies=read(PUBLIC/'PROVENANCE.json')['files']
    assert len(copies)==72 and all(sha_file(RUN/n)==sha_file(PUBLIC/n)==h for n,h in copies.items())
    assert v['passed'] and v['checkpoints']==20 and v['new_predictions_replayed']==1920 and v['results_sha256']==sha_file(PUBLIC/'results.json')
    assert s['plot_sha256']==sha_file(PUBLIC/'cache_comparison.png')
    assert s==summary(p,r,v,PUBLIC)|{'plot_sha256':s['plot_sha256']}
    assert (PUBLIC/'REVIEW.md').read_text(encoding='utf-8')==render(p,r,s)
    assert r['new_scientific_optimizer_updates']==s['new_scientific_optimizer_updates']==3000
    assert r['imported_reference_optimizer_updates']==s['imported_reference_optimizer_updates']==6000
    assert r['new_language_or_ppt_updates']==r['new_ood_predictions']==0 and r['graph_acceleration_claim'] is None
    assert not s['ood_allowed'] and r['original_random_rule_readiness']=='still_failed'
    assert sum(x['historical_reference_updates']+x['small_reference_updates'] for x in p['records'])==6000
    rows={split:read_jsonl(RUN/f'data/{split}.jsonl') for split in ('train','common','development')}
    tok=tokenizer();encoded={v['id']:encode(v,tok,1024) for v in rows['train']}
    for record,m in zip(p['records'],r['models']):
        arm=m['arm'];assert arm==record['arm'] and read(PUBLIC/f'models/{arm}/summary.json')==m
        trace=read_jsonl(RUN/f'models/{arm}/updates.jsonl');seen=Counter();counts=dict.fromkeys(p['expected_counts_per_arm'],0)
        assert len(trace)==600 and m['sampled_order_sha256']==p['sampled_order_sha256'] and m['schedule_sha256']==p['schedule_sha256']
        for i,update in enumerate(trace,1):
            batch=[encoded[x] for x in update['sampled_ids']];seen.update(update['sampled_ids'])
            actual={'input_tokens':sum(len(x['input']) for x in batch),'supervised_targets':sum(sum(v!=-100 for v in x['target']) for x in batch),
                    'padded_positions':len(batch)*max(len(x['input']) for x in batch)}
            assert update['step']==i and update['lr']==learning_rate(i,c) and math.isfinite(update['loss']) and math.isfinite(update['gradient_norm'])
            for key in counts:
                counts[key]+=actual[key];assert update[key]==actual[key] and update['cumulative_counts'][key]==counts[key]
        assert seen==Counter(p['exposures_per_example']) and set(seen.values())=={75}
        assert counts==p['expected_counts_per_arm']==m['counts']
        for variant,probe,step,metrics,name,split in groups(PUBLIC,m):
            values=rows[split];preds=read_jsonl(PUBLIC/name)
            if probe=='common16' and variant=='new64':
                ids={x['id'] for x in values};preds=[x for x in preds if x['id'] in ids]
            check_metrics(values,preds,metrics)
            # Complement inherited cell checker with independent per-node hop/sign partitions.
            a=annotate(values,preds,independent=True);counters=aggregate(a,metrics['teacher_forced']['cases'])
            for role in ROLES:
                assert sum(cell.get(role+'_nodes',0) for cell in metrics['generated']['cells'])==counters[role+'_nodes']
                assert sum(cell.get(role+'_correct',0) for cell in metrics['generated']['cells'])==counters[role+'_correct']
            assert counters['under']+counters['equal']+counters['over']==counters['valid_nodes']
            assert counters['first_wrong_vectors']==counters['valid_examples']-counters['exact_examples']
        assert m['cache_competent']==competent(m['curve'][-1]['metrics']['generated'],p['adaptation_config'])
        assert m['development_competent']==competent(m['development']['generated'],p['adaptation_config'])
        assert m['first_cache_crossing']==first_crossing(m['curve'],p['adaptation_config'])
    expected=csv_rows(PUBLIC,r);keys={k for row in expected for k in row}
    with (PUBLIC/'cells.csv').open(encoding='utf-8',newline='') as f:table=list(csv.DictReader(f))
    assert len(table)==220 and table==[{k:str(row[k]) if row.get(k) is not None else '' for k in keys} for row in expected]
    assert r['cache64_control_passed']==all(m['cache_competent'] for m in r['models'])
    events=[json.loads(x) for x in (ROOT/'logs/checks.jsonl').read_text(encoding='utf-8').splitlines()]
    labels=['cache-v17-freeze-before-first-update','cache-v17-five-fixed-budget-fits','cache-v17-separate-checkpoint-replay','cache-v17-export-verified-evidence']
    chain=[next(e for e in events if e['label']==label and e['exit_code']==0) for label in labels]
    for a,b in zip(chain[:-1],chain[1:]):assert datetime.fromisoformat(a['started_utc'])+timedelta(seconds=a['duration_seconds'])<datetime.fromisoformat(b['started_utc'])
    expected={k:h for k,h in p['source_sha256'].items() if k.startswith('cache_v17/')}
    assert all(e['cache_v17_sources_sha256']==expected for e in chain)
    assert datetime.fromisoformat(p['created_utc'])<datetime.fromisoformat(read(PUBLIC/'fit_started.json')['started_utc'])
    names=selected_files(ROOT)
    assert not any(n.startswith(('runs/','data/','Kaggle Outputs/','.uv-cache/')) for n in names)
    print('PASS:72 exact copies,20/1920 replay receipts, original numerical starts/config, two matched references,75 exposures/actual token budgets, independent node/hop/sign/cell accounting,220 CSV rows/review and frozen decisions/source/chronology/privacy.')


if __name__=='__main__':main()
