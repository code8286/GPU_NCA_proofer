"""Public-only ID data/order/baseline/cost and generated-curve reproduction."""
from collections import Counter
import csv
import hashlib
import io
import math
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ood_v12.plan import read
from ood_v12.data import generate,read_jsonl,write_jsonl
from ood_v12.graphs import floyd
from adaptation_v13.run import digest,dataset_audit
from adaptation_v13.training import order,learning_rate
from id_audit_v14.metrics import generated_errors
from accel_v18.core import marginal,baseline_predictions,semantic,target_met,crossing
from accel_v18.run import costs,pipeline_at,contrasts
from scripts.report_accel_v18 import compact,flatten,render
from scripts.check_overfit_v15_review import check_metrics
from scripts.check_id_audit_v14_review import decimal_vector
from transfer_v4.common import sha_file


def independent_semantic(rows,predictions):
    index={v['id']:v for v in predictions};counts={};valid=exact=covered=0
    assert len(index)==len(predictions)==len(rows) and set(index)=={r['id'] for r in rows}
    for r in rows:
        p=index[r['id']];parsed=decimal_vector(p['text'],r['graph']['n']) if p['status']=='ok' else None
        covered+=p['status']=='ok';valid+=parsed is not None;exact+=parsed==r['truth']
        key=r['task'],r['graph']['n'];c=counts.setdefault(key,Counter())
        for i,t in enumerate(r['truth']):
            if i==r['graph']['source']:continue
            c['nodes']+=1;c['correct']+=parsed is not None and parsed[i]==t
    return {'cells':[{'task':t,'n':n,**dict(c),'accuracy':c['correct']/c['nodes']} for (t,n),c in sorted(counts.items())],
            'valid_format':valid/len(rows),'coverage':covered/len(rows),'exact_match':exact/len(rows)}


def reproduce(root=ROOT):
    folder=root/'docs/results/accel_v18_run1';p=read(folder/'plan.json');r=read(folder/'results.json');c=p['config']
    assert sha_file(folder/'plan.json')==read(folder/'plan_receipt.json')['plan_sha256']==r['plan_sha256']
    assert sha_file(folder/'results.json')==read(folder/'results_receipt.json')['sha256']
    copies=read(folder/'PROVENANCE.json')['files']
    assert len(copies)==60 and all(sha_file(folder/name)==h for name,h in copies.items())
    assert p['config']==read(root/'configs/accel_v18.json')
    assert sha_file(root/'docs/ACCEL_V18_PROTOCOL.md')==p['protocol_sha256']==sha_file(folder/'FROZEN_PROTOCOL.md')
    assert all(sha_file(root/name)==h for name,h in p['source_sha256'].items())
    assert p['prior_plan_sha256']==sha_file(root/'docs/results/cache_v17_run1/plan.json')
    arrays,inventory=generate(p['data_config']);assert not arrays['assessment'] and inventory==p['inventory']
    rows={k:arrays[k] for k in ('train','development')};dataset_audit(rows)
    parent=root/'tmp/checks';parent.mkdir(parents=True,exist_ok=True);assert parent.resolve().is_relative_to(root.resolve())
    with tempfile.TemporaryDirectory(dir=parent) as temp:
        assert Path(temp).resolve().is_relative_to(parent.resolve())
        for split,values in rows.items():
            path=Path(temp)/(split+'.jsonl');write_jsonl(path,values)
            assert sha_file(path)==p['inputs_sha256'][f'data/{split}.jsonl']
            assert [v['id'] for v in values]==p['selected_ids'][split]
            for v in values:
                g=v['graph'];assert v['truth']==floyd(g['n'],g['edges'],g['source'],v['task']=='dijkstra_distances')
    indices=order(512,c['updates'],c['batch'],c['order_seed']);buf=io.BytesIO();np.save(buf,indices)
    assert hashlib.sha256(buf.getvalue()).hexdigest()==p['inputs_sha256']['order.npy']
    ids=[[rows['train'][i]['id'] for i in batch] for batch in indices]
    assert set(Counter(v for batch in ids for v in batch).values())=={9,10}
    assert digest(ids)==p['sampled_order_sha256'] and digest([learning_rate(i,c) for i in range(1,601)])==p['schedule_sha256']
    budgets={0:dict.fromkeys(p['expected_counts_per_arm'],0)};running=dict(budgets[0])
    for step,batch in enumerate(ids,1):
        vals=[p['per_example_counts'][key] for key in batch]
        running['input_tokens']+=sum(v['input_tokens'] for v in vals);running['supervised_targets']+=sum(v['supervised_targets'] for v in vals)
        running['padded_positions']+=8*max(v['input_tokens'] for v in vals)
        if step in c['evaluation_updates']:budgets[step]=dict(running)
    assert running==p['expected_counts_per_arm']
    modes=marginal(rows['train']);pred=baseline_predictions(rows['development'],modes)
    baseline={'modes':modes,'semantic':independent_semantic(rows['development'],pred)}
    assert baseline==read(folder/'baseline.json') and pred==read_jsonl(folder/'baseline_predictions.jsonl')
    # Independently verify the training mode, including deterministic tie handling.
    for mode in modes:
        values=[v for row in rows['train'] if (row['task'],row['graph']['n'])==(mode['task'],mode['n']) for i,v in enumerate(row['truth']) if i!=row['graph']['source']]
        cnt=Counter(values);assert mode['distance']==sorted(cnt,key=lambda x:(-cnt[x],x))[0]
    imported=costs(folder);assert imported==read(folder/'costs.json')
    oldlang={v['arm']:v for v in read(root/'docs/results/ood_v12_run1/plan.json')['legacy_checkpoints']}
    manifest=read(folder/'cost_sources/data_manifest.json')
    for account in imported:
        arm=account['arm'];raw=read(folder/f'cost_sources/{arm}_language.json')
        assert raw['checkpoint_sha256']==oldlang[arm]['checkpoint_sha256']
        assert account['language_updates']==2000 and account['language_tokens']==8192000
        assert account['original_cumulative_seconds']==raw['cumulative_seconds']
        assert account['language_stage_seconds']==raw['language_seconds']
        if arm=='scratch':assert account['ppt_updates']==account['ppt_tokens']==account['ppt_stage_seconds']==0
        else:
            ppt=read(folder/f'cost_sources/{arm}_ppt.json')
            assert account['ppt_updates']==1000 and account['ppt_tokens']==4096000
            assert account['ppt_stage_seconds']==ppt['stage_seconds'] and account['ppt_train_seconds']==ppt['train_seconds']
            assert account['paired_preparation_seconds']==sum(v['prepare_seconds'] for v in manifest['replicates']['59']['nca'].values())
        assert account['original_cumulative_seconds']>=account['language_stage_seconds']+account['ppt_stage_seconds']+account['paired_preparation_seconds']+account['text_preparation_seconds']
    for name in p['inputs_sha256']:
        if name.startswith('cost_sources/') or name in ('baseline.json','baseline_predictions.jsonl','costs.json'):
            assert sha_file(folder/name)==p['inputs_sha256'][name]
    total=0
    for record,m,cost in zip(p['records'],r['models'],imported):
        assert record['arm']==m['arm']==cost['arm'];arm=m['arm']
        old=next(v for v in read(root/'docs/results/cache_v17_run1/plan.json')['records'] if v['arm']==arm)
        assert record['initial_checkpoint_sha256']==old['initial_checkpoint_sha256'] and record['initial_model_sha256']==old['initial_model_sha256']
        assert m['cost']==cost and m['counts']==running and m['sampled_order_sha256']==p['sampled_order_sha256'] and m['schedule_sha256']==p['schedule_sha256']
        assert read(folder/f'models/{arm}/summary.json')==m
        for entry in m['curve']:
            name=f'models/{arm}/development_{entry["step"]}.jsonl';pred=read_jsonl(folder/name)
            assert sha_file(folder/name)==m['files_sha256'][name]
            assert generated_errors(rows['development'],pred)==entry['metrics']['generated']
            sem=independent_semantic(rows['development'],pred);assert sem==entry['semantic']
            baseline_cells={(v['task'],v['n']):v['accuracy'] for v in baseline['semantic']['cells']}
            manual=(sem['coverage']==1 and sem['valid_format']>=c['format_target'] and len(sem['cells'])==4 and
                    all(v['accuracy']>=baseline_cells[v['task'],v['n']]+c['baseline_margin'] for v in sem['cells']))
            assert manual==entry['target_met']
            # Shared TF counters are reaggregated; no neural logits are claimed.
            check_metrics(rows['development'],pred,entry['metrics'])
            assert target_met(sem,baseline['semantic'],c)==entry['target_met'] and entry['counts']==budgets[entry['step']]
            total+=len(pred)
        assert m['crossing']==crossing(m['curve'],baseline['semantic'],c)
        assert m['sustained_crossing']==crossing(m['curve'],baseline['semantic'],c,True)
        assert m['pipeline_at_crossing']==pipeline_at(m['crossing'],cost)
    assert total==4480 and r['contrasts']==contrasts(r['models'])
    s=read(folder/'summary.json');assert s==compact(r)|{'plot_sha256':sha_file(folder/'id_curves.png')}
    # render uses module PUBLIC; root is the running checkout, as in standalone.
    assert root==ROOT and (folder/'REVIEW.md').read_text(encoding='utf-8')==render(p,r,s)
    expected=flatten(r,baseline['semantic']);keys={k for row in expected for k in row}
    with (folder/'cells.csv').open(encoding='utf-8',newline='') as f:table=list(csv.DictReader(f))
    assert len(table)==140 and table==[{k:str(row[k]) if row.get(k) is not None else '' for k in keys} for row in expected]
    return rows


if __name__=='__main__':
    reproduce();print('PASS: ID-only data/order, training-only baseline, imported costs, 4480 independently parsed generated scores, 140 rows, targets/censoring/contrasts reproduce without private weights/data/cache/network. Teacher logits require weights.')
