"""Public data/probe, independent semantic and full trace/counter reproduction."""
from collections import Counter
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ood_v12.plan import read
from ood_v12.data import generate,read_jsonl,write_jsonl
from adaptation_v13.training import order,learning_rate
from accel_v18.core import marginal,baseline_predictions,semantic
from fit_audit_v19.core import select_training,trace_summary,decide
from id_audit_v14.metrics import generated_errors
from scripts.reproduce_accel_v18 import independent_semantic
from scripts.check_overfit_v15_review import check_metrics
from scripts.report_fit_audit_v19 import summary,tables,render
from transfer_v4.common import sha_file
import tempfile


def reproduce(root=ROOT):
    folder=root/'docs/results/fit_audit_v19_run1';old=root/'docs/results/accel_v18_run1'
    p=read(folder/'plan.json');r=read(folder/'results.json');p18=read(old/'plan.json');r18=read(old/'results.json');c=p['measurement_config']
    assert sha_file(folder/'plan.json')==read(folder/'plan_receipt.json')['plan_sha256']==r['plan_sha256']
    assert sha_file(folder/'results.json')==read(folder/'results_receipt.json')['sha256']
    assert p['config']==read(root/'configs/fit_audit_v19.json') and c==p18['config']
    assert sha_file(folder/'FROZEN_PROTOCOL.md')==p['protocol_sha256']==sha_file(root/'docs/FIT_AUDIT_V19_PROTOCOL.md')
    assert all(sha_file(root/name)==h for name,h in p['source_sha256'].items())
    for name,h in p['prior_evidence_sha256'].items():assert sha_file(root/name.replace('runs/','docs/results/',1))==h
    copies=read(folder/'PROVENANCE.json')['files'];assert len(copies)==36
    assert all(sha_file(folder/name)==h for name,h in copies.items())
    arrays,_=generate(p18['data_config']);assert not arrays['assessment']
    rows={'train':select_training(arrays['train'],p['config']),'development':arrays['development']}
    parent=root/'tmp/checks';parent.mkdir(parents=True,exist_ok=True);assert parent.resolve().is_relative_to(root.resolve())
    with tempfile.TemporaryDirectory(dir=parent) as temp:
        assert Path(temp).resolve().is_relative_to(parent.resolve())
        for split,values in rows.items():
            path=Path(temp)/(split+'.jsonl');write_jsonl(path,values)
            assert sha_file(path)==p['inputs_sha256'][f'data/{split}.jsonl'] and [v['id'] for v in values]==p['selected_ids'][split]
    modes=marginal(arrays['train']);base=read(folder/'baselines.json');assert modes==base['modes']==read(old/'baseline.json')['modes']
    for split,values in rows.items():
        pred=baseline_predictions(values,modes)
        assert pred==read_jsonl(folder/f'baseline_{split}_predictions.jsonl')
        assert independent_semantic(values,pred)==base['probes'][split]
    assert base['probes']['development']==read(old/'baseline.json')['semantic']
    indices=order(512,600,8,c['order_seed']);actual_ids=[[arrays['train'][i]['id'] for i in batch] for batch in indices]
    for record,m in zip(p['records'],r['models']):
        arm=m['arm'];original=next(v for v in r18['models'] if v['arm']==arm)
        assert record['checkpoint_sha256']==original['files_sha256'][f'models/{arm}/step_600.pt']==m['checkpoint_sha256']
        trainpred=read_jsonl(folder/f'models/{arm}/train_predictions.jsonl')
        assert sha_file(folder/f'models/{arm}/train_predictions.jsonl')==m['prediction_sha256']
        assert read(folder/f'models/{arm}/summary.json')==m
        for split,pred,metrics in (('train',trainpred,m['train']),
                                   ('development',read_jsonl(folder/f'reference/{arm}/development_predictions.jsonl'),m['development'])):
            check_metrics(rows[split],pred,metrics)
            assert generated_errors(rows[split],pred)==metrics['generated']
            assert independent_semantic(rows[split],pred)==metrics['semantic']
        assert sha_file(folder/f'reference/{arm}/development_predictions.jsonl')==original['files_sha256'][f'models/{arm}/development_600.jsonl']
        assert m['development']==original['curve'][-1]['metrics']|{'semantic':original['curve'][-1]['semantic']}
        trace=read_jsonl(folder/f'optimization/{arm}_updates.jsonl')
        assert sha_file(folder/f'optimization/{arm}_updates.jsonl')==original['files_sha256'][f'models/{arm}/updates.jsonl']
        assert [v['sampled_ids'] for v in trace]==actual_ids
        counts=dict.fromkeys(p18['expected_counts_per_arm'],0)
        for step,v in enumerate(trace,1):
            vals=[p18['per_example_counts'][key] for key in v['sampled_ids']]
            expected={'input_tokens':sum(x['input_tokens'] for x in vals),'supervised_targets':sum(x['supervised_targets'] for x in vals),
                      'padded_positions':8*max(x['input_tokens'] for x in vals)}
            assert v['step']==step and v['lr']==learning_rate(step,c)
            for key in counts:
                counts[key]+=expected[key];assert v[key]==expected[key] and v['cumulative_counts'][key]==counts[key]
        assert counts==p18['expected_counts_per_arm']
        assert trace_summary(trace,c,p['selected_ids']['train'])==r['optimization'][arm]
        assert decide(m['train'],m['development'],base['probes'],c)==m['decision']
    s=read(folder/'summary.json');assert s==summary(r)|{'plot_sha256':sha_file(folder/'fit_and_trace.png')}
    for filename,values,n in zip(('cells.csv','trace_blocks.csv'),tables(r,base['probes']),(40,30)):
        keys={k for row in values for k in row}
        with (folder/filename).open(encoding='utf-8',newline='') as f:actual=list(csv.DictReader(f))
        assert len(actual)==n and actual==[{k:str(row[k]) if row.get(k) is not None else '' for k in keys} for row in values]
    assert root==ROOT and (folder/'REVIEW.md').read_text(encoding='utf-8')==render(p,r,s)
    assert r['new_optimizer_updates']==r['new_ood_predictions']==0 and not r['confirmation_allowed']
    return rows


if __name__=='__main__':
    reproduce();print('PASS: ID-only probes, baseline identities,1280 new/imported semantic scores,3000 actual trace rows,40 cells/30 blocks and decisions reproduce without private weights/data/cache/network. Teacher logits require weights.')
