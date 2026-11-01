"""Reproduce public stage 17 data/order, both references and generated scores."""
import hashlib
import io
from collections import Counter
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ood_v12.plan import read
from ood_v12.data import generate,write_jsonl,read_jsonl
from ood_v12.graphs import floyd
from adaptation_v13.training import order,learning_rate
from adaptation_v13.run import digest
from id_audit_v14.metrics import select_training,generated_errors
from cache_v17.core import reference_probe,competent,first_crossing,validate_nested
from transfer_v4.common import sha_file


def reproduce(root=ROOT):
    public=root/'docs/results/cache_v17_run1';p=read(public/'plan.json');r=read(public/'results.json')
    assert sha_file(public/'plan.json')==read(public/'plan_receipt.json')['plan_sha256']==r['plan_sha256']
    assert sha_file(public/'results.json')==read(public/'results_receipt.json')['sha256']
    copies=read(public/'PROVENANCE.json')['files']
    assert len(copies)==72 and all(sha_file(public/name)==h for name,h in copies.items())
    assert all(sha_file(root/name)==h for name,h in p['source_sha256'].items())
    assert sha_file(public/'FROZEN_PROTOCOL.md')==p['protocol_sha256']==sha_file(root/'docs/CACHE_V17_PROTOCOL.md')
    for name,h in p['prior_evidence_sha256'].items():
        assert sha_file(root/name.replace('runs/','docs/results/',1))==h
    data,_=generate(read(root/'configs/ood_v12.json'))
    old15=root/'docs/results/overfit_v15_run1';old14=root/'docs/results/id_audit_v14_run1'
    p15=read(old15/'plan.json')
    rows={'train':select_training(data['train'],p['config']),'development':data['development'],
          'common':select_training(data['train'],p15['config'])}
    validate_nested(rows['train'],rows['common'],read(old14/'plan.json')['selected_ids']['train'])
    parent=root/'tmp/checks';parent.mkdir(parents=True,exist_ok=True)
    assert parent.resolve().is_relative_to(root.resolve())
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        assert Path(tmp).resolve().is_relative_to(parent.resolve())
        for split,values in rows.items():
            path=Path(tmp)/(split+'.jsonl');write_jsonl(path,values)
            assert sha_file(path)==p['inputs_sha256'][f'data/{split}.jsonl']
            assert [v['id'] for v in values]==p['selected_ids'][split]
            for v in values:
                g=v['graph'];assert v['truth']==floyd(g['n'],g['edges'],g['source'],v['task']=='dijkstra_distances')
    c=p['adaptation_config'];indices=order(64,c['updates'],c['batch'],c['order_seed'])
    buf=io.BytesIO();np.save(buf,indices)
    assert hashlib.sha256(buf.getvalue()).hexdigest()==p['inputs_sha256']['order.npy']
    expected_ids=[[rows['train'][i]['id'] for i in batch] for batch in indices]
    assert Counter(x for batch in expected_ids for x in batch)==Counter(p['exposures_per_example'])
    assert set(p['exposures_per_example'].values())=={75}
    assert digest(expected_ids)==p['sampled_order_sha256']
    assert digest([learning_rate(step,c) for step in range(1,601)])==p['schedule_sha256']
    counters=dict.fromkeys(p['expected_counts_per_arm'],0)
    for batch in expected_ids:
        lengths=[p['per_example_counts'][key] for key in batch]
        counters['input_tokens']+=sum(v['input_tokens'] for v in lengths)
        counters['supervised_targets']+=sum(v['supervised_targets'] for v in lengths)
        counters['padded_positions']+=8*max(v['input_tokens'] for v in lengths)
    assert counters==p['expected_counts_per_arm']
    total=0
    for record,m in zip(p['records'],r['models']):
        assert record['arm']==m['arm'];arm=m['arm']
        assert m['sampled_order_sha256']==p['sampled_order_sha256'] and m['schedule_sha256']==p['schedule_sha256']
        assert m['counts']==counters
        inherited=next(v for v in read(root/'docs/results/adaptation_v13_run1/results.json')['models'] if v['arm']==arm)
        assert inherited['files_sha256'][f'models/{arm}/step_0.pt']==record['initial_checkpoint_sha256']
        assert inherited['initial_model_state_sha256']==record['initial_model_sha256']
        for name,h in record['reference_files_sha256'].items():assert sha_file(old14/name)==h
        for name,h in record['small_reference_files_sha256'].items():assert sha_file(old15/name)==h
        oldmetrics=read(old14/f'models/{arm}/metrics.json')
        for split in ('train','development','common'):
            source='train' if split=='common' else split
            expected,preds=reference_probe(rows[split],read_jsonl(old14/f'models/{arm}/{source}_predictions.jsonl'),oldmetrics[source]['teacher_forced'])
            assert expected==read(public/f'reference/{arm}/metrics.json')[split]
            assert preds==read_jsonl(public/f'reference/{arm}/{split}_predictions.jsonl')
        small=read(old15/f'models/{arm}/summary.json')
        for split,name,metrics in (('common','cache_predictions_600.jsonl',small['curve'][-1]['metrics']),
                                   ('development','development_predictions.jsonl',small['development'])):
            expected,preds=reference_probe(rows[split],read_jsonl(old15/f'models/{arm}/{name}'),metrics['teacher_forced'])
            assert expected==read(public/f'reference16/{arm}/metrics.json')[split]
            assert preds==read_jsonl(public/f'reference16/{arm}/{split}_predictions.jsonl')
        for entry in m['curve']:
            name=f'models/{arm}/cache_predictions_{entry["step"]}.jsonl'
            assert sha_file(public/name)==m['files_sha256'][name]
            pred=read_jsonl(public/name);assert generated_errors(rows['train'],pred)==entry['metrics']['generated'];total+=len(pred)
            if entry['step']==600:
                common,_=reference_probe(rows['common'],pred,entry['metrics']['teacher_forced']);assert common==m['common']
        name=f'models/{arm}/development_predictions.jsonl';assert sha_file(public/name)==m['files_sha256'][name]
        pred=read_jsonl(public/name);assert generated_errors(rows['development'],pred)==m['development']['generated'];total+=len(pred)
        assert m['cache_competent']==competent(m['curve'][-1]['metrics']['generated'],c)
        assert m['development_competent']==competent(m['development']['generated'],c)
        assert m['first_cache_crossing']==first_crossing(m['curve'],c)
    assert total==1920 and r['cache64_control_passed']==all(m['cache_competent'] for m in r['models'])
    return rows


if __name__=='__main__':
    reproduce()
    print('PASS: public cache64/common16/dev128 data/labels/order, both same-case references, 1920 new generated scores and fixed decisions reproduce without private weights/data/cache/network. Neural/teacher forwards require retained weights.')
