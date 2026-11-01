"""Public data, order, reference and generated-score reproduction without weights."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ood_v12.plan import read
from ood_v12.data import generate,write_jsonl,read_jsonl
from ood_v12.graphs import floyd
from adaptation_v13.training import order
from id_audit_v14.metrics import select_training,generated_errors
from overfit_v15.core import reference_probe,competent,first_crossing
from transfer_v4.common import sha_file


def reproduce(root=ROOT):
    public=root/'docs/results/overfit_v15_run1';p=read(public/'plan.json');r=read(public/'results.json')
    assert sha_file(public/'plan.json')==read(public/'plan_receipt.json')['plan_sha256']==r['plan_sha256']
    assert sha_file(public/'results.json')==read(public/'results_receipt.json')['sha256']
    assert all(sha_file(root/name)==h for name,h in p['source_sha256'].items())
    for name,h in p['prior_evidence_sha256'].items():
        assert sha_file(root/name.replace('runs/','docs/results/',1))==h
    data,_=generate(read(root/'configs/ood_v12.json'))
    rows={'train':select_training(data['train'],p['config']),'development':data['development']}
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
    c=p['adaptation_config'];indices=order(16,c['updates'],c['batch'],c['order_seed'])
    buf=io.BytesIO();np.save(buf,indices)
    assert hashlib.sha256(buf.getvalue()).hexdigest()==p['inputs_sha256']['order.npy']
    total=0
    for record,m in zip(p['records'],r['models']):
        assert record['arm']==m['arm'];arm=m['arm']
        old13=read(root/'docs/results/adaptation_v13_run1/results.json')
        inherited=next(v for v in old13['models'] if v['arm']==arm)
        assert inherited['files_sha256'][f'models/{arm}/step_0.pt']==record['initial_checkpoint_sha256']
        assert inherited['initial_model_state_sha256']==record['initial_model_sha256']
        for name,h in record['reference_files_sha256'].items():
            assert sha_file(root/'docs/results/id_audit_v14_run1'/name)==h
        old=root/f'docs/results/id_audit_v14_run1/models/{arm}'
        oldmetrics=read(old/'metrics.json')
        for split in ('train','development'):
            expected,preds=reference_probe(rows[split],read_jsonl(old/f'{split}_predictions.jsonl'),oldmetrics[split]['teacher_forced'])
            assert expected==read(public/f'reference/{arm}/metrics.json')[split]
            assert preds==read_jsonl(public/f'reference/{arm}/{split}_predictions.jsonl')
        for entry in m['curve']:
            name=f'models/{arm}/cache_predictions_{entry["step"]}.jsonl'
            assert sha_file(public/name)==m['files_sha256'][name]
            pred=read_jsonl(public/name)
            assert generated_errors(rows['train'],pred)==entry['metrics']['generated'];total+=len(pred)
        name=f'models/{arm}/development_predictions.jsonl'
        assert sha_file(public/name)==m['files_sha256'][name]
        pred=read_jsonl(public/name)
        assert generated_errors(rows['development'],pred)==m['development']['generated'];total+=len(pred)
        assert m['cache_competent']==competent(m['curve'][-1]['metrics']['generated'],c)
        assert m['development_competent']==competent(m['development']['generated'],c)
        assert m['first_cache_crossing']==first_crossing(m['curve'],c)
    assert total==960 and r['small_cache_control_passed']==all(m['cache_competent'] for m in r['models'])
    return rows


if __name__=='__main__':
    reproduce()
    print('PASS: public matched data/labels/order, same-cache historical references, 960 new generated-answer scores and frozen decisions reproduce without private weights/data, tokenizer cache or network. New neural/teacher-forced scores require retained weights.')
