"""Regenerate matched public probes and rescore errors without private weights."""
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ood_v12.plan import read
from ood_v12.data import generate,write_jsonl,read_jsonl
from ood_v12.graphs import floyd
from id_audit_v14.metrics import select_training,generated_errors
from transfer_v4.common import sha_file


def reproduce(root=ROOT):
    public = root/'docs/results/id_audit_v14_run1'
    p,r = read(public/'plan.json'),read(public/'results.json')
    assert sha_file(public/'plan.json') == read(public/'plan_receipt.json')['plan_sha256'] == r['plan_sha256']
    assert sha_file(public/'results.json') == read(public/'results_receipt.json')['sha256']
    assert all(sha_file(root/name) == checksum for name,checksum in p['source_sha256'].items())
    arrays,_ = generate(read(root/'configs/ood_v12.json'))
    probes = {'train':select_training(arrays['train'],p['config']),'development':arrays['development']}
    temporary = root/'tmp/checks'; temporary.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary) as tmp:
        for split,rows in probes.items():
            path = Path(tmp)/(split+'.jsonl'); write_jsonl(path,rows)
            assert sha_file(path) == p['inputs_sha256'][f'data/{split}.jsonl']
            assert [v['id'] for v in rows] == p['selected_ids'][split]
            for row in rows:
                g = row['graph']
                assert row['truth'] == floyd(g['n'],g['edges'],g['source'],row['task']=='dijkstra_distances')
    total = 0
    for model in r['models']:
        for split,rows in probes.items():
            path = f"models/{model['arm']}/{split}_predictions.jsonl"
            assert sha_file(public/path) == model['files_sha256'][path]
            predictions = read_jsonl(public/path)
            assert generated_errors(rows,predictions) == model['metrics'][split]['generated']
            total += len(predictions)
    assert total == 1280
    return probes


if __name__ == '__main__':
    reproduce()
    print('PASS: 256 matched probe bytes/IDs/independent labels and all 1280 public generated-error scores reproduce without private weights/data, tokenizer cache, email or network. Teacher-forced/neural replay still requires private checkpoints.')
