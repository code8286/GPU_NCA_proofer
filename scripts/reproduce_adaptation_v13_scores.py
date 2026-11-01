"""Recreate public ID data and all generated-answer scores WITHOUT weights."""
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from transfer_v4.common import sha_file
from ood_v12.plan import read
from ood_v12.data import generate, write_jsonl, read_jsonl
from ood_v12.scoring import score


def main():
    public = ROOT/'docs/results/adaptation_v13_run1'
    p, result = read(public/'plan.json'), read(public/'results.json')
    assert sha_file(public/'plan.json') == read(public/'plan_receipt.json')['plan_sha256']
    assert sha_file(public/'results.json') == read(public/'results_receipt.json')['sha256']
    assert result['plan_sha256'] == sha_file(public/'plan.json')
    assert sha_file(public/'FROZEN_PROTOCOL.md') == p['protocol_sha256']
    assert all(sha_file(ROOT/name) == value for name,value in p['source_sha256'].items())
    arrays, _ = generate(read(ROOT/'configs/ood_v12.json'))
    parent = ROOT/'tmp/checks'
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        for split in ('train','development'):
            path = Path(temporary)/(split+'.jsonl')
            write_jsonl(path,arrays[split])
            assert sha_file(path) == p['input_sha256'][f'data/{split}.jsonl']
    count = 0
    for model in result['models']:
        for checkpoint in model['curve']:
            name = f"models/{model['arm']}/predictions_{checkpoint['step']}.jsonl"
            assert sha_file(public/name) == model['files_sha256'][name]
            predictions = read_jsonl(public/name)
            value = score(arrays['development'],predictions)
            assert value['overall_exact_match'] == checkpoint['exact_match']
            assert value['overall_valid_format'] == checkpoint['valid_format']
            assert value['coverage'] == checkpoint['coverage']
            assert value['cells'] == checkpoint['cells']
            count += len(predictions)
    assert count == 2560
    print('PASS: regenerated ID train/development bytes match frozen hashes; all 2560 public generated-answer scores/cells reproduce without model weights, private data, tokenizer cache or network. This does not replay neural inference/NLL/training.')


if __name__ == '__main__':
    main()
