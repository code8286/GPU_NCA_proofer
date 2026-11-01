"""Freeze, analyze and independently rescore retained stage 15 endpoint outputs."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from ood_v12.plan import ROOT, read
from ood_v12.data import generate, read_jsonl
from id_audit_v14.metrics import select_training
from transfer_v4.common import dump, sha_file
from .core import annotate, aggregate, paired

PUBLIC = ROOT/'docs/results/error_audit_v16_run1'
PRIOR = ROOT/'docs/results/overfit_v15_run1'
PROTOCOL = ROOT/'docs/ERROR_AUDIT_V16_PROTOCOL.md'


def sources():
    paths = sorted((ROOT/'error_audit_v16').glob('*.py')) + [ROOT/'scripts/check_id_audit_v14_review.py']
    return {p.relative_to(ROOT).as_posix():sha_file(p) for p in paths}


def inputs():
    old = read(PRIOR/'PROVENANCE.json')['files']
    paths = [PRIOR/n for n in old] + [PRIOR/'PROVENANCE.json', ROOT/'configs/ood_v12.json']
    for name,h in old.items():
        if sha_file(PRIOR/name) != h:
            raise ValueError('Imported stage 15 evidence changed')
    plan = read(PRIOR/'plan.json')
    paths += [ROOT/n for n in plan['source_sha256']]
    for name,h in plan['source_sha256'].items():
        if sha_file(ROOT/name) != h:
            raise ValueError('Inherited frozen source changed')
    return {p.relative_to(ROOT).as_posix():sha_file(p) for p in paths}


def freeze():
    p = read(PRIOR/'plan.json'); v = read(PRIOR/'verification.json')
    if not v['passed'] or v['results_sha256'] != sha_file(PRIOR/'results.json'):
        raise ValueError('Previously verified endpoint required')
    bound = inputs()
    PUBLIC.mkdir(parents=True,exist_ok=False)
    (PUBLIC/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    plan = {'schema_version':16,'created_utc':datetime.now(timezone.utc).isoformat(),
            'protocol_sha256':sha_file(PROTOCOL),'source_sha256':sources(), 'inputs_sha256':bound,
            'arms':p['config']['arms'],'variants':['small16','reference512'], 'step':600,
            'probes':{'cache':16,'development':128},'groups':['pooled','bfs_distances/8','bfs_distances/10','dijkstra_distances/8','dijkstra_distances/10'],
            'new_optimizer_updates':0,'new_model_forwards':0,'new_predictions':0,'ood_allowed':False}
    dump(PUBLIC/'plan.json',plan)
    dump(PUBLIC/'plan_receipt.json',{'plan_sha256':sha_file(PUBLIC/'plan.json'),'created_utc':datetime.now(timezone.utc).isoformat()})
    print('FROZEN stage 16 retained-output audit:',sha_file(PUBLIC/'plan.json'))


def verify_plan():
    p = read(PUBLIC/'plan.json')
    if (sha_file(PUBLIC/'plan.json') != read(PUBLIC/'plan_receipt.json')['plan_sha256'] or
            sha_file(PUBLIC/'FROZEN_PROTOCOL.md') != p['protocol_sha256'] or
            sha_file(PROTOCOL) != p['protocol_sha256'] or sources() != p['source_sha256'] or
            inputs() != p['inputs_sha256']):
        raise ValueError('Frozen protocol, source or evidence changed')
    return p


def compute(independent=False):
    p = verify_plan(); oldplan = read(PRIOR/'plan.json'); old = read(PRIOR/'results.json')
    data,_ = generate(read(ROOT/'configs/ood_v12.json'))
    rows = {'cache':select_training(data['train'],oldplan['config']),'development':data['development']}
    for probe, values in rows.items():
        split = 'train' if probe == 'cache' else probe
        if [v['id'] for v in values] != oldplan['selected_ids'][split] or len(values) != p['probes'][probe]:
            raise ValueError('Retain exactly the original selected probes')
    summaries = []; comparisons = []; total = 0
    for arm in p['arms']:
        model = next(m for m in old['models'] if m['arm'] == arm)
        ref = read(PRIOR/f'reference/{arm}/metrics.json')
        for probe, values in rows.items():
            by_variant = {}
            for variant in p['variants']:
                if variant == 'small16':
                    metrics = model['curve'][-1]['metrics'] if probe == 'cache' else model['development']
                    name = 'cache_predictions_600.jsonl' if probe == 'cache' else 'development_predictions.jsonl'
                    path = PRIOR/f'models/{arm}'/name
                else:
                    split = 'train' if probe == 'cache' else probe
                    metrics = ref[split]; path = PRIOR/f'reference/{arm}/{split}_predictions.jsonl'
                pred = read_jsonl(path); cases = annotate(values,pred,independent)
                # The independent path additionally checks all imported semantic cases.
                stored = {v['id']:v for v in metrics['generated']['cases']}
                if set(stored) != {v['id'] for v in cases}:
                    raise ValueError('Imported semantic case coverage differs')
                for case in cases:
                    if case['parsed'] != stored[case['id']]['parsed'] or all(case['correct']) != stored[case['id']]['exact_match']:
                        raise ValueError('Rescored generated semantics differ from imported evidence')
                by_variant[variant] = cases; total += len(cases)
                teachers = {v['id']:v for v in metrics['teacher_forced']['cases']}
                if len(teachers) != len(cases) or set(teachers) != {v['id'] for v in cases}:
                    raise ValueError('Imported teacher case coverage differs')
                for group in p['groups']:
                    subset = cases if group == 'pooled' else [v for v in cases if f"{v['task']}/{v['n']}" == group]
                    summaries.append({'arm':arm,'variant':variant,'probe':probe,'group':group,
                                      **aggregate(subset,[teachers[v['id']] for v in subset])})
            for group in p['groups']:
                selected = lambda vv: vv if group == 'pooled' else [v for v in vv if f"{v['task']}/{v['n']}" == group]
                comparisons.append({'arm':arm,'probe':probe,'group':group,
                                    **paired(selected(by_variant['small16']),selected(by_variant['reference512']))})
    if len(summaries) != 100 or len(comparisons) != 50 or total != 1440:
        raise ValueError('All arms, variants, probes and cells required')
    return {'plan_sha256':sha_file(PUBLIC/'plan.json'),'rows':summaries,'paired':comparisons,
            'retained_generated_outputs_rescored':total,'new_optimizer_updates':0,'new_model_forwards':0,
            'new_predictions':0,'ood_allowed':False,'graph_acceleration_claim':None}


def analyze():
    if (PUBLIC/'results.json').exists():
        raise ValueError('Preserve completed endpoint; no overwrite')
    value = compute(); dump(PUBLIC/'results.json',value)
    dump(PUBLIC/'results_receipt.json',{'sha256':sha_file(PUBLIC/'results.json'),'created_utc':datetime.now(timezone.utc).isoformat()})
    flat = []
    for row in value['rows']:
        flat.append({k:v for k,v in row.items() if k not in ('teacher','first_wrong_counts')} |
                    {'teacher_'+k:v for k,v in row['teacher'].items()} |
                    {'first_wrong_counts':json.dumps(row['first_wrong_counts'],sort_keys=True)})
    with (PUBLIC/'cells.csv').open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=sorted({k for row in flat for k in row}));w.writeheader();w.writerows(flat)
    with (PUBLIC/'paired.csv').open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(value['paired'][0]));w.writeheader();w.writerows(value['paired'])
    print('ANALYZED 1440 retained outputs; 100 cells and 50 paired rows; no new inference/training')


def verify():
    r = read(PUBLIC/'results.json'); receipt = read(PUBLIC/'results_receipt.json')
    if sha_file(PUBLIC/'results.json') != receipt['sha256'] or r != compute(independent=True):
        raise ValueError('Separate generated-error replay differs')
    if read(PUBLIC/'plan.json')['created_utc'] > receipt['created_utc']:
        raise ValueError('Freeze must precede analysis')
    # Validate all serialized CSV values, including empty-bin nulls and histograms.
    with (PUBLIC/'cells.csv').open(encoding='utf-8',newline='') as f:
        cells = list(csv.DictReader(f))
    if len(cells) != len(r['rows']):raise ValueError('CSV coverage differs')
    for row, csvrow in zip(r['rows'],cells):
        expected = {k:v for k,v in row.items() if k not in ('teacher','first_wrong_counts')} | {'teacher_'+k:v for k,v in row['teacher'].items()} | {'first_wrong_counts':json.dumps(row['first_wrong_counts'],sort_keys=True)}
        if csvrow != {k:'' if v is None else str(v) for k,v in expected.items()}:raise ValueError('Cell CSV differs')
    with (PUBLIC/'paired.csv').open(encoding='utf-8',newline='') as f:
        if list(csv.DictReader(f)) != [{k:str(v) for k,v in row.items()} for row in r['paired']]:raise ValueError('Paired CSV differs')
    dump(PUBLIC/'verification.json',{'passed':True,'plan_sha256':r['plan_sha256'],'results_sha256':receipt['sha256'],
         'verified_utc':datetime.now(timezone.utc).isoformat(),'retained_outputs_rescored':1440,
         'independent_generated_parser_and_hops':True,'independent_neural_replay':False,
         'scope':'Independent generated semantic parser/hops; shared aggregation; imported teacher counters only'})
    print('PASS: 1440 retained outputs independently parsed/hop-scored; 100 cell and 50 paired CSV rows reproduce. No new model forwards.')


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('freeze','analyze','verify'));a=ap.parse_args()
    {'freeze':freeze,'analyze':analyze,'verify':verify}[a.command]()


if __name__ == '__main__':main()
