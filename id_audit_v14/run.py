"""Freeze, score and replay five existing final models; zero optimizer updates."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time
import torch
from transfer_v4.common import dump, sha_file, state_sha
from adaptation_v13.run import verify_plan as verify_prior, model_from, digest
from ood_v12.plan import ROOT, read, tokenizer
from ood_v12.data import read_jsonl, write_jsonl, public_input
from ood_v12.graphs import floyd
from ood_v12.inference import greedy
from .metrics import select_training, covariates, generated_errors, teacher_forced, interpret

PRIOR = ROOT/'runs/adaptation_v13_run1'
CONFIG = ROOT/'configs/id_audit_v14.json'
PROTOCOL = ROOT/'docs/ID_AUDIT_V14_PROTOCOL.md'


def sources():
    prior = verify_prior(PRIOR)['source_sha256']
    return prior | {p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted((ROOT/'id_audit_v14').glob('*.py'))}


def prior_records(c):
    old = verify_prior(PRIOR)
    r,v = read(PRIOR/'results.json'),read(PRIOR/'verification.json')
    if (sha_file(PRIOR/'results.json') != read(PRIOR/'results_receipt.json')['sha256']
            or not v['passed'] or v['results_sha256'] != sha_file(PRIOR/'results.json')
            or v['plan_sha256'] != sha_file(PRIOR/'plan.json') or r['id_calibration_passed']):
        raise ValueError('Require unchanged verified historical failed calibration')
    records = []
    exposure = None
    for arm in c['arms']:
        summary = next(m for m in r['models'] if m['arm'] == arm)
        folder = PRIOR/'models'/arm
        trace = folder/'updates.jsonl'
        if sha_file(trace) != summary['files_sha256'][f'models/{arm}/updates.jsonl']:
            raise ValueError('Original exposure trace changed')
        sampled = [v['sampled_ids'] for v in read_jsonl(trace)]
        if digest(sampled) != old['sampled_order_sha256'] or len(sampled) != c['endpoint']:
            raise ValueError('Original matched sample order differs')
        counts = Counter(i for batch in sampled for i in batch)
        if exposure is not None and exposure != counts:
            raise ValueError('Training exposures differ between arms')
        exposure = counts
        path = folder/f"step_{c['endpoint']}.pt"
        pred = folder/f"predictions_{c['endpoint']}.jsonl"
        for file in (path,pred):
            if sha_file(file) != summary['files_sha256'][file.relative_to(PRIOR).as_posix()]:
                raise ValueError('Fixed historical endpoint changed')
        records.append({'arm':arm,'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha_file(path),
                        'development_predictions':pred.relative_to(ROOT).as_posix(),'development_predictions_sha256':sha_file(pred)})
    return records,dict(exposure)


def freeze(out):
    c = read(CONFIG)
    records,exposure = prior_records(c)
    train = read_jsonl(PRIOR/'data/train.jsonl')
    probes = {'train':select_training(train,c),'development':read_jsonl(PRIOR/'data/development.jsonl')}
    inventory = {s:covariates(rows) for s,rows in probes.items()}
    for split,rows in probes.items():
        if len(rows) != 128 or any(cell['examples'] != 32 for cell in inventory[split]) or len(inventory[split]) != 4:
            raise ValueError('Require four matched 32-example cells per split')
        for row in rows:
            g = row['graph']
            if (row['truth'] != floyd(g['n'],g['edges'],g['source'],row['task']=='dijkstra_distances')
                    or (exposure.get(row['id'],0) <= 0 if split == 'train' else exposure.get(row['id'],0) != 0)):
                raise ValueError('Independent label or original seen/held-out identity failed')
    tok = tokenizer()
    if any(len(tok.encode(r['prompt']))+c['generation_tokens'] > 1024 for rows in probes.values() for r in rows):
        raise ValueError('Complete input and generation budget must fit')
    out.mkdir(parents=True,exist_ok=False)
    for split,rows in probes.items():
        write_jsonl(out/f'data/{split}.jsonl',rows)
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    plan = {'schema_version':14,'created_utc':datetime.now(timezone.utc).isoformat(),'config':c,
            'source_sha256':sources(),'protocol_sha256':sha_file(PROTOCOL),
            'prior_plan_sha256':sha_file(PRIOR/'plan.json'),'prior_results_sha256':sha_file(PRIOR/'results.json'),
            'prior_verification_sha256':sha_file(PRIOR/'verification.json'),'checkpoints':records,
            'inputs_sha256':{f'data/{s}.jsonl':sha_file(out/f'data/{s}.jsonl') for s in probes},
            'selected_ids':{s:[r['id'] for r in rows] for s,rows in probes.items()},'inventory':inventory,
            'original_training_exposures':{r['id']:exposure[r['id']] for r in probes['train']},
            'environment':{'torch':str(torch.__version__)},'new_optimizer_updates':0,
            'scope':'Post-stage13 development checkpoint-only matched seen/held-out diagnostic; no confirmation or new fitting'}
    dump(out/'plan.json',plan)
    dump(out/'plan_receipt.json',{'plan_sha256':sha_file(out/'plan.json')})
    print('FROZEN',sha_file(out/'plan.json'),'; 5 fixed final models; 128 train +128 development; zero updates',flush=True)


def verify_plan(out):
    p = read(out/'plan.json')
    if (sha_file(out/'plan.json') != read(out/'plan_receipt.json')['plan_sha256'] or p['config'] != read(CONFIG)
            or p['source_sha256'] != sources() or p['protocol_sha256'] != sha_file(PROTOCOL)
            or p['protocol_sha256'] != sha_file(out/'FROZEN_PROTOCOL.md')):
        raise ValueError('Frozen plan/config/source/protocol changed')
    records, exposure = prior_records(p['config'])
    if (p['checkpoints'] != records or p['prior_plan_sha256'] != sha_file(PRIOR/'plan.json')
            or p['prior_results_sha256'] != sha_file(PRIOR/'results.json')
            or p['prior_verification_sha256'] != sha_file(PRIOR/'verification.json')):
        raise ValueError('Prior evidence changed')
    for name,checksum in p['inputs_sha256'].items():
        if sha_file(out/name) != checksum:
            raise ValueError('Frozen matched probes changed')
    if any(exposure.get(i) != n for i,n in p['original_training_exposures'].items()):
        raise ValueError('Prior exposure counts changed')
    return p


def evaluate_record(out,record,p):
    tok = tokenizer()
    saved = torch.load(ROOT/record['checkpoint'],map_location='cpu',weights_only=True)
    if (saved['arm'] != record['arm'] or saved['adaptation_step'] != p['config']['endpoint']
            or saved['plan_sha256'] != p['prior_plan_sha256']
            or {int(s['step']) for s in saved['optimizer']['state'].values()} != {600}):
        raise ValueError('Fixed endpoint/model/optimizer identity differs')
    model = model_from(saved)
    before = state_sha(model.state_dict())
    result,predictions = {},{}
    for split in ('train','development'):
        rows = read_jsonl(out/f'data/{split}.jsonl')
        predictions[split] = greedy(model,tok,[public_input(r) for r in rows],p['config']['generation_tokens'])
        if split == 'development' and predictions[split] != read_jsonl(ROOT/record['development_predictions']):
            raise ValueError('Original development free generations changed')
        result[split] = {'generated':generated_errors(rows,predictions[split]),
                         'teacher_forced':teacher_forced(model,tok,rows,p['config']['batch'])}
    if before != state_sha(model.state_dict()):
        raise ValueError('Read-only audit changed model tensors')
    result['interpretation'] = interpret(result['train']['generated'],result['development']['generated'],p['config'])
    train = {(r['task'],r['n']):r for r in result['train']['generated']['cells']}
    train_tf = {(r['task'],r['n']):r for r in result['train']['teacher_forced']['cells']}
    dev_tf = {(r['task'],r['n']):r for r in result['development']['teacher_forced']['cells']}
    result['matched_differences'] = [
        {'task':r['task'],'n':r['n'],
         **{'train_minus_development_'+k:train[r['task'],r['n']][k]-r[k] for k in ('exact_match','valid_format','node_accuracy')},
         'development_minus_train_nll':dev_tf[r['task'],r['n']]['nll']-train_tf[r['task'],r['n']]['nll'],
         'train_minus_development_digit_accuracy':train_tf[r['task'],r['n']]['digit_accuracy']-dev_tf[r['task'],r['n']]['digit_accuracy']}
        for r in result['development']['generated']['cells']]
    return result,predictions


def evaluate(out):
    p = verify_plan(out)
    if (out/'evaluation_started.json').exists():
        raise FileExistsError('Preserve first prediction attempt; no silent overwrite')
    dump(out/'evaluation_started.json',{'plan_sha256':sha_file(out/'plan.json'),'started_utc':datetime.now(timezone.utc).isoformat()})
    records = []
    for record in p['checkpoints']:
        tick = time.perf_counter()
        metrics,predictions = evaluate_record(out,record,p)
        files = {}
        for split,values in predictions.items():
            name = f"models/{record['arm']}/{split}_predictions.jsonl"
            write_jsonl(out/name,values); files[name] = sha_file(out/name)
        name = f"models/{record['arm']}/metrics.json"
        dump(out/name,metrics); files[name] = sha_file(out/name)
        row = {'arm':record['arm'],'files_sha256':files,'metrics':metrics,'inference_scoring_seconds':time.perf_counter()-tick}
        records.append(row)
        print(record['arm'], 'train/dev exact', metrics['train']['generated']['exact_match'],metrics['development']['generated']['exact_match'],
              'node',metrics['train']['generated']['node_accuracy'],metrics['development']['generated']['node_accuracy'],
              metrics['interpretation']['interpretation'],flush=True)
    dump(out/'results.json',{'plan_sha256':sha_file(out/'plan.json'),'models':records,'new_optimizer_updates':0,
                            'original_random_rule_readiness':'still_failed','ood_evaluation_allowed':False,'graph_acceleration_claim':None})
    dump(out/'results_receipt.json',{'sha256':sha_file(out/'results.json')})
    verify_plan(out)
    print('COMPLETE: all five matched diagnostics; no new fitting or OOD predictions',flush=True)


def verify(out):
    p = verify_plan(out)
    r = read(out/'results.json')
    if sha_file(out/'results.json') != read(out/'results_receipt.json')['sha256'] or [m['arm'] for m in r['models']] != p['config']['arms']:
        raise ValueError('Results/arm identities differ')
    total = 0
    for record,model in zip(p['checkpoints'],r['models']):
        for name,checksum in model['files_sha256'].items():
            if sha_file(out/name) != checksum:
                raise ValueError('Audit output changed')
        metrics,predictions = evaluate_record(out,record,p)
        if metrics != model['metrics']:
            raise ValueError('Matched errors/teacher scores/interpretation replay differs')
        for split,values in predictions.items():
            if values != read_jsonl(out/f"models/{record['arm']}/{split}_predictions.jsonl"):
                raise ValueError('Free-generated token/text/stop replay differs')
            total += len(values)
        print('replayed',record['arm'],flush=True)
    dump(out/'verification.json',{'passed':True,'models':len(r['models']),'predictions_replayed':total,
                                  'plan_sha256':sha_file(out/'plan.json'),'results_sha256':sha_file(out/'results.json'),
                                  'new_optimizer_updates':0,'scope':'Separate same-code checkpoint replay, not independent neural implementation or fitting'})
    print('VERIFIED',total,'new audit predictions plus all teacher-forced/error metrics',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','evaluate','verify'))
    parser.add_argument('--out',type=Path,default=ROOT/'runs/id_audit_v14_run1')
    a = parser.parse_args()
    torch.set_num_threads(read(CONFIG)['threads'])
    torch.use_deterministic_algorithms(True)
    globals()[a.command](a.out)


if __name__ == '__main__':
    main()
