"""Freeze, fit and replay a one-factor intermediate-cache experiment with historical reference."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import time
import numpy as np
import torch
from transfer_v4.common import dump,sha_file,state_sha
from adaptation_v13.run import model_from,digest
from adaptation_v13.training import encode,order,learning_rate
from id_audit_v14.run import verify_plan as verify_prior
from id_audit_v14.metrics import select_training,covariates,generated_errors,teacher_forced
from ood_v12.plan import ROOT,read,tokenizer
from ood_v12.data import read_jsonl,write_jsonl,public_input
from ood_v12.graphs import floyd
from ood_v12.inference import greedy
from overfit_v15.run import verify_plan as verify_small
from .core import training_step,reference_probe,competent,first_crossing,validate_nested

CONFIG=ROOT/'configs/cache_v17.json'
PROTOCOL=ROOT/'docs/CACHE_V17_PROTOCOL.md'
OLD=ROOT/'runs/adaptation_v13_run1'
AUDIT=ROOT/'runs/id_audit_v14_run1'
SMALL=ROOT/'runs/overfit_v15_run1'


def sources():
    return verify_small(SMALL)['source_sha256'] | {
        p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted((ROOT/'cache_v17').glob('*.py'))}


def prior(c):
    p14=verify_prior(AUDIT)
    r14,v14=read(AUDIT/'results.json'),read(AUDIT/'verification.json')
    if (sha_file(AUDIT/'results.json')!=read(AUDIT/'results_receipt.json')['sha256']
            or not v14['passed'] or v14['results_sha256']!=sha_file(AUDIT/'results.json')
            or v14['plan_sha256']!=sha_file(AUDIT/'plan.json')):
        raise ValueError('Verified unchanged stage 14 evidence required')
    p13,r13=read(OLD/'plan.json'),read(OLD/'results.json')
    p15=verify_small(SMALL);r15,v15=read(SMALL/'results.json'),read(SMALL/'verification.json')
    if (not v15['passed'] or v15['results_sha256']!=sha_file(SMALL/'results.json')
            or sha_file(SMALL/'results.json')!=read(SMALL/'results_receipt.json')['sha256']
            or p15['config']['selection_salt']!=c['selection_salt']):
        raise ValueError('Verified unchanged stage 15 reference required')
    base=read(ROOT/c['adaptation_config'])
    if base!=p13['config'] or c['arms']!=[m['arm'] for m in r13['models']] or c['selection_salt']!=p14['config']['selection_salt']:
        raise ValueError('Same original settings, arms and nested selection policy required')
    records=[]
    for arm in c['arms']:
        m13=next(m for m in r13['models'] if m['arm']==arm)
        m14=next(m for m in r14['models'] if m['arm']==arm)
        start=OLD/f'models/{arm}/step_0.pt'
        if sha_file(start)!=m13['files_sha256'][f'models/{arm}/step_0.pt']:
            raise ValueError('Historical complete initial checkpoint changed')
        reference={name:checksum for name,checksum in m14['files_sha256'].items()}
        if any(sha_file(AUDIT/name)!=checksum for name,checksum in reference.items()):
            raise ValueError('Imported reference evidence changed')
        m15=next(m for m in r15['models'] if m['arm']==arm)
        if read(SMALL/f'models/{arm}/summary.json')!=m15:
            raise ValueError('Small16 summary must match verified results')
        smallfiles={name:h for name,h in m15['files_sha256'].items()
                    if name.endswith('/cache_predictions_600.jsonl') or name.endswith('/development_predictions.jsonl')}
        smallfiles[f'models/{arm}/summary.json']=sha_file(SMALL/f'models/{arm}/summary.json')
        if any(sha_file(SMALL/name)!=h for name,h in smallfiles.items()):
            raise ValueError('Small16 reference prediction identity changed')
        records.append({'arm':arm,'initial_checkpoint':start.relative_to(ROOT).as_posix(),
                        'initial_checkpoint_sha256':sha_file(start),'initial_model_sha256':m13['initial_model_state_sha256'],
                        'reference_files_sha256':reference,'historical_reference_updates':m13['updates'],
                        'historical_reference_counts':m13['counts'],'small_reference_files_sha256':smallfiles,
                        'small_reference_updates':600,'small_reference_counts':m15['counts']})
    identities={str(p.relative_to(ROOT)).replace(chr(92),'/'):sha_file(p) for p in
                (OLD/'plan.json',OLD/'results.json',OLD/'verification.json',AUDIT/'plan.json',AUDIT/'results.json',AUDIT/'verification.json',
                 SMALL/'plan.json',SMALL/'results.json',SMALL/'verification.json')}
    return base,records,identities


def probes(c):
    train=select_training(read_jsonl(OLD/'data/train.jsonl'),c)
    dev=read_jsonl(OLD/'data/development.jsonl')
    if not {r['id'] for r in train}<=set(read(AUDIT/'plan.json')['selected_ids']['train']):
        raise ValueError('Small cache must be nested in the declared stage 14 probe')
    if len(train)!=64 or len(dev)!=128 or any(v['examples']!=16 for v in covariates(train)):
        raise ValueError('Require fixed balanced 64-example cache and original full ID development')
    common=read_jsonl(SMALL/'data/train.jsonl')
    validate_nested(train,common,read(AUDIT/'plan.json')['selected_ids']['train'])
    return {'train':train,'development':dev,'common':common}


def reference_metrics(rows,record):
    arm=record['arm'];original=read(AUDIT/f'models/{arm}/metrics.json')
    result,predictions={},{}
    for split in ('train','development','common'):
        source='train' if split=='common' else split
        result[split],predictions[split]=reference_probe(rows[split],read_jsonl(AUDIT/f'models/{arm}/{source}_predictions.jsonl'),
                                                       original[source]['teacher_forced'])
    return result,predictions


def small_reference_metrics(rows,record):
    arm=record['arm'];model=read(SMALL/f'models/{arm}/summary.json');result={};predictions={}
    for split,name,metrics in (('common','cache_predictions_600.jsonl',model['curve'][-1]['metrics']),
                               ('development','development_predictions.jsonl',model['development'])):
        result[split],predictions[split]=reference_probe(rows[split],read_jsonl(SMALL/f'models/{arm}/{name}'),metrics['teacher_forced'])
    return result,predictions


def freeze(out):
    c=read(CONFIG);base,records,identities=prior(c)
    rows=probes(c);tok=tokenizer()
    encoded=[encode(r,tok,1024) for r in rows['train']]
    indices=order(64,base['updates'],base['batch'],base['order_seed'])
    exposures=Counter(encoded[i]['id'] for batch in indices for i in batch)
    if set(exposures.values())!={75}: raise ValueError('Every small-cache example must receive exactly 75 presentations')
    for split,values in rows.items():
        for r in values:
            g=r['graph']
            if r['truth']!=floyd(g['n'],g['edges'],g['source'],r['task']=='dijkstra_distances'):
                raise ValueError('Independent selected-data oracle differs')
            if len(tok.encode(r['prompt']))+base['generation_tokens']>1024:
                raise ValueError('Whole input and generation budget must fit')
    for record in records:
        saved=torch.load(ROOT/record['initial_checkpoint'],map_location='cpu',weights_only=True)
        if (saved['adaptation_step']!=0 or saved['optimizer']['state'] or saved['arm']!=record['arm']
                or state_sha(saved['model'])!=record['initial_model_sha256']):
            raise ValueError('Must start at unchanged full pre-adaptation model with empty optimizer')
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    for split,values in rows.items():
        name=f'data/{split}.jsonl';write_jsonl(out/name,values);hashes[name]=sha_file(out/name)
    np.save(out/'order.npy',indices);hashes['order.npy']=sha_file(out/'order.npy')
    for record in records:
        metrics,predictions=reference_metrics(rows,record)
        name=f"reference/{record['arm']}/metrics.json";dump(out/name,metrics);hashes[name]=sha_file(out/name)
        for split,values in predictions.items():
            name=f"reference/{record['arm']}/{split}_predictions.jsonl";write_jsonl(out/name,values);hashes[name]=sha_file(out/name)
        metrics,predictions=small_reference_metrics(rows,record)
        name=f"reference16/{record['arm']}/metrics.json";dump(out/name,metrics);hashes[name]=sha_file(out/name)
        for split,values in predictions.items():
            name=f"reference16/{record['arm']}/{split}_predictions.jsonl";write_jsonl(out/name,values);hashes[name]=sha_file(out/name)
    counts=dict.fromkeys(('input_tokens','supervised_targets','padded_positions'),0)
    per_example={r['id']:{'input_tokens':len(r['input']),'supervised_targets':sum(t!=-100 for t in r['target'])} for r in encoded}
    for batch in indices:
        values=[per_example[encoded[i]['id']] for i in batch]
        counts['input_tokens']+=sum(v['input_tokens'] for v in values)
        counts['supervised_targets']+=sum(v['supervised_targets'] for v in values)
        counts['padded_positions']+=len(batch)*max(v['input_tokens'] for v in values)
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    plan={'schema_version':17,'created_utc':datetime.now(timezone.utc).isoformat(),'config':c,'adaptation_config':base,
          'base_config_sha256':sha_file(ROOT/c['adaptation_config']),'source_sha256':sources(),'protocol_sha256':sha_file(PROTOCOL),
          'prior_evidence_sha256':identities,'records':records,'inputs_sha256':hashes,
          'selected_ids':{s:[r['id'] for r in v] for s,v in rows.items()},'inventory':{s:covariates(v) for s,v in rows.items()},
          'exposures_per_example':dict(exposures),'per_example_counts':per_example,'expected_counts_per_arm':counts,
          'sampled_order_sha256':digest([[encoded[i]['id'] for i in batch] for batch in indices]),
          'schedule_sha256':digest([learning_rate(s,base) for s in range(1,base['updates']+1)]),
          'environment':{'torch':str(torch.__version__),'numpy':np.__version__},'ood_evaluation_allowed':False,
          'scope':'One historical cohort; new64 CPU task fitting with imported16/512 references; common16/dev128 three-way comparison, not graph acceleration'}
    dump(out/'plan.json',plan);dump(out/'plan_receipt.json',{'plan_sha256':sha_file(out/'plan.json')})
    print('FROZEN',sha_file(out/'plan.json'),'; 5 x600 updates;',counts,'per arm; each example 75 times',flush=True)


def verify_plan(out):
    p=read(out/'plan.json');base,records,identities=prior(read(CONFIG))
    if (sha_file(out/'plan.json')!=read(out/'plan_receipt.json')['plan_sha256'] or p['config']!=read(CONFIG)
            or p['adaptation_config']!=base or p['base_config_sha256']!=sha_file(ROOT/p['config']['adaptation_config'])
            or p['source_sha256']!=sources() or p['protocol_sha256']!=sha_file(PROTOCOL)
            or p['protocol_sha256']!=sha_file(out/'FROZEN_PROTOCOL.md') or p['records']!=records or p['prior_evidence_sha256']!=identities):
        raise ValueError('Frozen config/source/protocol/start/reference identity changed')
    for name,h in p['inputs_sha256'].items():
        if sha_file(out/name)!=h: raise ValueError('Frozen data/order/reference copy changed')
    return p


def measure(model,tok,rows,c):
    predictions=greedy(model,tok,[public_input(r) for r in rows],c['generation_tokens'])
    return {'generated':generated_errors(rows,predictions),'teacher_forced':teacher_forced(model,tok,rows,c['batch'])},predictions


def fit(out):
    p=verify_plan(out);c=p['adaptation_config']
    if (out/'fit_started.json').exists(): raise FileExistsError('Preserve first fitting attempt; no silent restart')
    dump(out/'fit_started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha_file(out/'plan.json')})
    rows={s:read_jsonl(out/f'data/{s}.jsonl') for s in ('train','development','common')}
    tok=tokenizer();encoded=[encode(r,tok,1024) for r in rows['train']]
    indices=np.load(out/'order.npy',allow_pickle=False)
    if not np.array_equal(indices,order(64,c['updates'],c['batch'],c['order_seed'])): raise ValueError('Actual order changed')
    models=[]
    for record in p['records']:
        arm=record['arm'];folder=out/'models'/arm;folder.mkdir(parents=True,exist_ok=False)
        saved=torch.load(ROOT/record['initial_checkpoint'],map_location='cpu',weights_only=True)
        model=model_from(saved)
        if state_sha(model.state_dict())!=record['initial_model_sha256']: raise ValueError('Starting tensors differ')
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=c['weight_decay'])
        counts=dict.fromkeys(p['expected_counts_per_arm'],0);training_seconds=evaluation_seconds=0.
        curve=[];trace=[];files={};development=common=None
        for step in range(c['updates']+1):
            if step:
                tick=time.perf_counter()
                update=training_step(model,optimizer,[encoded[i] for i in indices[step-1]],step,c)
                for k in counts: counts[k]+=update[k]
                training_seconds+=time.perf_counter()-tick
                trace.append({'step':step,**update,'cumulative_counts':dict(counts)})
            if step in c['evaluation_updates']:
                path=folder/f'step_{step}.pt'
                torch.save({'config':saved['config'],'vocab':saved['vocab'],'seed':saved['seed'],'arm':arm,
                            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'counts':dict(counts),
                            'plan_sha256':sha_file(out/'plan.json'),'initial_checkpoint_sha256':record['initial_checkpoint_sha256']},path)
                files[path.relative_to(out).as_posix()]=sha_file(path)
                tick=time.perf_counter();metrics,predictions=measure(model,tok,rows['train'],c)
                name=f'models/{arm}/cache_predictions_{step}.jsonl';write_jsonl(out/name,predictions);files[name]=sha_file(out/name)
                if step==c['updates']:
                    common,_=reference_probe(rows['common'],predictions,metrics['teacher_forced'])
                    development,devpred=measure(model,tok,rows['development'],c)
                    name=f'models/{arm}/development_predictions.jsonl';write_jsonl(out/name,devpred);files[name]=sha_file(out/name)
                evaluation_seconds+=time.perf_counter()-tick
                curve.append({'step':step,'counts':dict(counts),'training_seconds':training_seconds,'evaluation_seconds':evaluation_seconds,'metrics':metrics})
                dump(folder/'progress.json',{'arm':arm,'step':step,'cache_exact':metrics['generated']['exact_match'],
                                            'cache_node_accuracy':metrics['generated']['node_accuracy'],'training_seconds':training_seconds,'evaluation_seconds':evaluation_seconds})
                print(arm,step,'cache exact',metrics['generated']['exact_match'],'node',metrics['generated']['node_accuracy'],flush=True)
        if counts!=p['expected_counts_per_arm']: raise ValueError('Actual exposure differs')
        name=f'models/{arm}/updates.jsonl';write_jsonl(out/name,trace);files[name]=sha_file(out/name)
        summary={'arm':arm,'curve':curve,'development':development,'common':common,'cache_competent':competent(curve[-1]['metrics']['generated'],c),
                 'development_competent':competent(development['generated'],c),'first_cache_crossing':first_crossing(curve,c),
                 'counts':counts,'training_seconds':training_seconds,'evaluation_seconds':evaluation_seconds,'files_sha256':files,
                 'sampled_order_sha256':digest([r['sampled_ids'] for r in trace]),'schedule_sha256':digest([r['lr'] for r in trace])}
        dump(folder/'summary.json',summary);models.append(summary)
        del model,optimizer,saved
    r={'plan_sha256':sha_file(out/'plan.json'),'models':models,'cache64_control_passed':all(m['cache_competent'] for m in models),
       'new_scientific_optimizer_updates':3000,'imported_reference_optimizer_updates':6000,'new_language_or_ppt_updates':0,
       'new_ood_predictions':0,'original_random_rule_readiness':'still_failed','graph_acceleration_claim':None}
    dump(out/'results.json',r);dump(out/'results_receipt.json',{'sha256':sha_file(out/'results.json')});verify_plan(out)
    print('COMPLETE: cache64 criterion',r['cache64_control_passed'],'; OOD still gated',flush=True)


def verify(out):
    p=verify_plan(out);r=read(out/'results.json');c=p['adaptation_config'];tok=tokenizer()
    if sha_file(out/'results.json')!=read(out/'results_receipt.json')['sha256'] or [m['arm'] for m in r['models']]!=p['config']['arms']:
        raise ValueError('Result/arm identities changed')
    rows={s:read_jsonl(out/f'data/{s}.jsonl') for s in ('train','development','common')}
    indices=np.load(out/'order.npy',allow_pickle=False)
    expected_order=[[rows['train'][i]['id'] for i in batch] for batch in indices]
    total=checkpoints=0
    for record,m in zip(p['records'],r['models']):
        arm=record['arm'];folder=out/'models'/arm
        for name,h in m['files_sha256'].items():
            if sha_file(out/name)!=h: raise ValueError('Retained artifact changed')
        trace=read_jsonl(folder/'updates.jsonl')
        if ([v['sampled_ids'] for v in trace]!=expected_order or len(trace)!=c['updates']
                or m['sampled_order_sha256']!=p['sampled_order_sha256'] or m['schedule_sha256']!=p['schedule_sha256']
                or m['counts']!=p['expected_counts_per_arm']): raise ValueError('Matched order/schedule/budget differs')
        counts=dict.fromkeys(p['expected_counts_per_arm'],0)
        for step,update in enumerate(trace,1):
            if update['step']!=step or update['lr']!=learning_rate(step,c): raise ValueError('Actual step/schedule differs')
            values=[p['per_example_counts'][i] for i in update['sampled_ids']]
            expected={'input_tokens':sum(v['input_tokens'] for v in values),'supervised_targets':sum(v['supervised_targets'] for v in values),
                      'padded_positions':len(values)*max(v['input_tokens'] for v in values)}
            for k in counts:
                counts[k]+=expected[k]
                if update[k]!=expected[k] or update['cumulative_counts'][k]!=counts[k]: raise ValueError('Actual token exposure differs')
        for entry in m['curve']:
            step=entry['step'];saved=torch.load(folder/f'step_{step}.pt',map_location='cpu',weights_only=True)
            if saved['step']!=step or saved['plan_sha256']!=sha_file(out/'plan.json') or saved['initial_checkpoint_sha256']!=record['initial_checkpoint_sha256']:
                raise ValueError('Checkpoint receipt differs')
            expected_counts=trace[step-1]['cumulative_counts'] if step else dict.fromkeys(p['expected_counts_per_arm'],0)
            if saved['counts']!=entry['counts'] or saved['counts']!=expected_counts:
                raise ValueError('Checkpoint exposure differs from actual updates')
            if step:
                if {int(v['step']) for v in saved['optimizer']['state'].values()}!={step} or len(saved['optimizer']['state'])!=len(saved['model']):
                    raise ValueError('All full-model parameters must receive all declared updates')
            elif saved['optimizer']['state'] or state_sha(saved['model'])!=record['initial_model_sha256']:
                raise ValueError('Original pre-adaptation state/empty optimizer changed')
            model=model_from(saved);metrics,predictions=measure(model,tok,rows['train'],c)
            if metrics!=entry['metrics'] or predictions!=read_jsonl(folder/f'cache_predictions_{step}.jsonl'):
                raise ValueError('Cache free-generation/teacher scores do not replay')
            total+=len(predictions);checkpoints+=1
            if step==c['updates']:
                common,_=reference_probe(rows['common'],predictions,metrics['teacher_forced'])
                if common!=m['common']: raise ValueError('Common16 subset differs')
                dev,predictions=measure(model,tok,rows['development'],c)
                if dev!=m['development'] or predictions!=read_jsonl(folder/'development_predictions.jsonl'):
                    raise ValueError('Fixed endpoint development replay differs')
                total+=len(predictions)
            del model,saved
        if m['cache_competent']!=competent(m['curve'][-1]['metrics']['generated'],c) or m['development_competent']!=competent(m['development']['generated'],c) or m['first_cache_crossing']!=first_crossing(m['curve'],c):
            raise ValueError('Fixed decisions differ')
        reference,preds=reference_metrics(rows,record)
        if reference!=read(out/f'reference/{arm}/metrics.json'): raise ValueError('Imported same-cache reference differs')
        reference,preds=small_reference_metrics(rows,record)
        if reference!=read(out/f'reference16/{arm}/metrics.json'): raise ValueError('Imported small16 reference differs')
        print('replayed',arm,flush=True)
    if r['cache64_control_passed']!=all(m['cache_competent'] for m in r['models']): raise ValueError('All-arm decision differs')
    dump(out/'verification.json',{'passed':True,'checkpoints':checkpoints,'new_predictions_replayed':total,
                                  'plan_sha256':sha_file(out/'plan.json'),'results_sha256':sha_file(out/'results.json'),
                                  'scope':'Separate same-code checkpoint/teacher replay and imported reference reconstruction; not retraining'})
    print('VERIFIED',checkpoints,'checkpoints;',total,'new generated predictions and imported references',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','fit','verify'))
    parser.add_argument('--out',type=Path,default=ROOT/'runs/cache_v17_run1');a=parser.parse_args()
    torch.set_num_threads(read(ROOT/read(CONFIG)['adaptation_config'])['threads']);torch.use_deterministic_algorithms(True)
    globals()[a.command](a.out)


if __name__=='__main__':main()
