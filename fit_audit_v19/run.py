"""Freeze, evaluate and replay fixed stage18 endpoints; zero optimizer updates."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import time
import torch
from ood_v12.plan import ROOT,read,tokenizer
from ood_v12.data import read_jsonl,write_jsonl
from adaptation_v13.run import model_from
from accel_v18.run import verify_plan as verify_previous,measure
from accel_v18.core import marginal,baseline_predictions,semantic,target_met
from id_audit_v14.metrics import covariates
from transfer_v4.common import dump,sha_file
from .core import select_training,trace_summary,decide

CONFIG=ROOT/'configs/fit_audit_v19.json';PROTOCOL=ROOT/'docs/FIT_AUDIT_V19_PROTOCOL.md'
OLD=ROOT/'runs/accel_v18_run1'


def sources():
    return verify_previous(OLD)['source_sha256']|{p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted((ROOT/'fit_audit_v19').glob('*.py'))}


def prior():
    p=verify_previous(OLD);r=read(OLD/'results.json');v=read(OLD/'verification.json')
    if not v['passed'] or v['checkpoints']!=35 or v['predictions']!=4480 or v['results_sha256']!=sha_file(OLD/'results.json') or sha_file(OLD/'results.json')!=read(OLD/'results_receipt.json')['sha256']:
        raise ValueError('Verified unchanged stage18 required')
    identities={path.relative_to(ROOT).as_posix():sha_file(path) for path in (OLD/'plan.json',OLD/'results.json',OLD/'verification.json',OLD/'baseline.json')}
    records=[]
    for m in r['models']:
        arm=m['arm'];names=[f'models/{arm}/step_600.pt',f'models/{arm}/development_600.jsonl',f'models/{arm}/updates.jsonl']
        for name in names:
            if sha_file(OLD/name)!=m['files_sha256'][name]:raise ValueError('Selected checkpoint/prediction/trace changed')
        records.append({'arm':arm,'checkpoint':(OLD/names[0]).relative_to(ROOT).as_posix(),
                        'checkpoint_sha256':m['files_sha256'][names[0]],'source_files_sha256':{n:m['files_sha256'][n] for n in names}})
    if [v['arm'] for v in records]!=read(CONFIG)['arms']:raise ValueError('Retain all five arms')
    return p,r,records,identities


def freeze(out):
    c=read(CONFIG);old,r,records,identities=prior();full=read_jsonl(OLD/'data/train.jsonl')
    rows={'train':select_training(full,c),'development':read_jsonl(OLD/'data/development.jsonl')}
    if len(rows['train'])!=128 or len(rows['development'])!=128:raise ValueError('Balanced128 probes required')
    modes=read(OLD/'baseline.json')['modes']
    if marginal(full)!=modes:raise ValueError('Preserve the original training-only marginal model')
    baselines={};basepred={}
    for split,values in rows.items():
        pred=baseline_predictions(values,modes);basepred[split]=pred;baselines[split]=semantic(values,pred)
        target_met(baselines[split],baselines[split],old['config'])
    if baselines['development']!=read(OLD/'baseline.json')['semantic']:raise ValueError('Heldout target must remain exactly stage18')
    out.mkdir(parents=True,exist_ok=False);hashes={}
    for split,values in rows.items():
        name=f'data/{split}.jsonl';write_jsonl(out/name,values);hashes[name]=sha_file(out/name)
        name=f'baseline_{split}_predictions.jsonl';write_jsonl(out/name,basepred[split]);hashes[name]=sha_file(out/name)
    dump(out/'baselines.json',{'modes':modes,'probes':baselines});hashes['baselines.json']=sha_file(out/'baselines.json')
    traces={}
    for record,m in zip(records,r['models']):
        arm=m['arm'];saved=torch.load(ROOT/record['checkpoint'],map_location='cpu',weights_only=True)
        if (saved['step']!=600 or saved['counts']!=m['counts'] or saved['plan_sha256']!=sha_file(OLD/'plan.json')
            or {int(v['step']) for v in saved['optimizer']['state'].values()}!={600}
            or len(saved['optimizer']['state'])!=len(saved['model'])):raise ValueError('Original full optimizer/count endpoint required')
        name=f'optimization/{arm}_updates.jsonl';target=out/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(OLD/f'models/{arm}/updates.jsonl',target);hashes[name]=sha_file(target)
        traces[arm]=trace_summary(read_jsonl(target),old['config'],[v['id'] for v in rows['train']])
        name=f'reference/{arm}/development_predictions.jsonl';target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD/f'models/{arm}/development_600.jsonl',target);hashes[name]=sha_file(target)
        name=f'reference/{arm}/metrics.json';dump(out/name,{'generated':m['curve'][-1]['metrics']['generated'],
            'teacher_forced':m['curve'][-1]['metrics']['teacher_forced'],'semantic':m['curve'][-1]['semantic']});hashes[name]=sha_file(out/name)
    dump(out/'optimization_summary.json',traces);hashes['optimization_summary.json']=sha_file(out/'optimization_summary.json')
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    plan={'schema_version':19,'created_utc':datetime.now(timezone.utc).isoformat(),'config':c,'measurement_config':old['config'],
          'protocol_sha256':sha_file(PROTOCOL),'source_sha256':sources(),'records':records,'prior_evidence_sha256':identities,
          'inputs_sha256':hashes,'selected_ids':{s:[v['id'] for v in values] for s,values in rows.items()},
          'covariates':{s:covariates(values) for s,values in rows.items()},'new_optimizer_updates':0,
          'original_random_rule_readiness':'still_failed','ood_allowed':False,'confirmation_allowed':False}
    dump(out/'plan.json',plan);dump(out/'plan_receipt.json',{'plan_sha256':sha_file(out/'plan.json')})
    print('FROZEN',sha_file(out/'plan.json'),'five endpoints;128 seen +128 imported heldout;zero updates',flush=True)


def verify_plan(out):
    p=read(out/'plan.json');old,r,records,identities=prior()
    if (sha_file(out/'plan.json')!=read(out/'plan_receipt.json')['plan_sha256'] or p['config']!=read(CONFIG)
        or p['measurement_config']!=old['config'] or p['source_sha256']!=sources() or p['records']!=records
        or p['prior_evidence_sha256']!=identities or p['protocol_sha256']!=sha_file(PROTOCOL)
        or sha_file(out/'FROZEN_PROTOCOL.md')!=p['protocol_sha256']):raise ValueError('Frozen identity changed')
    for name,h in p['inputs_sha256'].items():
        if sha_file(out/name)!=h:raise ValueError('Frozen probe/reference/trace changed')
    return p


def evaluate(out):
    p=verify_plan(out);c=p['measurement_config'];tok=tokenizer();rows=read_jsonl(out/'data/train.jsonl')
    base=read(out/'baselines.json')['probes']
    if (out/'evaluation_started.json').exists():raise FileExistsError('Preserve original evaluation attempt')
    dump(out/'evaluation_started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha_file(out/'plan.json')})
    models=[]
    for record in p['records']:
        arm=record['arm'];folder=out/'models'/arm;folder.mkdir(parents=True,exist_ok=False)
        saved=torch.load(ROOT/record['checkpoint'],map_location='cpu',weights_only=True);model=model_from(saved);tick=time.perf_counter()
        metrics,sem,pred=measure(model,tok,rows,c);elapsed=time.perf_counter()-tick
        train=metrics|{'semantic':sem};dev=read(out/f'reference/{arm}/metrics.json')
        write_jsonl(folder/'train_predictions.jsonl',pred)
        m={'arm':arm,'train':train,'development':dev,'decision':decide(train,dev,base,c),'evaluation_seconds':elapsed,
           'checkpoint_sha256':record['checkpoint_sha256'],'prediction_sha256':sha_file(folder/'train_predictions.jsonl')}
        dump(folder/'summary.json',m);models.append(m);del saved,model
        print(arm,m['decision'],'train exact',sem['exact_match'],flush=True)
    r={'plan_sha256':sha_file(out/'plan.json'),'models':models,'optimization':read(out/'optimization_summary.json'),
       'new_optimizer_updates':0,'new_predictions':640,'imported_development_predictions':640,'new_ood_predictions':0,
       'original_random_rule_readiness':'still_failed','confirmation_allowed':False}
    dump(out/'results.json',r);dump(out/'results_receipt.json',{'sha256':sha_file(out/'results.json')});verify_plan(out)
    print('COMPLETE fixed endpoint diagnostic;zero optimizer updates',flush=True)


def verify(out):
    p=verify_plan(out);r=read(out/'results.json');c=p['measurement_config'];tok=tokenizer()
    if sha_file(out/'results.json')!=read(out/'results_receipt.json')['sha256']:raise ValueError('Result changed')
    rows=read_jsonl(out/'data/train.jsonl');base=read(out/'baselines.json')['probes'];total=0
    for record,m in zip(p['records'],r['models']):
        arm=record['arm'];saved=torch.load(ROOT/record['checkpoint'],map_location='cpu',weights_only=True)
        if saved['step']!=600 or {int(v['step']) for v in saved['optimizer']['state'].values()}!={600}:raise ValueError('Endpoint optimizer identity differs')
        model=model_from(saved);metrics,sem,pred=measure(model,tok,rows,c)
        if metrics|{'semantic':sem}!=m['train'] or pred!=read_jsonl(out/f'models/{arm}/train_predictions.jsonl') or sha_file(out/f'models/{arm}/train_predictions.jsonl')!=m['prediction_sha256']:
            raise ValueError('New seen-probe predictions/teacher scores do not replay')
        dev=read(out/f'reference/{arm}/metrics.json')
        if dev!=m['development'] or decide(m['train'],dev,base,c)!=m['decision']:raise ValueError('Reference/decision differs')
        if trace_summary(read_jsonl(out/f'optimization/{arm}_updates.jsonl'),c,p['selected_ids']['train'])!=r['optimization'][arm]:raise ValueError('Trace summaries differ')
        total+=len(pred);del model,saved;print('replayed',arm,flush=True)
    if total!=640:raise ValueError('All five complete probes required')
    dump(out/'verification.json',{'passed':True,'checkpoints':5,'new_predictions_replayed':640,
         'results_sha256':sha_file(out/'results.json'),'scope':'Separate same-code five-endpoint seen-probe replay; development imported;no fitting'})
    print('VERIFIED5 endpoints and640 new predictions;zero updates',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('freeze','evaluate','verify'))
    ap.add_argument('--out',type=Path,default=ROOT/'runs/fit_audit_v19_run1');a=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);globals()[a.command](a.out)


if __name__=='__main__':main()
