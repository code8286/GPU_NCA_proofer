"""Freeze, fit and replay paired exploratory ID learning curves and costs."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from ood_v12.plan import ROOT,read,tokenizer,LEGACY
from ood_v12.data import generate,write_jsonl,read_jsonl,public_input
from adaptation_v13.run import model_from,digest,dataset_audit
from adaptation_v13.training import encode,order,learning_rate
from cache_v17.run import verify_plan as verify_previous
from overfit_v15.core import training_step
from id_audit_v14.metrics import generated_errors,teacher_forced
from ood_v12.inference import greedy
from transfer_v4.common import dump,sha_file,state_sha
from .core import marginal,baseline_predictions,semantic,target_met,crossing,ratios

CONFIG=ROOT/'configs/accel_v18.json';PROTOCOL=ROOT/'docs/ACCEL_V18_PROTOCOL.md'
PREVIOUS=ROOT/'runs/cache_v17_run1'


def data_config(c):
    old=read(ROOT/'configs/ood_v12.json')
    return old|{'data_seed':c['data_seed'],'assessment_graphs_per_cell':0}


def sources():
    return verify_previous(PREVIOUS)['source_sha256']|{p.relative_to(ROOT).as_posix():sha_file(p) for p in sorted((ROOT/'accel_v18').glob('*.py'))}


def cost_inputs(out):
    names={'data_manifest.json':LEGACY.parents[1]/'data/development/manifest.json'}
    for arm in read(CONFIG)['arms']:
        names[f'{arm}_language.json']=LEGACY/f'{arm}_59/language_summary.json'
        if arm!='scratch':names[f'{arm}_ppt.json']=LEGACY/f'{arm}_59/ppt_summary.json'
    for name,path in names.items():
        target=out/'cost_sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    return {f'cost_sources/{name}':sha_file(out/'cost_sources'/name) for name in names}


def costs(folder):
    manifest=read(folder/'cost_sources/data_manifest.json');rep=manifest['replicates']['59'];records=[]
    for arm in read(CONFIG)['arms']:
        lang=read(folder/f'cost_sources/{arm}_language.json')
        ppt=None if arm=='scratch' else read(folder/f'cost_sources/{arm}_ppt.json')
        prepare=sum(v['prepare_seconds'] for v in rep['nca'].values()) if ppt else 0.
        records.append({'arm':arm,'hardware':'imported original GPU run',
          'language_updates':lang['language_updates'],'language_tokens':lang['language_tokens'],
          'language_train_seconds':lang['language_train_seconds'],'language_stage_seconds':lang['language_seconds'],
          'original_cumulative_seconds':lang['cumulative_seconds'],
          'ppt_updates':ppt['ppt_updates'] if ppt else 0,'ppt_tokens':ppt['ppt_tokens'] if ppt else 0,
          'ppt_train_seconds':ppt['train_seconds'] if ppt else 0.,'ppt_stage_seconds':ppt['stage_seconds'] if ppt else 0.,
          'paired_generation_seconds':sum(v['generation_seconds'] for v in rep['nca'].values()) if ppt else 0.,
          'paired_preparation_seconds':prepare,'text_preparation_seconds':manifest['text']['seconds'],
          'language_head_initial_sha256':lang['language_head_initial_sha256'],
          'language_order_sha256':lang['language_train_order_sha256'],'language_schedule_sha256':lang['language_schedule_sha256']})
    for key in ('language_updates','language_tokens','language_head_initial_sha256','language_order_sha256','language_schedule_sha256'):
        if len({r[key] for r in records})!=1:raise ValueError('Imported language stages must be identical: '+key)
    return records


def initial_records():
    p=verify_previous(PREVIOUS)
    return [{'arm':v['arm'],'initial_checkpoint':v['initial_checkpoint'],
             'initial_checkpoint_sha256':v['initial_checkpoint_sha256'],'initial_model_sha256':v['initial_model_sha256']} for v in p['records']]


def freeze(out):
    c=read(CONFIG);records=initial_records();tok=tokenizer();dc=data_config(c)
    arrays,inventory=generate(dc)
    if arrays['assessment'] or len(arrays['train'])!=512 or len(arrays['development'])!=128:raise ValueError('Fixed ID-only sizes required')
    rows={k:arrays[k] for k in ('train','development')};dataset_audit(rows)
    encoded={k:[encode(r,tok,1024) for r in values] for k,values in rows.items()}
    if any(len(tok.encode(r['prompt']))+c['generation_tokens']>1024 for r in rows['development']):raise ValueError('No context truncation')
    modes=marginal(rows['train']);preds=baseline_predictions(rows['development'],modes);base=semantic(rows['development'],preds)
    target_met(base,base,c)  # Validates all four prospective targets are feasible.
    indices=order(512,c['updates'],c['batch'],c['order_seed']);counts=dict.fromkeys(('input_tokens','supervised_targets','padded_positions'),0)
    per={r['id']:{'input_tokens':len(r['input']),'supervised_targets':sum(t!=-100 for t in r['target'])} for r in encoded['train']}
    for batch in indices:
        values=[per[encoded['train'][i]['id']] for i in batch]
        counts['input_tokens']+=sum(v['input_tokens'] for v in values);counts['supervised_targets']+=sum(v['supervised_targets'] for v in values)
        counts['padded_positions']+=len(batch)*max(v['input_tokens'] for v in values)
    for record in records:
        saved=torch.load(ROOT/record['initial_checkpoint'],map_location='cpu',weights_only=True)
        if saved['adaptation_step']!=0 or saved['optimizer']['state'] or state_sha(saved['model'])!=record['initial_model_sha256']:
            raise ValueError('Original complete unadapted starts and empty optimizer required')
    out.mkdir(parents=True,exist_ok=False);hashes={}
    for split,values in rows.items():
        name=f'data/{split}.jsonl';write_jsonl(out/name,values);hashes[name]=sha_file(out/name)
    np.save(out/'order.npy',indices);hashes['order.npy']=sha_file(out/'order.npy')
    dump(out/'baseline.json',{'modes':modes,'semantic':base});write_jsonl(out/'baseline_predictions.jsonl',preds)
    for name in ('baseline.json','baseline_predictions.jsonl'):hashes[name]=sha_file(out/name)
    hashes|=cost_inputs(out);imported=costs(out);dump(out/'costs.json',imported);hashes['costs.json']=sha_file(out/'costs.json')
    (out/'FROZEN_PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    plan={'schema_version':18,'created_utc':datetime.now(timezone.utc).isoformat(),'config':c,'data_config':dc,
          'data_config_base_sha256':sha_file(ROOT/'configs/ood_v12.json'),'source_sha256':sources(),
          'protocol_sha256':sha_file(PROTOCOL),'records':records,'inputs_sha256':hashes,'inventory':inventory,
          'selected_ids':{s:[r['id'] for r in values] for s,values in rows.items()},'per_example_counts':per,
          'expected_counts_per_arm':counts,'sampled_order_sha256':digest([[encoded['train'][i]['id'] for i in batch] for batch in indices]),
          'schedule_sha256':digest([learning_rate(s,c) for s in range(1,c['updates']+1)]),
          'prior_plan_sha256':sha_file(PREVIOUS/'plan.json'),'environment':{'torch':str(torch.__version__),'numpy':np.__version__},
          'original_random_rule_readiness':'still_failed','confirmation_allowed':False,'ood_allowed':False}
    dump(out/'plan.json',plan);dump(out/'plan_receipt.json',{'plan_sha256':sha_file(out/'plan.json')})
    print('FROZEN',sha_file(out/'plan.json'),counts,'baseline cells',base['cells'],flush=True)


def verify_plan(out):
    p=read(out/'plan.json')
    if (sha_file(out/'plan.json')!=read(out/'plan_receipt.json')['plan_sha256'] or p['config']!=read(CONFIG)
        or p['source_sha256']!=sources() or p['records']!=initial_records() or p['protocol_sha256']!=sha_file(PROTOCOL)
        or sha_file(out/'FROZEN_PROTOCOL.md')!=p['protocol_sha256'] or p['prior_plan_sha256']!=sha_file(PREVIOUS/'plan.json')
        or p['data_config_base_sha256']!=sha_file(ROOT/'configs/ood_v12.json') or p['data_config']!=data_config(p['config'])):
        raise ValueError('Frozen source/config/data policy/starts/protocol changed')
    for name,h in p['inputs_sha256'].items():
        if sha_file(out/name)!=h:raise ValueError('Frozen input changed: '+name)
    if costs(out)!=read(out/'costs.json'):raise ValueError('Imported cost extraction differs')
    return p


def measure(model,tok,rows,c):
    pred=greedy(model,tok,[public_input(r) for r in rows],c['generation_tokens'])
    return {'generated':generated_errors(rows,pred),'teacher_forced':teacher_forced(model,tok,rows,c['batch'])},semantic(rows,pred),pred


def pipeline_at(cross,cost):
    if not cross['reached']:return {'reached':False,'total_tokens':None,'total_updates':None,'accounted_mixed_hardware_seconds':None,'reason':'right_censored'}
    return {'reached':True,'total_tokens':cost['ppt_tokens']+cost['language_tokens']+cross['counts']['input_tokens'],
            'total_updates':cost['ppt_updates']+cost['language_updates']+cross['step'],
            'accounted_mixed_hardware_seconds':cost['original_cumulative_seconds']+cross['elapsed_seconds'],
            'reason':'component account only; imported GPU + new CPU, not controlled throughput'}


def contrasts(models):
    table={m['arm']:m for m in models};g=table['graph_nca']
    return [{'treatment':'graph_nca','reference':arm,**ratios(g['crossing'],table[arm]['crossing'])} for arm in ('grid_nca','prefix_graph','scratch','prefix_grid')]


def fit(out):
    p=verify_plan(out);c=p['config'];base=read(out/'baseline.json')['semantic'];tok=tokenizer()
    if (out/'fit_started.json').exists():raise FileExistsError('Preserve first attempt; no silent restart')
    dump(out/'fit_started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha_file(out/'plan.json')})
    rows={s:read_jsonl(out/f'data/{s}.jsonl') for s in ('train','development')};encoded=[encode(r,tok,1024) for r in rows['train']]
    indices=np.load(out/'order.npy',allow_pickle=False)
    if not np.array_equal(indices,order(512,c['updates'],c['batch'],c['order_seed'])):raise ValueError('Actual shared order changed')
    models=[]
    for record,cost in zip(p['records'],read(out/'costs.json')):
        arm=record['arm'];folder=out/'models'/arm;folder.mkdir(parents=True,exist_ok=False)
        saved=torch.load(ROOT/record['initial_checkpoint'],map_location='cpu',weights_only=True);model=model_from(saved)
        if state_sha(model.state_dict())!=record['initial_model_sha256']:raise ValueError('Starting tensors differ')
        opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=c['weight_decay'])
        counts=dict.fromkeys(p['expected_counts_per_arm'],0);train_seconds=eval_seconds=0.;curve=[];trace=[];files={};start=time.perf_counter()
        for step in range(c['updates']+1):
            if step:
                tick=time.perf_counter();update=training_step(model,opt,[encoded[i] for i in indices[step-1]],step,c)
                train_seconds+=time.perf_counter()-tick
                for key in counts:counts[key]+=update[key]
                trace.append({'step':step,**update,'cumulative_counts':dict(counts)})
            if step in c['evaluation_updates']:
                path=folder/f'step_{step}.pt';torch.save({'config':saved['config'],'vocab':saved['vocab'],'seed':saved['seed'],'arm':arm,
                  'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'counts':dict(counts),'plan_sha256':sha_file(out/'plan.json'),
                  'initial_checkpoint_sha256':record['initial_checkpoint_sha256']},path);files[path.relative_to(out).as_posix()]=sha_file(path)
                tick=time.perf_counter();metrics,sem,pred=measure(model,tok,rows['development'],c)
                name=f'models/{arm}/development_{step}.jsonl';write_jsonl(out/name,pred);files[name]=sha_file(out/name)
                eval_seconds+=time.perf_counter()-tick
                entry={'step':step,'counts':dict(counts),'training_seconds':train_seconds,'evaluation_seconds':eval_seconds,
                       'elapsed_seconds':time.perf_counter()-start,'metrics':metrics,'semantic':sem,'target_met':target_met(sem,base,c)}
                curve.append(entry);dump(folder/'progress.json',{'arm':arm,'step':step,'target_met':entry['target_met'],
                   'node_accuracy':metrics['generated']['node_accuracy'],'exact_match':sem['exact_match'],'cell_non_source':[v['accuracy'] for v in sem['cells']]})
                print(arm,step,'target',entry['target_met'],'node',metrics['generated']['node_accuracy'],flush=True)
        if counts!=p['expected_counts_per_arm']:raise ValueError('Exposure mismatch')
        name=f'models/{arm}/updates.jsonl';write_jsonl(out/name,trace);files[name]=sha_file(out/name)
        cross=crossing(curve,base,c);summary={'arm':arm,'curve':curve,'crossing':cross,'sustained_crossing':crossing(curve,base,c,True),
          'pipeline_at_crossing':pipeline_at(cross,cost),'cost':cost,'counts':counts,'training_seconds':train_seconds,'evaluation_seconds':eval_seconds,
          'files_sha256':files,'sampled_order_sha256':digest([v['sampled_ids'] for v in trace]),'schedule_sha256':digest([v['lr'] for v in trace])}
        dump(folder/'summary.json',summary);models.append(summary);del model,opt,saved
    result={'plan_sha256':sha_file(out/'plan.json'),'models':models,'contrasts':contrasts(models),'new_scientific_updates':3000,
      'new_language_or_ppt_updates':0,'new_ood_predictions':0,'confirmation_allowed':False,'scope':'Exploratory single historic cohort/new ID sample; original confirmation gates unchanged'}
    dump(out/'results.json',result);dump(out/'results_receipt.json',{'sha256':sha_file(out/'results.json')});verify_plan(out)
    print('COMPLETE exploratory ID curves; censored',sum(not m['crossing']['reached'] for m in models),'of5',flush=True)


def verify(out):
    p=verify_plan(out);r=read(out/'results.json');c=p['config'];tok=tokenizer();base=read(out/'baseline.json')['semantic']
    if sha_file(out/'results.json')!=read(out/'results_receipt.json')['sha256']:raise ValueError('Results changed')
    rows={s:read_jsonl(out/f'data/{s}.jsonl') for s in ('train','development')};dataset_audit(rows)
    modes=marginal(rows['train']);bp=baseline_predictions(rows['development'],modes)
    if {'modes':modes,'semantic':semantic(rows['development'],bp)}!=read(out/'baseline.json') or bp!=read_jsonl(out/'baseline_predictions.jsonl'):
        raise ValueError('Training-only marginal baseline differs')
    indices=order(512,c['updates'],c['batch'],c['order_seed']);expected_order=[[rows['train'][i]['id'] for i in batch] for batch in indices]
    total=checkpoints=0
    for record,m,cost in zip(p['records'],r['models'],read(out/'costs.json')):
        arm=record['arm'];folder=out/'models'/arm
        if m['arm']!=arm or m['cost']!=cost:raise ValueError('Arms/cost identities differ')
        for name,h in m['files_sha256'].items():
            if sha_file(out/name)!=h:raise ValueError('Artifact changed')
        trace=read_jsonl(folder/'updates.jsonl');counts=dict.fromkeys(p['expected_counts_per_arm'],0)
        if [v['sampled_ids'] for v in trace]!=expected_order or m['sampled_order_sha256']!=p['sampled_order_sha256'] or m['schedule_sha256']!=p['schedule_sha256']:
            raise ValueError('Actual order/schedule differs')
        for step,v in enumerate(trace,1):
            values=[p['per_example_counts'][key] for key in v['sampled_ids']]
            expected={'input_tokens':sum(x['input_tokens'] for x in values),'supervised_targets':sum(x['supervised_targets'] for x in values),
                      'padded_positions':len(values)*max(x['input_tokens'] for x in values)}
            if v['step']!=step or v['lr']!=learning_rate(step,c):raise ValueError('Update/LR differs')
            for key in counts:
                counts[key]+=expected[key]
                if v[key]!=expected[key] or v['cumulative_counts'][key]!=counts[key]:raise ValueError('Actual tokens differ')
        if counts!=m['counts'] or counts!=p['expected_counts_per_arm']:raise ValueError('Final budget differs')
        for entry in m['curve']:
            step=entry['step'];saved=torch.load(folder/f'step_{step}.pt',map_location='cpu',weights_only=True)
            ec=trace[step-1]['cumulative_counts'] if step else dict.fromkeys(counts,0)
            if saved['step']!=step or saved['counts']!=ec or entry['counts']!=ec or saved['plan_sha256']!=sha_file(out/'plan.json') or saved['initial_checkpoint_sha256']!=record['initial_checkpoint_sha256']:
                raise ValueError('Checkpoint receipt differs')
            if step:
                if {int(v['step']) for v in saved['optimizer']['state'].values()}!={step} or len(saved['optimizer']['state'])!=len(saved['model']):raise ValueError('Full optimizer budget differs')
            elif saved['optimizer']['state'] or state_sha(saved['model'])!=record['initial_model_sha256']:raise ValueError('Original start differs')
            model=model_from(saved);metrics,sem,pred=measure(model,tok,rows['development'],c)
            if metrics!=entry['metrics'] or sem!=entry['semantic'] or pred!=read_jsonl(folder/f'development_{step}.jsonl') or target_met(sem,base,c)!=entry['target_met']:
                raise ValueError('Neural prediction/teacher/semantic replay differs')
            total+=len(pred);checkpoints+=1;del model,saved
        if m['crossing']!=crossing(m['curve'],base,c) or m['sustained_crossing']!=crossing(m['curve'],base,c,True) or m['pipeline_at_crossing']!=pipeline_at(m['crossing'],cost):raise ValueError('Crossing/cost differs')
        print('replayed',arm,flush=True)
    if total!=4480 or checkpoints!=35 or r['contrasts']!=contrasts(r['models']):raise ValueError('Coverage/contrasts differ')
    dump(out/'verification.json',{'passed':True,'checkpoints':checkpoints,'predictions':total,'results_sha256':sha_file(out/'results.json'),
         'scope':'Separate same-code neural/teacher replay; original timings retained, not retraining'})
    print('VERIFIED35 checkpoints and4480 predictions/costs/decisions',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('freeze','fit','verify'));ap.add_argument('--out',type=Path,default=ROOT/'runs/accel_v18_run1');a=ap.parse_args()
    torch.set_num_threads(read(CONFIG)['threads']);torch.use_deterministic_algorithms(True);globals()[a.command](a.out)


if __name__=='__main__':main()
