"""Reuse the original numerical procedure; aggregate the imported reference."""
from collections import Counter,defaultdict
import torch
from adaptation_v13.training import collate,answer_loss,learning_rate
from id_audit_v14.metrics import generated_errors


def training_step(model,optimizer,examples,step,c):
    model.train()
    x,y=collate(examples)
    lr=learning_rate(step,c)
    for group in optimizer.param_groups: group['lr']=lr
    optimizer.zero_grad(set_to_none=True)
    loss=answer_loss(model,x,y)
    if not torch.isfinite(loss): raise ValueError('Nonfinite loss; preserve failed fit')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
    optimizer.step()
    return {'loss':float(loss.detach()),'gradient_norm':float(norm),'lr':lr,
            'input_tokens':sum(len(r['input']) for r in examples),'supervised_targets':int((y!=-100).sum()),
            'padded_positions':x.numel(),'sampled_ids':[r['id'] for r in examples]}


def subset_teacher(rows,original):
    index={r['id']:r for r in original['cases']}
    if len(index)!=len(original['cases']) or not {r['id'] for r in rows}<=set(index):
        raise ValueError('Reference teacher cases must cover every selected identity exactly once')
    cases=[index[r['id']] for r in rows]
    counts=defaultdict(Counter)
    for row,case in zip(rows,cases):
        counts[row['task'],row['graph']['n']].update({k:v for k,v in case.items() if k!='id'})
    cells=[]
    for (task,n),values in sorted(counts.items()):
        cell={'task':task,'n':n,**dict(values)}
        for prefix in ('','digit_','other_'):
            count=values[prefix+'targets']
            cell[prefix+'accuracy']=values[prefix+'correct']/count if count else None
            cell[prefix+'nll']=values[prefix+'nll_sum']/count if count else None
        cells.append(cell)
    return {'cases':cases,'cells':cells}


def reference_probe(rows,predictions,teacher):
    index={r['id']:r for r in predictions}
    if len(index)!=len(predictions) or not {r['id'] for r in rows}<=set(index):
        raise ValueError('Reference free predictions must cover selected identities')
    selected=[index[r['id']] for r in rows]
    return {'generated':generated_errors(rows,selected),'teacher_forced':subset_teacher(rows,teacher)},selected


def competent(g,c):
    return (len(g['cells'])==4 and all(r['exact_match']>=c['id_exact_match_target'] for r in g['cells'])
            and g['valid_format']>=c['id_format_target'] and g['coverage']==1)


def first_crossing(curve,c):
    if [r['step'] for r in curve]!=c['evaluation_updates']:
        raise ValueError('Retain all fixed cache evaluations')
    for row in curve:
        if row['metrics']['generated']['exact_match']>=c['id_exact_match_target']:
            return {'reached':True,'step':row['step'],'input_tokens':row['counts']['input_tokens'],
                    'supervised_targets':row['counts']['supervised_targets'],'training_seconds':row['training_seconds']}
    return {'reached':False,'step':None,'input_tokens':None,'supervised_targets':None,'training_seconds':None}
